"""Pure-JAX log-likelihood of the unit cube for one galaxy:

    loglike(u) = FitModel.loglike(FitModel.prior_transform(u))     (u inside the cube)
               = -inf                                              (u outside)

Built FROM the numpy FitModel and its PhotEmulator (weights already sorted to the
photometry's bands, emulator sigma, valid-band mask, observed variance), so every input is
shared with the numpy reference; only the arithmetic is re-written. The network forward
pass is the same as E-FSPS origin/PhotModel (efsps/models/photmodel.py); the likelihood
follows efsps_photfit/model.py (model_fluxes, loglike) line for line.

    ll = JaxLogLike(model)
    ll(u)          # (n_dim,) -> scalar; jit/vmap-able
    ll.batch(u)    # (N, n_dim) -> (N,), jitted vmap
"""
import jax
import jax.numpy as jnp
import numpy as np

from efsps_photfit.emulator import ARSINH_A, ARSINH_B, ARSINH_MU0, REF_LOGMASS
from jax_prior import JaxPriorTransform


# a Python float, not a numpy float64 scalar: the same value, but it keeps x's dtype (a numpy
# scalar would silently promote float32 activations back to float64)
_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))


def _gelu_tanh(x):
    """emulator.gelu_tanh, same order of operations (= jax.nn.gelu approximate=True)."""
    return 0.5 * x * (1.0 + jnp.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)))


EMU_DTYPES = ("float64", "float32")


class JaxLogLike:
    def __init__(self, model, emu_dtype=None):
        """emu_dtype: precision of the network's hidden layers. "float64" (default) is the
        numpy reference's arithmetic, bit for bit. "float32" is the precision the network was
        trained in (its weights are float32 on file): inputs are normalised in float64, cast,
        run through the layers in float32, and the output is cast back before the float64
        likelihood. OPT-IN, for timing: it differs from float64 by ~1e-6 mag (3e-4 at worst),
        ~1e-4 nats in log L at posterior draws (PRODUCTION_PLAN.md task S4), so fits are not
        bitwise equal to float64 ones. None reads the environment variable NSS_EMU_DTYPE."""
        import os
        self.pt = JaxPriorTransform(model)
        emu = model.emu
        if emu.dtype != np.float64:
            raise ValueError("the numpy reference runs the network in float64; so does this.")
        emu_dtype = emu_dtype or os.environ.get("NSS_EMU_DTYPE") or "float64"
        if emu_dtype not in EMU_DTYPES:
            raise ValueError(f"emu_dtype / NSS_EMU_DTYPE must be one of {EMU_DTYPES}, got {emu_dtype!r}")
        self.emu_dtype = emu_dtype
        self._f32 = emu_dtype == "float32"
        if self._f32:
            print("[jax_model] network hidden layers in float32 (NSS_EMU_DTYPE / emu_dtype): "
                  "not bitwise equal to float64 fits", flush=True)
        wd = jnp.float32 if self._f32 else jnp.float64
        self._W = [jnp.asarray(w, wd) for w in emu._W]
        self._B = [jnp.asarray(b, wd) for b in emu._B]
        self._in_mu = jnp.asarray(emu._in_mu)
        self._in_sig = jnp.asarray(emu._in_sig)
        self._out_mu = jnp.asarray(emu._out_mu)
        self._out_sig = jnp.asarray(emu._out_sig)

        # emulator input row: fixed values, then the free columns scattered in
        fixed = np.zeros(emu.n_param)
        emu_cols, theta_cols = [], []
        for j, p in enumerate(emu.parameters):
            if p in model.col:
                emu_cols.append(j)
                theta_cols.append(model.col[p])
            else:
                fixed[j] = model.fixed[p]
        self._fixed = jnp.asarray(fixed)
        self._emu_cols = np.asarray(emu_cols, int)
        self._theta_cols = np.asarray(theta_cols, int)
        self._c_mass = model.col["logmass"]
        self.mu = float(model.mu)

        v = model.valid
        self._valid = np.flatnonzero(v)
        self._sigma_mag = jnp.asarray(model.sigma_mag[v])
        self._fv = jnp.asarray(model._fv)
        self._var_obs = jnp.asarray(model._var_obs)

        self.batch = jax.jit(jax.vmap(self.__call__))

    def f_net(self, theta):
        """emulator.predict_maggies for one row -> maggies per 1e10 Msun, all bands."""
        x = self._fixed.at[self._emu_cols].set(theta[self._theta_cols])
        x = (x - self._in_mu) / self._in_sig
        if self._f32:
            x = x.astype(jnp.float32)
        for W, B in zip(self._W[:-1], self._B[:-1]):
            x = _gelu_tanh(x @ W + B)
        x = x @ self._W[-1] + self._B[-1]
        if self._f32:
            x = x.astype(jnp.float64)
        mag = self._out_sig * x + self._out_mu
        return 2.0 * ARSINH_B * jnp.sinh(-(mag - ARSINH_MU0) / ARSINH_A)

    def loglike_theta(self, theta):
        f_net = self.f_net(theta)[self._valid]
        scale = 10.0 ** (theta[self._c_mass] - REF_LOGMASS) * self.mu
        sig = (self._sigma_mag / ARSINH_A) * jnp.sqrt(f_net ** 2 + 4.0 * ARSINH_B ** 2) * scale
        fm = f_net * scale
        var = self._var_obs + sig ** 2
        r2 = (self._fv - fm) ** 2 / var
        return -0.5 * jnp.sum(r2 + jnp.log(2.0 * np.pi * var))

    def __call__(self, u):
        inside = jnp.all((u >= 0.0) & (u <= 1.0))
        ll = self.loglike_theta(self.pt(jnp.clip(u, 0.0, 1.0)))
        return jnp.where(inside & jnp.isfinite(ll), ll, -jnp.inf)
