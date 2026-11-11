"""Linear logmass coordinate for NSS: the SAME prior, in coordinates the slice sampler can move in.

Standard coordinates: every parameter is a unit-cube coordinate u pushed through its prior's
inverse CDF, so the prior is uniform in the cube. For logmass, the inverse CDF of the
mass-function prior p(logM | z) squeezes real galaxies, which sit in its high-mass tail, into
a sliver next to u = 1 (1 - u ~ 1e-4; logmass grows like log(1 - u)). Mass correlates with
almost every parameter, so near-linear degeneracies in theta become strongly curved in u and
the slice chains barely move (diagnose_nss.py, STATUS.md section 4).

Here the logmass coordinate is instead linear between the prior bounds,
    logmass = lo + (hi - lo) x_m,     x_m in [0, 1],
and the mass prior enters as a log-density, the rest of the cube is unchanged:
    log prior(x) = log p(logmass | z) + log(hi - lo)        (0 outside the cube: -inf)
p(logM | z) is exactly the density the tabulated inverse CDF inverts: the piecewise-linear
interpolation of the two weighted table rows (TabulatedMassPrior.rows_and_weights), divided
by their trapezoid integral. So the distribution of theta is the same as with u, to rounding.
blackjax's slice step compares the candidate's prior log-density with the slice level
(mcmc/slice.py _univariate_slice), so NSS samples this prior inside each likelihood contour.

    lm = LinearMass(model)             # model with the tabulated mass prior
    lm.loglike(x), lm.logprior(x)      # for blackjax.nss
    lm.theta(x)                        # (n_dim,) -> theta
    lm.x_from_u(u)                     # prior draws in the standard cube -> x (numpy, batched)
"""
import jax
import jax.numpy as jnp
import numpy as np

from jax_model import JaxLogLike


class LinearMass:
    def __init__(self, model):
        self.ll = JaxLogLike(model)
        self.pt = self.ll.pt
        if not self.pt._m_tab:
            raise ValueError("linear logmass needs the tabulated mass prior "
                             "(use_tabulated_mass_prior(model)).")
        self.n_dim = model.n_dim
        self.im = model.col["logmass"]
        self.lo, self.hi = float(model.mprior.lo), float(model.mprior.hi)
        self.log_width = float(np.log(self.hi - self.lo))
        self.batch_loglike = jax.jit(jax.vmap(self.loglike))
        self.batch_logprior = jax.jit(jax.vmap(self.logprior))
        self.batch_theta = jax.jit(jax.vmap(self.theta))
        self._batch_u_to_m = jax.jit(jax.vmap(self._u_to_m))

    def _z(self, x):
        pt = self.pt
        if pt._z_fixed is None:
            return pt.z_ppf(x[pt.col["zred"]])
        return jnp.asarray(pt._z_fixed, dtype=x.dtype)

    def logmass(self, x):
        return self.lo + (self.hi - self.lo) * x[self.im]

    def log_mass_prior(self, m, z):
        """log p(logM = m | z): the piecewise-linear density that _mass_ppf_tab inverts."""
        pt = self.pt
        ra, rb, wa, wb = pt.tab_rows_and_weights(z)
        pdf = wa * pt._tab_pdf[ra] + wb * pt._tab_pdf[rb]
        norm = wa * pt._tab_cum[ra, -1] + wb * pt._tab_cum[rb, -1]
        return jnp.log(jnp.interp(m, pt._m_grid, pdf)) - jnp.log(norm)

    def theta(self, x):
        xc = jnp.clip(x, 0.0, 1.0)
        return self.pt(xc, logmass=self.logmass(xc))

    def logprior(self, x):
        inside = jnp.all((x >= 0.0) & (x <= 1.0))
        xc = jnp.clip(x, 0.0, 1.0)
        lp = self.log_mass_prior(self.logmass(xc), self._z(xc)) + self.log_width
        return jnp.where(inside & jnp.isfinite(lp), lp, -jnp.inf)

    def loglike(self, x):
        inside = jnp.all((x >= 0.0) & (x <= 1.0))
        ll = self.ll.loglike_theta(self.theta(x))
        return jnp.where(inside & jnp.isfinite(ll), ll, -jnp.inf)

    def _u_to_m(self, u):
        return self.pt.mass_ppf(u[self.im], self._z(u))

    def x_from_u(self, u):
        """Standard-cube prior draws -> linear-mass coordinates (same theta)."""
        u = np.asarray(u, dtype=float)
        m = np.asarray(self._batch_u_to_m(jnp.asarray(u)))
        x = u.copy()
        x[:, self.im] = (m - self.lo) / (self.hi - self.lo)
        return x
