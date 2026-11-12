"""NSS with stellar mass integrated out of the likelihood, exactly: same model, same prior,
same posterior and evidence; 25 sampled parameters instead of 26.

Why: the photometry pins the flux normalization, i.e. mass x L/M(age, dust, Z, z), to a few
hundredths of a dex. In the 26-d space the likelihood region is a thin sheet curved along
that ridge, and the slice chains barely move (diagnose_nss.py; STATUS.md section 4). With
mass integrated out, the remaining 25-d region is as roomy as an ellipsoid of the live
covariance.

The joint prior is the efsps-photfit (Prospector-beta) one, unchanged:
    pi(z) p(m | z) p(Z | m) prod_j p(r_j | z, m) pi(rest)
  p(m | z)    the tabulated mass-function prior (tabulated_mass_prior.py): the
              piecewise-linear density its inverse CDF inverts
  p(Z | m)    truncated normal, loc / scale from the mass-metallicity table at m
  r_j         = clip(y_j, lo_r, hi_r), with the UNCLIPPED y_j = expected_j(z, m) + o_j and
              o_j a Student-t(nu=2, scale s) truncated to |o_j| <= hw. y_j is sampled
              (not r_j), so the prior's point masses at the clip bounds are kept exactly.
Sampling coordinates x (25-d, unit cube): zred and every mass-independent parameter keep
their standard unit-cube coordinate (inverse CDF); logzsol and y_j are linear in their
physical range (y_j over [min expected_j - hw, max expected_j + hw]).

For each x:
    q(m; x)     = p(m | z) p(Z | m) prod_j p(y_j | z, m) x (constant Jacobians)
    log prior   = log g(x),   g(x) = integral q(m; x) dm               (the marginal prior)
    log like    = log integral L(m, x) q(m; x) dm - log g(x)
L(m, x) needs the emulator once: logmass is not an emulator input (it predicts flux per
1e10 Msun), so only the flux scale 10^(m - REF) changes with m. The m integrals use a
coarse grid over the whole prior range (prior integral, and where the likelihood peaks),
then a fine grid around the peak for the likelihood integral (trapezoid; test_mass_marginal.py
measures the error against brute force). The mass posterior is recovered by drawing m from
L(m, x) q(m; x) for each posterior sample (sample_logmass).

    mm = MassMarginal(model)              # model with the tabulated mass prior
    mm.loglike(x), mm.logprior(x)         # for blackjax.nss
    mm.x_from_u(u)                        # standard 26-d prior draws -> (x, logmass)
    mm.theta(x, logmass)                  # -> 26-d theta
    mm.sample_logmass(key, x)             # one posterior draw of logmass given x

Optional dust prior (dust_prior.py, MassMarginal(model, dust_prior="mass_sfr")): the soft-ceiling
factor w(dust2; m, SFR_100(m)) multiplies L q inside the mass integral (it depends on m through
the ceiling and through SFR_100 = 10^m x f(ratios, z)). It is part of the likelihood, so the
prior draws and log g are unchanged; logZ includes the global constant ln C = ln E_pi[w].
dust_prior=None (default) leaves every computation exactly as before.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

from efsps_photfit.emulator import ARSINH_A, ARSINH_B, REF_LOGMASS
from jax_model import JaxLogLike
from jax_prior import truncated_t2_ppf, truncnorm_ppf, uniform_ppf, loguniform_ppf, simple_ppf

N_COARSE = 111          # 0.05 dex over the 5.5-dex prior range: the prior integral g(x), and
                        # the expected SFH ratios (interpolated to the fine grid)
N_FINE = 41             # the likelihood integral, over +-N_SIG sigma around the peak in mass
N_SIG = 8.0
MIN_HALF = 0.02         # dex; the fine grid never spans less than +-MIN_HALF
N_S_TABLE = 20001       # expected-ratio table in the shifted age s (~1.4 dex: ~7e-5 dex steps)


class MassMarginal:
    def __init__(self, model, n_coarse=N_COARSE, n_fine=N_FINE, dust_prior=None, dust_cfg=None):
        self.ll = JaxLogLike(model)
        pt = self.pt = self.ll.pt
        if not pt._m_tab:
            raise ValueError("needs the tabulated mass prior (use_tabulated_mass_prior).")
        col = model.col
        if "logzsol" not in col or pt._ratio_cols.size == 0:
            raise ValueError("expects free logzsol and SFH ratios.")
        self.n_full = model.n_dim
        self.im = col["logmass"]
        self.keep = np.array([c for c in range(model.n_dim) if c != self.im])   # x[k] <-> theta[keep[k]]
        self.n_dim = self.keep.size
        xi = {c: k for k, c in enumerate(self.keep)}
        self.x_z = xi[col["zred"]] if "zred" in col else None
        self.c_z = col.get("zred")
        self.c_zsol = col["logzsol"]
        self.x_zsol = xi[self.c_zsol]
        self.zsol_lo, self.zsol_hi = (float(v) for v in model.bounds["logzsol"])
        self.c_r = pt._ratio_cols                              # theta columns of the ratios
        self.x_r = np.array([xi[c] for c in self.c_r])
        self.r_j = pt._ratio_j
        self.r_lo, self.r_hi = pt._ratio_lo, pt._ratio_hi
        self.hw, self.ts = pt._hw, pt._ts
        tab = np.asarray(pt._ratio_table)[:, self.r_j]          # expected ratios on z_eff nodes
        self.y_lo = jnp.asarray(tab.min(axis=0) - self.hw)
        self.y_hi = jnp.asarray(tab.max(axis=0) + self.hw)
        self.simple = [(xi[c], c, kind, lo, hi, args) for c, kind, lo, hi, args in pt._simple]
        self.m_lo, self.m_hi = float(model.mprior.lo), float(model.mprior.hi)
        mg = np.asarray(model.mprior.grid)
        self.m_dx = float(mg[1] - mg[0])                       # the mass prior's node spacing
        if not np.allclose(np.diff(mg), self.m_dx) or abs(mg[0] - self.m_lo) > 1e-12:
            raise ValueError("expects the mass prior on a uniform grid starting at its lower bound")
        self.mc = jnp.linspace(self.m_lo, self.m_hi, n_coarse)
        self.wc = self._trap_w(self.mc)
        self.n_fine = int(n_fine)
        self.log_jac = float(np.log(self.zsol_hi - self.zsol_lo)
                             + np.sum(np.log(np.asarray(self.y_hi - self.y_lo))))
        # t2 normalization: D = hw / sqrt(2 s^2 + hw^2) (CDF mass inside +-hw)
        self._log_t2_norm = float(np.log(self.ts ** 2)
                                  - np.log(self.hw / np.sqrt(2 * self.ts ** 2 + self.hw ** 2)))
        self._build_ratio_table()
        # optional soft-ceiling dust prior (dust_prior.py)
        self.dust_prior = dust_prior
        if dust_prior is not None:
            import dust_prior as DPR
            if dust_prior not in DPR.KINDS:
                raise ValueError(f"dust_prior must be one of {DPR.KINDS} or None")
            if "dust2" not in col:
                raise ValueError("the dust prior needs a free dust2")
            self.dust_cfg = dict(DPR.CEILING, **(dust_cfg or {}))
            self.c_dust2 = col["dust2"]
            self.c_r_all = np.array([col[n] for n in model.ratio_names])   # all 9, in order
            self._dpr = DPR
        self.batch_loglike = jax.jit(jax.vmap(self.loglike))
        self.batch_logprior = jax.jit(jax.vmap(self.logprior))
        self.batch_both = jax.jit(jax.vmap(self.logprior_and_loglike))
        self._batch_x_from_u = jax.jit(jax.vmap(self._x_from_u))
        self.batch_theta = jax.jit(jax.vmap(self.theta))

    # ------------------------------------------------------------------ expected-ratio table
    def _build_ratio_table(self, n_s=N_S_TABLE):
        """The expected SFH ratios depend on (z, m) only through the shifted age
        s = log10 age(z) + delta_t_dex(m) (sfh.py ExpectedRatios.z_eff: shift, clip, invert
        age -> z_eff, clamp to Z_SHIFT_CLAMP, interpolate the log z_eff table). Tabulate
        E_j(s) once with that same chain, on a uniform s grid between the ages at the clamp
        redshifts (E is constant outside); test_mass_marginal.py measures the error."""
        pt = self.pt
        from efsps_photfit import sfh as S
        zlo, zhi = S.Z_SHIFT_CLAMP
        s0, s1 = np.log10(pt._age_zmax), np.log10(pt._age_zmin)
        s = jnp.linspace(s0, s1, n_s)

        def exact(si):
            shifted = jnp.clip(10.0 ** si, 0.1, pt._age0 - 0.1)
            ze = pt.z_of_age(shifted)
            ze = jnp.where(shifted <= pt._age_zmax, zhi,
                           jnp.where(shifted >= pt._age_zmin, zlo, ze))
            lz = jnp.log(jnp.clip(ze, zlo, zhi))
            return jax.vmap(lambda col: jnp.interp(lz, pt._log_zeff, col),
                            in_axes=1)(pt._ratio_table)[self.r_j]

        self._s0, self._ds = float(s0), float((s1 - s0) / (n_s - 1))
        self._n_s = int(n_s)
        self._etab = jax.jit(jax.vmap(exact))(s)                   # (n_s, n_r)

    def log10_age(self, z):
        return jnp.log10(self.pt.age(z))

    def exp_ratios_tab(self, log10_age_z, m):
        """Tabulated expected ratios for a vector of m at one z: (M,) -> (M, n_r)."""
        from jax_prior import delta_t_dex
        s = log10_age_z + delta_t_dex(m)
        t = jnp.clip((s - self._s0) / self._ds, 0.0, self._n_s - 1.0)
        i = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, self._n_s - 2)
        w = (t - i)[:, None]
        return (1.0 - w) * self._etab[i] + w * self._etab[i + 1]

    @staticmethod
    def _trap_w(g):
        h = g[1] - g[0]
        return jnp.full(g.shape, h).at[0].set(h / 2).at[-1].set(h / 2)

    # ------------------------------------------------------------------ coordinates
    def _z(self, x):
        if self.x_z is None:
            return jnp.asarray(self.pt._z_fixed, dtype=x.dtype)
        return self.pt.z_ppf(x[self.x_z])

    def zsol(self, x):
        return self.zsol_lo + (self.zsol_hi - self.zsol_lo) * x[self.x_zsol]

    def y(self, x):
        return self.y_lo + (self.y_hi - self.y_lo) * x[self.x_r]

    def theta(self, x, logmass):
        """26-d theta for the network and the outputs (mass-independent given x and m)."""
        th = jnp.zeros(self.n_full, dtype=x.dtype)
        z = self._z(x)
        if self.c_z is not None:
            th = th.at[self.c_z].set(z)
        th = th.at[self.im].set(logmass)
        th = th.at[self.c_zsol].set(self.zsol(x))
        th = th.at[self.c_r].set(jnp.clip(self.y(x), self.r_lo, self.r_hi))
        for k, c, kind, lo, hi, args in self.simple:
            th = th.at[c].set(simple_ppf(kind, x[k], lo, hi, args))
        return th

    def _x_from_u(self, u):
        if u.shape[-1] != self.n_full:          # JAX would clamp out-of-range indices silently
            raise ValueError(f"x_from_u needs {self.n_full}-d standard cube draws, got {u.shape[-1]}")
        pt = self.pt
        z = pt._z_fixed if self.c_z is None else pt.z_ppf(u[self.c_z])
        m = pt.mass_ppf(u[self.im], z)
        th = pt(u)                                              # standard transform
        off = truncated_t2_ppf(u[self.c_r], self.hw, self.ts)
        y = self.exp_ratios_tab(self.log10_age(z), jnp.atleast_1d(m))[0] + off
        x = u[self.keep]
        x = x.at[self.x_zsol].set((th[self.c_zsol] - self.zsol_lo) / (self.zsol_hi - self.zsol_lo))
        x = x.at[self.x_r].set((y - self.y_lo) / (self.y_hi - self.y_lo))
        return x, m

    def x_from_u(self, u):
        """Standard 26-d unit-cube prior draws -> (x, logmass): the same prior draw."""
        x, m = self._batch_x_from_u(jnp.asarray(np.asarray(u, dtype=float)))
        return np.asarray(x), np.asarray(m)

    # ------------------------------------------------------------------ densities
    def log_q(self, x, m):
        """log q(m; x) on a vector of m (prior part, including the constant Jacobians)."""
        pt = self.pt
        z = self._z(x)
        ra, rb, wa, wb = pt.tab_rows_and_weights(z)
        pdf = wa * pt._tab_pdf[ra] + wb * pt._tab_pdf[rb]
        norm = wa * pt._tab_cum[ra, -1] + wb * pt._tab_cum[rb, -1]
        lp_m = jnp.log(jnp.interp(m, pt._m_grid, pdf)) - jnp.log(norm)
        # p(Z | m): truncated normal (jax_prior.JaxPriorTransform.__call__)
        zs = self.zsol(x)
        loc = jnp.interp(m, pt._mm_x, pt._mm_p50)
        sig = jnp.interp(m, pt._mm_x, pt._mm_p84) - jnp.interp(m, pt._mm_x, pt._mm_p16)
        a = ndtr((self.zsol_lo - loc) / sig)
        b = ndtr((self.zsol_hi - loc) / sig)
        t = (zs - loc) / sig
        lp_z = -0.5 * t ** 2 - 0.5 * np.log(2 * np.pi) - jnp.log(sig) - jnp.log(b - a)
        # p(y_j | z, m): truncated t2 around the expected ratios
        exp_r = jax.vmap(lambda mm: pt.expected_ratios(z, mm)[self.r_j])(m)      # (M, n_r)
        o = self.y(x)[None, :] - exp_r
        lp_y = self._log_t2_norm - 1.5 * jnp.log(2 * self.ts ** 2 + o ** 2)
        lp_y = jnp.where(jnp.abs(o) <= self.hw, lp_y, -jnp.inf).sum(axis=1)
        return lp_m + lp_z + lp_y + self.log_jac

    def log_like_m(self, x, m):
        """log L(m, x) on a vector of m: one emulator call."""
        ll = self.ll
        th = self.theta(x, jnp.asarray(REF_LOGMASS, dtype=x.dtype))
        f = ll.f_net(th)[ll._valid]
        sig0 = (ll._sigma_mag / ARSINH_A) * jnp.sqrt(f ** 2 + 4.0 * ARSINH_B ** 2)
        scale = (10.0 ** (m - REF_LOGMASS) * ll.mu)[:, None]
        var = ll._var_obs[None, :] + (sig0[None, :] * scale) ** 2
        r2 = (ll._fv[None, :] - f[None, :] * scale) ** 2 / var
        return -0.5 * jnp.sum(r2 + jnp.log(2.0 * np.pi * var), axis=1)

    # ------------------------------------------------------------------ fast path
    def _log_q_given(self, x, z, m, exp_r):
        """log q(m; x) on a vector of m, with the expected ratios exp_r (M, n_r) given."""
        pt = self.pt
        ra, rb, wa, wb = pt.tab_rows_and_weights(z)
        pdf = wa * pt._tab_pdf[ra] + wb * pt._tab_pdf[rb]
        norm = wa * pt._tab_cum[ra, -1] + wb * pt._tab_cum[rb, -1]
        lp_m = jnp.log(jnp.interp(m, pt._m_grid, pdf)) - jnp.log(norm)
        zs = self.zsol(x)
        loc = jnp.interp(m, pt._mm_x, pt._mm_p50)
        sig = jnp.interp(m, pt._mm_x, pt._mm_p84) - jnp.interp(m, pt._mm_x, pt._mm_p16)
        a = ndtr((self.zsol_lo - loc) / sig)
        b = ndtr((self.zsol_hi - loc) / sig)
        t = (zs - loc) / sig
        lp_z = -0.5 * t ** 2 - 0.5 * np.log(2 * np.pi) - jnp.log(sig) - jnp.log(b - a)
        o = self.y(x)[None, :] - exp_r
        lp_y = self._log_t2_norm - 1.5 * jnp.log(2 * self.ts ** 2 + o ** 2)
        lp_y = jnp.where(jnp.abs(o) <= self.hw, lp_y, -jnp.inf).sum(axis=1)
        return lp_m + lp_z + lp_y + self.log_jac

    def _prior_part(self, xc):
        """z, expected ratios on the coarse grid, log g (the marginal prior)."""
        z = self._z(xc)
        exp_c = self.exp_ratios_tab(self.log10_age(z), self.mc)
        lq_c = self._log_q_given(xc, z, self.mc, exp_c)
        log_g = jax.scipy.special.logsumexp(lq_c + jnp.log(self.wc))
        return z, exp_c, log_g, lq_c

    def _fine(self, xc, z, exp_c, lq_c):
        """log(L q) on a fine grid of n_fine masses over +-N_SIG sigma around the likelihood's
        peak in mass, found analytically: the model flux is linear in A = 10^(m - REF) mu,
        so A is the weighted least-squares amplitude (two passes, the emulator variance
        evaluated at the first estimate) and sigma_A comes from its Fisher information."""
        ll = self.ll
        th = self.theta(xc, jnp.asarray(REF_LOGMASS, dtype=xc.dtype))
        f = ll.f_net(th)[ll._valid]
        sig0 = (ll._sigma_mag / ARSINH_A) * jnp.sqrt(f ** 2 + 4.0 * ARSINH_B ** 2)
        A0 = jnp.sum(f * ll._fv / ll._var_obs) / jnp.sum(f ** 2 / ll._var_obs)
        var1 = ll._var_obs + (sig0 * jnp.abs(A0)) ** 2
        info = jnp.sum(f ** 2 / var1)
        A1 = jnp.sum(f * ll._fv / var1) / info
        ok = A1 > 0
        A1s = jnp.where(ok, A1, 1.0)
        m_pk = jnp.log10(A1s / ll.mu) + REF_LOGMASS
        s_m = 1.0 / (jnp.sqrt(info) * A1s * np.log(10.0))
        # the prior can be steep where L peaks (deep in the mass-function tail): centre on
        # L q with one Newton step, shift = s_m^2 d log q / dm, and widen by the shift
        dm = self.mc[1] - self.mc[0]
        slope_c = jnp.diff(lq_c) / dm
        slope_c = jnp.where(jnp.isfinite(slope_c), slope_c, 0.0)
        slope = jnp.interp(m_pk, 0.5 * (self.mc[1:] + self.mc[:-1]), slope_c)
        shift = jnp.clip(s_m ** 2 * slope, -0.5, 0.5)
        centre = m_pk + shift
        half = jnp.maximum(N_SIG * s_m + jnp.abs(shift), MIN_HALF)
        a = jnp.where(ok, jnp.clip(centre - half, self.m_lo, self.m_hi), self.m_lo)
        b = jnp.where(ok, jnp.clip(centre + half, self.m_lo, self.m_hi), self.m_hi)
        b = jnp.maximum(b, a + 1e-6)
        mf = a + (b - a) * jnp.linspace(0.0, 1.0, self.n_fine)
        wf = self._trap_w(mf)
        # from the s table at the fine masses themselves (delta_t_dex has kinks at m = 9, 12)
        exp_f = self.exp_ratios_tab(self.log10_age(z), mf)
        scale = (10.0 ** (mf - REF_LOGMASS) * ll.mu)[:, None]
        var = ll._var_obs[None, :] + (sig0[None, :] * scale) ** 2
        r2 = (ll._fv[None, :] - f[None, :] * scale) ** 2 / var
        l_f = -0.5 * jnp.sum(r2 + jnp.log(2.0 * np.pi * var), axis=1)
        tot = l_f + self._log_q_given(xc, z, mf, exp_f)
        if self.dust_prior is not None:
            tot = tot + self.dust_lnw(th, z, mf)
        return mf, wf, tot

    def dust_lnw(self, th, z, m):
        """ln w of the soft-ceiling dust prior on a vector of masses m, at theta th (any
        logmass; the SFH ratios, dust2 and z are mass-independent given x)."""
        D = self._dpr
        lsf = D.log10_sfr100_per_mass_jax(self.pt.age(z), th[self.c_r_all])
        return D.lnw_jax(th[self.c_dust2], self.dust_prior, m, m + lsf, self.dust_cfg)

    def logprior_and_loglike(self, x):
        inside = jnp.all((x >= 0.0) & (x <= 1.0))
        xc = jnp.clip(x, 0.0, 1.0)
        z, exp_c, log_g, lq_c = self._prior_part(xc)
        mf, wf, tot = self._fine(xc, z, exp_c, lq_c)
        log_i = jax.scipy.special.logsumexp(tot + jnp.log(wf))
        lp = jnp.where(inside & jnp.isfinite(log_g), log_g, -jnp.inf)
        ll = jnp.where(inside & jnp.isfinite(log_g) & jnp.isfinite(log_i), log_i - log_g,
                       -jnp.inf)
        return lp, ll

    def logprior(self, x):
        inside = jnp.all((x >= 0.0) & (x <= 1.0))
        log_g = self._prior_part(jnp.clip(x, 0.0, 1.0))[2]
        return jnp.where(inside & jnp.isfinite(log_g), log_g, -jnp.inf)

    def loglike(self, x):
        return self.logprior_and_loglike(x)[1]

    def sample_logmass(self, key, x):
        """One draw of logmass from L(m, x) q(m; x): a fine-grid cell by its trapezoid mass,
        then uniform within the cell."""
        xc = jnp.clip(x, 0.0, 1.0)
        z, exp_c, _, lq_c = self._prior_part(xc)
        mf, _, tot = self._fine(xc, z, exp_c, lq_c)
        cell = jnp.logaddexp(tot[:-1], tot[1:])
        k1, k2 = jax.random.split(key)
        i = jax.random.categorical(k1, cell)
        return mf[i] + (mf[i + 1] - mf[i]) * jax.random.uniform(k2, dtype=x.dtype)
