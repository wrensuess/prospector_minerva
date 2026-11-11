"""JAX port of the efsps-photfit prior transform (FitModel.prior_transform).

Built FROM a numpy FitModel: every table (p(z) nodes, the mass grid, Gallazzi+05, the
expected-ratio table on z_eff, the age-of-universe splines) is taken from the model's own
objects, so the two cannot drift apart in their inputs. Only the arithmetic is re-written.

    pt = JaxPriorTransform(model)
    theta = pt(u)            # one row, (n_dim,) -> (n_dim,); jit/vmap-able
    theta = pt.batch(u)      # (N, n_dim) -> (N, n_dim), jitted vmap

Line references are to efsps_photfit/{model,priors,sfh}.py. The numpy code is the
reference; tests are in test_jax_prior.py. Requires jax_enable_x64.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr, ndtri

from efsps_photfit import priors as P
from efsps_photfit import sfh as S


def _need_x64():
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("JaxPriorTransform needs jax.config.update('jax_enable_x64', True).")


# ---------------------------------------------------------------- spline evaluation
class _Spline:
    """scipy CubicSpline evaluation (extrapolating), in scipy's own order of operations:
    interval by x[i] <= xq < x[i+1] (end intervals extend outwards), then
    c3 + c2*s + c1*s**2 + c0*s**3 summed with running powers of s (scipy _ppoly)."""

    def __init__(self, spl):
        self.x = jnp.asarray(spl.x)
        self.c = jnp.asarray(spl.c)          # (4, n-1)
        self.n = int(spl.x.size)

    def __call__(self, xq):
        i = jnp.clip(jnp.searchsorted(self.x, xq, side="right") - 1, 0, self.n - 2)
        s = xq - self.x[i]
        res = self.c[3, i]
        z = s
        res = res + self.c[2, i] * z
        z = z * s
        res = res + self.c[1, i] * z
        z = z * s
        return res + self.c[0, i] * z


# ---------------------------------------------------------------- 1-D pieces (priors.py)
def truncnorm_ppf(u, mu, sig, lo, hi):
    a = ndtr((lo - mu) / sig)
    b = ndtr((hi - mu) / sig)
    return jnp.clip(mu + sig * ndtri(a + u * (b - a)), lo, hi)


def truncated_t2_ppf(u, hw, sig):
    const2 = 2.0 * sig * hw
    const3 = const2 ** 2
    const4 = 2.0 * hw ** 2
    num = -1.0 * (const3 * u ** 2 - const3 * u + (sig * hw) ** 2)
    den = const4 * u ** 2 - const4 * u - sig ** 2
    f = jnp.sqrt(jnp.clip(num / den, 0.0, None))
    return jnp.where(u > 0.5, f, -f)


def uniform_ppf(u, lo, hi):
    return jnp.clip(lo + u * (hi - lo), lo, hi)


def loguniform_ppf(u, lo, hi):
    return jnp.clip(jnp.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))), lo, hi)


def _invert_cell(target, cum, pdf, x, h):
    """Shared tail of inverse_cdf_piecewise_linear (priors.py:184-193) for ONE density:
    i = number of cum[1:] <= target, then the stable quadratic root in cell i."""
    i = jnp.clip(jnp.sum(cum[1:] <= target), 0, h.size - 1)
    return _solve_cell(target, cum[i], pdf[i], pdf[i + 1], i, x, h)


def _solve_cell(target, c0, p0, p1, i, x, h):
    """The stable quadratic root of the piecewise-linear CDF in cell i (priors.py:187-193)."""
    rem = jnp.clip(target - c0, 0.0, None)
    s = (p1 - p0) / h[i]
    disc = jnp.clip(p0 * p0 + 2.0 * s * rem, 0.0, None)
    denom = p0 + jnp.sqrt(disc)
    t = jnp.where(denom > 0, 2.0 * rem / jnp.where(denom > 0, denom, 1.0), 0.0)
    return jnp.clip(x[i] + jnp.clip(t, 0.0, h[i]), x[0], x[-1])


def _cum_trapz(pdf, h):
    """Trapezoid CDF at the nodes, summed SEQUENTIALLY as np.cumsum does. jnp.cumsum
    adds in a different order; its last-bit differences move the inverse by up to
    0.02 dex in logmass at u ~ 1 - 1e-16, where the CDF is flat to machine precision."""
    area = 0.5 * (pdf[1:] + pdf[:-1]) * h

    def step(c, a):
        c = c + a
        return c, c

    _, cum = jax.lax.scan(step, jnp.zeros((), area.dtype), area)
    return jnp.concatenate([jnp.zeros(1, area.dtype), cum])


# ---------------------------------------------------------------- mass function (priors.py)
_PARS = {k: np.asarray(v, float) for k, v in P.PARS.items()}
_PHI_T18 = jnp.asarray(P.PHI_T18)
_M_T18 = jnp.asarray(P.M_T18)
_ALPHA_T18 = jnp.asarray(P.ALPHA_T18)


def _schechter(logm, logphi, logmstar, alpha):
    return ((10 ** logphi) * np.log(10) * 10 ** ((logm - logmstar) * (alpha + 1))
            * jnp.exp(-10 ** (logm - logmstar)))


def _low_z_mass_func(z0, logm):
    z0 = jnp.clip(z0, 0.2, 3.0)
    lphi1 = P.parameter_at_z0(_PARS["logphi1"], z0)
    lphi2 = P.parameter_at_z0(_PARS["logphi2"], z0)
    lmstar = P.parameter_at_z0(_PARS["logmstar"], z0)
    return (_schechter(logm, lphi1, lmstar, _PARS["alpha1"][0])
            + _schechter(logm, lphi2, lmstar, _PARS["alpha2"][0]))


def _t18_at_integer(k, m):
    i = k - 4
    phi_star, m_star, alpha = _PHI_T18[i], _M_T18[i], _ALPHA_T18[i]
    return phi_star * (m / m_star) ** (alpha + 1) * jnp.exp(-m / m_star)


def _cos_weight(z, zlo, zhi):
    zs = (z - zlo) / (zhi - zlo) * (np.pi / 2)
    w = jnp.cos(zs) ** 2
    return w, 1 - w


def _high_z_mass_func(z0, m):
    k = jnp.clip(jnp.floor(z0).astype(jnp.int32), 4, 11)
    w0, w1 = _cos_weight(z0, k, k + 1)
    phi = _t18_at_integer(k, m) * w0 + _t18_at_integer(k + 1, m) * w1
    phi = jnp.where(z0 <= 4.0, _t18_at_integer(4, m), phi)
    return jnp.where(z0 >= 12.0, _t18_at_integer(12, m), phi)


def mass_function(z, logm, const_phi, bounds):
    """priors.py:123-137 for scalar z and a grid of logm."""
    if const_phi:
        phi = _low_z_mass_func(z, logm)
    else:
        m = 10 ** logm
        lowz = _low_z_mass_func(z, logm)
        w0, w1 = _cos_weight(z, 3.0, 4.0)
        blend = _low_z_mass_func(3.0, logm) * w0 + _t18_at_integer(4, m) * w1
        highz = _high_z_mass_func(z, m)
        phi = jnp.where(z <= 3.0, lowz, jnp.where(z < 4.0, blend, highz))
    return jnp.where((logm < bounds[0]) | (logm > bounds[1]), 0.0, phi)


def delta_t_dex(m):
    """sfh.py:111-115."""
    slope = (0.8 - (-0.2)) / (12.0 - 9.0)
    intercept = -0.2 - (slope * 9.0)
    return jnp.clip(slope * m + intercept, -0.2, 0.8)


# ---------------------------------------------------------------- the transform (model.py)
class JaxPriorTransform:
    def __init__(self, model):
        _need_x64()
        self.n_dim = model.n_dim
        col = model.col
        self.col = dict(col)

        # zred: free (p(z) inverse CDF on the model's own nodes) or fixed
        if "zred" in col:
            zp = model.zprior
            self._z_x = jnp.asarray(zp.z_nodes)
            self._z_pdf = jnp.asarray(zp.pdf_nodes)
            h = np.diff(zp.z_nodes)
            self._z_h = jnp.asarray(h)
            # static: built once in numpy exactly as priors.py:173-174 builds it
            area = 0.5 * (zp.pdf_nodes[1:] + zp.pdf_nodes[:-1]) * h
            self._z_cum = jnp.asarray(np.concatenate([[0.0], np.cumsum(area)]))
            self._z_fixed = None
        else:
            self._z_fixed = float(model.fixed["zred"])

        # logmass | z
        mp = model.mprior
        self._m_grid = jnp.asarray(mp.grid)
        self._m_h = jnp.asarray(np.diff(mp.grid))
        self._m_const_phi = mp.const_phi
        self._m_bounds = (mp.lo, mp.hi)
        # tabulated mass prior (tabulated_mass_prior.TabulatedMassPrior): same tables
        self._m_tab = hasattr(mp, "cum_table")
        if self._m_tab:
            self._tab_pdf = jnp.asarray(mp.pdf_table)
            self._tab_cum = jnp.asarray(mp.cum_table)
            self._tab_z = jnp.asarray(mp.z_nodes)
            self._tab_n_nodes = int(mp.n_nodes)
            self._tab_n_rows = int(mp.pdf_table.shape[0])
            self._tab_row_l3 = int(getattr(mp, "row_l3", -1))
            self._tab_row_t18_4 = int(getattr(mp, "row_t18_4", -1))

        # logzsol | mass
        self._met = (model.bounds["logzsol"] if "logzsol" in col else None)
        mm = P.massmet
        self._mm_x = jnp.asarray(mm[:, 0])
        self._mm_p50 = jnp.asarray(mm[:, 1])
        self._mm_p16 = jnp.asarray(mm[:, 2])
        self._mm_p84 = jnp.asarray(mm[:, 3])

        # SFH ratios | z, mass
        self._ratio_cols, self._ratio_j, self._ratio_lo, self._ratio_hi = [], [], [], []
        for name in model.ratio_names:
            if name in col:
                self._ratio_cols.append(col[name])
                self._ratio_j.append(int(name.rsplit("_", 1)[1]) - 1)
                lo, hi = model.bounds[name]
                self._ratio_lo.append(lo)
                self._ratio_hi.append(hi)
        self._ratio_cols = np.asarray(self._ratio_cols, int)
        self._ratio_j = np.asarray(self._ratio_j, int)
        self._ratio_lo = jnp.asarray(self._ratio_lo)
        self._ratio_hi = jnp.asarray(self._ratio_hi)
        self._hw = float(model.nzsfh["logsfr_ratio_halfwidth"])
        self._ts = float(model.nzsfh["logsfr_ratio_tscale"])
        er = model.ratios
        self._age_fwd = _Spline(er._age._fwd)
        self._age_inv = _Spline(er._age._inv)
        self._age0 = float(er._age.age0)
        self._age_zmax = float(er._age_zmax)
        self._age_zmin = float(er._age_zmin)
        self._log_zeff = jnp.asarray(er.log_zeff)
        self._ratio_table = jnp.asarray(er.table)          # (n_zeff, 9)

        # the 1-D priors
        self._simple = []
        for name in model.free:
            r = model.rows[name]
            if r.kind.startswith("nzsfh"):
                continue
            lo, hi = model.bounds[name]
            if r.kind not in ("uniform", "loguniform", "truncnormal"):
                raise ValueError(f"{name}: no transform for prior kind {r.kind!r}")
            self._simple.append((col[name], r.kind, float(lo), float(hi), tuple(r.args)))

        self.batch = jax.jit(jax.vmap(self.__call__))

    # pieces, exposed for testing ----------------------------------------------------
    def z_ppf(self, u):
        target = u * self._z_cum[-1]
        return _invert_cell(target, self._z_cum, self._z_pdf, self._z_x, self._z_h)

    def tab_rows_and_weights(self, z):
        """TabulatedMassPrior.rows_and_weights for one z."""
        n = self._tab_n_nodes
        zc = jnp.clip(z, 0.2, 3.0)
        j = jnp.clip(jnp.searchsorted(self._tab_z, zc, side="right") - 1, 0, n - 2)
        t = (zc - self._tab_z[j]) / (self._tab_z[j + 1] - self._tab_z[j])
        ra, rb, wa, wb = j, j + 1, 1.0 - t, t
        if self._m_const_phi:
            return ra, rb, wa, wb
        b0, b1 = _cos_weight(z, 3.0, 4.0)
        k = jnp.clip(jnp.floor(z).astype(jnp.int32), 4, 11)
        v0, v1 = _cos_weight(z, k, k + 1)
        hk = self._tab_row_t18_4 + k - 4
        top = z >= 12.0
        hk = jnp.where(top, self._tab_row_t18_4 + 8, hk)
        v0 = jnp.where(top, 1.0, v0)
        v1 = jnp.where(top, 0.0, v1)
        blend = (z > 3.0) & (z < 4.0)
        high = z >= 4.0
        ra = jnp.where(blend, self._tab_row_l3, jnp.where(high, hk, ra))
        rb = jnp.where(blend, self._tab_row_t18_4, jnp.where(high, hk + 1, rb))
        wa = jnp.where(blend, b0, jnp.where(high, v0, wa))
        wb = jnp.where(blend, b1, jnp.where(high, v1, wb))
        rb = jnp.minimum(rb, self._tab_n_rows - 1)
        return ra, rb, wa, wb

    def mass_ppf(self, u, z):
        if self._m_tab:
            return self._mass_ppf_tab(u, z)
        pdf = mass_function(z, self._m_grid, self._m_const_phi, self._m_bounds)
        cum = _cum_trapz(pdf, self._m_h)
        return _invert_cell(u * cum[-1], cum, pdf, self._m_grid, self._m_h)

    def _mass_ppf_tab(self, u, z):
        """TabulatedMassPrior.ppf for one (u, z): two stored rows, no mass function."""
        ra, rb, wa, wb = self.tab_rows_and_weights(z)
        cum = wa * self._tab_cum[ra] + wb * self._tab_cum[rb]
        target = u * cum[-1]
        h = self._m_h
        i = jnp.clip(jnp.sum(cum[1:] <= target), 0, h.size - 1)
        p0 = wa * self._tab_pdf[ra, i] + wb * self._tab_pdf[rb, i]
        p1 = wa * self._tab_pdf[ra, i + 1] + wb * self._tab_pdf[rb, i + 1]
        return _solve_cell(target, cum[i], p0, p1, i, self._m_grid, h)

    def age(self, z):
        return jnp.exp(self._age_fwd(jnp.log1p(z)))

    def z_of_age(self, age_gyr):
        return jnp.expm1(self._age_inv(jnp.log(age_gyr)))

    def z_eff(self, z, m):
        """sfh.py ExpectedRatios.z_eff."""
        lo, hi = S.Z_SHIFT_CLAMP
        base = self.age(z)
        shifted = 10 ** (jnp.log10(base) + delta_t_dex(m))
        shifted = jnp.clip(shifted, 0.1, self._age0 - 0.1)
        ze = self.z_of_age(shifted)
        ze = jnp.where(shifted <= self._age_zmax, hi,
                       jnp.where(shifted >= self._age_zmin, lo, ze))
        return jnp.clip(ze, lo, hi)

    def expected_ratios(self, z, m):
        """sfh.py ExpectedRatios.__call__ for one row: -> (9,)."""
        lz = jnp.log(self.z_eff(z, m))
        return jax.vmap(lambda col: jnp.interp(lz, self._log_zeff, col),
                        in_axes=1)(self._ratio_table)

    # the transform -------------------------------------------------------------------
    def __call__(self, u, logmass=None):
        """u -> theta. With logmass given (linear_mass.py), it replaces the mass inverse CDF
        and u[logmass] is ignored; with logmass=None (default) nothing changes."""
        th = jnp.zeros(self.n_dim, dtype=u.dtype)
        col = self.col
        if self._z_fixed is None:
            z = self.z_ppf(u[col["zred"]])
            th = th.at[col["zred"]].set(z)
        else:
            z = jnp.asarray(self._z_fixed, dtype=u.dtype)
        m = self.mass_ppf(u[col["logmass"]], z) if logmass is None else logmass
        th = th.at[col["logmass"]].set(m)
        if self._met is not None:
            lo, hi = self._met
            loc = jnp.interp(m, self._mm_x, self._mm_p50)
            scale = jnp.interp(m, self._mm_x, self._mm_p84) - jnp.interp(m, self._mm_x,
                                                                          self._mm_p16)
            th = th.at[col["logzsol"]].set(truncnorm_ppf(u[col["logzsol"]], loc, scale,
                                                         lo, hi))
        if self._ratio_cols.size:
            expect = self.expected_ratios(z, m)[self._ratio_j]
            off = truncated_t2_ppf(u[self._ratio_cols], self._hw, self._ts)
            th = th.at[self._ratio_cols].set(jnp.clip(expect + off, self._ratio_lo,
                                                      self._ratio_hi))
        for c, kind, lo, hi, args in self._simple:
            x = u[c]
            if kind == "uniform":
                v = uniform_ppf(x, lo, hi)
            elif kind == "loguniform":
                v = loguniform_ppf(x, lo, hi)
            else:
                v = truncnorm_ppf(x, args[0], args[1], lo, hi)
            th = th.at[c].set(v)
        return th
