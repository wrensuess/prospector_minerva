"""Candidate exact speed-ups of the mass inverse CDF, drop-in replacements for
JaxPriorTransform.mass_ppf, for profiling (profile_nss.py variants fix_twopoint and
fix_tables). NOT yet validated by test_jax_prior.py; jax_prior.py is unchanged.
  (1) mass_ppf_twopoint: p0, p1 by evaluating the mass function at grid[i], grid[i+1]
      instead of gathering them from the 551-point pdf (inside the slice loop that
      gather cost ~0.9 ms per batched call on the M2 Pro; cause not pinned down).
  (2) mass_ppf_tables_twopoint: (1) plus every z > 3 piece of the mass function from
      tables numpy computes once (T18 at z = 4..12, Leja+20 at z = 3)."""
import numpy as np, jax, jax.numpy as jnp
import jax_prior as J
from efsps_photfit import priors as P


def mass_ppf_twopoint(pt):
    """(1) p0, p1 = the mass function evaluated at grid[i], grid[i+1] (not gathered)."""
    grid, h = pt._m_grid, pt._m_h
    def f(u, z):
        pdf = J.mass_function(z, grid, pt._m_const_phi, pt._m_bounds)
        cum = J._cum_trapz(pdf, h)
        target = u * cum[-1]
        i = jnp.clip(jnp.sum(cum[1:] <= target), 0, h.size - 1)
        p01 = J.mass_function(z, jnp.stack([grid[i], grid[i + 1]]), pt._m_const_phi, pt._m_bounds)
        return _tail(target, cum[i], p01[0], p01[1], grid[i], h[i], grid)
    return f


def _tail(target, c0, p0, p1, x_i, h_i, grid):
    rem = jnp.clip(target - c0, 0.0, None)
    s = (p1 - p0) / h_i
    disc = jnp.clip(p0 * p0 + 2.0 * s * rem, 0.0, None)
    denom = p0 + jnp.sqrt(disc)
    t = jnp.where(denom > 0, 2.0 * rem / jnp.where(denom > 0, denom, 1.0), 0.0)
    return jnp.clip(x_i + jnp.clip(t, 0.0, h_i), grid[0], grid[-1])


def make_tables(pt):
    """numpy's own values: T18 at z = 4..12 and Leja+20 at z = 3, on the mass grid."""
    g = np.asarray(pt._m_grid)
    m = 10 ** g
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        t18 = np.stack([P._t18_at_integer(k, m) for k in range(4, 13)])     # (9, M)
        l3 = P.low_z_mass_func(3.0, g)
    return jnp.asarray(t18), jnp.asarray(l3)


def mass_function_tables(pt, t18, l3):
    """(2) priors.mass_function (const_phi=False) with every z>3 piece from tables."""
    grid, lo, hi = pt._m_grid, *pt._m_bounds
    def f(z, logm, idx=None):
        T = t18 if idx is None else t18[:, idx]
        L3 = l3 if idx is None else l3[idx]
        lowz = J._low_z_mass_func(z, logm)
        w0, w1 = J._cos_weight(z, 3.0, 4.0)
        blend = L3 * w0 + T[0] * w1
        k = jnp.clip(jnp.floor(z).astype(jnp.int32), 4, 11)
        v0, v1 = J._cos_weight(z, k, k + 1)
        highz = T[k - 4] * v0 + T[k - 3] * v1
        highz = jnp.where(z <= 4.0, T[0], highz)
        highz = jnp.where(z >= 12.0, T[8], highz)
        phi = jnp.where(z <= 3.0, lowz, jnp.where(z < 4.0, blend, highz))
        return jnp.where((logm < lo) | (logm > hi), 0.0, phi)
    return f


def mass_ppf_tables_twopoint(pt):
    """(1) + (2)."""
    grid, h = pt._m_grid, pt._m_h
    t18, l3 = make_tables(pt)
    mf = mass_function_tables(pt, t18, l3)
    def f(u, z):
        pdf = mf(z, grid)
        cum = J._cum_trapz(pdf, h)
        target = u * cum[-1]
        i = jnp.clip(jnp.sum(cum[1:] <= target), 0, h.size - 1)
        idx = jnp.stack([i, i + 1])
        p01 = mf(z, grid[idx], idx)
        return _tail(target, cum[i], p01[0], p01[1], grid[i], h[i], grid)
    return f, mf
