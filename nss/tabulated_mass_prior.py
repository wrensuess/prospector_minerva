"""Tabulated Prospector-beta mass prior: a drop-in for efsps_photfit.priors.MassPrior that
needs no mass-function evaluation per call.

The mass prior p(logM | z) is the stellar mass function at z (priors.mass_function). Its
redshift dependence has a simple structure:
  z <= 0.2        Leja+20 with z clamped to 0.2: ONE fixed curve
  0.2 < z <= 3    Leja+20, parameters quadratic in z: varies smoothly
  3 < z < 4       cos^2 blend of TWO FIXED curves, Leja+20(z=3) and Tacchella+18(z=4)
  4 <= z < 12     cos^2 blend of TWO FIXED curves, Tacchella+18 at floor(z) and floor(z)+1
  z >= 12         Tacchella+18(z=12): ONE fixed curve
(const_phi=True: Leja+20 at every z, clamped to [0.2, 3].)

So every z is "weight_a x row_a + weight_b x row_b" of a table of curves on the mass grid:
  - above z = 3 (and below 0.2) the rows are the fixed curves and the weights are the code's
    own cos^2 weights: the SAME prior as priors.mass_function (to rounding, ~1e-16);
  - for 0.2 < z <= 3 the rows are the normalised prior at z nodes spaced dz_low apart and the
    weights interpolate linearly in z. This is the one approximation. At dz_low = 0.01 the KS
    distance to the exact prior is <= 2e-7 and masses at the same u move by <= 4.4e-6 dex
    (measured 2026-10-08; test_tabulated_mass_prior.py).
The cumulative distribution of each row is computed once (trapezoid, as priors.py does), so
a draw is: combine two stored CDF rows, find the cell, solve the same per-cell quadratic as
priors.inverse_cdf_piecewise_linear. No pow/exp, no running sum per call.

    mp = TabulatedMassPrior.from_mass_prior(model.mprior)     # or: use_tabulated_mass_prior(model)
    mp.ppf(u, z)        # numpy reference; jax_prior.JaxPriorTransform uses the same tables
"""
import numpy as np

from efsps_photfit import priors as P

Z_LOW_CLAMP, Z_LEJA_MAX = 0.2, 3.0          # priors.low_z_mass_func clamps z to [0.2, 3]


class TabulatedMassPrior:
    def __init__(self, lo, hi, const_phi=False, n_grid=551, dz_low=0.01):
        self.lo, self.hi, self.const_phi = float(lo), float(hi), bool(const_phi)
        self.grid = np.linspace(self.lo, self.hi, int(n_grid))
        self.h = np.diff(self.grid)
        self.dz_low = float(dz_low)
        n_nodes = int(round((Z_LEJA_MAX - Z_LOW_CLAMP) / self.dz_low)) + 1
        self.z_nodes = np.linspace(Z_LOW_CLAMP, Z_LEJA_MAX, n_nodes)
        bounds = (self.lo, self.hi)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            # rows 0 .. n_nodes-1: the exact prior at each node, normalised
            low = P.mass_function(self.z_nodes[:, None], self.grid[None, :], self.const_phi, bounds)
            low = low / self._cum(low)[:, -1:]
            rows = [low]
            if not self.const_phi:
                m = 10 ** self.grid
                inside = (self.grid >= self.lo) & (self.grid <= self.hi)
                # row n_nodes: Leja+20 at z = 3 (the low end of the 3-4 blend), unnormalised as
                # priors.mass_function uses it; rows n_nodes+1 ..: Tacchella+18 at z = 4 .. 12
                l3 = P.low_z_mass_func(3.0, self.grid)
                t18 = np.stack([P._t18_at_integer(k, m) for k in range(4, 13)])
                rows += [np.where(inside, l3, 0.0)[None, :], np.where(inside, t18, 0.0)]
        self.pdf_table = np.concatenate(rows)                       # (n_rows, n_grid)
        self.cum_table = self._cum(self.pdf_table)                  # (n_rows, n_grid)
        self.n_nodes = n_nodes
        self.row_l3 = n_nodes
        self.row_t18_4 = n_nodes + 1                                # T18 at k: row_t18_4 + k - 4

    @classmethod
    def from_mass_prior(cls, mp, dz_low=0.01):
        return cls(mp.lo, mp.hi, mp.const_phi, mp.grid.size, dz_low)

    def _cum(self, pdf):
        area = 0.5 * (pdf[..., 1:] + pdf[..., :-1]) * self.h
        return np.concatenate([np.zeros(pdf.shape[:-1] + (1,)), np.cumsum(area, axis=-1)], axis=-1)

    # ------------------------------------------------------------------ the two rows
    def rows_and_weights(self, z):
        """(N,) z -> row_a, row_b (int), w_a, w_b: the prior at z is w_a*row_a + w_b*row_b."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        # 0.2 < z <= 3 (and the clamped ends): linear between nodes
        zc = np.clip(z, Z_LOW_CLAMP, Z_LEJA_MAX)
        j = np.clip(np.searchsorted(self.z_nodes, zc, side="right") - 1, 0, self.n_nodes - 2)
        t = (zc - self.z_nodes[j]) / (self.z_nodes[j + 1] - self.z_nodes[j])
        ra, rb, wa, wb = j, j + 1, 1.0 - t, t
        if self.const_phi:
            return ra, rb, wa, wb
        # 3 < z < 4: Leja(3) x w0 + T18(4) x w1, the weights of priors.mass_function
        b0, b1 = P.cos_weight(z, 3.0, 4.0)
        # z >= 4: T18(k) x v0 + T18(k+1) x v1; T18(4) at z <= 4, T18(12) at z >= 12
        k = np.clip(np.floor(z).astype(int), 4, 11)
        v0, v1 = P.cos_weight(z, k, k + 1)
        hk = self.row_t18_4 + k - 4
        hk = np.where(z >= 12.0, self.row_t18_4 + 8, hk)
        v0 = np.where(z >= 12.0, 1.0, v0)
        v1 = np.where(z >= 12.0, 0.0, v1)
        blend = (z > 3.0) & (z < 4.0)
        high = z >= 4.0
        ra = np.where(blend, self.row_l3, np.where(high, hk, ra))
        rb = np.where(blend, self.row_t18_4, np.where(high, hk + 1, rb))
        wa = np.where(blend, b0, np.where(high, v0, wa))
        wb = np.where(blend, b1, np.where(high, v1, wb))
        rb = np.minimum(rb, self.pdf_table.shape[0] - 1)            # z >= 12: weight 0 anyway
        return ra, rb, wa, wb

    # ------------------------------------------------------------------ prior
    def pdf_rows(self, z):
        """(N,) z -> (N, n_grid) the (unnormalised) prior density on the mass grid."""
        ra, rb, wa, wb = self.rows_and_weights(z)
        return wa[:, None] * self.pdf_table[ra] + wb[:, None] * self.pdf_table[rb]

    def ppf(self, u, z):
        """Inverse CDF, the same per-cell solution as priors.inverse_cdf_piecewise_linear."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        z = np.broadcast_to(np.atleast_1d(np.asarray(z, dtype=float)), u.shape)
        ra, rb, wa, wb = self.rows_and_weights(z)
        cum = wa[:, None] * self.cum_table[ra] + wb[:, None] * self.cum_table[rb]
        target = u * cum[:, -1]
        h, x = self.h, self.grid
        i = np.clip((cum[:, 1:] <= target[:, None]).sum(axis=1), 0, h.size - 1)
        r = np.arange(u.size)
        p0 = wa * self.pdf_table[ra, i] + wb * self.pdf_table[rb, i]
        p1 = wa * self.pdf_table[ra, i + 1] + wb * self.pdf_table[rb, i + 1]
        c0 = cum[r, i]
        rem = np.clip(target - c0, 0.0, None)
        s = (p1 - p0) / h[i]
        disc = np.clip(p0 * p0 + 2.0 * s * rem, 0.0, None)
        denom = p0 + np.sqrt(disc)
        with np.errstate(invalid="ignore", divide="ignore"):
            tt = np.where(denom > 0, 2.0 * rem / denom, 0.0)
        return np.clip(x[i] + np.clip(tt, 0.0, h[i]), x[0], x[-1])


def use_tabulated_mass_prior(model, dz_low=0.01):
    """Switch a FitModel's mass prior to the tabulated one (prior_transform then uses it).
    Returns the model."""
    model.mprior = TabulatedMassPrior.from_mass_prior(model.mprior, dz_low)
    return model
