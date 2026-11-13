"""How far is the tabulated mass prior (tabulated_mass_prior.py) from the exact one
(efsps_photfit.priors.MassPrior)? By redshift bin, for both mass functions (const_phi False
and True):
  1. masses at the same u: max |dlogM| over ordinary u in [1e-6, 1 - 1e-6], and over all u
     including u within 1e-12 of 0 and 1 (where the CDF is flat to machine precision, so
     last-bit differences between summing the CDF and combining stored CDFs move the mass);
  2. KS distance: max over the mass grid of |F_tabulated - F_exact|;
  3. continuity across every switch point (z = 0.2, 3, 4, ..., 12): the tabulated prior's
     masses at z -/+ 1e-9 must agree as closely as the exact prior's do;
  4. edges u in {0, 1e-300, 1, ...}: finite and inside [lo, hi].
Pass: ordinary-u |dlogM| <= 1e-4 dex, KS <= 1e-6, jumps <= 1e-4 dex.

Run:  ~/anaconda3/envs/yaargh/bin/python test_tabulated_mass_prior.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("EFSPS_PARENT", "/Users/wren/Projects/MINERVA/efsps-photfit"))
sys.path.insert(0, HERE)
from efsps_photfit.priors import MassPrior
from tabulated_mass_prior import TabulatedMassPrior

TOL_DLOGM, TOL_KS, TOL_JUMP = 1e-4, 1e-6, 1e-4
Z_BINS = [(0.01, 0.2), (0.2, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 8.0),
          (8.0, 10.0), (10.0, 12.0), (12.0, 15.0)]
FAILS = []


def cdf_rows(mp, z):
    pdf = mp.pdf_rows(z)
    h = np.diff(mp.grid)
    cum = np.concatenate([np.zeros((z.size, 1)),
                          np.cumsum(0.5 * (pdf[:, 1:] + pdf[:, :-1]) * h, axis=1)], axis=1)
    return cum / cum[:, -1:]


def run(const_phi, rng, n=20000, lo_m=7.0):
    n_grid = int(round((12.5 - lo_m) / 0.01)) + 1           # 0.01-dex steps (nss_priors)
    ex = MassPrior(lo_m, 12.5, const_phi=const_phi, n_grid=n_grid)
    tp = TabulatedMassPrior(lo_m, 12.5, const_phi=const_phi, n_grid=n_grid)
    print(f"\n== const_phi={const_phi}, logM {lo_m}-12.5 ({n_grid} grid points): "
          f"{tp.pdf_table.shape[0]} table rows, node spacing {tp.dz_low} in 0.2 < z <= 3")
    print(f"  {'z bin':12s} {'max|dlogM| ordinary u':>22s} {'all u (incl. 1e-12 edges)':>26s} "
          f"{'KS distance':>12s}")
    edges_u = np.array([1e-12, 1e-9, 1 - 1e-9, 1 - 1e-12])
    for lo, hi in Z_BINS:
        z = rng.uniform(lo, hi, n)
        u = rng.uniform(1e-6, 1 - 1e-6, n)
        d_ord = np.abs(ex.ppf(u, z) - tp.ppf(u, z)).max()
        ue = rng.choice(edges_u, n)
        d_all = max(d_ord, np.abs(ex.ppf(ue, z) - tp.ppf(ue, z)).max())
        ks = np.abs(cdf_rows(ex, z[:5000]) - cdf_rows(tp, z[:5000])).max()
        ok = d_ord <= TOL_DLOGM and ks <= TOL_KS
        if not ok:
            FAILS.append(f"const_phi={const_phi} z {lo}-{hi}")
        print(f"  {lo:5.2f}-{hi:<6.2f} {d_ord:22.1e} {d_all:26.1e} {ks:12.1e}  "
              f"{'ok' if ok else 'FAIL'}")
    # continuity across the switch points
    us = rng.uniform(1e-6, 1 - 1e-6, 2000)
    worst = (0.0, None)
    for zs in [0.2, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
        a, b = np.full(us.size, zs - 1e-9), np.full(us.size, zs + 1e-9)
        jump_tab = np.abs(tp.ppf(us, a) - tp.ppf(us, b)).max()
        jump_ex = np.abs(ex.ppf(us, a) - ex.ppf(us, b)).max()
        if jump_tab - jump_ex > worst[0]:
            worst = (jump_tab - jump_ex, zs)
    ok = worst[0] <= TOL_JUMP
    if not ok:
        FAILS.append(f"const_phi={const_phi} jump at z={worst[1]}")
    print(f"  switch points 0.2, 3, 4..12: largest extra jump vs the exact prior "
          f"{worst[0]:.1e} dex{'' if worst[1] is None else f' (z={worst[1]})'}  {'ok' if ok else 'FAIL'}")
    # edges
    ue = np.array([0.0, 1e-300, 1e-15, 0.5, 1 - 1e-16, 1.0] * 400)
    ze = rng.uniform(0.0, 15.0, ue.size)
    m = tp.ppf(ue, ze)
    ok = bool(np.all(np.isfinite(m)) & np.all((m >= tp.lo) & (m <= tp.hi)))
    if not ok:
        FAILS.append(f"const_phi={const_phi} edges")
    print(f"  edges u in {{0, 1e-300, 1e-15, 0.5, 1-1e-16, 1}}: finite and in bounds: "
          f"{'ok' if ok else 'FAIL'}")


def main():
    rng = np.random.default_rng(7)
    for cp, lo_m in ((False, 7.0), (True, 7.0), (True, 6.0)):   # last = the NSS default
        run(cp, rng, lo_m=lo_m)
    print("\nRESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
