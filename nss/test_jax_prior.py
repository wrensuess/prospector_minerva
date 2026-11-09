"""Validate jax_prior.JaxPriorTransform against the numpy FitModel.prior_transform.

Same u in, compare theta out. Every comparison is reported by redshift bin.
  1. free z: random u + the u of the NSS dead points of 58858 (free z, L=1000), also with
     the downstream log L (58858's photometry, numpy likelihood on both thetas)
  2. fixed z on a grid 0.01-15 that hits every switch: the z_eff clamps (z_eff 0.15, 10),
     the mass function (z = 3, 4, each integer 4-12), and the p(z) table ends
  3. z windows (the windows of the windowed NSS run, plus narrow ones on the switches)
  4. unit-cube edges: u in {0, 1e-300, 1e-15, 1e-9, 0.5, 1-1e-9, 1-1e-16, 1}; where theta
     differs, judged in probability (see tail_check)
  4b. const_phi=True (the Leja+20-only mass function and p(z))
  5. internals in isolation: z_eff and expected ratios vs sfh.ExpectedRatios
  6. single-row call (un-jitted) == jitted vmap; timing

Run:  ~/anaconda3/envs/yaargh/bin/python test_jax_prior.py [--n 200000]
"""
import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.environ.get("EFSPS_PARENT", "/Users/wren/Projects/MINERVA/efsps-photfit")
sys.path.insert(0, PARENT)
sys.path.insert(0, HERE)
from efsps_photfit import config as C
from efsps_photfit.catalog import read_catalog
from photometry import read_photometry
from efsps_photfit.emulator import PhotEmulator, EmulatorRefusal
from efsps_photfit.model import FitModel, table_for_run
from jax_prior import JaxPriorTransform
from tabulated_mass_prior import use_tabulated_mass_prior

EMULATOR = os.path.join(PARENT, "efsps_photfit/data/v4_n22000000_w256_d5_s0_parrot.npy")
Z_EDGES = [0.01, 0.04, 0.15, 0.5, 1.0, 2.0, 2.9, 3.1, 3.9, 4.1, 6.0, 8.0, 10.0, 12.0, 15.0001]
TOL_THETA = 1e-9      # max |delta theta| in parameter units, anywhere
TOL_LOGL = 1e-6       # max |delta log L|

FAILS = []
MASS_PRIOR = "exact"   # or "tabulated" (tabulated_mass_prior.py); set by --mass-prior


def check(ok, what):
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "FAIL"


def build(phot, fix_z=None, z_window=None, emu_cache={}):
    key = tuple(phot.bands)
    if key not in emu_cache:
        emu_cache[key] = PhotEmulator(EMULATOR, bands=phot.bands,
                                      expected_md5=list(C.KNOWN_EMULATOR_MD5))
    table = table_for_run(C.PARAMETER_TABLE, fix_z, z_window)
    m = FitModel(emu_cache[key], phot.bands, phot.flux, phot.err, phot.mu, table=table)
    return use_tabulated_mass_prior(m) if MASS_PRIOR == "tabulated" else m


def compare(model, pt, u):
    th_np = model.prior_transform(u)
    th_jx = np.asarray(pt.batch(jnp.asarray(u)))
    return th_np, th_jx


def groups(model):
    g = {"zred": ["zred"], "logmass": ["logmass"], "logzsol": ["logzsol"],
         "sfh ratios": [n for n in model.free if n.startswith("logsfr")],
         "1-D priors": [n for n in model.free if model.rows[n].kind in
                        ("uniform", "loguniform", "truncnormal")]}
    return {k: [model.col[n] for n in v if n in model.col] for k, v in g.items()
            if any(n in model.col for n in v)}


def diff_stats(a, b):
    d = np.abs(a - b)
    return float(d.max()) if d.size else 0.0, float(np.mean(a == b)) if a.size else 1.0


def report_by_z(model, u, z, th_np, th_jx, title, logl=None):
    print(f"\n== {title}")
    g = groups(model)
    hdr = f"  {'z bin':13s} {'N':>7s} " + " ".join(f"{k:>18s}" for k in g)
    if logl is not None:
        hdr += f" {'|dlogL| max':>12s}"
    print(hdr)
    print(f"  {'':13s} {'':7s} " + " ".join(f"{'max|d|  bitwise':>18s}" for _ in g))
    worst = 0.0
    for lo, hi in zip(Z_EDGES[:-1], Z_EDGES[1:]):
        s = (z >= lo) & (z < hi)
        if not s.any():
            continue
        line = f"  {lo:5.2f}-{hi:<7.2f} {s.sum():7d} "
        for k, cols in g.items():
            dm, fb = diff_stats(th_np[s][:, cols], th_jx[s][:, cols])
            worst = max(worst, dm)
            line += f" {dm:9.1e} {100 * fb:6.1f}%  "
        if logl is not None:
            dl = np.abs(logl[0][s] - logl[1][s])
            dl = dl[np.isfinite(dl)]
            line += f" {dl.max() if dl.size else 0:12.1e}"
        print(line)
    finite = bool(np.all(np.isfinite(th_jx)))
    print(f"  {verdict(model, u, th_np, th_jx, title + ' theta')} (tol {TOL_THETA:g});"
          f"  all JAX theta finite: {check(finite, title + ' finite')}")
    return worst


TOL_CDF = 1e-13       # max |delta F| for logmass where theta differs (see tail_check)


def mass_cdf(model, z, m):
    """numpy's own conditional CDF F(logmass | z), normalised, at m (per row)."""
    mp = model.mprior
    pdf = mp.pdf_rows(z)
    h = np.diff(mp.grid)
    cum = np.concatenate([np.zeros((z.size, 1)),
                          np.cumsum(0.5 * (pdf[:, 1:] + pdf[:, :-1]) * h, axis=1)], axis=1)
    i = np.clip(np.searchsorted(mp.grid, m, side="right") - 1, 0, h.size - 1)
    r = np.arange(z.size)
    p0, p1 = pdf[r, i], pdf[r, i + 1]
    dx = m - mp.grid[i]
    F = cum[r, i] + p0 * dx + 0.5 * (p1 - p0) / h[i] * dx ** 2
    return F / cum[:, -1]


def tail_check(model, u, th_np, th_jx):
    """Where theta differs by more than TOL_THETA. XLA contracts u*cum[-1] - c0 in the
    mass inversion into a fused multiply-add (it does so when the CDF is shared across
    the batch, i.e. fixed z), which is MORE exact than numpy's rounded product. Where
    target - c0 cancels (the flat far tail of the mass function, u within ~1e-9 of 1)
    that last-bit difference moves logmass visibly. Judge those rows fairly:
      (a) logmass: numpy's own CDF at both values must agree to TOL_CDF;
      (b) everything else: numpy's transform CONDITIONED ON JAX's logmass must match
          JAX's theta to TOL_THETA.
    Returns (n rows over tol, max |dF|, max |d theta| after conditioning, ok)."""
    bad = np.flatnonzero(np.abs(th_np - th_jx).max(axis=1) > TOL_THETA)
    if bad.size == 0:
        return 0, 0.0, 0.0, True
    c = model.col
    z = th_np[bad, c["zred"]] if "zred" in c else np.full(bad.size, model.fixed["zred"])
    zj = th_jx[bad, c["zred"]] if "zred" in c else z
    dz = float(np.abs(z - zj).max())
    dF = float(np.max(np.abs(mass_cdf(model, z, th_np[bad, c["logmass"]])
                             - mass_cdf(model, z, th_jx[bad, c["logmass"]]))))
    real = model.mprior
    m_jax = th_jx[bad, c["logmass"]]

    class _Fixed:          # numpy transform with logmass taken from JAX
        def ppf(self, u_, z_):
            return m_jax
    model.mprior = _Fixed()
    try:
        th_cond = model._prior_transform_block(u[bad])     # unblocked: m_jax is per row
    finally:
        model.mprior = real
    dcond = float(np.abs(th_cond - th_jx[bad]).max())
    ok = dz <= TOL_THETA and dF <= TOL_CDF and dcond <= TOL_THETA
    return bad.size, dF, dcond, ok


def verdict(model, u, a, b, label):
    """PASS if theta agrees to TOL_THETA, or if every row that does not passes
    tail_check. Returns the printable verdict."""
    raw = float(np.abs(a - b).max())
    if raw <= TOL_THETA:
        return f"max|d theta| {raw:.1e} [{check(True, label)}]"
    n, dF, dcond, ok = tail_check(model, u, a, b)
    return (f"max|d theta| {raw:.1e} in {n} rows -> tail check: max|dF(logmass)| {dF:.1e}"
            f" (tol {TOL_CDF:g}), downstream given JAX logmass {dcond:.1e} "
            f"[{check(ok, label)}]")


def in_bounds(model, th):
    ok = True
    for n, c in model.col.items():
        lo, hi = model.bounds[n]
        ok &= bool(np.all((th[:, c] >= lo) & (th[:, c] <= hi)))
    return ok


def loglike_safe(model, th):
    out = np.empty(th.shape[0])
    for i in range(0, th.shape[0], 5000):
        blk = th[i:i + 5000]
        try:
            out[i:i + 5000] = model.loglike(blk)
        except EmulatorRefusal:
            out[i:i + 5000] = [_one_loglike(model, t) for t in blk]
    return out


def _one_loglike(model, t):
    try:
        return float(model.loglike(t[None])[0])
    except EmulatorRefusal:
        return -np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--n-fixed", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mass-prior", choices=["exact", "tabulated"], default="exact")
    ap.add_argument("--catalog", default=os.path.join(HERE, "uncover_test.csv"))
    ap.add_argument("--id", default="58858")
    args = ap.parse_args()
    global MASS_PRIOR
    MASS_PRIOR = args.mass_prior
    rng = np.random.default_rng(args.seed)

    phot = read_photometry(args.catalog, [args.id])[0]
    model = build(phot)
    pt = JaxPriorTransform(model)
    d = model.n_dim
    print(f"{args.id}: {d} free params, {int(model.valid.sum())} bands, mu {model.mu:.4f}, jax {jax.__version__}, numpy {np.__version__}, "
          f"mass prior: {MASS_PRIOR}")

    # ---- 1. free z ------------------------------------------------------------------
    u = rng.random((args.n, d))
    th_np, th_jx = compare(model, pt, u)
    report_by_z(model, u, th_np[:, 0], th_np, th_jx, f"1a. free z, {args.n} random u")
    print(f"  JAX theta inside final bounds: {check(in_bounds(model, th_jx), '1a bounds')}")

    dead = np.load(os.path.join(HERE, "results/58858_free_L1000.npz"))
    ud = dead["u"]
    th_np, th_jx = compare(model, pt, ud)
    ll = (loglike_safe(model, th_np), loglike_safe(model, th_jx))
    report_by_z(model, ud, th_np[:, 0], th_np, th_jx,
                f"1b. free z, u of the {ud.shape[0]} NSS dead points of 58858 "
                f"(+ log L of {args.id})", logl=ll)
    fin = np.isfinite(ll[0]) & np.isfinite(ll[1])
    same_inf = bool(np.all(np.isfinite(ll[0]) == np.isfinite(ll[1])))
    dll = float(np.max(np.abs(ll[0][fin] - ll[1][fin])))
    print(f"  max |delta log L| {dll:.2e} (tol {TOL_LOGL:g}): "
          f"{check(dll <= TOL_LOGL, '1b logL')}; same -inf set: {check(same_inf, '1b -inf')}")

    # ---- 2. fixed z grid ------------------------------------------------------------
    z_grid = [0.01, 0.02, 0.04, 0.1, 0.149, 0.15, 0.151, 0.2, 0.3, 0.5, 1.0, 1.6, 2.0, 2.5,
              2.999, 3.0, 3.001, 3.5, 3.999, 4.0, 4.001, 4.5, 5.0, 5.5, 6.0, 7.0, 7.999, 8.0,
              9.0, 10.0, 11.0, 11.999, 12.0, 12.001, 13.0, 14.0, 15.0]
    print(f"\n== 2. fixed z: {len(z_grid)} redshifts, {args.n_fixed} random u each")
    print(f"  {'z':>7s} " + " ".join(f"{k:>18s}" for k in ["logmass", "logzsol",
                                                         "sfh ratios", "1-D priors"]))
    worst2 = 0.0
    for zf in z_grid:
        mz = build(phot, fix_z=zf)
        pz = JaxPriorTransform(mz)
        uz = rng.random((args.n_fixed, mz.n_dim))
        a, b = compare(mz, pz, uz)
        line = f"  {zf:7.3f} "
        for k, cols in groups(mz).items():
            dm, fb = diff_stats(a[:, cols], b[:, cols])
            worst2 = max(worst2, dm)
            line += f" {dm:9.1e} {100 * fb:6.1f}%  "
        if not (np.all(np.isfinite(b)) and in_bounds(mz, b)):
            line += "  NONFINITE/OUT OF BOUNDS"
            FAILS.append(f"2 fixed z={zf} finite/bounds")
        v = verdict(mz, uz, a, b, f"2 fixed z={zf}")
        print(line + ("" if "FAIL" not in v else "  " + v))
    print(f"  worst |delta theta| {worst2:.2e} (tol {TOL_THETA:g}); "
          f"{'all pass' if not any(f.startswith('2 ') for f in FAILS) else 'FAILURES above'}")

    # ---- 3. z windows ---------------------------------------------------------------
    windows = [(0.01, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 3.6), (3.6, 4.16), (4.16, 6.0),
               (6.0, 15.0), (2.9, 3.1), (3.9, 4.1), (11.9, 12.1), (0.01, 0.2), (14.0, 15.0)]
    print(f"\n== 3. z windows, {args.n_fixed} random u each")
    for w in windows:
        mw = build(phot, z_window=w)
        pw = JaxPriorTransform(mw)
        uw = rng.random((args.n_fixed, mw.n_dim))
        a, b = compare(mw, pw, uw)
        dm = float(np.abs(a - b).max())
        dz = float(np.abs(a[:, 0] - b[:, 0]).max())
        ok = np.all(np.isfinite(b)) and in_bounds(mw, b)
        print(f"  window {w[0]:5.2f}-{w[1]:<6.2f} z range [{b[:, 0].min():.3f}, "
              f"{b[:, 0].max():.3f}]  max|d zred| {dz:.1e}  "
              f"{verdict(mw, uw, a, b, f'3 window {w}')}"
              f"{'' if ok else '  NONFINITE/OUT OF BOUNDS'}")
        if not ok:
            FAILS.append(f"3 window {w} finite/bounds")

    # ---- 4. unit-cube edges ---------------------------------------------------------
    edge = np.array([0.0, 1e-300, 1e-15, 1e-9, 0.5, 1 - 1e-9, 1 - 1e-16, 1.0])
    print(f"\n== 4. unit-cube edges: u values {edge.tolist()}")
    for label, kw in [("free z", {}), ("fixed z=0.01", dict(fix_z=0.01)),
                      ("fixed z=3", dict(fix_z=3.0)), ("fixed z=4", dict(fix_z=4.0)),
                      ("fixed z=15", dict(fix_z=15.0))]:
        me = build(phot, **kw)
        pe = JaxPriorTransform(me)
        ue = np.concatenate([np.repeat(edge[:, None], me.n_dim, axis=1),
                             rng.choice(edge, size=(20000, me.n_dim))])
        a, b = compare(me, pe, ue)
        dm = float(np.abs(a - b).max())
        ok = bool(np.all(np.isfinite(b)) and in_bounds(me, b))
        print(f"  {label:13s} {ue.shape[0]} rows, {100 * np.mean(a == b):.1f}% of values "
              f"bitwise: {verdict(me, ue, a, b, '4 edges ' + label)}; finite & in bounds "
              f"[{check(ok, '4 edges finite ' + label)}]")
        if dm > 0:
            i, j = np.unravel_index(np.argmax(np.abs(a - b)), a.shape)
            print(f"      worst: {me.free[j]} at u={ue[i, j]!r} (z u={ue[i, 0]!r}): "
                  f"numpy {a[i, j]!r} jax {b[i, j]!r}")

    # ---- 4b. the other mass function (const_phi=True: Leja+20 only) ------------------
    mc = FitModel(model.emu, phot.bands, phot.flux, phot.err, phot.mu,
                  nzsfh=dict(C.NZSFH, const_phi=True))
    if MASS_PRIOR == "tabulated":
        use_tabulated_mass_prior(mc)
    uc = rng.random((args.n_fixed * 5, mc.n_dim))
    a, b = compare(mc, JaxPriorTransform(mc), uc)
    report_by_z(mc, uc, a[:, 0], a, b, f"4b. const_phi=True, free z, {uc.shape[0]} random u")

    # ---- 5. internals ---------------------------------------------------------------
    print("\n== 5. internals vs sfh.ExpectedRatios, by z bin")
    zz = np.concatenate([rng.uniform(0.01, 15, args.n // 2),
                         10 ** rng.uniform(-2, np.log10(15), args.n // 2)])
    mm = rng.uniform(7.0, 12.5, zz.size)
    ze_np = model.ratios.z_eff(zz, mm)
    ze_jx = np.asarray(jax.jit(jax.vmap(pt.z_eff))(zz, mm))
    er_np = model.ratios(zz, mm)
    er_jx = np.asarray(jax.jit(jax.vmap(pt.expected_ratios))(zz, mm))
    print(f"  {'z bin':13s} {'max|d z_eff|/z_eff':>19s} {'max|d ratio|':>13s}")
    for lo, hi in zip(Z_EDGES[:-1], Z_EDGES[1:]):
        s = (zz >= lo) & (zz < hi)
        print(f"  {lo:5.2f}-{hi:<7.2f} {np.max(np.abs(ze_np[s] - ze_jx[s]) / ze_np[s]):19.1e} "
              f"{np.max(np.abs(er_np[s] - er_jx[s])):13.1e}")

    # ---- 6. single row == batch; timing ---------------------------------------------
    print("\n== 6. un-jitted single row vs jitted batch; timing")
    us = jnp.asarray(u[:50])
    single = np.stack([np.asarray(pt(r)) for r in us])
    batch = np.asarray(pt.batch(us))
    dm = float(np.abs(single - batch).max())
    print(f"  max|single - batch| {dm:.1e}: {check(dm <= TOL_THETA, '6 single vs batch')}")
    for nrow in (1000, 10000, 100000):
        ub = jnp.asarray(rng.random((nrow, d)))
        pt.batch(ub).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(3):
            pt.batch(ub).block_until_ready()
        tj = (time.perf_counter() - t0) / 3
        un = np.asarray(ub)
        t0 = time.perf_counter()
        model.prior_transform(un)
        tn = time.perf_counter() - t0
        print(f"  {nrow:6d} rows: jax {1e6 * tj / nrow:6.2f} us/row, numpy "
              f"{1e6 * tn / nrow:6.2f} us/row")

    print("\nRESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
