"""Validate jax_model.JaxLogLike against numpy FitModel.loglike(prior_transform(u)) for
58858, by redshift bin:
  1. free z, random u
  2. the NSS dead points of the numpy-callback run (also against the log L stored by
     that run)
  3. fixed z on a grid 0.01-15
  4. outside the cube -> -inf; on the cube's faces -> finite and equal
  5. timing per row at batch sizes 100 (= num_delete), 1000, 10000

Pass: |delta log L| <= 1e-8 * max(1, |log L|) everywhere.

Run:  ~/anaconda3/envs/yaargh/bin/python test_jax_loglike.py
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

import test_jax_prior as TJP
from test_jax_prior import HERE, Z_EDGES, build, loglike_safe
from photometry import read_photometry
from jax_model import JaxLogLike

REL_TOL = 1e-8
FAILS = []


def ref_loglike(model, u):
    th = model.prior_transform(u)
    return loglike_safe(model, th), th


def report(model, ll, u, label, z=None):
    a, th = ref_loglike(model, u)
    b = np.asarray(ll.batch(jnp.asarray(u)))
    z = th[:, model.col["zred"]] if z is None else z
    rel = np.abs(a - b) / np.maximum(1.0, np.abs(a))
    same_inf = np.array_equal(np.isfinite(a), np.isfinite(b))
    rel = np.where(np.isfinite(a) & np.isfinite(b), rel, 0.0)
    print(f"\n== {label}")
    print(f"  {'z bin':13s} {'N':>7s} {'median logL':>12s} {'max|dlogL|':>11s} "
          f"{'max rel':>9s} {'bitwise':>8s}")
    for lo, hi in zip(Z_EDGES[:-1], Z_EDGES[1:]):
        s = (z >= lo) & (z < hi)
        if not s.any():
            continue
        f = s & np.isfinite(a)
        print(f"  {lo:5.2f}-{hi:<7.2f} {s.sum():7d} {np.median(a[f]):12.4g} "
              f"{np.max(np.abs(a[f] - b[f])):11.1e} {rel[s].max():9.1e} "
              f"{100 * np.mean(a[s] == b[s]):7.1f}%")
    ok = rel.max() <= REL_TOL and same_inf
    if not ok:
        FAILS.append(label)
    print(f"  max rel {rel.max():.1e} (tol {REL_TOL:g}), same -inf set {same_inf}: "
          f"{'ok' if ok else 'FAIL'}")
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--n-fixed", type=int, default=5000)
    ap.add_argument("--mass-prior", choices=["exact", "tabulated"], default="exact")
    ap.add_argument("--catalog", default=os.path.join(HERE, "uncover_test.csv"))
    ap.add_argument("--id", default="58858")
    args = ap.parse_args()
    TJP.MASS_PRIOR = args.mass_prior
    rng = np.random.default_rng(2)
    phot = read_photometry(args.catalog, [args.id])[0]
    model = build(phot)
    ll = JaxLogLike(model)
    d = model.n_dim
    print(f"{args.id}: {d} free params, {int(model.valid.sum())} valid bands, mu {model.mu:.4f}, "
          f"mass prior: {args.mass_prior}")

    report(model, ll, rng.random((args.n, d)), f"1. free z, {args.n} random u")

    dead = np.load(os.path.join(HERE, "results/58858_free_L1000.npz"))
    a, b = report(model, ll, dead["u"], f"2. u of the {dead['u'].shape[0]} NSS dead points of 58858")
    stored = dead["logl"]
    f = np.isfinite(stored)
    if args.id == "58858":
        print(f"  vs log L stored by the numpy-callback run: max|d| numpy-now "
              f"{np.max(np.abs(a[f] - stored[f])):.1e}, jax {np.max(np.abs(b[f] - stored[f])):.1e}")

    for zf in [0.01, 0.1, 0.15, 0.5, 1.0, 2.0, 3.0, 3.5, 3.88, 4.0, 5.0, 7.0, 10.0, 12.0, 15.0]:
        mz = build(phot, fix_z=zf)
        report(mz, JaxLogLike(mz), rng.random((args.n_fixed, mz.n_dim)),
               f"3. fixed z={zf}, {args.n_fixed} random u", z=np.full(args.n_fixed, zf))

    print("\n== 4. outside / on the cube")
    u = rng.random((4000, d))
    out = u.copy()
    i = rng.integers(0, d, 4000)
    out[np.arange(4000), i] = np.where(rng.random(4000) < 0.5, -1e-12, 1 + 1e-12)
    bo = np.asarray(ll.batch(jnp.asarray(out)))
    ok_out = bool(np.all(bo == -np.inf))
    print(f"  4000 rows just outside one face: all -inf: {'ok' if ok_out else 'FAIL'}")
    if not ok_out:
        FAILS.append("4 outside")
    face = rng.choice(np.array([0.0, 1.0, 0.5]), size=(4000, d))
    report(model, ll, face, "4. rows on the cube's faces/corners (u in {0, 0.5, 1})")

    print("\n== 5. timing (jitted vmap, CPU)")
    for n in (100, 1000, 10000):
        ub = jnp.asarray(rng.random((n, d)))
        ll.batch(ub).block_until_ready()
        reps = max(3, 20000 // n)
        t0 = time.perf_counter()
        for _ in range(reps):
            ll.batch(ub).block_until_ready()
        t = (time.perf_counter() - t0) / reps
        tp = time.perf_counter()
        for _ in range(reps):
            ll.pt.batch(ub).block_until_ready()
        tp = (time.perf_counter() - tp) / reps
        print(f"  batch {n:6d}: {1e6 * t / n:6.2f} us/row (of which prior transform "
              f"{1e6 * tp / n:5.2f})")

    print("\nRESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
