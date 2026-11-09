"""Offline check of a "large initial prior sample" start for NSS.

Idea under test: draw N >> n_live points from the prior, delete the worst N - n_live
without replacement (no MCMC needed), and start ordinary NSS from the best n_live. The
first live set is then the top n_live of N prior draws instead of n_live random draws.

This script only evaluates prior draws, no sampling. For each N it reports how many of the
top n_live lie near the galaxy's best known redshift solution (``--z-target``), their best
log L, and the threshold (the n_live-th best log L) they must beat. With N = n_live and
``--blanca-seeds`` it rebuilds exactly the initial live sets of the Blanca runs
(nss_jax.run_nss_jax: PRNGKey(seed) -> split -> uniform), so they can be matched with the
recorded outcome of each seed.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nss_efsps import build
from photometry import read_photometry
from jax_model import JaxLogLike
from tabulated_mass_prior import use_tabulated_mass_prior


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--catalog", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--z-target", type=float, required=True,
                   help="redshift of the best known solution")
    p.add_argument("--dz", type=float, default=0.1,
                   help="window half-width in units of (1+z_target)")
    p.add_argument("--n-live", type=int, default=1000)
    p.add_argument("--n", type=float, nargs="+", default=[1e3, 1e4, 1e5, 1e6])
    p.add_argument("--repeats", type=int, nargs="+", default=[20, 20, 20, 5])
    p.add_argument("--batch", type=int, default=20000)
    p.add_argument("--mass-prior", choices=["exact", "tabulated"], default="tabulated")
    p.add_argument("--blanca-seeds", type=int, default=20,
                   help="also rebuild the initial live sets of Blanca seeds 0..N-1")
    p.add_argument("--blanca-results", default=None,
                   help="glob of Blanca per-seed JSONs for this id, to attach outcomes")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    phot = read_photometry(args.catalog, [args.id])[0]
    model = build(phot, None, None)
    if args.mass_prior == "tabulated":
        use_tabulated_mass_prior(model)
    ll_fn = JaxLogLike(model)
    iz = model.col["zred"]
    d = model.n_dim

    @jax.jit
    def eval_rows(u):
        return jax.vmap(lambda r: (ll_fn(r), ll_fn.pt(r)[iz]))(u)

    def evaluate(u):
        out_l, out_z = [], []
        for s in range(0, u.shape[0], args.batch):
            chunk = u[s:s + args.batch]
            pad = args.batch - chunk.shape[0]
            if pad:  # keep one compiled shape
                chunk = jnp.concatenate([chunk, jnp.full((pad, d), 0.5)])
            l, z = eval_rows(chunk)
            out_l.append(np.asarray(l)[:args.batch - pad])
            out_z.append(np.asarray(z)[:args.batch - pad])
        return np.concatenate(out_l), np.concatenate(out_z)

    zt, half = args.z_target, args.dz * (1 + args.z_target)

    def stats(l, z):
        top = np.argsort(l)[::-1][:args.n_live]
        thr = l[top[-1]]
        win = np.abs(z - zt) < half
        top_win = win[top]
        return dict(threshold=float(thr), n_top_in_window=int(top_win.sum()),
                    best_in_window=float(l[win].max()) if win.any() else None,
                    n_window_total=int(win.sum()),
                    best_overall=float(l[top[0]]), z_best_overall=float(z[top[0]]),
                    top_z_hist=np.histogram(z[top], bins=[0, .5, 1, 2, 3, 4, 5, 7, 10, 20])[0]
                    .tolist())

    res = dict(id=phot.id, z_target=zt, window=[zt - half, zt + half], n_live=args.n_live,
               mass_prior=args.mass_prior, by_n={}, blanca_seeds=[])
    t0 = time.time()
    for n, reps in zip(args.n, args.repeats):
        n = int(n)
        rows = []
        for r in range(reps):
            key = jax.random.fold_in(jax.random.PRNGKey(10_000 + r), n)
            u = jax.random.uniform(key, (n, d), dtype=jnp.float64)
            rows.append(stats(*evaluate(u)))
        res["by_n"][str(n)] = rows
        k = np.array([x["n_top_in_window"] for x in rows])
        b = np.array([x["best_in_window"] if x["best_in_window"] is not None else -np.inf
                      for x in rows])
        t = np.array([x["threshold"] for x in rows])
        print(f"[{phot.id}] N={n:>8d} x{reps:2d}: top-{args.n_live} in window: median "
              f"{np.median(k):.0f} (range {k.min()}-{k.max()}, zero in {np.sum(k == 0)}); "
              f"best in window median {np.median(b):.0f}; threshold median {np.median(t):.0f}"
              f"  [{time.time() - t0:.0f} s]", flush=True)

    outcomes = {}
    if args.blanca_results:
        for f in glob.glob(args.blanca_results):
            j = json.load(open(f))
            s = j["settings"]["seed"] if "seed" in j.get("settings", {}) else None
            k_in = j["settings"].get("num_inner_steps")
            ok = abs(j["z_at_max_logL"] - zt) < half
            outcomes.setdefault(s, []).append((k_in, bool(ok), j["max_logL"]))
    for s in range(args.blanca_seeds):
        key = jax.random.PRNGKey(s)
        key, k0 = jax.random.split(key)
        u = jax.random.uniform(k0, (args.n_live, d), dtype=jnp.float64)
        st = stats(*evaluate(u))
        st["seed"] = s
        st["outcomes"] = sorted(outcomes.get(s, []))
        res["blanca_seeds"].append(st)
    if res["blanca_seeds"]:
        print(f"[{phot.id}] Blanca initial live sets (seed: n in window, best in window, "
              f"found target at k=...):")
        for st in res["blanca_seeds"]:
            found = [k for k, ok, _ in st["outcomes"] if ok]
            print(f"   seed {st['seed']:2d}: {st['n_top_in_window']:3d}  "
                  f"{st['best_in_window'] if st['best_in_window'] is not None else float('nan'):8.1f}"
                  f"  found at k={found} of {[k for k, _, _ in st['outcomes']]}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{phot.id}_prior_start.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
