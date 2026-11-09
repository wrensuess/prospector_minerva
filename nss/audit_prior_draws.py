"""Brute-force audit: evaluate many prior draws of the efsps model for one galaxy and
record the best log-likelihood found in each redshift bin. A nested-sampling fit whose
posterior misses a bin where this audit found a clearly better likelihood has missed
a mode.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nss_efsps import build, read_catalog

p = argparse.ArgumentParser()
p.add_argument("--catalog", required=True)
p.add_argument("--id", required=True)
p.add_argument("--n", type=float, default=1e6)
p.add_argument("--batch", type=int, default=10000)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out", default="results")
args = p.parse_args()

phot = read_catalog(args.catalog, ids=[args.id])[0]
model = build(phot)
rng = np.random.default_rng(args.seed)
edges = np.array([0.0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15.1])
best = np.full(len(edges) - 1, -np.inf)
best_theta = [None] * (len(edges) - 1)
iz = model.col["zred"]
t0 = time.time()
n = int(args.n)
for start in range(0, n, args.batch):
    u = rng.uniform(size=(min(args.batch, n - start), model.n_dim))
    th = model.prior_transform(u)
    ll = model.loglike(th)
    b = np.clip(np.digitize(th[:, iz], edges) - 1, 0, len(best) - 1)
    for k in np.unique(b):
        m = b == k
        j = np.argmax(np.where(m, ll, -np.inf))
        if ll[j] > best[k]:
            best[k], best_theta[k] = float(ll[j]), th[j]
wall = time.time() - t0
out = {"id": phot.id, "n_draws": n, "wall_s": wall,
       "best_logL_by_z_bin": {f"{a:g}-{c:g}": v for a, c, v in zip(edges[:-1], edges[1:], best)
                              if np.isfinite(v)},
       "z_of_best_in_bin": {f"{a:g}-{c:g}": float(t[iz]) for a, c, t in
                            zip(edges[:-1], edges[1:], best_theta) if t is not None}}
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, f"{phot.id}_audit.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
print(f"{n} draws in {wall:.0f} s ({wall / n * 1e6:.0f} us per draw)")
