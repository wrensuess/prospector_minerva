"""Shrink a 20-fit run directory to what the convergence analysis needs: N equal-weight
posterior draws per fit (resampled from the fit's weighted dead points), in one small file
per directory, so 8.6 GB of per-fit .npz files need not be copied.

  python posterior_draws.py results/blanca_layout_*_k*          -> <dir>/draws.npz

draws.npz: draws (n_fits, N, n_dim) float32, names, seeds, logZ, logZ_err, wall_s.
"""
import glob
import json
import os
import re
import sys

import numpy as np

N = 2000
for d in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(d, "*_seed*.npz")),
                   key=lambda f: int(re.search(r"_seed(\d+)\.npz$", f).group(1)))
    if not files:
        print(f"{d}: no fits, skipped")
        continue
    rng = np.random.default_rng(0)
    draws, seeds, lz, lze, wall = [], [], [], [], []
    names = None
    for f in files:
        z = np.load(f)
        w = z["w"] / z["w"].sum()
        draws.append(z["theta"][rng.choice(w.size, N, p=w)].astype(np.float32))
        names = [str(n) for n in z["names"]]
        s = json.load(open(f[:-4] + ".json"))
        seeds.append(int(re.search(r"_seed(\d+)\.npz$", f).group(1)))
        lz.append(s["logZ"]); lze.append(s["logZ_err"]); wall.append(s["wall_s"])
    np.savez_compressed(os.path.join(d, "draws.npz"), draws=np.stack(draws), names=np.array(names),
                        seeds=np.array(seeds), logZ=np.array(lz), logZ_err=np.array(lze),
                        wall_s=np.array(wall))
    print(f"{d}: {len(files)} fits -> draws.npz", flush=True)
