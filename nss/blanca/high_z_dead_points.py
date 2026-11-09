"""For every fit of one galaxy: which dead points ever sat at high redshift, how well they
fitted, and when they were discarded. Small output, so the per-fit .npz (dead points in
deletion order, then the final live points) can stay on Blanca.

  python high_z_dead_points.py --id 1152437 --zmin 4.0 results/blanca_layout_*_1152437_k*_L*
  -> <dir>/high_z_dead_points_<id>.npz per run directory:
     seed (per point), order (index in deletion order / number of dead points: 0 = first
     discarded, 1 = end of run), zred, logl, theta (all parameters); plus per-fit
     n_dead, max_logl, z_at_max.
"""
import argparse
import glob
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--id", required=True)
ap.add_argument("--zmin", type=float, default=4.0)
ap.add_argument("dirs", nargs="+")
args = ap.parse_args()

for d in args.dirs:
    files = sorted(glob.glob(os.path.join(d, f"{args.id}_seed*.npz")),
                   key=lambda f: int(re.search(r"_seed(\d+)\.npz$", f).group(1)))
    seed, order, zred, logl, theta = [], [], [], [], []
    n_dead, max_logl, z_at_max, seeds = [], [], [], []
    for f in files:
        s = int(re.search(r"_seed(\d+)\.npz$", f).group(1))
        z = np.load(f)
        th, ll = z["theta"], z["logl"]
        zc = list(z["names"]).index("zred")
        sel = th[:, zc] >= args.zmin
        n = th.shape[0]
        seed.append(np.full(sel.sum(), s)); order.append(np.flatnonzero(sel) / n)
        zred.append(th[sel, zc]); logl.append(ll[sel]); theta.append(th[sel])
        seeds.append(s); n_dead.append(n); i = int(np.argmax(ll))
        max_logl.append(float(ll[i])); z_at_max.append(float(th[i, zc]))
    out = os.path.join(d, f"high_z_dead_points_{args.id}.npz")
    np.savez_compressed(out, seed=np.concatenate(seed), order=np.concatenate(order),
                        zred=np.concatenate(zred), logl=np.concatenate(logl),
                        theta=np.concatenate(theta).astype(np.float32),
                        fit_seed=np.array(seeds), n_dead=np.array(n_dead),
                        max_logl=np.array(max_logl), z_at_max=np.array(z_at_max),
                        names=np.array([str(n) for n in z["names"]]), zmin=args.zmin)
    print(f"{d}: {len(files)} fits, {sum(len(x) for x in zred)} dead points at z >= {args.zmin} -> {out}",
          flush=True)
