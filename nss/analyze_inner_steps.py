"""Convergence of NSS versus inner slice steps (k), from the 20-seed Blanca runs.

For each galaxy and k (results/blanca_layout_<job>_<id>_k<k>_L<L>_<prior>/draws.npz):
  logZ      mean and std over the 20 seeds, mean quoted error, R = std / quoted error
            (R ~ 1 when the quoted error is honest; 20 seeds give std to ~16%)
  S_med, S_max   fit-to-fit scatter of each parameter's median, in units of the pooled
            posterior half-width (16-84 / 2): median over the 26 parameters, and the worst one.
            Monte Carlo floor from 2000 draws per fit: ~0.03.
  shift     |pooled median(k) - pooled median(k=260)| / half-width, worst parameter: does the
            pooled posterior still move as k grows?
  z range   min - max of the 20 fits' redshift medians (catastrophic photo-z jumps show here)

  python analyze_inner_steps.py blanca_runs/results
"""
import glob
import os
import re
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "blanca_runs/results"
runs = {}
for d in glob.glob(os.path.join(root, "blanca_layout_*_k*_L*")):
    m = re.search(r"blanca_layout_\d+_(\d+)_k(\d+)_L(\d+)_(\w+)$", d)
    runs[(m.group(1), int(m.group(2)))] = np.load(os.path.join(d, "draws.npz"))


def stats(z):
    D = z["draws"].astype(float)                              # (fits, N, dim)
    pooled = D.reshape(-1, D.shape[-1])
    p16, p50, p84 = np.percentile(pooled, [16, 50, 84], axis=0)
    hw = 0.5 * (p84 - p16)
    med = np.median(D, axis=1)                                 # (fits, dim)
    s = med.std(axis=0, ddof=1) / hw
    return dict(p50=p50, hw=hw, s=s, med=med, lz=z["logZ"], lze=z["logZ_err"],
                names=[str(n) for n in z["names"]])


gals = sorted({g for g, _ in runs})
ks = sorted({k for _, k in runs})
print(f"{len(runs)} runs: galaxies {gals}, k {ks}\n")
summary = {}
for g in gals:
    ref = stats(runs[(g, max(ks))]) if (g, max(ks)) in runs else None
    print(f"== {g}")
    print(f"  {'k':>4s} {'logZ mean':>9s} {'std':>5s} {'quoted':>6s} {'R':>5s} | {'S_med':>5s} "
          f"{'S_max':>5s} {'(param)':24s} | {'shift vs k=' + str(max(ks)):>14s} {'(param)':24s} | z range")
    for k in ks:
        if (g, k) not in runs:
            continue
        st = stats(runs[(g, k)])
        R = st["lz"].std(ddof=1) / st["lze"].mean()
        i = int(np.argmax(st["s"]))
        sh = np.abs(st["p50"] - ref["p50"]) / ref["hw"] if ref is not None else np.zeros_like(st["s"])
        j = int(np.argmax(sh))
        zc = st["names"].index("zred")
        zr = f"{st['med'][:, zc].min():.3f}-{st['med'][:, zc].max():.3f}"
        summary[(g, k)] = dict(R=R, S_med=float(np.median(st["s"])), S_max=float(st["s"][i]),
                               shift=float(sh[j]))
        print(f"  {k:4d} {st['lz'].mean():9.2f} {st['lz'].std(ddof=1):5.2f} {st['lze'].mean():6.2f} "
              f"{R:5.1f} | {np.median(st['s']):5.2f} {st['s'][i]:5.2f} {st['names'][i]:24s} | "
              f"{sh[j]:14.2f} {st['names'][j] if k != max(ks) else '':24s} | {zr}")
    print()
print("== all galaxies: median (worst) over the 9 galaxies")
print(f"  {'k':>4s} {'R':>12s} {'S_med':>12s} {'S_max':>12s} {'shift':>12s}")
for k in ks:
    v = [summary[(g, k)] for g in gals if (g, k) in summary]
    f = lambda key: f"{np.median([x[key] for x in v]):5.2f} ({max(x[key] for x in v):5.2f})"
    print(f"  {k:4d} {f('R'):>12s} {f('S_med'):>12s} {f('S_max'):>12s} {f('shift'):>12s}")
