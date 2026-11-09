"""Compare two NSS result files (results/<id>_<tag>.json/.npz) side by side.

  python compare_runs.py results/58858_free_L1000 results/58858_jax_free_L1000

Per parameter: 16/50/84 percentiles of each, and the shift of the median in units of the
reference run's half 16-84 width. Plus logZ, max log L, chi2, ESS, z by bin, wall time.
"""
import json
import sys

import numpy as np


def load(stem):
    with open(stem + ".json") as fh:
        return json.load(fh)


def main(a_stem, b_stem):
    a, b = load(a_stem), load(b_stem)
    print(f"A = {a_stem}\nB = {b_stem}\n")
    dz = b["logZ"] - a["logZ"]
    err = np.hypot(a["logZ_err"], b["logZ_err"])
    print(f"  {'':22s} {'A':>12s} {'B':>12s}")
    print(f"  {'logZ':22s} {a['logZ']:8.2f}±{a['logZ_err']:.2f} {b['logZ']:8.2f}±{b['logZ_err']:.2f}"
          f"   B-A {dz:+.2f} ({dz / err:+.1f} sigma)")
    for k in ("max_logL", "chi2_at_max", "ess", "iterations", "wall_s", "z_at_max_logL"):
        va, vb = a.get(k), b.get(k)
        print(f"  {k:22s} {va:12.4g} {vb:12.4g}")
    if b.get("compile_s") is not None:
        print(f"  {'compile_s (B)':22s} {'':12s} {b['compile_s']:12.4g}")
    print(f"  {'z weight by bin':22s} A {a.get('z_weight_by_bin')}\n  {'':22s} B "
          f"{b.get('z_weight_by_bin')}\n")
    print(f"  {'parameter':26s} {'A p16 / p50 / p84':>30s} {'B p16 / p50 / p84':>30s} "
          f"{'dmed/width':>10s}")
    worst = 0.0
    for n, pa in a["percentiles"].items():
        pb = b["percentiles"][n]
        width = 0.5 * (pa[2] - pa[0])
        s = (pb[1] - pa[1]) / width if width > 0 else 0.0
        worst = max(worst, abs(s))
        print(f"  {n:26s} {pa[0]:9.4g} {pa[1]:9.4g} {pa[2]:9.4g}  "
              f"{pb[0]:9.4g} {pb[1]:9.4g} {pb[2]:9.4g}  {s:+9.2f}")
    print(f"\n  largest median shift: {worst:.2f} x (half 16-84 width of A)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
