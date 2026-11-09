"""Combine redshift-window NSS runs into one posterior and evidence, and compare with
free-redshift runs.

Each window run used efsps's p(z) restricted to the table nodes inside the window and
renormalised, so

    Z_total = sum_w P_w Z_w,    P_w = integral of the full-range p(z) over the window,

provided the window edges are table nodes (they then tile the full range exactly).
The combined posterior weights each window's samples by P_w Z_w / Z_total.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, os.environ.get("EFSPS_PARENT", "/Users/wren/Projects/MINERVA/efsps-photfit"))
from efsps_photfit.priors import RedshiftPrior

p = argparse.ArgumentParser()
p.add_argument("--id", required=True)
p.add_argument("--results", default="results")
p.add_argument("--windows", nargs="+", required=True, help="run tags of the window runs")
p.add_argument("--free", nargs="*", default=[], help="run tags of free-redshift runs")
args = p.parse_args()

full = RedshiftPrior(0.01, 15.0)
zn, pn = full.z_nodes, full.pdf_nodes


def prior_mass(lo, hi):
    m = (zn >= lo - 1e-9) & (zn <= hi + 1e-9)
    return float(np.trapezoid(pn[m], zn[m]))


edges = np.array([0.0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15.1])


def zhist(z, w):
    return np.array([w[(z >= a) & (z < b)].sum() for a, b in zip(edges[:-1], edges[1:])])


rows, parts = [], []
for tag in args.windows:
    s = json.load(open(os.path.join(args.results, f"{args.id}_{tag}.json")))
    d = np.load(os.path.join(args.results, f"{args.id}_{tag}.npz"))
    lo, hi = s["settings"]["z_window"]
    lo_node = zn[zn >= lo - 1e-9][0]
    hi_node = zn[zn <= hi + 1e-9][-1]
    P = prior_mass(lo_node, hi_node)
    rows.append((tag, lo_node, hi_node, P, s["logZ"], s["logZ_err"], s["max_logL"],
                 s["model_rows_evaluated"], s["wall_s"]))
    iz = list(d["names"]).index("zred")
    parts.append((d["theta"][:, iz], d["w"], s["logZ"] + np.log(P), s["logZ_err"]))

cover = sum(r[3] for r in rows)
logZ_terms = np.array([pt[2] for pt in parts])
logZ_tot = logsumexp(logZ_terms)
# error: propagate each window's logZ error through the sum
frac = np.exp(logZ_terms - logZ_tot)
logZ_err = float(np.sqrt(np.sum((frac * np.array([pt[3] for pt in parts])) ** 2)))

print(f"{'window':>14} {'z range':>13} {'P(window)':>10} {'logZ_w':>9} {'+-':>5} "
      f"{'log(P Z)':>9} {'share':>8} {'max logL':>9} {'rows':>10}")
for r, f in zip(rows, frac):
    tag, lo, hi, P, lz, le, ml, nr, wall = r
    print(f"{tag:>14} {lo:5.2f}-{hi:5.2f}  {P:10.4f} {lz:9.2f} {le:5.2f} {lz + np.log(P):9.2f} "
          f"{f:8.2e} {ml:9.2f} {nr:10d}")
print(f"prior mass covered by windows: {cover:.6f} (should be 1)")
print(f"COMBINED windows: logZ = {logZ_tot:.2f} +- {logZ_err:.2f}, "
      f"total model rows {sum(r[7] for r in rows)}")

zc = np.concatenate([pt[0] for pt in parts])
wc = np.concatenate([pt[1] * f for pt, f in zip(parts, frac)])
hc = zhist(zc, wc)
labels = [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]
print("combined p(z) by bin:", {l: f"{v:.3g}" for l, v in zip(labels, hc) if v > 1e-4})

for tag in args.free:
    fn = os.path.join(args.results, f"{args.id}_{tag}.json")
    if not os.path.exists(fn):
        print(f"[{tag}] not finished")
        continue
    s = json.load(open(fn))
    d = np.load(os.path.join(args.results, f"{args.id}_{tag}.npz"))
    iz = list(d["names"]).index("zred")
    h = zhist(d["theta"][:, iz], d["w"])
    print(f"FREE {tag}: logZ = {s['logZ']:.2f} +- {s['logZ_err']:.2f}, max logL "
          f"{s['max_logL']:.2f}, rows {s['model_rows_evaluated']}, wall {s['wall_s']:.0f} s")
    print("   p(z) by bin:", {l: f"{v:.3g}" for l, v in zip(labels, h) if v > 1e-4})
