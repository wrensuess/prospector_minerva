"""Mock photometry from the efsps w256 emulator with known truth, written as an
efsps CSV (nJy, mu = 1). Noise per band: the median UNCOVER error of the template
object's valid bands (applied to every band), plus the efsps 5% floor is NOT added
to the data (the likelihood adds it).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nss_efsps import build, read_catalog
from efsps_photfit.catalog import DR3_BAND_MAP

MOCKS = {
    # name: overrides of the drawn prior sample
    "mockA_z4.5_dusty": dict(zred=4.5, logmass=9.8, dust2=1.2, gas_logu=-2.5),
    "mockB_z2.8_dusty": dict(zred=2.8, logmass=9.3, dust2=1.6, gas_logu=-2.5),
    "mockC_z6.2_eelg": dict(zred=6.2, logmass=9.2, dust2=0.1, gas_logu=-1.5),
}

p = argparse.ArgumentParser()
p.add_argument("--template-csv", default="uncover_test.csv")
p.add_argument("--template-id", default="25665")
p.add_argument("--out", default="mocks.csv")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

tmpl = read_catalog(args.template_csv, ids=[args.template_id])[0]
model = build(tmpl)
rng = np.random.default_rng(args.seed)
err = np.nanmedian(tmpl.err[tmpl.valid])          # maggies, one depth for all bands
inv = {v: k for k, v in DR3_BAND_MAP.items()}
rows, truths = [], {}
for name, over in MOCKS.items():
    theta = model.prior_transform(rng.uniform(0.2, 0.8, size=(1, model.n_dim)))[0]
    for k, v in over.items():
        theta[model.col[k]] = v
    model.mu = 1.0
    f, _, _ = model.model_fluxes(theta[None, :])
    f = f[0]
    noisy = f + rng.normal(0.0, err, size=f.size)
    row = {"id": name, "mu": 1.0, "z": over["zred"]}
    for b, fv in zip(tmpl.bands, noisy):
        row["f_" + inv[b]] = fv * 3631e9
        row["e_" + inv[b]] = err * 3631e9
    rows.append(row)
    truths[name] = {n: float(theta[i]) for i, n in enumerate(model.free)}
    snr = f / err
    print(f"{name}: S/N median {np.median(snr):.1f}, max {snr.max():.1f}")

with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(os.path.splitext(args.out)[0] + "_truth.json", "w") as fh:
    json.dump(truths, fh, indent=1)
