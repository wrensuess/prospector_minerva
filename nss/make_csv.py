"""Write an efsps-photfit CSV (nJy) for a few UNCOVER objects.

Photometry from the v5.3.0 SUPER catalogue (fluxes in 10 nJy), magnification
(mu_num_50) and z_16/z_50/z_84 from the matching Prospector SPS catalogue.
"""
import argparse
import csv
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, "/Users/wren/Projects/MINERVA/efsps-photfit")
from efsps_photfit.catalog import DR3_BAND_MAP

p = argparse.ArgumentParser()
p.add_argument("--phot", default="/Users/wren/Surveys/UNCOVER/UNCOVER_v5.3.0_LW_SUPER_CATALOG.fits")
p.add_argument("--sps", default="/Users/wren/Surveys/UNCOVER/UNCOVER_v5.3.0_LW_SUPER_SPScatalog_spsv1.0.fits")
p.add_argument("--ids", nargs="+", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

phot = fits.open(args.phot)[1]
sps = fits.open(args.sps)[1].data
units = {phot.columns[i].name: phot.columns[i].unit for i in range(len(phot.columns))}
d = phot.data
bands = [c[2:] for c in d.columns.names if c.startswith("f_") and c[2:] in DR3_BAND_MAP]
for b in bands:
    if units.get("f_" + b) not in (None, "10**1 nJy", "10 nJy"):
        raise SystemExit(f"unexpected unit {units['f_' + b]!r} for f_{b}")

rows = []
for i in args.ids:
    r = np.where(d["id"] == int(i))[0][0]
    s = np.where(sps["id"] == int(i))[0][0]
    row = {"id": i, "mu": float(sps["mu_num_50"][s]), "z": float(sps["z_50"][s]),
           "z_16": float(sps["z_16"][s]), "z_84": float(sps["z_84"][s])}
    for b in bands:
        f, e = float(d["f_" + b][r]), float(d["e_" + b][r])
        row["f_" + b] = f * 10.0 if np.isfinite(f) else ""
        row["e_" + b] = e * 10.0 if (np.isfinite(e) and e > 0) else ""
    rows.append(row)
    nvalid = sum(1 for b in bands if row["e_" + b] != "")
    print(f"{i}: mu={row['mu']:.3f} z16/50/84={row['z_16']:.3f}/{row['z']:.3f}/{row['z_84']:.3f} "
          f"bands={nvalid}/{len(bands)}")

with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
