"""Write a small MINERVA catalogue with only the requested rows: same columns, same units
(header TUNITs), so photometry.read_fits treats it exactly like the full catalogue. For
shipping test galaxies to Blanca without the 345,792-row file.

  python make_minerva_subset.py --out minerva_uds_test9.fits          (the 9 test galaxies)

The 15-galaxy set (the 9 plus the 6 AGN test galaxies), cut from n3.0_m3.1_v1.2.1 but
keeping only the v1.2 columns, i.e. no MIRI bands:
  python make_minerva_subset.py --catalog V121 --ids 9 agn \
      --columns-from minerva_uds_test9.fits --out minerva_uds_test15.fits
"""
import argparse

import numpy as np
from astropy.io import fits

from photometry import (MINERVA_AGN_TEST_IDS, MINERVA_TEST_IDS, MINERVA_UDS,
                        MINERVA_UDS_V121)

ap = argparse.ArgumentParser()
ap.add_argument("--catalog", default=MINERVA_UDS,
                help="catalogue path; 'V121' = the n3.0_m3.1_v1.2.1 wMIRI catalogue")
ap.add_argument("--ids", nargs="+", default=MINERVA_TEST_IDS,
                help="ids; '9' = the 9 test galaxies, 'agn' = the 6 AGN test galaxies")
ap.add_argument("--columns-from", default=None,
                help="keep only this FITS table's columns (in its order)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
if args.catalog == "V121":
    args.catalog = MINERVA_UDS_V121
ids_str = []
for i in args.ids:
    ids_str += MINERVA_TEST_IDS if i == "9" else MINERVA_AGN_TEST_IDS if i == "agn" else [i]

with fits.open(args.catalog, memmap=True) as hdul:
    hdu = hdul[1]
    ids = np.array([int(i) for i in ids_str])
    rows = np.flatnonzero(np.isin(hdu.data["id"], ids))
    found = set(int(v) for v in hdu.data["id"][rows])
    missing = [i for i in ids if int(i) not in found]
    if missing:
        raise SystemExit(f"id(s) {missing} not in {args.catalog}")
    if args.columns_from:
        with fits.open(args.columns_from) as ref:
            keep = ref[1].columns.names
        absent = [c for c in keep if c not in hdu.columns.names]
        if absent:
            raise SystemExit(f"columns {absent} of {args.columns_from} not in {args.catalog}")
        cols = fits.ColDefs([hdu.columns[c] for c in keep])
        sub = fits.BinTableHDU.from_columns(cols, nrows=len(rows))
        for c in keep:
            sub.data[c] = hdu.data[c][rows]
        for k in ("EXTNAME",):
            if k in hdu.header:
                sub.header[k] = hdu.header[k]
        dropped = [c for c in hdu.columns.names if c not in keep]
        sub.header["HISTORY"] = f"columns as in {args.columns_from}; dropped {len(dropped)}"
    else:
        sub = fits.BinTableHDU(data=hdu.data[rows], header=hdu.header)
    sub.header["HISTORY"] = f"subset of {args.catalog}: ids {' '.join(ids_str)}"
    fits.HDUList([fits.PrimaryHDU(header=hdul[0].header), sub]).writeto(args.out, overwrite=True)
print(f"wrote {args.out}: {len(rows)} rows of {args.catalog}")
