"""Write a small MINERVA catalogue with only the requested rows: same columns, same units
(header TUNITs), so photometry.read_fits treats it exactly like the full catalogue. For
shipping test galaxies to Blanca without the 345,792-row file.

  python make_minerva_subset.py --out minerva_uds_test9.fits          (the 9 test galaxies)
"""
import argparse

import numpy as np
from astropy.io import fits

from photometry import MINERVA_TEST_IDS, MINERVA_UDS

ap = argparse.ArgumentParser()
ap.add_argument("--catalog", default=MINERVA_UDS)
ap.add_argument("--ids", nargs="+", default=MINERVA_TEST_IDS)
ap.add_argument("--out", required=True)
args = ap.parse_args()

with fits.open(args.catalog, memmap=True) as hdul:
    hdu = hdul[1]
    ids = np.array([int(i) for i in args.ids])
    rows = np.flatnonzero(np.isin(hdu.data["id"], ids))
    found = set(int(v) for v in hdu.data["id"][rows])
    missing = [i for i in ids if int(i) not in found]
    if missing:
        raise SystemExit(f"id(s) {missing} not in {args.catalog}")
    sub = fits.BinTableHDU(data=hdu.data[rows], header=hdu.header)
    sub.header["HISTORY"] = f"subset of {args.catalog}: ids {' '.join(args.ids)}"
    fits.HDUList([fits.PrimaryHDU(header=hdul[0].header), sub]).writeto(args.out, overwrite=True)
print(f"wrote {args.out}: {len(rows)} rows of {args.catalog}")
