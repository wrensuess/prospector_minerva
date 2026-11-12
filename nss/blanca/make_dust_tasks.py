"""Task list for the dust-prior tests (STATUS.md section 6, item 5b): every galaxy in the given
MINERVA subset catalogues x dust prior settings x seeds, one line per fit:
    CATALOG ID DUST SEED
Lines are shuffled (fixed seed) so every chunk of tasklist_layout.sbatch mixes galaxies and settings.

  python blanca/make_dust_tasks.py [--settings none,mass,sfr,mass_sfr] [--seeds 0,1,2] CATALOG ...
e.g. (run from nss/)
  the 48-galaxy test, 2026-11-11 (576 fits):
    python blanca/make_dust_tasks.py minerva_uds_dust27.fits minerva_cosmos_dust21.fits > blanca/dust_tasks.txt
  the same galaxies at the adopted setting (144 fits):
    python blanca/make_dust_tasks.py --settings mass minerva_uds_dust27.fits minerva_cosmos_dust21.fits \
        > blanca/dust48_adopted_tasks.txt
  the spec-z validation, with and without the prior, one seed (1346 fits):
    python blanca/make_dust_tasks.py --settings none,mass --seeds 0 minerva_uds_specval.fits \
        minerva_cosmos_specval.fits > blanca/specval_tasks.txt
"""
import argparse
import sys

import numpy as np
from astropy.io import fits

ap = argparse.ArgumentParser()
ap.add_argument("--settings", default="none,mass,sfr,mass_sfr", help="comma-separated")
ap.add_argument("--seeds", default="0,1,2", help="comma-separated")
ap.add_argument("catalogs", nargs="+")
a = ap.parse_args()
a.settings = a.settings.split(",")
a.seeds = [int(v) for v in a.seeds.split(",")]
lines = [f"{cat} {int(i)} {d} {s}" for cat in a.catalogs for i in fits.open(cat)[1].data["id"]
         for d in a.settings for s in a.seeds]
rng = np.random.default_rng(11)
for k in rng.permutation(len(lines)):
    print(lines[k])
print(f"{len(lines)} tasks", file=sys.stderr)
