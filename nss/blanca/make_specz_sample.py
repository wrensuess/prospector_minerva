"""The expanded spec-z validation sample (2026-11-13, ~1050 galaxies, UDS + COSMOS): every
galaxy of the earlier dust-prior spec-z validation (673; minerva_{uds,cosmos}_specval.fits,
STATUS.md section 6 item 5b), so the prior change can be compared fit for fit, plus a random
sample stratified in z_spec that fills the redshifts that set never covered (it has no
galaxies at z 1-3).

Pool: photoz_specz/sample_<field>.fits (private; catalogue z_spec > 0, use_phot = 1, ACS
coverage, >= 4 medium bands, F444W S/N > 3), without zspec_dup sources (one spectrum matched to
several neighbours; STATUS / memory). spec_suspect sources are kept and flagged in the CSV.
Per z_spec bin, NEW galaxies are split between the fields in proportion to what each has.

Writes (private: MINERVA photometry never goes in the repo; nss/minerva_*.fits is git-ignored)
  nss/minerva_uds_specz1k.fits, nss/minerva_cosmos_specz1k.fits   cut from the same catalogue
      files and with the same columns as the specval subsets (checked: the 673 old rows are
      identical)
  <private>/specz1k/specz1k_sample.csv   field, id, z_spec, group, in_specval, spec_suspect,
      mag444, production z_50 / logM / dust2, eazy z

  python blanca/make_specz_sample.py          (from nss/; needs /Volumes/DarkPhoenix mounted)
"""
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

NSS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = "/Users/wren/Projects/minerva/photoz_specz/sample_{}.fits"
OLD_GROUPS = "/Users/wren/Projects/minerva/dust_prior/specz_validation_sample.csv"
PRIVATE = "/Users/wren/Projects/minerva/nss_private_handoff/specz1k"
CATS = {
    "UDS": ("/Volumes/DarkPhoenix/Surveys/MINERVA/UDS/catalogs/n3.0_m3.1_v1.2.1/"
            "MINERVA-UDS_n3.0_m3.1_v1.2.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.fits",
            "minerva_uds_specval.fits", "minerva_uds_specz1k.fits"),
    "COSMOS": ("/Volumes/DarkPhoenix/Surveys/MINERVA/COSMOS/catalogs/n3.0_m3.0_v1.0.1/"
               "MINERVA-COSMOS_n3.0_m3.0_v1.0.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.fits",
               "minerva_cosmos_specval.fits", "minerva_cosmos_specz1k.fits"),
}
#: new galaxies per z_spec bin, both fields together (z > 5.5: all that are left)
BINS = [(0.0, 0.5, 30), (0.5, 1.0, 40), (1.0, 1.5, 60), (1.5, 2.0, 60), (2.0, 3.0, 70),
        (3.0, 4.0, 60), (4.0, 5.5, 50), (5.5, 20.0, None)]
SEED = 20261113


def cut(catalog, ids, columns_from, out):
    """make_minerva_subset.py's cut: the rows of `ids`, the columns of `columns_from`."""
    with fits.open(catalog, memmap=True) as hdul, fits.open(columns_from) as ref:
        hdu = hdul[1]
        keep = ref[1].columns.names
        rows = np.flatnonzero(np.isin(hdu.data["id"], ids))
        if len(rows) != len(ids):
            raise SystemExit(f"{len(ids) - len(rows)} ids not in {catalog}")
        rows = rows[np.argsort(hdu.data["id"][rows])]
        sub = fits.BinTableHDU.from_columns(fits.ColDefs([hdu.columns[c] for c in keep]),
                                            nrows=len(rows))
        for c in keep:
            sub.data[c] = hdu.data[c][rows]
        if "EXTNAME" in hdu.header:
            sub.header["EXTNAME"] = hdu.header["EXTNAME"]
        sub.header["HISTORY"] = f"columns as in {columns_from}"
        sub.header["HISTORY"] = f"subset of {catalog}: {len(rows)} spec-z galaxies (make_specz_sample.py)"
        fits.HDUList([fits.PrimaryHDU(header=hdul[0].header), sub]).writeto(out, overwrite=True)
    # the old rows must be identical to the specval subset
    a, b = fits.open(out)[1].data, fits.open(columns_from)[1].data
    ia = {int(v): k for k, v in enumerate(a["id"])}
    for c in keep:
        x = np.asarray([a[c][ia[int(v)]] for v in b["id"]])
        y = np.asarray(b[c])
        same = np.array_equal(x, y) or (x.dtype.kind == "f" and np.array_equal(x, y, equal_nan=True))
        if not same:
            raise SystemExit(f"{out}: column {c} differs from {columns_from} on the old rows")
    return len(rows)


def main():
    rng = np.random.default_rng(SEED)
    old_groups = pd.read_csv(OLD_GROUPS).set_index("id")["group"].to_dict()
    pools = {}
    for f, (_, old_file, _) in CATS.items():
        s = Table.read(POOL.format(f)).to_pandas()
        s["field"] = f
        s["in_specval"] = np.isin(s["id"], fits.open(os.path.join(NSS, old_file))[1].data["id"])
        pools[f] = s
    picks = {f: list(pools[f].loc[pools[f]["in_specval"], "id"]) for f in CATS}
    group = {int(i): f"specval_{old_groups.get(int(i), '?')}" for f in CATS for i in picks[f]}
    for lo, hi, n in BINS:
        avail = {f: pools[f][(~pools[f]["in_specval"]) & (pools[f]["zspec_dup"] == 0)
                             & (pools[f]["z_spec"] >= lo) & (pools[f]["z_spec"] < hi)]
                 for f in CATS}
        tot = sum(len(v) for v in avail.values())
        for f, v in avail.items():
            k = len(v) if n is None else min(len(v), int(round(n * len(v) / tot)))
            ch = rng.choice(np.asarray(v["id"]), k, replace=False) if k else []
            picks[f] += list(ch)
            group.update({int(i): f"zbin_{lo:g}-{hi:g}" for i in ch})
    rows = []
    for f, (cat, old_file, out) in CATS.items():
        ids = np.array(sorted(int(i) for i in picks[f]))
        n = cut(cat, ids, os.path.join(NSS, old_file), os.path.join(NSS, out))
        s = pools[f].set_index("id").loc[ids]
        for i, r in s.iterrows():
            rows.append(dict(field=f, id=int(i), z_spec=r["z_spec"], group=group[int(i)],
                             in_specval=bool(r["in_specval"]), spec_suspect=bool(r["spec_suspect"]),
                             mag444=r["mag444"], prod_z50=r["pr_z_50"], prod_logM=r["pr_mstar_50"],
                             prod_dust2=r["pr_dust2_50"], eazy_z=r["ez_z_phot"]))
        print(f"{f}: {n} galaxies -> {out}", file=sys.stderr)
    os.makedirs(PRIVATE, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PRIVATE, "specz1k_sample.csv"), index=False, float_format="%.5g")
    print(df.groupby(["field", "group"]).size().to_string(), file=sys.stderr)
    print(f"total {len(df)}; spec_suspect {int(df['spec_suspect'].sum())}", file=sys.stderr)


if __name__ == "__main__":
    main()
