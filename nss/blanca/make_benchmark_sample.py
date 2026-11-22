"""The standing spec-z benchmark sample (PRODUCTION_PLAN task Z1), rebuilt on Z1b's truth
(2026-11-19).

Truth: dja_specz_all.csv from dja_specz_catalog.py (Z1b): DJA nirspec_public_v4.5 grade-3
redshifts attached to MINERVA sources by segmentation map and F444W light in a 0.10" aperture
at the spectrum position, one spec-z per source, use_phot = 1 only. See that script's docstring
for every rule. No photo-z is used in any selection or flag that defines `clean`.

Sample: every row of dja_specz_all.csv (Wren 2026-11-19: only use_phot = 1 galaxies, never fit
anything with use_phot = 0; the old "z > 7 with use_phot = 0" exception is gone). Each id is
checked against its catalogue: present, use_phot = 1, same ra/dec. Nothing else is cut; the
flags below select subsets.

Flags (columns of benchmark_sample.csv)
  possible_blend, zspec_conflict, blend_ids, own_frac, second_frac, n_spec, gratings, dz_max,
  all_z, owner_how, spec_sep, grating, dja_root, dja_srcid, dja_file
                 copied from dja_specz_all.csv (definitions there)
  cat_z_spec     the catalogue's own z_spec (NaN = none); cat_z_agrees = within 0.01 (1+z)
  blend_unchecked
                 no F444W light in the aperture for any of the source's spectra (own_frac NaN;
                 owner from the seg pixel, mostly EGS where the F444W mosaic on disk has no
                 data), so possible_blend could not be evaluated (it is False there)
  zspec_pair     another benchmark source within 2" with the same z to 0.01 (1+z) (a physical
                 pair or two pieces of one object hit by different spectra); informational
  clean          ~possible_blend & ~zspec_conflict & ~blend_unchecked (Wren 2026-11-22: the
                 strict definition; a source whose blend check could not run is not clean)
  spec_suspect   the photoz_specz (2026-11-09) definition: eazy and production both outliers,
                 agreeing within 0.05 (1+z), eazy 95% interval excluding z_spec; 1/0, -1 where
                 production is missing (EGS). Photo-z based: reported, NEVER part of clean
  half           'tune' / 'confirm': sha1 of the id string, last hex digit even / odd (the same
                 hash as the 2026-11-17 sample, so an id in both samples keeps its half)

Old (DJA v3, nearest-centroid) criteria and what became of them
  dja_grade3     dropped: every row is grade 3 by construction
  one_to_one     dropped: one spectrum -> one source by construction (Z1b)
  zspec_dup      replaced by zspec_pair: the catalogue's duplicated z_spec (one spectrum copied
                 to neighbours) cannot occur any more; same-z neighbours now come from different
                 spectra, so it is informational, not a clean cut
  use_phot == 1  now true of every row
  neighbour_05, blend_candidate (sources within 0.5" of the target), slit_* / owner_changes /
  plausible_owner_id / owner_conf (nearest-centroid ownership, which used eazy z_phot)
                 dropped: superseded by Z1b's light-based owner and possible_blend; the photo-z
                 based owner choice would break the "no photo-z in the truth" rule
  clean_strict   dropped: the old one removed blend candidates, which possible_blend now does
                 inside `clean`; `clean` also excludes the no-light (blend_unchecked) case

Other columns: F444W magnitude (28.9 - 2.5 log10 f, 10 nJy units), total S/N (quadrature sum of
f/e over valid BB+MB bands), n_bb / n_mb / n_miri valid bands (D3 lists below; valid = the
reader's mask: e > 0, f finite, not f + 5 e < 0; MIRI only COUNTED, for the third band set:
flag_mophongo = 0, e = max(e_mophongo, e_ap)), n_bands_mb (catalogue's own), eazy z_phot / z160
/ z840 (row-matched zout; the baseline), production dynesty z_16/50/84 (spsv1.0, UDS and
COSMOS; row-matched, ids and ra/dec checked identical), in_specz1k, specz1k_group,
spec_suspect_specz1k.

Writes (PRIVATE; MINERVA data never goes in this repo):
  <OUT>/benchmark_sample.csv                      the validation file: z_spec here is the
        benchmark truth (DJA v4.5, Z1b), joined to fit results by (field, id)
  <OUT>/minerva_{uds,cosmos,egs}_benchmark.fits   catalogue rows (sorted by id), all columns,
        byte-for-byte the catalogue's values, z_spec included (Wren 2026-11-22: the MINERVA
        catalogues and copies of them are never changed; the truth lives only in the csv)
  <OUT>/benchmark_tasks.txt                       CATALOG ID DUST SEED (blanca/
        tasklist_layout.sbatch), DUST = mass, SEED = 0

  ~/envs/nss/bin/python blanca/make_benchmark_sample.py [--specz CSV] [--out DIR]
  (needs DarkPhoenix; ~1 min)
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

D = "/Volumes/DarkPhoenix/Surveys/MINERVA/"
CATS = {
    "UDS": dict(
        cat=D + "UDS/catalogs/n3.0_m3.1_v1.2.1/MINERVA-UDS_n3.0_m3.1_v1.2.1_ACS+WEBB_Kf444w_SUPER_CATALOG_mophongo.fits",
        zout=D + "UDS/catalogs/n3.0_m3.1_v1.2.1/MINERVA-UDS_n3.0_m3.1_v1.2.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.eazypy.zout.fits",
        sps=D + "UDS/sps/n3.0_v1.2/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_CATALOG_SPScatalog_spsv1.0.fits"),
    "COSMOS": dict(
        cat=D + "COSMOS/catalogs/n3.0_m3.0_v1.0.1/MINERVA-COSMOS_n3.0_m3.0_v1.0.1_ACS+WEBB_Kf444w_SUPER_CATALOG_mophongo.fits",
        zout=D + "COSMOS/catalogs/n3.0_v1.1/MINERVA-COSMOS_n3.0_m3.0_v1.0.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.eazypy.zout.fits",
        sps=D + "COSMOS/sps/n3.0_v1.0/MINERVA-COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_CATALOG_SPScatalog_spsv1.0.fits"),
    "EGS": dict(
        cat=D + "EGS/catalogs/n2.0_m2.1_v1.3.1/MINERVA-EGS_n2.0_m2.1_v1.3.1_ACS+WEBB_Kf444w_SUPER_CATALOG_mophongo.fits",
        zout=D + "EGS/catalogs/n2.0_v1.3/MINERVA-EGS_n2.0_m2.1_v1.3.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.eazypy.zout.fits",
        sps=None),
}
PRIV = "/Users/wren/Projects/minerva/nss_private_handoff/"
SPECZ = PRIV + "benchmark/specz_v45/dja_specz_all.csv"
SPECZ1K = PRIV + "specz1k/specz1k_sample.csv"
OUT = PRIV + "benchmark/v45"

ZTOL = 0.01       # |dz| / (1 + z)
PAIR_R = 2.0      # arcsec, zspec_pair
COPY = ["grating", "dja_root", "dja_srcid", "dja_file", "n_spec", "gratings", "zspec_conflict",
        "dz_max", "all_z", "possible_blend", "blend_ids", "own_frac", "second_frac",
        "owner_how", "spec_sep", "cat_z_spec"]

# D3 band sets (PRODUCTION_PLAN): BB = every HST band + JWST wide bands + F410M;
# MB = the other NIRCam medium bands; F098M and F470N have no emulator band (not counted).
BB = ["f275wu", "f336wu", "f435w", "f475w", "f606w", "f775w", "f814w", "f850lp", "f105w",
      "f125w", "f140w", "f160w", "f070w", "f090w", "f115w", "f150w", "f200w", "f277w",
      "f356w", "f444w", "f410m"]
MB = ["f140m", "f162m", "f182m", "f210m", "f250m", "f300m", "f335m", "f360m", "f430m",
      "f460m", "f480m"]


def valid_flux(f, e):
    f = np.asarray(f, float)
    e = np.asarray(e, float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(f) & (e > 0) & ~((f < 0) & (f + 5 * e < 0))


def half_of(i):
    return "tune" if int(hashlib.sha1(str(int(i)).encode()).hexdigest()[-1], 16) % 2 == 0 \
        else "confirm"


def field_block(field, P, sz, s1k):
    """Return (rows DataFrame for this field, catalogue row indices of the sample, HDUList)."""
    hdul = fits.open(P["cat"], memmap=True)
    d = hdul[1].data
    names = hdul[1].columns.names
    n = len(d)
    ids = np.asarray(d["id"]).astype(np.int64)
    ra = np.asarray(d["ra"], float)
    dec = np.asarray(d["dec"], float)
    up_all = np.asarray(d["use_phot"]).astype(int)
    row_of = {int(v): k for k, v in enumerate(ids)}

    z = fits.open(P["zout"], memmap=True)[1].data
    if len(z) != n or not np.array_equal(np.asarray(z["id"]), ids):
        raise SystemExit(f"{field}: eazy zout not row-matched to the catalogue")
    sps = None
    if P["sps"]:
        s = fits.open(P["sps"], memmap=True)[1].data
        if len(s) != n or not np.array_equal(np.asarray(s["id"]), ids) or \
                not (np.array_equal(np.asarray(s["ra"], float), ra)
                     and np.array_equal(np.asarray(s["dec"], float), dec)):
            raise SystemExit(f"{field}: production catalogue not row-matched (id / ra / dec)")
        sps = s

    S = sz[sz.field == field].sort_values("id").reset_index(drop=True)
    miss = [i for i in S.id if int(i) not in row_of]
    if miss:
        raise SystemExit(f"{field}: spec-z ids not in the catalogue: {miss[:10]}")
    rows = np.array([row_of[int(i)] for i in S.id], int)
    if not (np.allclose(ra[rows], S.ra, rtol=0, atol=1e-9)
            and np.allclose(dec[rows], S.dec, rtol=0, atol=1e-9)):
        raise SystemExit(f"{field}: spec-z ra/dec differ from the catalogue")
    bad = (up_all[rows] != 1) | (S.use_phot.values != 1)
    if bad.any():
        raise SystemExit(f"{field}: {bad.sum()} spec-z sources with use_phot != 1 "
                         f"({S.id[bad].tolist()[:10]}); the benchmark never fits those")
    print(f"[{field}] catalogue {n} rows; spec-z sources {len(S)} (all use_phot = 1)",
          file=sys.stderr)

    zs = S.z_spec.values.astype(float)
    out = pd.DataFrame(dict(field=field, id=ids[rows], ra=ra[rows], dec=dec[rows], z_spec=zs,
                            use_phot=up_all[rows]))
    for c in COPY:
        out[c] = S[c].values
    out["blend_ids"] = out.blend_ids.fillna("")
    out["cat_z_agrees"] = np.abs(out.cat_z_spec - zs) / (1 + zs) < ZTOL

    # ---- photometry summaries
    f444 = np.asarray(d["f_f444w"], float)[rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mag444"] = np.where(f444 > 0, 28.9 - 2.5 * np.log10(f444), np.nan)
    snr2 = np.zeros(len(rows))
    for key, lst in (("n_bb", BB), ("n_mb", MB)):
        cnt = np.zeros(len(rows), int)
        for b in lst:
            if "f_" + b not in names:
                continue
            fb = np.asarray(d["f_" + b], float)[rows]
            eb = np.asarray(d["e_" + b], float)[rows]
            v = valid_flux(fb, eb)
            cnt += v
            snr2 += np.where(v, (fb / np.where(v, eb, 1)) ** 2, 0)
        out[key] = cnt
    out["snr_tot"] = np.sqrt(snr2)
    # MIRI: counted only (for the BB+MB+MIRI band set); never read into a fit here
    miri = sorted({c[2:-9] for c in names if c.startswith("f_") and c.endswith("_mophongo")})
    cnt = np.zeros(len(rows), int)
    for b in miri:
        fm = np.asarray(d[f"f_{b}_mophongo"], float)[rows]
        em = np.asarray(d[f"e_{b}_mophongo"], float)[rows]
        ea = np.asarray(d[f"e_{b}_ap"], float)[rows] if f"e_{b}_ap" in names else np.full(len(rows), np.nan)
        e = np.fmax(em, ea)
        cnt += valid_flux(fm, e) & (np.asarray(d[f"flag_mophongo_{b}"])[rows] == 0)
    out["n_miri"] = cnt
    out["n_bands_mb"] = np.asarray(d["n_bands_mb"])[rows]

    # ---- baselines (no fits): eazy and production dynesty
    ez = np.asarray(z["z_phot"], float)[rows]
    out["eazy_z"] = ez
    out["eazy_z16"] = np.asarray(z["z160"], float)[rows]
    out["eazy_z84"] = np.asarray(z["z840"], float)[rows]
    ez025 = np.asarray(z["z025"], float)[rows]
    ez975 = np.asarray(z["z975"], float)[rows]
    for c in ("z_16", "z_50", "z_84"):
        out["prod_" + c] = np.asarray(sps[c], float)[rows] if sps is not None else np.nan

    # ---- flags
    out["blend_unchecked"] = ~np.isfinite(out.own_frac.values.astype(float))
    cs = SkyCoord(out.ra.values * u.deg, out.dec.values * u.deg)
    i1, i2, _, _ = cs.search_around_sky(cs, PAIR_R * u.arcsec)
    k = (i1 != i2) & (np.abs(zs[i1] - zs[i2]) / (1 + zs[i1]) < ZTOL)
    partner = {}
    for a, b in zip(i1[k], i2[k]):
        partner.setdefault(a, []).append(int(out.id.iat[b]))
    out["zspec_pair"] = [kq in partner for kq in range(len(out))]
    out["zspec_pair_with"] = [" ".join(map(str, partner.get(kq, []))) for kq in range(len(out))]
    if sps is not None:
        pz = out.prod_z_50.values
        oe = ~(ez > 0) | (np.abs(ez - zs) / (1 + zs) > 0.15)
        op = np.abs(pz - zs) / (1 + zs) > 0.15
        agree = np.abs(ez - pz) / (1 + zs) < 0.05
        excl = ~((zs >= ez025) & (zs <= ez975))
        out["spec_suspect"] = (oe & op & agree & excl).astype(int)
    else:
        out["spec_suspect"] = -1
    out["clean"] = (~out.possible_blend.astype(bool) & ~out.zspec_conflict.astype(bool)
                    & ~out.blend_unchecked.astype(bool))
    out["half"] = [half_of(i) for i in out.id]
    m = s1k[s1k.field == field].set_index("id")
    out["in_specz1k"] = out.id.isin(m.index)
    out["specz1k_group"] = [m.group.get(i, "") for i in out.id]
    out["spec_suspect_specz1k"] = [bool(m.spec_suspect.get(i, False)) for i in out.id]
    return out, rows, hdul


def write_subset(hdul, rows, path):
    """Catalogue rows, all columns, values unchanged (z_spec included; the benchmark truth is
    only in benchmark_sample.csv). Checked column by column after writing."""
    hdu = hdul[1]
    data = hdu.data[rows]
    sub = fits.BinTableHDU(data=data, header=hdu.header)
    sub.header["HISTORY"] = (f"Z1 benchmark subset ({len(rows)} rows) of "
                             f"{os.path.basename(hdul.filename())} (make_benchmark_sample.py); "
                             "values unchanged")
    fits.HDUList([fits.PrimaryHDU(header=hdul[0].header), sub]).writeto(path, overwrite=True)
    a = fits.open(path)[1].data
    if a.columns.names != hdu.columns.names:
        raise SystemExit(f"{path}: columns differ from the catalogue")
    for c in hdu.columns.names:
        x, y = np.asarray(a[c]), np.asarray(data[c])
        if not (np.array_equal(x, y) or (x.dtype.kind == "f" and np.array_equal(x, y, equal_nan=True))):
            raise SystemExit(f"{path}: column {c} differs from the catalogue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specz", default=SPECZ, help="dja_specz_all.csv from dja_specz_catalog.py")
    ap.add_argument("--specz1k", default=SPECZ1K)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sz = pd.read_csv(a.specz)
    if sz.duplicated(["field", "id"]).any():
        raise SystemExit("spec-z table has duplicate (field, id)")
    s1k = pd.read_csv(a.specz1k)
    allrows, tasks = [], []
    for field, P in CATS.items():
        df, rows, hdul = field_block(field, P, sz, s1k)
        fn = f"minerva_{field.lower()}_benchmark.fits"
        write_subset(hdul, rows, os.path.join(a.out, fn))
        df["catalog_file"] = fn
        tasks += [f"{fn} {i} mass 0" for i in df.id]
        allrows.append(df)
        print(f"[{field}] {len(df)} rows -> {fn}; clean {df.clean.sum()}", file=sys.stderr)
    out = pd.concat(allrows, ignore_index=True)
    if len(out) != len(sz):
        raise SystemExit(f"{len(sz) - len(out)} spec-z rows not in UDS / COSMOS / EGS")
    out.to_csv(os.path.join(a.out, "benchmark_sample.csv"), index=False, float_format="%.10g")
    with open(os.path.join(a.out, "benchmark_tasks.txt"), "w") as fh:
        fh.write("\n".join(tasks) + "\n")
    print(f"total {len(out)} rows, {len(tasks)} tasks", file=sys.stderr)


if __name__ == "__main__":
    main()
