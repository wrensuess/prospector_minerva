"""One photometry reader for the NSS fits: UNCOVER and MINERVA SUPER catalogues (FITS),
and the efsps-photfit CSV format.

FITS, both surveys (same layout, same units):
  - fluxes f_<band>, errors e_<band> in 10 nJy (AB zero point 28.9); the header unit
    '10**1 nJy' is checked -> maggies;
  - bands via efsps_photfit.catalog.DR3_BAND_MAP (= prospector_minerva's
    utils.filter_dictionary: F606W/F814W are ACS/WFC). A band the emulator lacks (MINERVA's
    F098M) is dropped and listed in meta["dropped_bands"] and printed, never silently;
  - MIRI (F560W-F2550W) is EXCLUDED by default, although the emulator has those bands: Wren's
    rule is to fit MIRI only when explicitly asked (include_miri=True; nss_jax.py
    --include-miri). Excluded MIRI bands are printed and listed in meta["excluded_miri"].
    This matters for the wMIRI catalogues (e.g. UDS n3.0_m3.1_v1.2.1), which would
    otherwise add MIRI to the fit silently;
  - magnification: the 'mu' column if there is one (UNCOVER), else mu = 1 (MINERVA; UDS is
    not lensed). mu only scales the model fluxes;
  - mask (prospector_minerva load_obs, used there for both surveys): a band counts only if
    err > 0, flux finite, and not (flux < 0 and flux + 5 err < 0); masked bands get
    err = NaN, efsps's "missing". mask_negative=False reproduces efsps-photfit's reader.
NOT copied from prospector: its error floor max(unc, 0.05 flux). The efsps likelihood adds
5% of the observed flux in quadrature (config.LIKELIHOOD) instead.

CSV: efsps_photfit.catalog.read_csv, unchanged (that is what uncover_test.csv is).

    phot = read_photometry(path, ["1212906"])[0]       # efsps_photfit.catalog.Photometry
"""
import os
from typing import List, Sequence

import numpy as np

from efsps_photfit.catalog import (DR3_BAND_MAP, DR3_FITS_UNIT, FLUX_UNITS, IGNORED_BANDS,
                                   CatalogRefusal, Photometry, _norm_id, read_csv)

#: JWST/MIRI imaging bands (efsps_photfit.catalog lists them among its JWST bands).
MIRI_BANDS = frozenset(("f560w", "f770w", "f1000w", "f1130w", "f1280w", "f1500w", "f1800w",
                        "f2100w", "f2550w"))

MINERVA_UDS = os.path.expanduser(
    "~/Surveys/MINERVA/UDS/catalogs/n3.0_v1.2/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_CATALOG.fits")

#: n3.0_m3.1_v1.2.1 adds MIRI columns (f770w..f1800w, which ARE emulator bands) and leaves
#: every v1.2 column identical, except that a missing z_spec is -99 instead of NaN (checked
#: 2026-11-09 for the 9 test ids). Subsets cut from it for tests use --columns-from
#: minerva_uds_test9.fits, so they keep the v1.2 (no-MIRI) band set.
MINERVA_UDS_V121 = ("/Volumes/DarkPhoenix/Surveys/MINERVA/UDS/catalogs/n3.0_m3.1_v1.2.1/"
                    "MINERVA-UDS_n3.0_m3.1_v1.2.1_ACS+WEBB_Kf444w_SUPER_CATALOG_wMIRI.fits")

#: the prospector-minerva timing/validation set (speedup_runs/blanca_test/make_taskfiles.sh)
MINERVA_TEST_IDS = ["1212906", "1174326", "1113001", "1169924", "1192179", "1198917",
                    "1241601", "1197855", "1152437"]

#: AGN test set (Wren, 2026-11-09). In the production spsv1.0 catalogue (log fagn uniform from
#: -5 to log 3), the first four go to low z with high fagn while eazy and spec-z agree on high z;
#: the last two are controls, at the right z but needing high fagn (logfagn_16 > 0).
MINERVA_AGN_FAIL_IDS = ["1200052", "1252985", "1079135", "1188204"]
MINERVA_AGN_CONTROL_IDS = ["1057478", "1171434"]
MINERVA_AGN_TEST_IDS = MINERVA_AGN_FAIL_IDS + MINERVA_AGN_CONTROL_IDS


def read_fits(path: str, ids: Sequence[str], mask_negative: bool = True,
              verbose: bool = True, include_miri: bool = False) -> List[Photometry]:
    from astropy.io import fits
    fac = FLUX_UNITS["10nJy"]
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[1]
        cols, names = hdu.columns, hdu.columns.names
        shorts, bands, dropped, excluded_miri = [], [], [], []
        for n in names:
            if not n.startswith("f_"):
                continue
            s = n[2:]
            if s in IGNORED_BANDS:
                continue
            if s in MIRI_BANDS and not include_miri:
                excluded_miri.append(s)
                continue
            if s not in DR3_BAND_MAP:
                dropped.append(s)
                continue
            if "e_" + s not in names:
                raise CatalogRefusal(f"{path}: flux column {n} has no error column e_{s}.")
            unit = (cols[n].unit or "").strip()
            if unit != DR3_FITS_UNIT:
                raise CatalogRefusal(f"{path}: column {n} declares unit {unit!r}, not "
                                     f"{DR3_FITS_UNIT!r}; refusing to guess the conversion.")
            shorts.append(s)
            bands.append(DR3_BAND_MAP[s])
        if verbose and dropped:
            print(f"[photometry] {os.path.basename(path)}: no emulator band for {dropped}; "
                  "dropped.", flush=True)
        if verbose and excluded_miri:
            print(f"[photometry] {os.path.basename(path)}: MIRI bands {excluded_miri} "
                  "excluded (fit them only if explicitly asked: include_miri=True / "
                  "--include-miri).", flush=True)
        data = hdu.data
        index = {_norm_id(v): i for i, v in enumerate(data["id"])}
        want = [_norm_id(i) for i in ids]
        missing = [i for i in want if i not in index]
        if missing:
            raise CatalogRefusal(f"{path}: id(s) {missing[:20]} not found.")
        has_mu = "mu" in names
        out = []
        for i in want:
            r = index[i]
            row = data[r]
            flux = np.array([row["f_" + s] for s in shorts], dtype=float) * fac
            err = np.array([row["e_" + s] for s in shorts], dtype=float) * fac
            use = (err > 0) & np.isfinite(flux)
            n_neg = 0
            if mask_negative:
                neg = use & (flux < 0) & (flux + 5 * err < 0)
                n_neg = int(neg.sum())
                use &= ~neg
            err = np.where(use, err, np.nan)
            mu = float(row["mu"]) if has_mu else 1.0
            if not np.isfinite(mu) or mu <= 0:
                raise CatalogRefusal(f"{path}: object {i} has no valid magnification mu={mu}.")
            meta = {k: (float(row[k]) if np.issubdtype(np.asarray(row[k]).dtype, np.number)
                        else str(row[k]))
                    for k in ("use_phot", "z_phot", "z_spec", "flag_star", "flag_artifact",
                              "flag_lowsnr", "ra", "dec") if k in names}
            zs = meta.get("z_spec", np.nan)
            zc = zs if (np.isfinite(zs) and zs > 0) else meta.get("z_phot", None)
            meta.update(source=os.path.abspath(path), row=int(r), flux_unit="10nJy",
                        dropped_bands=dropped, masked_negative=n_neg,
                        excluded_miri=list(excluded_miri), include_miri=include_miri,
                        mu_source="catalog" if has_mu else "assumed 1 (no mu column)")
            out.append(Photometry(i, list(bands), flux, err, mu, zc, meta))
    return out


def _drop_miri(p: Photometry) -> Photometry:
    """Remove MIRI bands from a Photometry (CSV path), recording them in meta."""
    keep = [k for k, b in enumerate(p.bands)
            if not (b.startswith("jwst_") and b[5:] in MIRI_BANDS)]
    excl = [p.bands[k][5:] for k in range(len(p.bands)) if k not in keep]
    meta = dict(p.meta or {})
    meta.update(excluded_miri=excl, include_miri=False)
    if excl:
        print(f"[photometry] {p.id}: MIRI bands {excl} excluded (fit them only if explicitly "
              "asked: include_miri=True / --include-miri).", flush=True)
    return Photometry(p.id, [p.bands[k] for k in keep], np.asarray(p.flux)[keep],
                      np.asarray(p.err)[keep], p.mu, p.z_catalog, meta)


def read_photometry(path: str, ids: Sequence[str], mask_negative: bool = True,
                    include_miri: bool = False) -> List[Photometry]:
    """FITS (UNCOVER or MINERVA) via read_fits; CSV via efsps-photfit's read_csv.
    MIRI bands are excluded unless include_miri=True (see the module docstring)."""
    low = path.lower()
    if low.endswith((".fits", ".fit", ".fits.gz")):
        return read_fits(path, ids, mask_negative, include_miri=include_miri)
    if low.endswith(".csv"):
        out = read_csv(path, ids)
        return out if include_miri else [_drop_miri(p) for p in out]
    raise CatalogRefusal(f"{path}: unrecognised catalogue type (need .fits or .csv).")
