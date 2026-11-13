"""Gather every per-galaxy postprocessing file (<id>_<tag>_post.h5, postprocess_nss.py) into
the collected HDF5 files and the FITS SPS catalogue. One script, one read of each file:
it replaces the dynesty pipeline's save_perc.py, save_sfh.py, save_chain.py (+ split runs
and combine_chain.py), save_spec.py and save_final_catalog.py.

Outputs in --out (dataset names follow the old files wherever the quantity exists):
  quant_<prior>.h5   objid; (n, 7) percentiles: zred, total_mass, stellar_mass, met, mwa,
                     sfr10/30/100, ssfr10/30/100, dust2, dust_index, dust1_fraction,
                     log_fagn, log_agn_tau, gas_logz, duste_qpah, duste_umin, log_duste_gamma,
                     and the E-FSPS parameters the old model did not have (logsfr_ratios_1-9,
                     gas_logu, gas_logn, delta_*); zred_ml, zred_spec (-99 = none),
                     logZ, logZ_err, ess; ok (row filled)
  sfh_<prior>.h5     objid, percentiles (fractions), agebins_max (log10 yr),
                     sfh (n, 7, 100) Msun/yr on logspace(1, agebins_max, 100) yr;
                     mformed_cum (n, 7, 100) Msun formed before each lookback time and
                     mformed_cum_frac (n, 7, 100) as a fraction of the total formed mass.
                     The draws' per-bin SFRs and bin edges are NOT stored here: they follow
                     exactly from chains' zred, total_mass and logsfr_ratios_1-9
                     (postprocess_nss.sfh_bins_from_chains; ~47 KB per galaxy saved). The
                     per-galaxy files keep them (draws/sfr_bins, draws/agebins).
  chains_<prior>.h5  objid, theta_labels, chains (n, 1000, n_col) float32: the old 20 columns
                     (zred ... log_duste_gamma), then logsfr_ratios_1-9, gas_logu, gas_logn,
                     delta_*, loglike (likelihood of the draw at its own mass)
  spec_<prior>.h5    objid, bands, chi2 (best fit, the fit's own variance), nbands, zred
                     (best fit), obsmag / obsmag_unc (observed maggies, NaN = not fitted),
                     modmag_map (best-fit model maggies), modmag_perc (n, nb, 7)
  sps_catalog/<catalogue>_SPScatalog_<version>.fits   every catalogue row; the old columns
                     (id, ra, dec, z_spec, z_ml, z/mtot/mstar/met/mwa/dust2/dust_index/
                     dust1_fraction/logfagn/sfr10/30/100/ssfr10/30/100 _16/_50/_84, chi2,
                     nbands) plus logZ, logZ_err; rows without a fit are masked.
                     Rest-frame U V J u g i columns appear once the per-galaxy files carry them.
  failed_files.txt, duplicate_ids.txt   (only if there are any)
Rows are sorted by id. Unreadable files keep their row with ok = 0 and NaN data, and are
listed in failed_files.txt; they are masked in the FITS catalogue.

    python make_catalog_nss.py --indir RESULTS --catalog PHOT.fits --out OUT [--workers 20]
"""
import argparse
import glob
import json
import os
import re
import time
from collections import OrderedDict
from datetime import date
from multiprocessing import Pool

import h5py
import numpy as np

OLD_QUANT = OrderedDict([          # quant/chains name -> per-galaxy percentiles name
    ("zred", "zred"), ("total_mass", "logmass"), ("stellar_mass", "stellar_mass"),
    ("met", "logzsol"), ("mwa", "mwa"),
    ("sfr10", ("sfr", 0)), ("sfr30", ("sfr", 1)), ("sfr100", ("sfr", 2)),
    ("ssfr10", ("ssfr", 0)), ("ssfr30", ("ssfr", 1)), ("ssfr100", ("ssfr", 2)),
    ("dust2", "dust2"), ("dust_index", "dust_index"), ("dust1_fraction", "dust1_fraction"),
    ("log_fagn", "log_fagn"), ("log_agn_tau", "log_agn_tau"), ("gas_logz", "gas_logz"),
    ("duste_qpah", "duste_qpah"), ("duste_umin", "duste_umin"),
    ("log_duste_gamma", "log_duste_gamma")])
NEW_PARAMS = ([f"logsfr_ratios_{i}" for i in range(1, 10)] +
              ["gas_logu", "gas_logn", "delta_ionspec_logLratio2", "delta_ionspec_logLratio3",
               "delta_gas_logqion"])
CHAIN_COLS = [("logzsol" if k == "met" else k) for k in OLD_QUANT] + NEW_PARAMS + ["loglike"]
REST_KEYS = ["rest_UVJugi", "rest_UVJugi_colors", "rest_gz", "rest_gz_colors",
             "rest_NUVrJ", "rest_NUVrJ_colors"]
BLOCK = 256                        # rows per write


def file_id(path):
    return int(re.match(r"(\d+)_", os.path.basename(path)).group(1))


def _pick(g, spec):
    if isinstance(spec, tuple):
        return g[spec[0]][spec[1]]
    return g[spec][()]


def read_one(path):
    """-> dict of everything one galaxy contributes, or {'error': ...}."""
    try:
        with h5py.File(path, "r") as f:
            a = f.attrs
            pg, dg = f["percentiles"], f["draws"]
            out = {"path": path, "id": int(a["id"]), "logZ": float(a["logZ"]),
                   "logZ_err": float(a["logZ_err"]), "ess": float(a["ess"]),
                   "spectra": str(a["spectra"])}
            out["quant"] = {k: _pick(pg, s) for k, s in OLD_QUANT.items()}
            out["quant"].update({k: pg[k][()] for k in NEW_PARAMS})
            out["rest"] = {k: pg[k][()] for k in REST_KEYS if k in pg}
            out["agebins_max"] = float(f["sfh_grid"].attrs["agebins_max"])
            out["sfh"] = pg["sfh"][()].T                       # (7, 100), old layout
            out["mformed_cum"] = pg["mformed_cum"][()].T
            out["mformed_cum_frac"] = pg["mformed_cum_frac"][()].T
            cols = [dg[OLD_QUANT[c][0]][:, OLD_QUANT[c][1]] if isinstance(OLD_QUANT.get(c), tuple)
                    else dg[c][()] for c in CHAIN_COLS]
            out["chains"] = np.stack(cols, axis=1).astype(np.float32)
            b = f["bestfit"]
            out.update(zred_ml=float(b.attrs["zred"]), chi2=float(b.attrs["chi2"]),
                       nbands=int(b.attrs["nbands"]), modmag_map=b["model_flux"][()],
                       modmag_perc=f["model_flux_percentiles"][()],
                       obsmag=f["obs/flux"][()], obsmag_unc=f["obs/err"][()],
                       bands=[x.decode() if isinstance(x, bytes) else str(x)
                              for x in f["obs/bands"][()]],
                       percents=a["percents"][()],
                       summary=f["fit"].attrs["summary_json"])
        if out["id"] != file_id(path):
            raise ValueError(f"file name id {file_id(path)} != stored id {out['id']}")
        return out
    except Exception as e:                                     # recorded, never silent
        return {"path": path, "error": f"{type(e).__name__}: {e}"}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--indir", required=True, help="searched recursively for --pattern")
    p.add_argument("--pattern", default="*_post.h5")
    p.add_argument("--catalog", required=True, help="the photometric catalogue the fits used")
    p.add_argument("--out", required=True)
    p.add_argument("--prior", default="nss", help="tag in the collected file names")
    p.add_argument("--sps-version", default="spsv2.0")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    files = glob.glob(os.path.join(args.indir, "**", args.pattern), recursive=True)
    if not files:
        raise SystemExit(f"no {args.pattern} under {args.indir}")
    # one file per id: the most recently written (a rerun replaces an older fit); the rest
    # are listed in duplicate_ids.txt
    files.sort(key=lambda f: (file_id(f), -os.path.getmtime(f), f))
    ids_all = np.array([file_id(f) for f in files])
    first = np.r_[True, ids_all[1:] != ids_all[:-1]]
    dups = [f for f, keep in zip(files, first) if not keep]
    files = [f for f, keep in zip(files, first) if keep]
    ids = ids_all[first]
    n = len(files)
    print(f"{n} galaxies ({len(dups)} duplicate files skipped) under {args.indir}", flush=True)
    if dups:
        with open(os.path.join(args.out, "duplicate_ids.txt"), "w") as fh:
            fh.write("\n".join(dups) + "\n")

    # reference layout (band set, draw count): the most common among the first 50 readable
    # files, so one odd file at the start cannot reject the rest
    sample = [r for r in (read_one(f) for f in files[:50]) if "error" not in r]
    if not sample:
        raise SystemExit("none of the first 50 files is readable")
    keys = [(tuple(r["bands"]), r["chains"].shape) for r in sample]
    best = max(set(keys), key=keys.count)
    ref = sample[keys.index(best)]
    K, ncol = ref["chains"].shape
    nb, npc = ref["modmag_perc"].shape
    has_rest = sorted(ref["rest"])
    summ0 = json.loads(ref["summary"])

    fq = h5py.File(os.path.join(args.out, f"quant_{args.prior}.h5"), "w")
    fs = h5py.File(os.path.join(args.out, f"sfh_{args.prior}.h5"), "w")
    fc = h5py.File(os.path.join(args.out, f"chains_{args.prior}.h5"), "w")
    fp = h5py.File(os.path.join(args.out, f"spec_{args.prior}.h5"), "w")
    kw = dict(compression="gzip", shuffle=True)
    for h in (fq, fs, fc, fp):
        h.create_dataset("objid", data=ids.astype(np.int64))
        h.create_dataset("ok", shape=(n,), dtype=np.int8)
        h.attrs["percents"] = ref["percents"]
        h.attrs["source_dir"] = os.path.abspath(args.indir)
        h.attrs["created"] = str(date.today())
    dq = {k: fq.create_dataset(k, shape=(n, npc), dtype=np.float32, fillvalue=np.nan,
                               chunks=(min(n, 4096), npc), **kw)
          for k in list(OLD_QUANT) + NEW_PARAMS}
    for k in has_rest:
        dq[k] = fq.create_dataset(k, shape=(n,) + ref["rest"][k].shape, dtype=np.float32,
                                  fillvalue=np.nan, **kw)
    sc = {k: fq.create_dataset(k, shape=(n,), dtype=np.float32, fillvalue=np.nan, **kw)
          for k in ("zred_ml", "logZ", "logZ_err", "ess")}
    fs.create_dataset("percentiles", data=ref["percents"] / 100.0)
    ds_age = fs.create_dataset("agebins_max", shape=(n,), dtype=np.float32, fillvalue=np.nan)
    ds_sfh = fs.create_dataset("sfh", shape=(n,) + ref["sfh"].shape, dtype=np.float32,
                               fillvalue=np.nan, chunks=(min(n, 64),) + ref["sfh"].shape, **kw)
    sfh_more = {k: fs.create_dataset(k, shape=(n,) + ref[k].shape, dtype=np.float32,
                                     fillvalue=np.nan, chunks=(min(n, 64),) + ref[k].shape, **kw)
                for k in ("mformed_cum", "mformed_cum_frac")}
    fc.create_dataset("theta_labels", data=np.array(CHAIN_COLS, dtype="S"))
    ds_ch = fc.create_dataset("chains", shape=(n, K, ncol), dtype=np.float32, fillvalue=np.nan,
                              chunks=(min(n, 16), K, ncol), **kw)
    fp.create_dataset("bands", data=np.array(ref["bands"], dtype="S"))
    sp = {"chi2": fp.create_dataset("chi2", shape=(n,), dtype=np.float32, fillvalue=np.nan),
          "nbands": fp.create_dataset("nbands", shape=(n,), dtype=np.int32),
          "zred": fp.create_dataset("zred", shape=(n,), dtype=np.float32, fillvalue=np.nan)}
    for k in ("obsmag", "obsmag_unc", "modmag_map"):
        sp[k] = fp.create_dataset(k, shape=(n, nb), dtype=np.float32, fillvalue=np.nan,
                                  chunks=(min(n, 4096), nb), **kw)
    sp["modmag_perc"] = fp.create_dataset("modmag_perc", shape=(n, nb, npc), dtype=np.float32,
                                          fillvalue=np.nan, chunks=(min(n, 1024), nb, npc), **kw)

    ok = np.zeros(n, np.int8)
    failed = []
    buf = []

    def flush(rows):
        a, b = rows[0][0], rows[-1][0] + 1
        sl = slice(a, b)
        blk = {k: np.full((b - a,) + ds.shape[1:], np.nan, np.float32) for k, ds in dq.items()}
        sca = {k: np.full(b - a, np.nan, np.float32) for k in sc}
        age = np.full(b - a, np.nan, np.float32)
        sfh = np.full((b - a,) + ds_sfh.shape[1:], np.nan, np.float32)
        more = {k: np.full((b - a,) + ds.shape[1:], np.nan, np.float32) for k, ds in sfh_more.items()}
        ch = np.full((b - a, K, ncol), np.nan, np.float32)
        spb = {k: np.full((b - a,) + ds.shape[1:], np.nan if k != "nbands" else 0,
                          np.float32 if k != "nbands" else np.int32) for k, ds in sp.items()}
        for i, r in rows:
            j = i - a
            if "error" in r:
                continue
            for k in dq:
                blk[k][j] = r["quant"][k] if k in r["quant"] else r["rest"][k]
            for k in sc:
                sca[k][j] = r[k]
            age[j] = r["agebins_max"]
            sfh[j] = r["sfh"]
            for k in more:
                more[k][j] = r[k]
            ch[j] = r["chains"]
            spb["chi2"][j], spb["nbands"][j], spb["zred"][j] = r["chi2"], r["nbands"], r["zred_ml"]
            for k in ("obsmag", "obsmag_unc", "modmag_map", "modmag_perc"):
                spb[k][j] = r[k]
        for k, ds in dq.items():
            ds[sl] = blk[k]
        for k, ds in sc.items():
            ds[sl] = sca[k]
        ds_age[sl], ds_sfh[sl], ds_ch[sl] = age, sfh, ch
        for k, ds in sfh_more.items():
            ds[sl] = more[k]
        for k, ds in sp.items():
            ds[sl] = spb[k]

    done = 0
    pool = Pool(args.workers) if args.workers > 1 else None
    it = pool.imap(read_one, files, chunksize=32) if pool else map(read_one, files)
    for i, r in enumerate(it):
        if "error" not in r:
            if r["bands"] != ref["bands"] or r["chains"].shape != (K, ncol):
                r = {"path": r["path"], "error": "bands or draw count differ from the common layout "
                     f"({len(r['bands'])} bands, {r['chains'].shape} vs {nb}, {(K, ncol)})"}
        if "error" in r:
            failed.append(f"{r['path']}\t{r['error']}")
        else:
            ok[i] = 1
        buf.append((i, r))
        if len(buf) == BLOCK:
            flush(buf)
            buf = []
            done += BLOCK
            if done % (BLOCK * 40) == 0:
                print(f"  {done}/{n} ({done / (time.time() - t0):.0f} files/s)", flush=True)
    if buf:
        flush(buf)
    if pool:
        pool.close()
        pool.join()
    for h in (fq, fs, fc, fp):
        h["ok"][:] = ok
    zml = fq["zred_ml"][:]
    lz, lze = fq["logZ"][:], fq["logZ_err"][:]
    quant = {k: fq[k][:] for k in OLD_QUANT}
    rest = {k: fq[k][:] for k in has_rest}
    chi2, nbands = fp["chi2"][:], fp["nbands"][:]

    # ------------------------------------------------------------------ FITS catalogue
    import warnings
    from astropy.io import fits
    from astropy.table import Table, MaskedColumn
    from astropy import units as u
    # 'log Msol' / 'log Zsol' are not FITS units; astropy warns once per column (the dynesty
    # pipeline's save_final_catalog.py silenced all warnings for the same reason)
    warnings.filterwarnings("ignore", category=u.UnitsWarning)
    with fits.open(args.catalog, memmap=True) as hd:
        cd = hd[1].data
        cat = Table({"id": np.asarray(cd["id"]).astype(np.int64),
                     "ra": np.asarray(cd["ra"], float), "dec": np.asarray(cd["dec"], float)})
        zspec_cat = np.asarray(cd["z_spec"], float) if "z_spec" in cd.names else np.full(len(cat), np.nan)
    row_of = {int(v): i for i, v in enumerate(cat["id"])}
    rows = np.array([row_of.get(int(v), -1) for v in ids])
    missing = ids[rows < 0]
    if missing.size:
        print(f"WARNING: {missing.size} fitted ids are not in {args.catalog}: {missing[:10]}")
    good = (rows >= 0) & (ok == 1)
    src, dst = np.flatnonzero(good), rows[good]
    zsp = np.where(np.isfinite(zspec_cat) & (zspec_cat > 0), zspec_cat, -99.0)
    zred_spec = np.where(rows >= 0, zsp[np.maximum(rows, 0)], -99.0)   # per galaxy, -99 = none
    fq.create_dataset("zred_spec", data=zred_spec.astype(np.float32))

    def col(name, data, unit=None):
        arr = np.full(len(cat), np.nan)
        arr[dst] = data[src]
        m = np.ones(len(cat), bool)
        m[dst] = False
        cat.add_column(MaskedColumn(data=arr, name=name, mask=m, unit=unit))

    col("z_spec", zred_spec)                                   # as save_final_catalog: -99 = none
    col("z_ml", zml)
    i16, i50, i84 = 2, 3, 4                                    # 15.9, 50, 84.1 in PERCENTS
    names = [("zred", "z", None), ("total_mass", "mtot", "log Msol"),
             ("stellar_mass", "mstar", "log Msol"), ("met", "met", "log Zsol"),
             ("mwa", "mwa", u.Gyr), ("dust2", "dust2", None), ("dust_index", "dust_index", None),
             ("dust1_fraction", "dust1_fraction", None), ("log_fagn", "logfagn", None),
             ("sfr10", "sfr10", u.solMass / u.yr), ("sfr30", "sfr30", u.solMass / u.yr),
             ("sfr100", "sfr100", u.solMass / u.yr), ("ssfr10", "ssfr10", 1 / u.yr),
             ("ssfr30", "ssfr30", 1 / u.yr), ("ssfr100", "ssfr100", 1 / u.yr)]
    for key, short, unit in names:
        for suf, ii in (("16", i16), ("50", i50), ("84", i84)):
            col(f"{short}_{suf}", quant[key][:, ii], unit)
    if "rest_UVJugi" in rest:
        for t, nm in enumerate(["rest_U", "rest_V", "rest_J", "rest_u", "rest_g", "rest_i"]):
            for suf, ii in (("16", i16), ("50", i50), ("84", i84)):
                col(f"{nm}_{suf}", rest["rest_UVJugi"][:, t, ii], u.ABmag)
    if "rest_UVJugi_colors" in rest:
        for t, nm in enumerate(["UV", "VJ", "gi", "ug"]):
            for suf, ii in (("16", i16), ("50", i50), ("84", i84)):
                col(f"{nm}_{suf}", rest["rest_UVJugi_colors"][:, t, ii], u.ABmag)
    col("chi2", chi2.astype(float))
    col("nbands", nbands.astype(float))
    col("logZ", lz.astype(float))
    col("logZ_err", lze.astype(float))
    meta = OrderedDict()
    meta["AUTHOR"] = "Suess research group"
    meta["CREATED"] = str(date.today())
    meta["SPSCODE"] = "NSS + E-FSPS w256 emulator"
    emu = summ0.get("emulator", {})
    meta["EMU_MD5"] = str(emu.get("md5", "unknown"))
    meta["NSS_GIT"] = str(summ0.get("postprocess_code", {}).get("git", "unknown"))[:40]
    meta["SPECTRA"] = "no" if "not computed" in ref["spectra"] else "yes"
    cat.meta = meta
    os.makedirs(os.path.join(args.out, "sps_catalog"), exist_ok=True)
    base = os.path.basename(args.catalog)
    base = base[:-5] if base.endswith(".fits") else base
    fcat = os.path.join(args.out, "sps_catalog", f"{base}_SPScatalog_{args.sps_version}.fits")
    cat.write(fcat, format="fits", overwrite=True)
    for h in (fq, fs, fc, fp):
        h.close()
    if failed:
        with open(os.path.join(args.out, "failed_files.txt"), "w") as fh:
            fh.write("\n".join(failed) + "\n")
    print(f"done in {time.time() - t0:.1f} s: {int(ok.sum())}/{n} galaxies; {len(failed)} failed "
          f"files; {len(dups)} duplicates; {missing.size} ids not in the catalogue; "
          f"{int(good.sum())} rows filled of {len(cat)} -> {fcat}", flush=True)


if __name__ == "__main__":
    main()
