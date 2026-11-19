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
                     logZ, logZ_err, ess; ok (row filled); fit_status (int8, codes in
                     attrs fit_status_codes) and timed_out (int8) for EVERY fit found
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
                     nbands) plus logZ, logZ_err, ess, timed_out and fit_status ("ok",
                     "timed_out", "capped", "nonfinite", "bad_input", "post_unusable",
                     "unknown"; "" = no fit found); rows without a usable fit are masked.
                     Rest-frame U V J u g i and UV/VJ/gi/ug columns come from spectra_nss.py
                     (masked for galaxies it has not reached yet).
  quant_ also gets rest_UVJugi, rest_gz, rest_NUVrJ (+ _colors, _map), and spec_ gets
  modspec_map (best-fit spectrum, observed f_nu per rest pixel) and wavspec, as before.
  missing_spectra.txt lists galaxies without spectra: spectra_nss.py --list catches up.
  failed_files.txt, duplicate_ids.txt   (only if there are any)
Rows are sorted by id. Unreadable files keep their row with ok = 0 and NaN data, and are
listed in failed_files.txt; they are masked in the FITS catalogue.

Fit status (nss_jax.py STATUSES): every fit is found from its <id>_<tag>.json as well as its
_post.h5, so a fit with no _post.h5 (timed out, capped, non-finite, bad input, or
postprocessing failed) still gets its row, masked, with its status; never a silently
missing or silently good row. The status comes from the json ("status"), else from the
_post.h5's copy of the summary; files from before the status field count as "ok" if they
have logZ and timed_out is false. "post_unusable": the fit says ok but its _post.h5 is
missing, unreadable or not in the common layout (failed_files.txt says which).
Provenance (provenance.py): the distinct nss commits, dirty flags, efsps-photfit commits,
emulator md5s and library versions of all fits, with counts, go into the HDF5 attrs
("provenance", JSON) and the FITS header; a run whose fits disagree on the nss commit or
emulator md5 is reported with a WARNING.

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
REST_BASE = ["rest_UVJugi", "rest_UVJugi_colors", "rest_gz", "rest_gz_colors",
             "rest_NUVrJ", "rest_NUVrJ_colors"]
REST_KEYS = REST_BASE + [k + "_map" for k in REST_BASE]     # spectra_nss.py; old names and shapes
BLOCK = 256                        # rows per write
#: fit_status codes in quant_<prior>.h5 (attrs fit_status_codes); FITS fit_status has the names
FIT_STATUS = ["ok", "timed_out", "capped", "nonfinite", "bad_input", "post_unusable", "unknown"]
#: provenance entries collected over the run -> FITS header keywords
PROV_KEYS = OrderedDict([("nss_commit", "NSS_GIT"), ("nss_dirty", "NSS_DIRT"),
                         ("nss_source", "NSS_SRC"), ("efsps_commit", "EFSPS_GT"),
                         ("emulator_md5", "EMU_MD5"), ("v_python", "V_PYTHON"),
                         ("v_jax", "V_JAX"), ("v_jaxlib", "V_JAXLIB"), ("v_blackjax", "V_BLACKJ"),
                         ("v_numpy", "V_NUMPY"), ("v_scipy", "V_SCIPY"), ("v_h5py", "V_H5PY"),
                         ("v_astropy", "V_ASTROP")])


def file_id(path):
    return int(re.match(r"(\d+)_", os.path.basename(path)).group(1))


def fit_status(summary):
    """Status of a fit from its summary (json or the _post.h5 copy). Before the status field
    existed: "ok" if it has logZ and did not time out."""
    st = summary.get("status")
    if st is not None:
        return st if st in FIT_STATUS else "unknown"
    if summary.get("timed_out"):
        return "timed_out"
    return "ok" if "logZ" in summary else "unknown"


def prov_brief(summary, prov):
    """The provenance entries the collected products report (PROV_KEYS), from a fit's
    provenance (provenance.py), or, for fits made before it, from what the summary has."""
    p = prov or {}
    pc = summary.get("postprocess_code") or {}
    out = {"nss_commit": p.get("nss_commit") or pc.get("git") or "unknown",
           "nss_dirty": p.get("nss_dirty", pc.get("dirty")),
           "nss_source": p.get("nss_source") or ("git" if pc.get("git") else "unknown"),
           "efsps_commit": p.get("efsps_commit") or "unknown",
           "emulator_md5": p.get("emulator_md5") or (summary.get("emulator") or {}).get("md5")
           or "unknown"}
    for k, v in (p.get("versions") or {}).items():
        out[f"v_{k}"] = v
    return {k: str(v) for k, v in out.items()}


def read_entry(entry):
    """entry = (id, _post.h5 path or None, json path or None) -> read_one's dict (rows with
    data) or a dict without data; always with status, timed_out and prov."""
    gid, post, js = entry
    summ, jerr = None, None
    if js is not None:
        try:
            with open(js) as fh:
                summ = json.load(fh)
        except Exception as e:
            jerr = f"{type(e).__name__}: {e}"
    status = fit_status(summ) if summ is not None else None
    if post is not None and status in (None, "ok"):
        r = read_one(post)
        if summ is None and "error" not in r:                   # no json: the copy in the h5
            summ = json.loads(r["summary"])
            status = fit_status(summ)
        prov = r.pop("provenance", None)
    else:
        r = {"path": js or post}
        prov = None
    summ = summ or {}
    if status is None:
        status = "unknown"
        r.setdefault("error", jerr or "no readable json or _post.h5")
    elif status == "ok" and "error" in r:
        status = "post_unusable"
    elif status == "ok" and post is None:
        status = "post_unusable"
        r["error"] = "status ok but no _post.h5"
    r.update(id=gid, status=status, timed_out=bool(summ.get("timed_out", False)),
             prov=prov_brief(summ, summ.get("provenance") or prov))
    if status != "ok":
        r["no_data"] = True
    return r


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
            out["has_spectra"] = "spectra/bestfit" in f
            out["spec_best"] = f["spectra/bestfit"][()] if out["has_spectra"] else None
            out["wave_rest"] = f["spectra/wavelength_rest"][()] if out["has_spectra"] else None
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
            if "provenance" in f["fit"].attrs:
                out["provenance"] = json.loads(f["fit"].attrs["provenance"])
        if out["id"] != file_id(path):
            raise ValueError(f"file name id {file_id(path)} != stored id {out['id']}")
        return out
    except Exception as e:                                     # recorded, never silent
        return {"path": path, "error": f"{type(e).__name__}: {e}"}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--indir", required=True, help="searched recursively for --pattern")
    p.add_argument("--pattern", default="*_post.h5")
    p.add_argument("--json-pattern", default=None,
                   help="the fits' summary jsons (default: --pattern with _post.h5 -> .json); "
                        "a fit with a json but no _post.h5 gets a masked row with its status")
    p.add_argument("--catalog", required=True, help="the photometric catalogue the fits used")
    p.add_argument("--out", required=True)
    p.add_argument("--prior", default="nss", help="tag in the collected file names")
    p.add_argument("--sps-version", default="spsv2.0")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    posts = glob.glob(os.path.join(args.indir, "**", args.pattern), recursive=True)
    jpat = args.json_pattern or (args.pattern[:-len("_post.h5")] + ".json"
                                 if args.pattern.endswith("_post.h5") else None)
    jsons = glob.glob(os.path.join(args.indir, "**", jpat), recursive=True) if jpat else []
    # one fit = one stem (<id>_<tag>) with its _post.h5 and/or its json
    fits_by_stem = {}
    for f in posts:
        fits_by_stem.setdefault(f[:-len("_post.h5")] if f.endswith("_post.h5") else f, [None, None])[0] = f
    for f in jsons:
        if re.match(r"\d+_", os.path.basename(f)):
            fits_by_stem.setdefault(f[:-len(".json")], [None, None])[1] = f
    if not fits_by_stem:
        raise SystemExit(f"no {args.pattern} or {jpat} under {args.indir}")

    def newest(pj):
        return max(os.path.getmtime(x) for x in pj if x is not None)

    # one fit per id: the most recently written (a rerun replaces an older fit); the rest
    # are listed in duplicate_ids.txt
    entries = sorted(((file_id(next(x for x in pj if x)), pj[0], pj[1])
                      for pj in fits_by_stem.values()),
                     key=lambda e: (e[0], -newest(e[1:]), str(e[1:])))
    ids_all = np.array([e[0] for e in entries])
    first = np.r_[True, ids_all[1:] != ids_all[:-1]]
    dups = [e[1] or e[2] for e, keep in zip(entries, first) if not keep]
    entries = [e for e, keep in zip(entries, first) if keep]
    ids = ids_all[first]
    n = len(entries)
    print(f"{n} galaxies ({len(dups)} duplicate fits skipped; {len(posts)} _post.h5, "
          f"{len(jsons)} json files) under {args.indir}", flush=True)
    if dups:
        with open(os.path.join(args.out, "duplicate_ids.txt"), "w") as fh:
            fh.write("\n".join(dups) + "\n")

    # reference layout (band set, draw count): the most common among the first 50 readable
    # ok fits, so one odd file at the start cannot reject the rest
    sample = [r for r in (read_entry(e) for e in [e for e in entries if e[1]][:50])
              if "error" not in r and not r.get("no_data")]
    if not sample:
        raise SystemExit("none of the first 50 _post.h5 files is a readable status-ok fit")
    keys = [(tuple(r["bands"]), r["chains"].shape) for r in sample]
    best = max(set(keys), key=keys.count)
    ref = sample[keys.index(best)]
    K, ncol = ref["chains"].shape
    nb, npc = ref["modmag_perc"].shape
    # spectra (spectra_nss.py) may be missing from some files: shapes from the first file that
    # has them, found cheaply from the file attribute
    spec_ref = next((r for r in sample if r["has_spectra"]), None)
    if spec_ref is None:
        for f in [e[1] for e in entries if e[1]][50:]:
            try:
                with h5py.File(f, "r") as h:
                    if "spectra/bestfit" in h:
                        spec_ref = read_one(f)
                        break
            except OSError:
                continue
    has_rest = [k for k in REST_KEYS if spec_ref is not None and k in spec_ref["rest"]]

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
        dq[k] = fq.create_dataset(k, shape=(n,) + spec_ref["rest"][k].shape, dtype=np.float32,
                                  fillvalue=np.nan, **kw)
    sc = {k: fq.create_dataset(k, shape=(n,), dtype=np.float32, fillvalue=np.nan, **kw)
          for k in ("zred_ml", "logZ", "logZ_err", "ess")}
    ds_status = fq.create_dataset("fit_status", shape=(n,), dtype=np.int8)
    ds_status.attrs["fit_status_codes"] = json.dumps({i: k for i, k in enumerate(FIT_STATUS)})
    ds_timed = fq.create_dataset("timed_out", shape=(n,), dtype=np.int8)
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
    if spec_ref is not None:          # best-fit spectrum, as the old spec file's modspec_map
        fp.create_dataset("wavspec", data=spec_ref["wave_rest"])
        fp["wavspec"].attrs["units"] = "rest-frame A; modspec_map is observed f_nu (maggies) per rest pixel"
        npix = spec_ref["wave_rest"].size
        sp["modspec_map"] = fp.create_dataset("modspec_map", shape=(n, npix), dtype=np.float32,
                                              fillvalue=np.nan, chunks=(min(n, 64), npix), **kw)
    sp["modmag_perc"] = fp.create_dataset("modmag_perc", shape=(n, nb, npc), dtype=np.float32,
                                          fillvalue=np.nan, chunks=(min(n, 1024), nb, npc), **kw)

    ok = np.zeros(n, np.int8)
    status = np.zeros(n, np.int8)
    timed = np.zeros(n, np.int8)
    provs = []
    failed, no_spec = [], []
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
            if "error" in r or r.get("no_data"):
                continue
            for k in dq:
                if k in r["quant"]:
                    blk[k][j] = r["quant"][k]
                elif k in r["rest"]:                           # NaN until spectra_nss.py has run
                    blk[k][j] = r["rest"][k]
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
            if "modspec_map" in spb and r["has_spectra"]:
                spb["modspec_map"][j] = r["spec_best"]
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
    it = pool.imap(read_entry, entries, chunksize=32) if pool else map(read_entry, entries)
    for i, r in enumerate(it):
        if "error" not in r and not r.get("no_data"):
            if r["bands"] != ref["bands"] or r["chains"].shape != (K, ncol):
                r = dict(path=r["path"], status="post_unusable", timed_out=r["timed_out"],
                         prov=r["prov"], error="bands or draw count differ from the common layout "
                         f"({len(r['bands'])} bands, {r['chains'].shape} vs {nb}, {(K, ncol)})")
        status[i] = FIT_STATUS.index(r["status"])
        timed[i] = int(r["timed_out"])
        provs.append(r["prov"])
        if "error" in r:
            failed.append(f"{r['path']}\t{r['error']}")
        elif not r.get("no_data"):
            ok[i] = 1
            if not r["has_spectra"]:
                no_spec.append(r["path"])
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
    ds_status[:] = status
    ds_timed[:] = timed
    # provenance over the run: distinct values with counts; warn if the code or emulator differ
    prov_counts = OrderedDict()
    for k in PROV_KEYS:
        c = {}
        for pb in provs:
            v = pb.get(k, "unknown")
            c[v] = c.get(v, 0) + 1
        prov_counts[k] = dict(sorted(c.items(), key=lambda kv: -kv[1]))
    for k in ("nss_commit", "emulator_md5"):
        if len(prov_counts[k]) > 1:
            print(f"WARNING: the fits disagree on {k}: {prov_counts[k]}", flush=True)
    for h in (fq, fs, fc, fp):
        h["ok"][:] = ok
        h.attrs["provenance"] = json.dumps(prov_counts)
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
    col("ess", fq["ess"][:].astype(float))
    # status and timed_out for every fit found, also the masked ones; "" = no fit found
    have = rows >= 0
    st = np.full(len(cat), "", dtype="U13")
    st[rows[have]] = np.array(FIT_STATUS)[status[have]]
    cat.add_column(st, name="fit_status")
    to = np.zeros(len(cat), np.int16)
    to[rows[have]] = timed[have]
    m = np.ones(len(cat), bool)
    m[rows[have]] = False
    cat.add_column(MaskedColumn(data=to, name="timed_out", mask=m))
    n_status = {k: int((status == i).sum()) for i, k in enumerate(FIT_STATUS) if (status == i).any()}
    meta = OrderedDict()
    meta["AUTHOR"] = "Suess research group"
    meta["CREATED"] = str(date.today())
    meta["SPSCODE"] = "NSS + E-FSPS w256 emulator"
    comments = []
    for k, key in PROV_KEYS.items():                       # one value, or MIXED + a comment
        vals = list(prov_counts[k])
        meta[key] = vals[0][:68] if len(vals) == 1 else f"MIXED ({len(vals)} values)"
        if len(vals) > 1:
            comments += [f"{k}: {v} ({c} fits)"[:72] for v, c in prov_counts[k].items()]
    meta["N_SPEC"] = int(ok.sum()) - len(no_spec)          # galaxies with spectra / rest colours
    comments.append("fit_status counts: " + ", ".join(f"{k} {v}" for k, v in n_status.items()))
    meta["comments"] = comments                          # FITS COMMENT cards
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
    if no_spec:                        # catch up with: spectra_nss.py --list missing_spectra.txt
        with open(os.path.join(args.out, "missing_spectra.txt"), "w") as fh:
            fh.write("\n".join(no_spec) + "\n")
    print(f"fit status: {n_status}", flush=True)
    print(f"done in {time.time() - t0:.1f} s: {int(ok.sum())}/{n} galaxies; {len(failed)} failed "
          f"files; {len(dups)} duplicates; {len(no_spec)} without spectra (missing_spectra.txt); "
          f"{missing.size} ids not in the catalogue; "
          f"{int(good.sum())} rows filled of {len(cat)} -> {fcat}", flush=True)


if __name__ == "__main__":
    main()
