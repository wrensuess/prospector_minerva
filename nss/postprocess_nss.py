"""Per-galaxy postprocessing of an NSS fit, run at the end of the fit (nss_jax.py) while the
model is in memory. Writes ONE file, <id>_<tag>_post.h5, with the products the dynesty +
Parrot pipeline made in three files (postprocess_parrot.py: perc, unw, spec), and with the
same definitions wherever the model allows. make_catalog_nss.py gathers these files into
the collected HDF5 files and the FITS catalogue.

What is computed, and from what
  - Percentiles (0.1, 2.3, 15.9, 50, 84.1, 97.7, 99.9; postprocess_parrot.percents) from
    the FULL weighted NSS posterior, not a resample. Weighted quantiles use efsps-photfit's
    convention (fit.weighted_quantiles: midpoint CDF, linear interpolation), so the 50th
    matches the fit summary's median exactly.
  - Every sample has all 26 parameters, logmass included: with mass integrated out of the
    likelihood, nss_jax draws one logmass per sample from its posterior given the other 25
    (MassMarginal.sample_logmass).
  - loglike: the photometric log-likelihood of each sample AT ITS OWN MASS, i.e.
    efsps-photfit's FitModel.loglike(theta) (observed error + 5% floor + emulator error).
    The fit's own logl is the likelihood with mass integrated out (and the dust prior
    factor), which is not the likelihood of a draw. No priors are included.
  - Best fit: the sample with the largest loglike (the dynesty pipeline's chain_ml was the
    last, i.e. highest-likelihood, dynesty sample). chi2 at it uses the same variance as
    the likelihood (agreed 2026-11-13; the dynesty pipeline clipped errors at 5% instead).
  - SFH (efsps-photfit sfh.py; 10 bins, Planck18): formed mass per bin, SFR averaged over
    the last 10, 30 and 100 Myr (exact overlap of the step SFH with [0, T]; the dynesty
    pipeline averaged a 2 Myr grid), formed-mass-weighted age (Gyr, bin midpoints, as
    postprocess_parrot.get_mwa), surviving stellar mass (efsps-photfit survival.py table)
    and sSFR = SFR / surviving mass (1/yr, as postprocess_parrot).
  - SFH(t) percentiles on postprocess_parrot's grid: 100 points log-spaced from 10 yr to
    the oldest bin edge of any sample (agebins_max = log10 of that, in yr), SFR of the bin
    containing t, 0 at and beyond the sample's oldest edge (postprocess_parrot set the last
    grid point to 0 and NaN -> 0 before taking percentiles; same result).
  - Cumulative formed-mass history on the same grid (replaces postprocess_parrot's
    mformed_t, which was the formed mass of the bin containing t; Wren, 2026-11-13): the
    mass formed before each lookback time t, exact from the step SFH, in Msun and as a
    fraction of the sample's total formed mass. Not surviving mass.
  - Model photometry (observed bands, magnified, maggies) for every sample, from the fit's
    own network, and its weighted percentiles.
  - K = 1000 equal-weight draws (multinomial resample, seeded by the fit seed), with every
    parameter, the derived quantities, the 10 per-bin SFRs and loglike.
  - NOT YET: spectra and rest-frame colours. They need E-FSPS SpecModel with the stellar
    library the w256 emulator was trained on (c3k_a-mist-chabrier does not reproduce it;
    STATUS.md). The file says so in attrs["spectra"], and the catalogue leaves those
    columns out.

Output layout (<id>_<tag>_post.h5; float64 unless noted; percentile axis last):
  attrs: id, tag, ra, dec, z_spec, z_catalog, mu, logZ, logZ_err, ess, n_samples, percents,
         spectra, format_version, created_utc
  percentiles/<param>     (7,)     each of the 26 free parameters (efsps names), plus
                                    log_fagn, log_agn_tau, log_duste_gamma (log10 of the
                                    linear parameters), total_mass (= logmass, formed)
  percentiles/stellar_mass (7,)    log10 surviving mass
  percentiles/mwa          (7,)    Gyr
  percentiles/sfr, ssfr    (3, 7)  rows 10, 30, 100 Myr; Msun/yr and 1/yr
  percentiles/sfh          (100, 7) Msun/yr on sfh_grid
  percentiles/mformed_cum  (100, 7) Msun formed before each sfh_grid lookback time
  percentiles/mformed_cum_frac (100, 7) the same as a fraction of the total formed mass
  sfh_grid (100,) lookback yr; attrs agebins_max (log10 yr)
  draws/<param> (K,) float32 for every parameter above and stellar_mass, mwa;
  draws/sfr, ssfr (K, 3) float32; draws/sfr_bins (K, 10) float32 (Msun/yr per bin, youngest
  first); draws/agebins (K, 10, 2) float32, the bins' log10 lookback-time edges in yr
  (efsps_photfit.sfh.agebins_batch(zred), the Prospector agebins layout);
  draws/loglike (K,) float64; draws/index (K,) row in the fit's sample list
  bestfit: theta (26,), attrs param names, loglike, chi2, nbands, zred, index;
           model_flux (nb,), stellar_mass, mwa, sfr (3,), ssfr (3,)
  obs: bands (nb,) strings, flux, err (maggies; NaN = not used), valid (nb,) int8, attrs mu
  model_flux_percentiles (nb, 7) maggies
  fit: attrs summary_json (the fit's summary, without its percentiles)

    from postprocess_nss import postprocess
    postprocess(model, phot, theta, w, summary, out_path, seed)

Standalone, from a fit's full sample file (<id>_<tag>.npz, nss_jax.py without --no-full-npz):
    python postprocess_nss.py --catalog CAT --id ID --npz RUN/ID_TAG.npz [--json RUN/ID_TAG.json]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("EFSPS_PARENT", "/Users/wren/Projects/MINERVA/efsps-photfit"))
from efsps_photfit.sfh import agebins_batch, mass_fractions   # noqa: E402
from efsps_photfit.survival import SurvivingFraction           # noqa: E402

FORMAT_VERSION = 2              # 2: + draws/agebins, percentiles/mformed_cum(_frac)
PERCENTS = np.array([0.1, 2.3, 15.9, 50, 84.1, 97.7, 99.9])     # postprocess_parrot.percents
N_DRAWS = 1000                                                 # postprocess_parrot.nsamp
SFR_TIMES_YR = np.array([1e7, 3e7, 1e8])                        # 10, 30, 100 Myr
SFH_TBINS = 100                                                # postprocess_parrot tbins
CHUNK = 5000              # rows per block: bounds memory, changes no number (~1-2 s per fit either way)
WEIGHT_KEEP = 1e-8           # samples with w <= this x max(w) are dropped, as nss_efsps.summarise
LOG_OF = {"log_fagn": "fagn", "log_agn_tau": "agn_tau", "log_duste_gamma": "duste_gamma"}
SPECTRA_STATUS = ("not computed: needs E-FSPS SpecModel with the stellar library the w256 "
                  "emulator was trained on (c3k_a-mist-chabrier does not reproduce it)")

_SURV = None


def surviving_fraction():
    global _SURV
    if _SURV is None:
        _SURV = SurvivingFraction()
    return _SURV


# ------------------------------------------------------------------ statistics
def weighted_percentiles(x, w, percents=PERCENTS):
    """Weighted percentiles along axis 0, efsps-photfit's convention (fit.weighted_quantiles:
    cumulative weight at each sorted point's midpoint, linear interpolation, clamped at the
    ends). x (N,) -> (P,); x (N, M) -> (M, P). Non-finite values are ignored per column."""
    x = np.asarray(x, float)
    one = x.ndim == 1
    x2 = x[:, None] if one else x.reshape(x.shape[0], -1)
    q = np.asarray(percents, float) / 100.0
    out = np.full((x2.shape[1], q.size), np.nan)
    w = np.asarray(w, float)
    for j in range(x2.shape[1]):
        col = x2[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            continue
        o = np.argsort(col[ok], kind="stable")
        xs, ws = col[ok][o], w[ok][o]
        c = np.cumsum(ws)
        c = (c - 0.5 * ws) / c[-1]
        out[j] = np.interp(q, c, xs)
    if one:
        return out[0]
    return out.reshape(x.shape[1:] + (q.size,))


# ------------------------------------------------------------------ derived quantities
def sfh_quantities(values, n):
    """Per sample: SFH bin edges, per-bin SFR, SFRs over SFR_TIMES_YR, mwa, surviving mass.
    values = FitModel.values(theta) (fixed parameters broadcast)."""
    z = np.broadcast_to(np.asarray(values["zred"], float), (n,))
    lm = np.broadcast_to(np.asarray(values["logmass"], float), (n,))
    lz = np.broadcast_to(np.asarray(values["logzsol"], float), (n,))
    ratios = np.stack([np.broadcast_to(np.asarray(values[f"logsfr_ratios_{i}"], float), (n,))
                       for i in range(1, 10)], axis=1)
    ab = agebins_batch(z)
    t0, t1 = 10.0 ** ab[..., 0], 10.0 ** ab[..., 1]
    dt = t1 - t0
    frac = mass_fractions(ratios, dt)
    mass_bins = frac * (10.0 ** lm)[:, None]
    sfr_bins = mass_bins / dt
    sfr = np.stack([(sfr_bins * np.clip(np.minimum(t1, T) - t0, 0.0, None)).sum(axis=1) / T
                    for T in SFR_TIMES_YR], axis=1)
    mwa = (frac * 0.5 * (t0 + t1)).sum(axis=1) / 1e9
    ms = surviving_fraction().surviving_mass(mass_bins, lz, t0, t1)
    return dict(t0=t0, t1=t1, agebins=ab, sfr_bins=sfr_bins, sfr=sfr, mwa=mwa,
                stellar_mass=np.log10(ms), ssfr=sfr / ms[:, None])


def sfh_bins_from_chains(chains, labels):
    """Per-bin SFRs and bin edges of collected draws (chains_<prior>.h5), which store zred,
    total_mass (formed) and logsfr_ratios_1-9 but not the bins themselves.
    chains (..., n_col), labels (n_col,) str or bytes -> (sfr_bins (..., 10) Msun/yr,
    agebins (..., 10, 2) log10 lookback yr), as draws/sfr_bins and draws/agebins in the
    per-galaxy files (same arithmetic; float32 inputs, so equal to ~1e-6 relative)."""
    labels = [x.decode() if isinstance(x, bytes) else str(x) for x in labels]
    c = np.asarray(chains, float)
    shape = c.shape[:-1]
    c = c.reshape(-1, c.shape[-1])
    z, lm = c[:, labels.index("zred")], c[:, labels.index("total_mass")]
    ratios = np.stack([c[:, labels.index(f"logsfr_ratios_{i}")] for i in range(1, 10)], axis=1)
    ab = agebins_batch(z)
    dt = 10.0 ** ab[..., 1] - 10.0 ** ab[..., 0]
    sfr_bins = mass_fractions(ratios, dt) * (10.0 ** lm)[:, None] / dt
    return sfr_bins.reshape(shape + (10,)), ab.reshape(shape + (10, 2))


def mass_formed_on_grid(t0, t1, sfr_bins, grid):
    """Cumulative formed mass: the mass formed BEFORE each lookback time t (i.e. already in
    place at t), exactly, from the step SFH: sum_k SFR_k x max(0, t1_k - max(t, t0_k)).
    Equals the total formed mass at t <= t0 of the youngest bin, 0 at and beyond the
    oldest edge. (N bins) x (G grid) -> (N, G) Msun."""
    out = np.zeros((t0.shape[0], grid.size))
    for k in range(t0.shape[1]):
        out += sfr_bins[:, k:k + 1] * np.clip(
            t1[:, k:k + 1] - np.maximum(grid[None, :], t0[:, k:k + 1]), 0.0, None)
    return out


def sfh_on_grid(t0, t1, sfr_bins, grid):
    """SFR of the bin containing each lookback time in grid; 0 at and beyond the oldest
    edge (postprocess_parrot.stepInterp, with its NaN -> 0; its last grid point, set to 0
    there, is the oldest edge of any sample, so it is 0 here too)."""
    out = np.zeros((t0.shape[0], grid.size))
    for k in range(t0.shape[1]):
        inside = (grid[None, :] >= t0[:, k:k + 1]) & (grid[None, :] < t1[:, k:k + 1])
        out = np.where(inside, sfr_bins[:, k:k + 1], out)
    return out


def model_photometry(model, theta):
    """Model fluxes (magnified maggies, all bands) and FitModel.loglike, from one network
    pass, in blocks. loglike is computed exactly as efsps_photfit.model.FitModel.loglike."""
    fm, ll = [], []
    v = model.valid
    for i in range(0, theta.shape[0], CHUNK):
        f_mod, sig, _ = model.model_fluxes(theta[i:i + CHUNK])
        var = model._var_obs[None, :] + sig[:, v] ** 2
        r2 = (model._fv[None, :] - f_mod[:, v]) ** 2 / var
        ll.append(-0.5 * np.sum(r2 + np.log(2.0 * np.pi * var), axis=1))
        fm.append(f_mod)
    return np.concatenate(fm), np.concatenate(ll)


def chi2_at(model, theta_row):
    f_mod, sig, _ = model.model_fluxes(theta_row[None, :])
    v = model.valid
    var = model._var_obs + sig[0, v] ** 2
    return float(np.sum((model._fv - f_mod[0, v]) ** 2 / var))


# ------------------------------------------------------------------ provenance
def code_version():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        if h.returncode == 0:
            d = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", here],
                               capture_output=True, text=True, timeout=10)
            return {"git": h.stdout.strip(), "dirty": bool(d.stdout.strip())}
    except Exception:                                             # pragma: no cover
        pass
    return {"git": "unknown (not a git checkout)", "dirty": None}


def _meta_float(meta, key):
    try:
        v = float(meta.get(key, np.nan))
    except (TypeError, ValueError):
        return np.nan
    return v


# ------------------------------------------------------------------ main entry
def postprocess(model, phot, theta, w, summary, out_path, seed=0, n_draws=N_DRAWS):
    """Compute every product from the weighted posterior (theta (N, n_dim) with model.free
    columns, w (N,) weights) and write out_path atomically. Returns a small dict of timings
    and headline numbers."""
    import h5py
    t_start = time.time()
    theta = np.asarray(theta, float)
    w = np.asarray(w, float)
    if theta.shape[0] != w.size or theta.shape[1] != model.n_dim:
        raise ValueError(f"theta {theta.shape} and w {w.shape} do not match the model "
                         f"({model.n_dim} free parameters).")
    keep = w > WEIGHT_KEEP * w.max()
    idx_kept = np.flatnonzero(keep)
    theta, w = theta[keep], w[keep] / w[keep].sum()
    n = theta.shape[0]
    names = list(model.free)

    # per-sample quantities (all kept samples)
    vals = model.values(theta)
    parts = []
    for a in range(0, n, CHUNK):           # the surviving-mass lookup is (25 Z x rows x 10 bins)
        b = min(a + CHUNK, n)
        parts.append(sfh_quantities(model.values(theta[a:b]), b - a))
    q = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
    f_mod, loglike = model_photometry(model, theta)

    # percentiles
    perc = {}
    for i, name in enumerate(names):
        perc[name] = weighted_percentiles(theta[:, i], w)
    for lname, lin in LOG_OF.items():
        if lin in model.col:
            perc[lname] = weighted_percentiles(np.log10(theta[:, model.col[lin]]), w)
    perc["total_mass"] = perc["logmass"]
    perc["stellar_mass"] = weighted_percentiles(q["stellar_mass"], w)
    perc["mwa"] = weighted_percentiles(q["mwa"], w)
    perc["sfr"] = weighted_percentiles(q["sfr"], w)            # (3, 7)
    perc["ssfr"] = weighted_percentiles(q["ssfr"], w)
    if "zred" not in model.col:                                # fixed-redshift fit
        perc["zred"] = np.full(PERCENTS.size, float(model.fixed["zred"]))

    # SFH(t) on postprocess_parrot's grid
    agebins_max = float(np.log10(q["t1"].max()))
    grid = np.logspace(1.0, agebins_max, SFH_TBINS)
    grid[-1] = q["t1"].max()                                   # exactly the oldest edge
    sfh_perc = np.empty((SFH_TBINS, PERCENTS.size))
    mf_perc = np.empty((SFH_TBINS, PERCENTS.size))
    mfrac_perc = np.empty((SFH_TBINS, PERCENTS.size))
    m_tot = (q["sfr_bins"] * (q["t1"] - q["t0"])).sum(axis=1)  # formed mass of each sample
    for a in range(0, SFH_TBINS, 25):                          # columns in blocks: memory
        b = min(a + 25, SFH_TBINS)
        sfh_perc[a:b] = weighted_percentiles(
            sfh_on_grid(q["t0"], q["t1"], q["sfr_bins"], grid[a:b]), w)
        mf = mass_formed_on_grid(q["t0"], q["t1"], q["sfr_bins"], grid[a:b])
        mf_perc[a:b] = weighted_percentiles(mf, w)
        mfrac_perc[a:b] = weighted_percentiles(mf / m_tot[:, None], w)
    perc["sfh"] = sfh_perc
    perc["mformed_cum"] = mf_perc
    perc["mformed_cum_frac"] = mfrac_perc

    flux_perc = weighted_percentiles(f_mod, w)                  # (nb, 7)

    # best fit: the sample with the largest likelihood at its own mass
    ib = int(np.argmax(loglike))
    best = theta[ib]
    chi2 = chi2_at(model, best)

    # equal-weight draws
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20261113]))
    pick = np.sort(rng.choice(n, size=int(n_draws), replace=True, p=w))

    zs = _meta_float(phot.meta or {}, "z_spec")
    s = {k: v for k, v in summary.items() if k != "percentiles"}
    s["postprocess_code"] = code_version()
    tmp = f"{out_path}.tmp.{os.getpid()}"
    written = False
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs.update({
                "id": str(phot.id), "tag": str(summary.get("tag", "")),
                "ra": _meta_float(phot.meta or {}, "ra"), "dec": _meta_float(phot.meta or {}, "dec"),
                "z_spec": zs if (np.isfinite(zs) and zs > 0) else np.nan,
                "z_catalog": np.nan if phot.z_catalog is None else float(phot.z_catalog),
                "mu": float(phot.mu),
                "logZ": float(summary.get("logZ", np.nan)),
                "logZ_err": float(summary.get("logZ_err", np.nan)),
                "ess": float(1.0 / np.sum(w ** 2)), "n_samples": n,
                "n_samples_dropped": int(keep.size - n),
                "percents": PERCENTS, "spectra": SPECTRA_STATUS,
                "format_version": FORMAT_VERSION,
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
            g = f.create_group("percentiles")
            for k, v in perc.items():
                g.create_dataset(k, data=np.asarray(v, float))
            f.create_dataset("sfh_grid", data=grid)
            f["sfh_grid"].attrs["agebins_max"] = agebins_max
            f["sfh_grid"].attrs["units"] = "lookback time, yr"
            f.create_dataset("model_flux_percentiles", data=flux_perc)

            d = f.create_group("draws")
            for i, name in enumerate(names):
                d.create_dataset(name, data=theta[pick, i].astype(np.float32))
            for lname, lin in LOG_OF.items():
                if lin in model.col:
                    d.create_dataset(lname, data=np.log10(theta[pick, model.col[lin]]).astype(np.float32))
            if "zred" not in model.col:
                d.create_dataset("zred", data=np.full(pick.size, float(model.fixed["zred"]), np.float32))
            d.create_dataset("total_mass", data=theta[pick, model.col["logmass"]].astype(np.float32))
            for k in ("stellar_mass", "mwa", "sfr", "ssfr", "sfr_bins", "agebins"):
                d.create_dataset(k, data=q[k][pick].astype(np.float32),
                                 compression="gzip", shuffle=True)
            d.create_dataset("loglike", data=loglike[pick])
            d.create_dataset("index", data=idx_kept[pick].astype(np.int64))

            b = f.create_group("bestfit")
            b.create_dataset("theta", data=best)
            b.attrs["names"] = json.dumps(names)
            b.attrs.update({"loglike": float(loglike[ib]), "chi2": chi2,
                            "nbands": int(model.valid.sum()),
                            "zred": float(np.broadcast_to(np.asarray(vals["zred"], float), (n,))[ib]),
                            "index": int(idx_kept[ib])})
            b.create_dataset("model_flux", data=f_mod[ib])
            for k in ("stellar_mass", "mwa"):
                b.attrs[k] = float(q[k][ib])
            b.create_dataset("sfr", data=q["sfr"][ib])
            b.create_dataset("ssfr", data=q["ssfr"][ib])

            o = f.create_group("obs")
            o.create_dataset("bands", data=np.array(phot.bands, dtype=h5py.string_dtype()))
            o.create_dataset("flux", data=np.where(model.valid, phot.flux, np.nan))
            o.create_dataset("err", data=np.where(model.valid, phot.err, np.nan))
            o.create_dataset("valid", data=model.valid.astype(np.int8))
            o.attrs["mu"] = float(phot.mu)
            o.attrs["units"] = "maggies (observed, i.e. magnified); model fluxes include mu"

            fg = f.create_group("fit")
            fg.attrs["summary_json"] = json.dumps(s, default=str)
        os.replace(tmp, out_path)
        written = True
    finally:
        if not written and os.path.exists(tmp):
            os.remove(tmp)                                     # never leave a partial file
    return {"postprocess_s": time.time() - t_start, "n_samples": n,
            "bestfit_loglike": float(loglike[ib]), "chi2_bestfit": chi2,
            "file": os.path.basename(out_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--catalog", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--npz", required=True, help="the fit's full sample file <id>_<tag>.npz")
    p.add_argument("--json", default=None, help="the fit's summary <id>_<tag>.json "
                   "(default: next to the npz)")
    p.add_argument("--out", default=None, help="output file (default <npz stem>_post.h5)")
    p.add_argument("--fix-z", type=float, default=None)
    p.add_argument("--include-miri", action="store_true")
    args = p.parse_args()
    from photometry import read_photometry
    from nss_efsps import build
    stem = args.npz[:-4] if args.npz.endswith(".npz") else args.npz
    js = args.json or stem + ".json"
    summary = json.load(open(js)) if os.path.exists(js) else {"tag": os.path.basename(stem)}
    if args.fix_z is None:
        args.fix_z = (summary.get("settings") or {}).get("fix_z")
    phot = read_photometry(args.catalog, [args.id], include_miri=args.include_miri)[0]
    model = build(phot, args.fix_z, None)
    d = np.load(args.npz, allow_pickle=False)
    if list(d["names"]) != list(model.free):
        raise SystemExit(f"{args.npz}: parameters {list(d['names'])} differ from the model's "
                         f"{model.free}")
    seed = int((summary.get("settings") or {}).get("seed", 0))
    out = args.out or stem + "_post.h5"
    r = postprocess(model, phot, d["theta"], d["w"], summary, out, seed=seed)
    print(f"[{phot.id}] postprocessed {r['n_samples']} samples in {r['postprocess_s']:.1f} s "
          f"-> {out}; best-fit chi2 {r['chi2_bestfit']:.1f}")


if __name__ == "__main__":
    main()
