"""Spectra and rest-frame colours for NSS fits, added to their per-galaxy files
(<id>_<tag>_post.h5, postprocess_nss.py). ONE process builds E-FSPS SpecModel once and loops
over many galaxies: run it as the last step of each Blanca job (tasklist_layout.sbatch), not
inside each fit, so SpecModel's ~1-2 GB is paid once per node instead of once per fit.

Restartable: a galaxy whose file already has spectra (attrs["spectra"] starts with "computed")
is skipped; each file is rewritten through a temporary copy and os.replace, so a job killed
mid-write (preemptable queue) leaves the old file intact, and the requeued job redoes only
the galaxies still missing. make_catalog_nss.py lists any galaxy without spectra; run this
script on that list to catch up (any node, preemptable or not).

Model: E-FSPS SpecModel with the stellar library the w256 photometric emulator was trained on
(c3k_hr-mist-afe, E-FSPS commit cc1b132), configured like the emulator's inputs: every
emulator input free, efsps-photfit's fixed values for the rest (afe 0, gas_logno 0,
gas_logco 0, eline_sigma 100 km/s), Kriek & Conroy dust (dust_type 4), dust emission,
nebular emission, AGN torus (fagn normalisation), IGM. Luminosity distances: COSMO (WMAP9,
E-FSPS's default when the training set was made; it matches the emulator better than
Planck18, but still leaves ~0.02-0.04 mag; the exact training settings are to be confirmed by
Joel, STATUS.md). Spectra are computed at the network's reference mass (10^10 Msun formed) and
scaled by 10^(logmass - 10) x mu per draw, as the photometric model is.

Products, per galaxy, from its 1,000 equal-weight draws and its best fit:
  spectra/wavelength_rest (n_pix,)   rest-frame wavelength, A (the library grid)
  spectra/percentiles (n_pix, 7) f4  observed-frame f_nu (maggies, magnified) per rest pixel,
                                     i.e. at lambda_rest x (1 + z) of each draw; percentiles over
                                     the draws (as postprocess_parrot's modspecs_perc)
  spectra/bestfit (n_pix,) f4        the same for the best fit (modspec_map); attrs zred
  spectra/phot_check_dmag (nb,)      per band, median over draws of m(SpecModel) - m(network)
                                     for the fit's own bands; attrs median / sigma over all
  draws/rest_mags (K, 10) f4         rest-frame AB mags (names in attrs)
  percentiles/rest_UVJugi (6, 7), rest_UVJugi_colors (4, 7) [U-V, V-J, g-i, u-g],
  rest_gz (2, 7), rest_gz_colors (7,), rest_NUVrJ (3, 7), rest_NUVrJ_colors (2, 7) [NUV-r,
  r-J], and *_map for the best fit: postprocess_parrot's names, shapes and filters (sedpy
  bessell_U/V, twomass_J, synthetic_u/g/i, sdss_g0/z0/r0, galex_NUV, integrated exactly as
  sedpy does; restframe_filters_<grid>.npz, make_restframe_filters.py). As there: f_lambda
  deredshifted to the rest frame (lambda / (1 + z), f x (1 + z)), so the magnitudes include the
  distance and only colours are distance-free.

    python spectra_nss.py --libdir LIB --dir RESULTS            # every *_post.h5 below RESULTS
    python spectra_nss.py --libdir LIB --files a_post.h5 b_post.h5
    python spectra_nss.py --libdir LIB --list L --progress L.started   # as the job script runs it
    python spectra_nss.py --make-slim FULL_LIB SLIM_LIB         # weights-only copy (~250 MB)
Environment: E-FSPS importable (EFSPS_SRC, default ../E-FSPS next to efsps-photfit) with its
dependencies (the efsps env, not the nss env); EFSPS_PARENT for efsps-photfit.
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EFSPS_PARENT = os.environ.get("EFSPS_PARENT", "/Users/wren/Projects/MINERVA/efsps-photfit")
sys.path.insert(0, EFSPS_PARENT)
EFSPS_SRC = os.environ.get("EFSPS_SRC", os.path.join(os.path.dirname(os.path.abspath(EFSPS_PARENT)), "E-FSPS"))

COSMO = "WMAP9"
SWITCHES = dict(add_dust=True, add_dust_emission=True, add_neb_emission=True, add_agn=True,
                agn_norm="fagn", add_igm_absorption=True, dust_type=4)
REST_FILTERS = ["bessell_U", "bessell_V", "twomass_J", "synthetic_u", "synthetic_g", "synthetic_i",
                "sdss_g0", "sdss_z0", "galex_NUV", "sdss_r0"]
QPAH_TEMPLATE_MAX = 4.5799       # E-FSPS's q_PAH template grid ends at float32(4.58)
AB_FNU = 3.631e-20               # sedpy ab_gnu, erg/s/cm^2/Hz
C_AA = 2.998e18                  # sedpy lightspeed, A/s
DONE = "computed"


def _p(x):
    return np.asarray(x, float)


# ------------------------------------------------------------------ weights-only library copy
def make_slim(src, dst):
    """Copy a library with the multi-GB training-residual tables dropped (what E-FSPS's
    sidecar does): ann/*.pkl hold only what predict() reads. Ship THIS to Blanca."""
    os.makedirs(os.path.join(dst, "ann"), exist_ok=True)
    shutil.copy2(os.path.join(src, "meta_data.pkl"), os.path.join(dst, "meta_data.pkl"))
    for f in sorted(glob.glob(os.path.join(src, "ann", "*_ann.pkl"))):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        d.pop("residuals", None)
        out = os.path.join(dst, "ann", os.path.basename(f))
        with open(out + ".tmp", "wb") as fh:
            pickle.dump(d, fh, protocol=4)
        os.replace(out + ".tmp", out)
        print(f"  {os.path.basename(f)}: {os.path.getsize(f) / 1e9:.2f} GB -> "
              f"{os.path.getsize(out) / 1e6:.0f} MB", flush=True)


# ------------------------------------------------------------------ the model
class Spectra:
    def __init__(self, libdir, batch=50, cosmo=COSMO, fast_lines=True):
        sys.path.insert(0, EFSPS_SRC)
        import jax
        import jax.numpy as jnp
        from efsps.models import SpecModel, ModelParameter, ParameterSet, priors
        from efsps.observation import Photometry
        from efsps.models.transforms import dust1_fraction_to_dust1
        from efsps_photfit import config as C
        from efsps_photfit.emulator import PhotEmulator
        from nss_efsps import EMULATOR
        self.jax, self.jnp, self.Photometry = jax, jnp, Photometry
        self.libdir, self.batch, self.cosmo = os.path.abspath(libdir), int(batch), cosmo
        self.emu = PhotEmulator(EMULATOR, expected_md5=list(C.KNOWN_EMULATOR_MD5))
        self.fixed = {r.name: float(r.value) for r in C.PARAMETER_TABLE
                      if r.status == "fixed" and r.name in self.emu.parameters}
        lim = dict(zip(self.emu.parameters, np.asarray(self.emu.limits)))
        self.free = [n for n in self.emu.parameters if n not in self.fixed]
        ratios = [n for n in self.free if n.startswith("logsfr_ratios_")]
        pl = [ModelParameter(name="logsfr_ratios", isfree=True, init=[0.0] * len(ratios),
                             prior=priors.MultivariateStudent(
                                 min=float(lim[ratios[0]][0]), max=float(lim[ratios[0]][1]),
                                 mean=[0.0] * len(ratios), df=2.0,
                                 scale=[[1.0 if i == j else 0.0 for j in range(len(ratios))]
                                        for i in range(len(ratios))]))]
        for n in self.free:
            if n in ratios:
                continue
            lo, hi = (float(v) for v in lim[n])
            if n == "duste_qpah":
                hi = min(hi, QPAH_TEMPLATE_MAX)
            pl.append(ModelParameter(name=n, isfree=True, init=0.5 * (lo + hi),
                                     prior=priors.Uniform(min=lo, max=hi)))
        for n, v in self.fixed.items():
            pl.append(ModelParameter(name=n, isfree=False, init=v))
        pl.append(ModelParameter(name="logmass", isfree=False, init=10.0))     # REF_LOGMASS
        pl.append(ModelParameter(name="dust1", isfree=False, init=0.0,
                                 depends_on=dust1_fraction_to_dust1))
        self.model = SpecModel(ParameterSet(pl), emulator_model_path=self.libdir, cosmo=cosmo,
                               **SWITCHES)
        self.order = list(self.model.free_parameter_order)
        with open(os.path.join(self.libdir, "meta_data.pkl"), "rb") as fh:
            meta = pickle.load(fh)
        self.wave = np.asarray(meta["wavelengths"], float)        # float64; the model runs float32
        if not np.allclose(np.asarray(self.model.spec_wavelengths, float), self.wave, rtol=1e-6, atol=0):
            raise ValueError("SpecModel's wavelength grid differs from the library's meta_data.pkl")
        if fast_lines:
            import efsps_fast_lines
            efsps_fast_lines.enable(self.model)
        self.fast_lines = bool(fast_lines)
        self._fns, self._emus = {}, {}
        rf = self._rest_filter_file()
        if list(rf["names"]) != REST_FILTERS or int(rf["n_wave"]) != self.wave.size:
            raise ValueError(f"rest-frame filter table does not match this library's grid")
        self.rest_w, self.rest_zero = rf["weights"], rf["zero_counts"]
        self.config = {"library": os.path.basename(self.libdir.rstrip("/")),
                       "stellar_library": str(meta.get("stellar library")),
                       "imf_type": np.asarray(meta.get("imf_type")).tolist(),
                       "cosmo": cosmo, "switches": SWITCHES, "fixed": self.fixed,
                       "fast_lines": self.fast_lines, "efsps_src": EFSPS_SRC,
                       "efsps_git": _git(EFSPS_SRC), "emulator_md5": self.emu.md5}

    def _rest_filter_file(self):
        import hashlib
        cands = sorted(glob.glob(os.path.join(HERE, "restframe_filters_*.npz")))
        sha = hashlib.sha1(np.ascontiguousarray(self.wave).tobytes()).hexdigest()
        for c in cands:
            d = np.load(c)
            if str(d["wave_sha1"]) == sha:
                return d
        raise FileNotFoundError(f"no restframe_filters_*.npz in {HERE} for this wavelength grid; "
                                f"make one with make_restframe_filters.py")

    def _fn(self, bands):
        key = tuple(bands)
        if key not in self._fns:
            obs = self.Photometry(filternames=list(bands), flux=np.ones(len(bands)),
                                  uncertainty=np.ones(len(bands)))
            model = self.model

            def run(th):
                phot = model.predict(th, observations=[obs])[0]
                return model.spec, phot

            self._fns[key] = self.jax.jit(run)
        return self._fns[key]

    def predict(self, values, bands):
        """values: dict name -> (N,) for every free emulator input (efsps names). Returns
        per-10^10-Msun-formed observed spectra (N, n_pix) and photometry (N, nb), unmagnified."""
        n = len(values["zred"])
        T = np.empty((n, len(self.order)), np.float32)
        for j, name in enumerate(self.order):
            v = _p(values[name])
            T[:, j] = np.minimum(v, QPAH_TEMPLATE_MAX) if name == "duste_qpah" else v
        fn = self._fn(bands)
        spec = np.empty((n, self.wave.size), np.float32)
        phot = np.empty((n, len(bands)), np.float32)
        B = self.batch
        for a in range(0, n, B):
            blk = T[a:a + B]
            m = blk.shape[0]
            if m < B:                                    # one compiled shape: pad the last batch
                blk = np.concatenate([blk, np.repeat(blk[-1:], B - m, axis=0)])
            s, ph = fn(self.jnp.asarray(blk))
            spec[a:a + m] = np.asarray(s)[:m]
            phot[a:a + m] = np.asarray(ph)[:m]
        return spec, phot

    def network(self, values, bands):
        key = tuple(bands)
        if key not in self._emus:
            from efsps_photfit.emulator import PhotEmulator
            from nss_efsps import EMULATOR
            self._emus[key] = PhotEmulator(EMULATOR, bands=list(bands))
        emu = self._emus[key]
        return emu.predict_maggies(emu.assemble({p: (values[p] if p in values else self.fixed[p])
                                                 for p in emu.parameters}), check=False)

    def rest_mags(self, fnu_obs, z):
        """postprocess_parrot: f_lambda at lambda_obs, deredshifted (lambda / (1+z), f x (1+z)),
        through sedpy's filters. fnu_obs (N, n_pix) maggies per rest pixel."""
        flam_rest = fnu_obs.astype(float) * AB_FNU * C_AA / (self.wave[None, :] ** 2 * (1.0 + _p(z)[:, None]))
        with np.errstate(divide="ignore", invalid="ignore"):
            return -2.5 * np.log10((flam_rest @ self.rest_w.T) / self.rest_zero[None, :])


def _git(path):
    import subprocess
    try:
        h = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return h.stdout.strip() if h.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ one galaxy
def _rest_products(m):
    """m (N, 10) mags in REST_FILTERS order -> postprocess_parrot's arrays (before percentiles)."""
    U, V, J, u, g, i, g0, z0, nuv, r0 = (m[:, k] for k in range(10))
    return {"rest_UVJugi": np.stack([U, V, J, u, g, i], axis=1),
            "rest_UVJugi_colors": np.stack([U - V, V - J, g - i, u - g], axis=1),
            "rest_gz": np.stack([g0, z0], axis=1),
            "rest_gz_colors": g0 - z0,
            "rest_NUVrJ": np.stack([nuv, r0, J], axis=1),
            "rest_NUVrJ_colors": np.stack([nuv - r0, r0 - J], axis=1)}


def process(sp, path, redo=False):
    import h5py
    for stale in glob.glob(path + ".spectra_tmp.*"):          # left by a killed earlier run
        os.remove(stale)
    with h5py.File(path, "r") as f:
        if str(f.attrs.get("spectra", "")).startswith(DONE) and not redo:
            return "skipped"
        percents = f.attrs["percents"][()]
        dg = f["draws"]
        names = [n for n in sp.free if n in dg]
        missing = [n for n in sp.free if n not in dg]
        if missing:
            raise ValueError(f"draws lack {missing}")
        vals = {n: dg[n][()].astype(float) for n in names}
        lm = dg["logmass"][()].astype(float)
        bnames = json.loads(f["bestfit"].attrs["names"])
        bt = f["bestfit/theta"][()]
        best = {n: np.array([bt[bnames.index(n)]]) if n in bnames else None for n in sp.free}
        if best["zred"] is None:                               # fixed-redshift fit
            best["zred"] = np.array([float(vals["zred"][0])])
        lm_best = float(bt[bnames.index("logmass")])
        bands = [x.decode() if isinstance(x, bytes) else str(x) for x in f["obs/bands"][()]]
        valid = f["obs/valid"][()].astype(bool)
        mu = float(f["obs"].attrs["mu"])
    allv = {n: np.concatenate([vals[n], best[n]]) for n in sp.free}
    spec, phot = sp.predict(allv, bands)                       # per 1e10 Msun formed
    net = sp.network(allv, bands)
    scale = 10.0 ** (np.concatenate([lm, [lm_best]]) - 10.0) * mu
    spec_obs = spec * scale[:, None].astype(np.float32)
    mags = sp.rest_mags(spec_obs, allv["zred"])
    K = len(lm)
    # SpecModel photometry vs the network the fit used, on the fit's valid, non-faint bands
    with np.errstate(divide="ignore", invalid="ignore"):
        dm = -2.5 * np.log10(phot / net)
    bright = (net > 1e-3 * net.max(axis=1, keepdims=True)) & valid[None, :] & np.isfinite(dm)
    dm_band = np.array([np.median(dm[:K, j][bright[:K, j]]) if bright[:K, j].any() else np.nan
                        for j in range(len(bands))])
    allb = dm[:K][bright[:K]]
    rp = _rest_products(mags[:K])
    rb = _rest_products(mags[K:])
    tmp = f"{path}.spectra_tmp.{os.getpid()}"
    shutil.copy2(path, tmp)
    try:
        with h5py.File(tmp, "r+") as f:
            for k in ("spectra",):
                if k in f:
                    del f[k]
            for k in list(rp) + [k + "_map" for k in rp]:
                if f"percentiles/{k}" in f:
                    del f[f"percentiles/{k}"]
            if "draws/rest_mags" in f:
                del f["draws/rest_mags"]
            g = f.create_group("spectra")
            g.create_dataset("wavelength_rest", data=sp.wave, compression="gzip", shuffle=True)
            g.create_dataset("percentiles", data=np.percentile(spec_obs[:K], percents, axis=0).T
                             .astype(np.float32), compression="gzip", shuffle=True)
            g.create_dataset("bestfit", data=spec_obs[K], compression="gzip", shuffle=True)
            g["bestfit"].attrs["zred"] = float(allv["zred"][K])
            g["percentiles"].attrs["units"] = ("observed f_nu, maggies (magnified), per rest-frame "
                                               "pixel: lambda_obs = wavelength_rest x (1 + z) per draw")
            g.create_dataset("phot_check_dmag", data=dm_band)
            g.attrs.update({"phot_check_median": float(np.median(allb)) if allb.size else np.nan,
                            "phot_check_sigma": float(0.5 * np.subtract(*np.percentile(allb, [84, 16])))
                            if allb.size else np.nan,
                            "phot_check_note": "m(SpecModel) - m(network), fit's valid non-faint bands",
                            "config": json.dumps(sp.config, default=str)})
            d = f["draws"].create_dataset("rest_mags", data=mags[:K].astype(np.float32))
            d.attrs["names"] = json.dumps(REST_FILTERS)
            pg = f["percentiles"]
            for k, v in rp.items():
                pg.create_dataset(k, data=np.percentile(v, percents, axis=0).T)
                pg.create_dataset(k + "_map", data=rb[k][0])
            f.attrs["spectra"] = (f"{DONE} {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}: "
                                  f"E-FSPS SpecModel, {sp.config['library']}, cosmo {sp.cosmo}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return f"done (phot check {np.median(allb):+.3f} mag)" if allb.size else "done"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--libdir", default=os.environ.get("EFSPS_LIB"))
    p.add_argument("--files", nargs="*", default=[])
    p.add_argument("--list", default=None, help="text file with one _post.h5 path per line")
    p.add_argument("--dir", default=None, help="every *_post.h5 below this directory")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--cosmo", default=COSMO)
    p.add_argument("--redo", action="store_true")
    p.add_argument("--progress", default=None,
                   help="append each galaxy's path to this file as it is STARTED (the job script "
                        "charges a failure only to galaxies a failed process had started)")
    p.add_argument("--no-fast-lines", action="store_true")
    p.add_argument("--make-slim", nargs=2, metavar=("FULL_LIB", "SLIM_LIB"))
    args = p.parse_args()
    if args.make_slim:
        make_slim(*args.make_slim)
        return
    files = list(args.files)
    if args.list:
        files += [l.strip() for l in open(args.list) if l.strip()]
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, "**", "*_post.h5"), recursive=True))
    files = [f for f in dict.fromkeys(files) if os.path.exists(f)]
    n_given = len(files)
    for f in files:                         # temporary copies left by a killed run
        for stale in glob.glob(f + ".spectra_tmp.*"):
            os.remove(stale)
    if not args.redo:                       # don't load SpecModel for galaxies already done
        import h5py
        todo = []
        for f in files:
            try:
                with h5py.File(f, "r") as h:
                    if str(h.attrs.get("spectra", "")).startswith(DONE):
                        continue
            except OSError:
                pass                        # unreadable: let process() report it
            todo.append(f)
        files = todo
    if not files:
        print(f"spectra_nss: nothing to do ({n_given} files, all with spectra or none found)", flush=True)
        return
    if not args.libdir:
        raise SystemExit("give --libdir or set EFSPS_LIB")
    t0 = time.time()
    sp = Spectra(args.libdir, batch=args.batch, cosmo=args.cosmo, fast_lines=not args.no_fast_lines)
    print(f"spectra_nss: SpecModel ready in {time.time() - t0:.1f} s ({sp.config['library']}, "
          f"cosmo {sp.cosmo}); {len(files)} files", flush=True)
    n_ok = n_skip = n_fail = 0
    for path in files:
        t = time.time()
        if args.progress:
            with open(args.progress, "a") as fh:
                fh.write(path + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        try:
            r = process(sp, path, redo=args.redo)
            n_skip += r == "skipped"
            n_ok += r != "skipped"
            print(f"  {os.path.basename(path)}: {r} [{time.time() - t:.1f} s]", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"  {os.path.basename(path)}: FAILED {e!r}\n{traceback.format_exc()}", flush=True)
    n_skip += n_given - len(files)
    print(f"spectra_nss: {n_ok} done, {n_skip} already had spectra, {n_fail} failed, "
          f"{time.time() - t0:.0f} s in all", flush=True)


if __name__ == "__main__":
    main()
