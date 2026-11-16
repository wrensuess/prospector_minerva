"""Checks spectra_nss.py outputs (per-galaxy *_post.h5 after the spectra step):

 1. spectra: recomputed with UNPATCHED E-FSPS (no efsps_fast_lines) from the same draws, the
    best-fit spectrum and every spectrum percentile agree to float32 rounding;
 2. rest-frame magnitudes of the best fit (rest_*_map) == sedpy.observate.getSED on the stored
    best-fit spectrum, deredshifted as postprocess_parrot.py does (run in --sedpy-python, the
    env the dynesty pipeline used);
 3. rest-frame percentiles == np.percentile of draws/rest_mags (and the colours);
 4. SpecModel vs network photometry (phot_check) within --max-offset mag (the training-config
    question, STATUS.md): reported, and a FAIL only beyond the threshold;
 5. restartability: a second run skips the file; a run killed mid-galaxy leaves the file intact
    and no temporary file once the next run starts.

    python test_spectra_nss.py --libdir SLIM_LIB --dir DIR [--sedpy-python /opt/anaconda3/envs/prosp/bin/python]
Ends RESULT: PASS or RESULT: FAIL.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h5py                                                     # noqa: E402
import spectra_nss as SN                                        # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--libdir", required=True)
p.add_argument("--dir", required=True)
p.add_argument("--sedpy-python", default="/opt/anaconda3/envs/prosp/bin/python")
p.add_argument("--max-offset", type=float, default=0.1)
p.add_argument("--skip-restart", action="store_true")
args = p.parse_args()
fails = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        fails.append(name)


files = sorted(glob.glob(os.path.join(args.dir, "*_post.h5")))
ref = SN.Spectra(args.libdir, fast_lines=False)                 # E-FSPS exactly as it is
SED = """
import sys, json, numpy as np
from sedpy import observate
d = np.load(sys.argv[1])
fl = observate.load_filters(json.loads(sys.argv[2]))
print(json.dumps(observate.getSED(d["wave"], d["flam"], filterlist=fl).tolist()))
"""
for path in files:
    with h5py.File(path, "r") as f:
        print(f"[{os.path.basename(path)}] z_ml {f['bestfit'].attrs['zred']:.3f}; {f.attrs['spectra'][:60]}")
        P = f.attrs["percents"][()]
        names = [n for n in ref.free]
        vals = {n: f[f"draws/{n}"][()].astype(float) for n in names}
        lm = f["draws/logmass"][()].astype(float)
        bn = json.loads(f["bestfit"].attrs["names"])
        bt = f["bestfit/theta"][()]
        best = {n: np.array([bt[bn.index(n)]]) if n in bn else np.array([vals["zred"][0]]) for n in names}
        bands = [x.decode() if isinstance(x, bytes) else str(x) for x in f["obs/bands"][()]]
        mu = float(f["obs"].attrs["mu"])
        s_best, s_perc = f["spectra/bestfit"][()], f["spectra/percentiles"][()]
        rest_mags = f["draws/rest_mags"][()]
        stored = {k: f[f"percentiles/{k}"][()] for k in ("rest_UVJugi", "rest_UVJugi_colors", "rest_gz",
                                                          "rest_gz_colors", "rest_NUVrJ", "rest_NUVrJ_colors")}
        maps = {k: f[f"percentiles/{k}_map"][()] for k in stored}
        g = f["spectra"]
        pc_med, pc_sig = g.attrs["phot_check_median"], g.attrs["phot_check_sigma"]
        wave = g["wavelength_rest"][()]
    # 1. unpatched E-FSPS
    allv = {n: np.concatenate([vals[n], best[n]]) for n in names}
    spec, _ = ref.predict(allv, bands)
    scale = 10.0 ** (np.concatenate([lm, [float(bt[bn.index('logmass')])]]) - 10.0) * mu
    so = spec * scale[:, None].astype(np.float32)
    K = len(lm)
    sig = so[K] > 1e-6 * so[K].max()
    e1 = np.max(np.abs(so[K][sig] - s_best[sig]) / np.abs(s_best[sig]))
    pref = np.percentile(so[:K], P, axis=0).T
    sigp = pref > 1e-6 * pref.max()
    e2 = np.max(np.abs(pref[sigp] - s_perc[sigp]) / np.abs(pref[sigp]))
    check("1a best-fit spectrum = unpatched E-FSPS", e1 < 1e-4, f"max rel diff {e1:.1e} (pixels above 1e-6 of max)")
    check("1b spectrum percentiles = unpatched E-FSPS", e2 < 1e-4, f"max rel diff {e2:.1e}")
    # 2. sedpy on the stored best-fit spectrum (postprocess_parrot's deredshift)
    z = float(allv["zred"][K])
    flam_rest = s_best.astype(float) * SN.AB_FNU * SN.C_AA / (wave ** 2 * (1.0 + z))
    with tempfile.TemporaryDirectory() as td:
        np.savez(os.path.join(td, "s.npz"), wave=wave, flam=flam_rest)
        out = subprocess.run([args.sedpy_python, "-c", SED, os.path.join(td, "s.npz"), json.dumps(SN.REST_FILTERS)],
                             capture_output=True, text=True)
    sed = np.array(json.loads(out.stdout.strip().splitlines()[-1]))
    mine = np.concatenate([maps["rest_UVJugi"], maps["rest_gz"], [maps["rest_NUVrJ"][0], maps["rest_NUVrJ"][1]]])
    sed_sel = np.concatenate([sed[:6], sed[6:8], [sed[8], sed[9]]])
    e3 = np.max(np.abs(mine - sed_sel))
    col = maps["rest_UVJugi_colors"]
    check("2 rest-frame mags of the best fit = sedpy getSED", e3 < 1e-4,
          f"max |diff| {e3:.1e} mag; U-V {col[0]:.3f}, V-J {col[1]:.3f}")
    # 3. percentiles of the draws' rest mags
    rp = SN._rest_products(rest_mags.astype(float))
    e4 = max(np.max(np.abs(np.atleast_1d(np.percentile(v, P, axis=0).T) - np.atleast_1d(stored[k])))
             for k, v in rp.items())
    check("3 rest-frame percentiles = np.percentile of the draws", e4 < 1e-5, f"max |diff| {e4:.1e} mag")
    check("4 SpecModel vs network photometry", abs(pc_med) < args.max_offset,
          f"median {pc_med:+.3f} mag, sigma {pc_sig:.3f} (threshold {args.max_offset})")

# 5. restartability, on a copy
if not args.skip_restart:
    env = dict(os.environ)
    with tempfile.TemporaryDirectory() as td:
        src = files[0]
        cp = os.path.join(td, os.path.basename(src))
        shutil.copy2(src, cp)
        cmd = [sys.executable, "-u", os.path.join(HERE, "spectra_nss.py"), "--libdir", args.libdir, "--files", cp]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        check("5a second run skips a finished file", "nothing to do" in r.stdout or " 1 already" in r.stdout,
              r.stdout.strip().splitlines()[-1])
        h0 = hashlib.md5(open(cp, "rb").read()).hexdigest()
        for when in ("computing", "writing"):
            # kill (a) while the galaxy is being computed, (b) the moment its temporary copy exists
            proc = subprocess.Popen(cmd + ["--redo"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, env=env)
            t0 = time.time()
            while time.time() - t0 < 300:
                if when == "computing" and time.time() - t0 > 12:          # after start-up, mid-galaxy
                    break
                if when == "writing" and glob.glob(cp + ".spectra_tmp.*"):
                    break
                time.sleep(0.01)
            proc.send_signal(signal.SIGKILL)
            proc.wait()
            h1 = hashlib.md5(open(cp, "rb").read()).hexdigest()
            try:
                with h5py.File(cp, "r") as f:
                    complete = str(f.attrs.get("spectra", "")).startswith(SN.DONE) and "spectra/bestfit" in f
            except Exception as e:
                complete = False
            state = "untouched" if h1 == h0 else ("replaced by a complete new file" if complete else "CORRUPT")
            check(f"5b killed while {when}: file untouched or complete", h1 == h0 or complete,
                  f"{state}; temporary copies left: {len(glob.glob(cp + '.spectra_tmp.*'))}")
            h0 = h1
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        check("5c next run removes the stale temporary copy", not glob.glob(cp + ".spectra_tmp.*"),
              r.stdout.strip().splitlines()[-1])
print(f"RESULT: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")
