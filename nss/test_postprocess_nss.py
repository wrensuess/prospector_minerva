"""Checks postprocess_nss.py outputs against independent reference computations, from the
fit's own full sample file. For each <stem>_post.h5 next to <stem>.npz and <stem>.json:

 1. draws/loglike == efsps-photfit FitModel.loglike(theta) for the same samples (exact code
    path of the fit's likelihood, run separately);
 2. best fit: its loglike is the maximum of FitModel.loglike over all kept samples, and chi2
    is recomputed from scratch (observed error, 5% floor, emulator error);
 3. parameter percentiles: the 50th equals the fit summary's median; 15.9/84.1 equal
    efsps_photfit.fit.weighted_quantiles;
 4. stellar mass, mwa, SFR10, SFR100, sSFR100 of the draws == efsps_photfit.fit.derived_quantities
    (float32 storage);
 5. SFR over 30 Myr against a brute-force integral of the step SFH (and 10, 100 Myr too);
    the per-bin SFRs integrate back to the formed mass;
 6. SFH(t) percentiles == postprocess_parrot's own stepInterp algorithm (copied below) on the
    full weighted posterior; the draws' bin edges == efsps agebins_batch(z); the cumulative
    formed-mass history (Msun and fraction) == an independent piecewise-linear
    interpolation of the cumulative bin masses, non-increasing with lookback time;
 7. model-flux percentiles == weighted quantiles of FitModel.model_fluxes;
 8. draws vs the weighted posterior: medians agree within 4 Monte Carlo sigma;
 9. observed photometry and the valid mask match the catalogue; fixed-z fits store z.

    python test_postprocess_nss.py --catalog CAT --id ID --stem RUN/ID_TAG [--fix-z Z]
Ends RESULT: PASS or RESULT: FAIL.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("EFSPS_PARENT", "/Users/wren/Projects/MINERVA/efsps-photfit"))
import h5py                                                    # noqa: E402
from efsps_photfit.fit import weighted_quantiles, derived_quantities   # noqa: E402
from efsps_photfit.survival import SurvivingFraction           # noqa: E402
from efsps_photfit.sfh import agebins_batch                    # noqa: E402
from photometry import read_photometry                         # noqa: E402
from nss_efsps import build                                    # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--catalog", required=True)
p.add_argument("--id", required=True)
p.add_argument("--stem", required=True)
p.add_argument("--fix-z", type=float, default=None)
args = p.parse_args()

fails = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        fails.append(name)


phot = read_photometry(args.catalog, [args.id])[0]
model = build(phot, args.fix_z, None)
d = np.load(args.stem + ".npz")
summ = json.load(open(args.stem + ".json"))
f = h5py.File(args.stem + "_post.h5", "r")
theta_all, w_all = d["theta"], d["w"] / d["w"].sum()
keep = w_all > 1e-8 * w_all.max()
theta, w = theta_all[keep], w_all[keep] / w_all[keep].sum()
kept_idx = np.flatnonzero(keep)
P = f.attrs["percents"]
idx = f["draws/index"][:]
pos = np.searchsorted(kept_idx, idx)                        # row of each draw among kept samples
z_str = f"z_spec {f.attrs['z_spec']:.3f}" if np.isfinite(f.attrs["z_spec"]) else "no z_spec"
print(f"[{args.id}] {z_str}; {theta.shape[0]} kept samples; fixed z {args.fix_z}")
check("summary marks postprocess ok", "error" not in summ.get("postprocess", {}),
      str(summ.get("postprocess", {}).get("error", "no error")))

# 1. loglike of the draws
ll_ref = model.loglike(theta_all[idx])
dl = np.max(np.abs(f["draws/loglike"][:] - ll_ref))
check("1 draw loglike = FitModel.loglike", dl < 1e-9, f"max |diff| {dl:.2e}")

# 2. best fit
ll_all = model.loglike(theta)
ib = int(np.argmax(ll_all))
b = f["bestfit"]
check("2a best fit = max loglike sample", int(b.attrs["index"]) == kept_idx[ib]
      and abs(b.attrs["loglike"] - ll_all[ib]) < 1e-9,
      f"index {b.attrs['index']} vs {kept_idx[ib]}, loglike {b.attrs['loglike']:.4f} vs {ll_all[ib]:.4f}")
fm, sig, _ = model.model_fluxes(theta[ib][None])
v = model.valid
var = phot.err[v] ** 2 + (0.05 * phot.flux[v]) ** 2 + sig[0, v] ** 2
chi2 = np.sum((phot.flux[v] - fm[0, v]) ** 2 / var)
check("2b chi2 from scratch", abs(chi2 - b.attrs["chi2"]) < 1e-8 * max(1.0, chi2),
      f"{b.attrs['chi2']:.4f} vs {chi2:.4f}; nbands {b.attrs['nbands']} vs {int(v.sum())}")

# 3. percentiles
worst50, worst = 0.0, 0.0
for i, n in enumerate(model.free):
    pp = f[f"percentiles/{n}"][:]
    worst50 = max(worst50, abs(pp[3] - summ["percentiles"][n][1]))
    ref = weighted_quantiles(theta[:, i], w, (0.001, 0.023, 0.159, 0.841, 0.977, 0.999))
    worst = max(worst, np.max(np.abs(pp[[0, 1, 2, 4, 5, 6]] - ref)))
check("3a medians = fit summary medians", worst50 < 1e-12, f"max |diff| {worst50:.1e}")
check("3b other percentiles = efsps weighted_quantiles", worst < 1e-12, f"max |diff| {worst:.1e}")
for ln, lin in (("log_fagn", "fagn"), ("log_agn_tau", "agn_tau"), ("log_duste_gamma", "duste_gamma")):
    ref = weighted_quantiles(np.log10(theta[:, model.col[lin]]), w, (0.5,))[0]
    check(f"3c {ln} median", abs(f[f'percentiles/{ln}'][3] - ref) < 1e-12,
          f"{f[f'percentiles/{ln}'][3]:.5f} vs {ref:.5f}")
if args.fix_z is not None:
    check("9b fixed z stored", np.allclose(f["percentiles/zred"][:], args.fix_z)
          and np.allclose(f["draws/zred"][:], args.fix_z), f"{f['percentiles/zred'][3]}")

# 4. derived quantities of the draws vs efsps-photfit
der = derived_quantities(model, theta_all[idx], SurvivingFraction())
pairs = [("stellar_mass", f["draws/stellar_mass"][:], der["logmstar_surviving"], 2e-6),
         ("mwa", f["draws/mwa"][:], der["mwa_gyr"], 2e-6),
         ("log sfr10", np.log10(f["draws/sfr"][:, 0]), der["logsfr10"], 2e-6),
         ("log sfr100", np.log10(f["draws/sfr"][:, 2]), der["logsfr100"], 2e-6),
         ("log ssfr100", np.log10(f["draws/ssfr"][:, 2]), der["logssfr100"], 2e-6)]
for n, a, r, tol in pairs:
    ok = np.isfinite(r)
    e = np.max(np.abs(a[ok] - r[ok]) / np.maximum(1.0, np.abs(r[ok])))
    check(f"4 {n} = efsps derived_quantities", e < tol, f"max rel diff {e:.1e}")

# 5. SFRs by brute force; mass conservation
vals = model.values(theta_all[idx])
z = np.broadcast_to(np.asarray(vals["zred"], float), (idx.size,))
ab = agebins_batch(z)
t0, t1 = 10 ** ab[..., 0], 10 ** ab[..., 1]
sb = f["draws/sfr_bins"][:].astype(float)
mform = (sb * (t1 - t0)).sum(axis=1)
lm = np.broadcast_to(np.asarray(vals["logmass"], float), (idx.size,))
e = np.max(np.abs(np.log10(mform) - lm))
check("5a per-bin SFRs integrate to the formed mass", e < 1e-6, f"max |dlog M| {e:.1e}")
for k, T in enumerate((1e7, 3e7, 1e8)):
    tt = (np.arange(200000) + 0.5) * (T / 200000)          # midpoints on [0, T]
    rows = np.arange(0, idx.size, 50)
    brute = np.array([sb[r][np.clip(np.searchsorted(t1[r], tt, side="right"), 0, 9)].mean()
                      for r in rows])
    ours = f["draws/sfr"][rows, k].astype(float)
    e = np.max(np.abs(ours - brute) / np.maximum(brute, 1e-30))
    check(f"5b SFR over {T / 1e6:.0f} Myr vs brute force", e < 1e-4, f"max rel diff {e:.1e}")


# 6. SFH(t) percentiles vs postprocess_parrot's algorithm
def stepInterp(ab, val, ts):                                 # postprocess_parrot.stepInterp
    newval = np.zeros_like(ts) + np.nan
    for i in range(0, len(ab)):
        newval[(ts >= ab[i, 0]) & (ts < ab[i, 1])] = val[i]
    newval[-1] = 0
    return newval


vk = model.values(theta)
zk = np.broadcast_to(np.asarray(vk["zred"], float), (theta.shape[0],))
abk = 10 ** agebins_batch(zk)
from postprocess_nss import sfh_quantities                   # noqa: E402
qk = sfh_quantities(vk, theta.shape[0])
grid = f["sfh_grid"][:]
check("6a grid end = oldest edge", np.isclose(np.log10(grid[-1]), np.log10(abk[..., 1].max()), atol=1e-12)
      and np.isclose(grid[0], 10.0), f"agebins_max {f['sfh_grid'].attrs['agebins_max']:.4f}")
old = np.array([stepInterp(abk[r], qk["sfr_bins"][r], grid) for r in range(theta.shape[0])])
old[np.isnan(old)] = 0
ref = np.array([weighted_quantiles(old[:, j], w, P / 100.0) for j in range(grid.size)])
e = np.max(np.abs(f["percentiles/sfh"][:] - ref) / np.maximum(np.abs(ref), 1e-30))
check("6b SFH(t) percentiles = postprocess_parrot stepInterp", e < 1e-10, f"max rel diff {e:.1e}")

# 6c. bin edges of the draws
e = np.max(np.abs(f["draws/agebins"][:] - agebins_batch(z)))
check("6c draws/agebins = efsps agebins_batch(z)", e < 1e-5, f"max |diff| {e:.1e} dex (float32)")

# 6d. cumulative formed mass, independently: piecewise linear in t between bin edges
mass_k = qk["sfr_bins"] * (abk[..., 1] - abk[..., 0])
edges = np.concatenate([abk[:, :1, 0], abk[..., 1]], axis=1)           # (N, 11) yr
mcum_edges = np.concatenate([mass_k[:, ::-1].cumsum(axis=1)[:, ::-1], np.zeros((len(w), 1))],
                            axis=1)                                     # mass older than each edge
ref_abs = np.array([np.interp(grid, edges[r], mcum_edges[r], left=mcum_edges[r, 0], right=0.0)
                    for r in range(len(w))])
ref_frac = ref_abs / mcum_edges[:, :1]
for key, ref_s in (("mformed_cum", ref_abs), ("mformed_cum_frac", ref_frac)):
    refp = np.array([weighted_quantiles(ref_s[:, j], w, P / 100.0) for j in range(grid.size)])
    got = f[f"percentiles/{key}"][:]
    e = np.max(np.abs(got - refp) / np.maximum(np.abs(refp), 1e-300 if key == "mformed_cum" else 1e-12))
    med = got[:, 3]
    mono = bool(np.all(np.diff(med) <= 1e-12 * max(1.0, med.max())))
    check(f"6d {key} = piecewise-linear reference", e < 1e-9,
          f"max rel diff {e:.1e}; median non-increasing in t: {mono}; last point {got[-1, 3]:.1e}")
    if not mono:
        fails.append(f"6d {key} monotonic")
fr = f["percentiles/mformed_cum_frac"][:]
check("6e fraction in [0, 1], ~1 at 10 yr", fr.min() >= 0 and fr.max() <= 1 + 1e-12 and fr[0, 3] > 0.99,
      f"median at 10 yr {fr[0, 3]:.6f}")

# 7. model-flux percentiles
fm_all = model.model_fluxes(theta)[0]
ref = np.array([weighted_quantiles(fm_all[:, j], w, P / 100.0) for j in range(fm_all.shape[1])])
e = np.max(np.abs(f["model_flux_percentiles"][:] - ref) / np.abs(ref))
check("7 model-flux percentiles", e < 1e-12, f"max rel diff {e:.1e}")

# 8. draws vs weighted posterior
for n, a, ref in (("zred", f["draws/zred"][:], f["percentiles/zred"][:]),
                  ("logmass", f["draws/logmass"][:], f["percentiles/logmass"][:]),
                  ("stellar_mass", f["draws/stellar_mass"][:], f["percentiles/stellar_mass"][:]),
                  ("dust2", f["draws/dust2"][:], f["percentiles/dust2"][:])):
    half = 0.5 * (ref[4] - ref[2])
    mc = 1.2533 * max(half, 1e-12) / np.sqrt(a.size)
    dm = abs(np.median(a) - ref[3])
    check(f"8 draws median {n}", dm < 4 * mc + 1e-6, f"|diff| {dm:.4f}, 4 MC sigma {4 * mc:.4f}")

# 9. observed photometry
o = f["obs"]
same = (np.array([x.decode() if isinstance(x, bytes) else x for x in o["bands"][:]]) == np.array(phot.bands)).all()
vv = o["valid"][:].astype(bool)
check("9a obs bands, flux, valid", same and (vv == model.valid).all()
      and np.allclose(o["flux"][:][vv], phot.flux[vv], rtol=0, atol=0),
      f"{int(vv.sum())}/{vv.size} valid")

print(f"RESULT: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")
