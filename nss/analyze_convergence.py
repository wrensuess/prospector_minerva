"""Single-solution convergence of NSS, per parameter, from the 20-seed Blanca summaries.

Reads only the per-fit JSON summaries (p16/p50/p84 per parameter, logZ, quoted error, ESS),
so it runs where the posterior draws are not available.

Groups: results/blanca_layout_<job>_<id>_k<k>_L<L>_<prior>[_ps<N>]/<id>_seed<s>.json,
searched recursively under the root. Start "std" = standard start, "ps1e6" = 10^6 prior start.

Per galaxy, setting and parameter, over the seeds that found the main redshift solution:
  hw     posterior half-width, (p84 - p16)/2, median over seeds
  S50    std over seeds of the median / hw. Gelman-Rubin R-hat ~ sqrt(1 + S50^2);
         the acceptance threshold R-hat < 1.01 is S50 <= 0.14 (STATUS.md section 5).
  S16, S84   the same for p16 and p84: do runs agree on width and tails?
  floor  scatter expected from finite samples alone, with perfect sampling:
         1.25/sqrt(ESS) for the median, 1.51/sqrt(ESS) for p16/p84 (Gaussian).
  ESSeff (1.25/S50)^2: how many independent posterior draws a run is "worth" for that median.
  width  hw(k) / hw(k_ref): is a single run too narrow at small k? (proxy for W; the real
         W needs a reference posterior)
  bias   (mean p50(k) - mean p50(k_ref)) / hw, and its t value against the seed scatter.
  rho    posterior hw / prior hw (prior at the galaxy's redshift): near 1 = prior-dominated.
  R      logZ std over seeds / mean quoted error.

  EFSPS_PARENT=... python analyze_convergence.py <results root> [--catalog minerva_uds_test9.fits]
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BIMODAL = {"1241601"}          # genuine second redshift solution: reported, not summarised
S_OK, S_WARN = 0.14, 0.28      # R-hat 1.01, and twice that


def load(root):
    groups = defaultdict(list)
    pat = re.compile(r"blanca_layout_\d+_(\d+)_k(\d+)_L(\d+)_([a-z]+)(?:_ps(\w+))?$")
    for d in sorted(glob.glob(os.path.join(root, "**", "blanca_layout_*"), recursive=True)):
        m = pat.search(os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        gal, k, L, prior, ps = m.groups()
        start = f"ps{ps}" if ps else "std"
        for f in sorted(glob.glob(os.path.join(d, f"{gal}_seed*.json"))):
            groups[(gal, int(k), int(L), start)].append(json.load(open(f)))
    return groups


def prior_halfwidths(catalog, gal, z, n=20000, seed=1):
    """Prior p16-p84 half-width per parameter, at the galaxy's redshift (zred: free-z prior)."""
    from nss_efsps import build
    from photometry import read_photometry
    phot = read_photometry(catalog, [gal])[0]
    rng = np.random.default_rng(seed)
    out = {}
    for fz in (None, z):
        m = build(phot, fz, None)
        th = m.prior_transform(rng.uniform(size=(n, m.n_dim)))
        p16, p84 = np.percentile(th, [16, 84], axis=0)
        for i, name in enumerate(m.free):
            if fz is None and name != "zred":
                continue
            out[name] = 0.5 * (p84[i] - p16[i])
    return out


def group_stats(runs, zref):
    names = list(runs[0]["percentiles"])
    zc = names.index("zred")
    P = np.array([[r["percentiles"][n] for n in names] for r in runs])    # (seeds, par, 3)
    ok = np.abs(P[:, zc, 1] - zref) < 0.3 * (1 + zref)
    P = P[ok]
    hw = np.median(0.5 * (P[:, :, 2] - P[:, :, 0]), axis=0)
    S = P.std(axis=0, ddof=1) / hw[:, None]                                 # (par, 3)
    lz = np.array([r["logZ"] for r in runs])[ok]
    lze = np.array([r["logZ_err"] for r in runs])[ok]
    ess = np.median([r["ess"] for r in runs])
    return dict(names=names, P=P, hw=hw, S16=S[:, 0], S50=S[:, 1], S84=S[:, 2],
                mean50=P[:, :, 1].mean(0), var50=P[:, :, 1].var(0, ddof=1), n=int(ok.sum()),
                n_all=len(runs), R=lz.std(ddof=1) / lze.mean(), logz_std=lz.std(ddof=1),
                logz_err=lze.mean(), ess=ess, chi2=np.median([r["chi2_at_max"] for r in runs]),
                nb=runs[0]["n_bands_used"], iters=np.median([r["iterations"] for r in runs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--catalog", default=os.path.join(os.path.dirname(__file__),
                                                      "minerva_uds_test9.fits"))
    ap.add_argument("--no-prior", action="store_true", help="skip the prior widths (rho)")
    a = ap.parse_args()

    groups = load(a.root)
    gals = sorted({g[0] for g in groups})
    settings = sorted({(k, L, s) for _, k, L, s in groups}, key=lambda x: (x[2] != "std", x))
    print(f"{sum(len(v) for v in groups.values())} fits in {len(groups)} groups; "
          f"galaxies {gals}; settings (k, L, start) {settings}\n")

    # main redshift solution per galaxy: median of the prior-start runs (all found it), else all
    zref = {}
    for g in gals:
        zs = [r["percentiles"]["zred"][1] for (gg, k, L, s), v in groups.items()
              if gg == g and s.startswith("ps") for r in v]
        if not zs:
            zs = [r["percentiles"]["zred"][1] for (gg, *_), v in groups.items() if gg == g for r in v]
        zref[g] = float(np.median(zs))

    st = {key: group_stats(v, zref[key[0]]) for key, v in groups.items()}
    names = st[next(iter(st))]["names"]
    kref = max(k for k, _, s in settings if s == "std")
    single = [g for g in gals if g not in BIMODAL]

    rho = {}
    if not a.no_prior:
        for g in gals:
            ph = prior_halfwidths(a.catalog, g, zref[g])
            ref = st[(g, kref, 1000, "std")] if (g, kref, 1000, "std") in st else st[
                next(key for key in st if key[0] == g)]
            rho[g] = np.array([ref["hw"][i] / ph[n] for i, n in enumerate(names)])

    # ---------------------------------------------------------------- A. per setting
    print("== A. Per setting, over the single-solution galaxies "
          f"({len(single)}: {', '.join(single)}; {', '.join(BIMODAL)} reported separately)")
    print("   S50/S16/S84: median over galaxies of the per-galaxy median over parameters;")
    print("   worst: median over galaxies of the worst parameter; frac ok: share of "
          "(galaxy, parameter) with S50 <= 0.14;")
    print(f"   frac warn: share above {S_WARN}; width and bias are relative to k={kref} std; "
          "|t|>3: share of (galaxy, parameter) with a significant bias.")
    hdr = (f"   {'k':>4s} {'start':>6s} {'runs':>7s} | {'S50':>5s} {'worst':>5s} {'S16':>5s} "
           f"{'S84':>5s} {'floor':>5s} | {'ok':>5s} {'warn':>5s} | {'R':>5s} | {'width':>5s} "
           f"{'|bias|':>6s} {'|t|>3':>5s}")
    print(hdr)
    for k, L, s in settings:
        rows = [st[(g, k, L, s)] for g in single if (g, k, L, s) in st]
        if not rows:
            continue
        S50 = np.array([r["S50"] for r in rows])
        med = lambda x: float(np.median(x))
        width, bias, tsig = [], [], []
        for g in single:
            if (g, k, L, s) in st and (g, kref, 1000, "std") in st:
                r, ref = st[(g, k, L, s)], st[(g, kref, 1000, "std")]
                width.append(np.median(r["hw"] / ref["hw"]))
                d = r["mean50"] - ref["mean50"]
                bias.append(np.median(np.abs(d) / ref["hw"]))
                se = np.sqrt(r["var50"] / r["n"] + ref["var50"] / ref["n"])
                tsig.append(np.mean(np.abs(d / se) > 3))
        nr = f"{sum(r['n'] for r in rows)}/{sum(r['n_all'] for r in rows)}"
        print(f"   {k:4d} {s:>6s} {nr:>7s} | {med([np.median(x) for x in S50]):5.2f} "
              f"{med(S50.max(1)):5.2f} {med([np.median(r['S16']) for r in rows]):5.2f} "
              f"{med([np.median(r['S84']) for r in rows]):5.2f} "
              f"{med([1.25 / np.sqrt(r['ess']) for r in rows]):5.3f} | "
              f"{np.mean(S50 <= S_OK):5.2f} {np.mean(S50 > S_WARN):5.2f} | "
              f"{med([r['R'] for r in rows]):5.1f} | "
              f"{med(width) if width else np.nan:5.2f} {med(bias) if bias else np.nan:6.2f} "
              f"{np.mean(tsig) if tsig else np.nan:5.2f}")
    print()

    # ---------------------------------------------------------------- B. per parameter
    ks_std = sorted(k for k, L, s in settings if s == "std" and L == 1000)
    cur = (130, 1000, "ps1e6")
    print("== B. Per parameter: S50 at the current default (k=130, 1e6 prior start), "
          "per galaxy, and its scaling with k (standard start)")
    print("   slope: d log S50 / d log k fitted over k = " + "/".join(map(str, ks_std)) +
          " (median over galaxies); k for 0.14: extrapolated k where the median S50 reaches 0.14")
    print("   rho: posterior/prior half-width (median over galaxies; ~1 = prior-dominated)")
    print("   " + f"{'parameter':26s}" + "".join(f"{g:>8s}" for g in single) +
          f" | {'median':>6s} {'S16':>5s} {'S84':>5s} | {'slope':>5s} {'k@0.14':>6s} | {'rho':>5s}")
    par_rows = []
    for i, n in enumerate(names):
        vals = [st[(g,) + cur]["S50"][i] if (g,) + cur in st else np.nan for g in single]
        s16 = np.nanmedian([st[(g,) + cur]["S16"][i] for g in single if (g,) + cur in st])
        s84 = np.nanmedian([st[(g,) + cur]["S84"][i] for g in single if (g,) + cur in st])
        slopes, curves = [], []
        for g in single:
            y = [st[(g, k, 1000, "std")]["S50"][i] for k in ks_std if (g, k, 1000, "std") in st]
            if len(y) == len(ks_std):
                slopes.append(np.polyfit(np.log(ks_std), np.log(y), 1)[0])
                curves.append(y)
        slope = float(np.median(slopes))
        ymed = np.median(curves, axis=0)
        b, c = np.polyfit(np.log(ks_std), np.log(ymed), 1)
        k14 = float(np.exp((np.log(S_OK) - c) / b)) if b < 0 else np.inf
        rr = np.median([rho[g][i] for g in single]) if rho else np.nan
        par_rows.append((n, np.nanmedian(vals), rr))
        flag = lambda v: "*" if v > S_WARN else ("+" if v > S_OK else " ")
        print("   " + f"{n:26s}" + "".join(f"{v:7.2f}{flag(v)}" for v in vals) +
              f" | {np.nanmedian(vals):6.2f} {s16:5.2f} {s84:5.2f} | {slope:5.2f} "
              f"{min(k14, 99999):6.0f} | {rr:5.2f}")
    print("   (+ above 0.14 = R-hat 1.01; * above 0.28 = warning)\n")

    # ---------------------------------------------------------------- C. per galaxy
    print("== C. Per galaxy at the current default (k=130, 1e6 prior start)")
    print(f"   {'galaxy':>8s} {'z':>5s} {'bands':>5s} {'chi2':>5s} {'iter':>5s} {'runs':>5s} | "
          f"{'S50 med':>7s} {'worst':>5s} {'(parameter)':24s} | {'R':>4s} {'lzstd':>5s} "
          f"{'quoted':>6s} | {'ESS':>5s} {'ESSeff':>6s}")
    for g in gals:
        if (g,) + cur not in st:
            continue
        r = st[(g,) + cur]
        i = int(np.argmax(r["S50"]))
        esse = np.median((1.25 / np.maximum(r["S50"], 1e-6)) ** 2)
        tag = "  (bimodal)" if g in BIMODAL else ""
        print(f"   {g:>8s} {zref[g]:5.2f} {r['nb']:5d} {r['chi2']:5.1f} {r['iters']:5.0f} "
              f"{r['n']:2d}/{r['n_all']:2d} | {np.median(r['S50']):7.2f} {r['S50'][i]:5.2f} "
              f"{names[i]:24s} | {r['R']:4.1f} {r['logz_std']:5.2f} {r['logz_err']:6.2f} | "
              f"{r['ess']:5.0f} {esse:6.0f}{tag}")
    print()

    # ---------------------------------------------------------------- D. what predicts S
    if rho:
        x, y = [], []
        for g in single:
            if (g,) + cur in st:
                x += list(rho[g])
                y += list(st[(g,) + cur]["S50"])
        r_s, p = spearmanr(x, y)
        print("== D. Does scatter track how prior-dominated a parameter is? "
              f"Spearman(rho, S50) over (galaxy, parameter) at k=130 ps1e6: {r_s:+.2f} (p={p:.1g})")
        x, y = np.array(x), np.array(y)
        for lo, hi in [(0, 0.3), (0.3, 0.7), (0.7, 10)]:
            m = (x >= lo) & (x < hi)
            if m.any():
                print(f"   rho {lo:.1f}-{hi:.1f}: {m.sum():3d} (galaxy, parameter) pairs, "
                      f"median S50 {np.median(y[m]):.2f}, share > 0.14: {np.mean(y[m] > S_OK):.2f}")
        print()

    # ---------------------------------------------------------------- E. solution weights
    print("== E. Weight outside the main redshift solution, ALL runs (none dropped)")
    print("   off = posterior weight in 0.5-wide z bins that miss zref +- 0.3(1+zref), per run;")
    print("   shown as mean, std over seeds, min-max. std is what a seed changes; compare it with")
    print("   the mean to see whether the secondary solution's weight is reproducible.")
    for g in gals:
        lo, hi = zref[g] - 0.3 * (1 + zref[g]), zref[g] + 0.3 * (1 + zref[g])
        print(f"   {g} (zref {zref[g]:.2f})")
        for k, L, s in settings:
            if (g, k, L, s) not in groups:
                continue
            off = []
            for r in groups[(g, k, L, s)]:
                wb = r["z_weight_by_bin"]
                on = sum(v for b, v in wb.items()
                         if float(b.split("-")[1]) > lo and float(b.split("-")[0]) < hi)
                off.append(max(0.0, 1.0 - on))
            off = np.array(off)
            print(f"     k={k:3d} {s:>5s}: off {off.mean():.4f}  std {off.std(ddof=1):.4f}  "
                  f"range {off.min():.4f}-{off.max():.4f}  runs >1%: {np.sum(off > 0.01):2d}/{off.size}")


if __name__ == "__main__":
    main()
