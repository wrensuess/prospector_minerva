"""Compare prior-start NSS runs with standard-start runs, galaxy by galaxy, on the three
criteria agreed on 2026-11-09 (reported as numbers, not a strict pass/fail):

  1. found rate: runs whose max-likelihood point is at the best known redshift
     (|z - z_best| < 0.1 (1 + z_best)); for 1152437 the target was >= 19/20.
  2. no harm, for the other galaxies:
     a. found rate not lower than standard;
     b. mean logZ within 2 combined standard errors of standard;
     c. no shift of per-seed posterior medians that survives Bonferroni (Mann-Whitney U,
        two-sided, every free parameter).
  3. speed: wall time per fit, iterations, and prior-draw time. The Blanca jobs land on
     different CPU types, so the iteration ratio (CPU-independent) is reported next to
     the raw wall times, plus the measured ms per (iteration x inner step) of each run.

z_best per galaxy is the redshift of the highest log L found by ANY run given (standard or
prior start).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu


def load_runs(pattern):
    runs = []
    for d in sorted(glob.glob(pattern)):
        for f in sorted(glob.glob(os.path.join(d, "*_seed*.json"))):
            j = json.load(open(f))
            j["_dir"] = d
            runs.append(j)
    return runs


def job_cpus(out_globs):
    cpu = {}
    for g in out_globs:
        for f in glob.glob(g):
            m = re.search(r"_(\d{8})\.out$", f)
            if not m:
                continue
            for line in open(f, errors="replace"):
                if line.startswith("CPU:"):
                    cpu[m.group(1)] = line.split(":", 1)[1].split(";")[0].strip()
                    break
    return cpu


def job_of(run):
    m = re.search(r"blanca_layout_(\d+)_", run["_dir"])
    return m.group(1) if m else "?"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base", required=True, help="folder holding the run directories")
    p.add_argument("--standard", default="blanca_layout_*_{id}_k{k}_L1000_tabulated",
                   help="glob (relative to --base) of the standard runs")
    p.add_argument("--ps", default="results/blanca_layout_*_{id}_k{k}_L1000_tabulated_ps{n}")
    p.add_argument("--starts", nargs="+", default=["1e6", "1e7"])
    p.add_argument("--k", type=int, default=130)
    p.add_argument("--outs", nargs="+", default=["blanca_out/*.out", "blanca/*.out"])
    p.add_argument("--json", default=None)
    args = p.parse_args()

    os.chdir(args.base)
    cpus = job_cpus(args.outs)
    ids = sorted({re.search(r"blanca_layout_\d+_(\d+)_k", d).group(1)
                  for d in glob.glob(args.ps.format(id="*", k=args.k, n="*"))})
    all_rows, mw_all = [], []
    print(f"standard: {args.standard}; prior start: {args.ps}; inner steps {args.k}\n")
    for gid in ids:
        groups = {"standard": load_runs(args.standard.format(id=gid, k=args.k))}
        for n in args.starts:
            groups[f"ps{n}"] = load_runs(args.ps.format(id=gid, k=args.k, n=n))
        best = max((r for g in groups.values() for r in g), key=lambda r: r["max_logL"])
        zt = best["z_at_max_logL"]
        half = 0.1 * (1 + zt)
        params = list(best["percentiles"])
        print(f"=== {gid}: z_best {zt:.3f} (max log L {best['max_logL']:.1f}, "
              f"{'prior start' if 'prior_start' in best else 'standard'} run)")
        print(f"  {'start':9s} {'n':>3s} {'found':>6s} {'logZ mean':>10s} {'+-SE':>6s} "
              f"{'scatter':>7s} {'quoted':>6s} {'dZ/SEc':>7s} {'min p':>8s} {'max|dmed|/hw':>12s} "
              f"{'iters':>6s} {'it ratio':>8s} {'wall min':>8s} {'prior s':>7s} {'ms/it/k':>7s}")
        std = groups["standard"]
        for name, runs in groups.items():
            if not runs:
                print(f"  {name:9s} no runs")
                continue
            found = np.array([abs(r["z_at_max_logL"] - zt) < half for r in runs])
            lz = np.array([r["logZ"] for r in runs])
            lze = np.array([r["logZ_err"] for r in runs])
            se = lz.std(ddof=1) / np.sqrt(len(lz))
            its = np.array([r["iterations"] for r in runs])
            wall = np.array([r["wall_s"] for r in runs])
            prior_s = np.array([r.get("prior_start", {}).get("eval_s", 0.0) for r in runs])
            ms = (wall - prior_s) / (its * args.k) * 1e3
            row = dict(id=gid, z_best=zt, start=name, n=len(runs), found=int(found.sum()),
                       logZ_mean=lz.mean(), logZ_se=se, logZ_scatter=lz.std(ddof=1),
                       logZ_quoted=lze.mean(), iters_median=float(np.median(its)),
                       wall_median_min=float(np.median(wall) / 60),
                       prior_s_median=float(np.median(prior_s)),
                       ms_per_it_step_median=float(np.median(ms)),
                       cpus=sorted({cpus.get(job_of(r), "?") for r in runs}))
            dz = minp = maxshift = itr = None
            if name != "standard" and std:
                lzs = np.array([r["logZ"] for r in std])
                ses = lzs.std(ddof=1) / np.sqrt(len(lzs))
                dz = (lz.mean() - lzs.mean()) / np.hypot(se, ses)
                ps_, shifts = [], []
                for q in params:
                    a = np.array([r["percentiles"][q][1] for r in std])
                    b = np.array([r["percentiles"][q][1] for r in runs])
                    hw = np.median([(r["percentiles"][q][2] - r["percentiles"][q][0]) / 2
                                    for r in std + runs])
                    pv = mannwhitneyu(a, b, alternative="two-sided").pvalue
                    ps_.append(pv)
                    shifts.append(abs(np.median(b) - np.median(a)) / hw if hw > 0 else 0.0)
                    mw_all.append((gid, name, q, pv))
                minp = float(np.min(ps_))
                maxshift = float(np.max(shifts))
                row["min_p_param"] = params[int(np.argmin(ps_))]
                row["max_shift_param"] = params[int(np.argmax(shifts))]
                itr = float(np.median(its) / np.median([r["iterations"] for r in std]))
                row.update(dlogZ_over_SEc=dz, min_p=minp, max_median_shift_over_hw=maxshift,
                           iters_ratio=itr)
            all_rows.append(row)
            f = lambda v, fmt: (fmt.format(v) if v is not None else "-")
            print(f"  {name:9s} {len(runs):3d} {found.sum():3d}/{len(runs):<2d} "
                  f"{lz.mean():10.2f} {se:6.2f} {lz.std(ddof=1):7.2f} {lze.mean():6.2f} "
                  f"{f(dz, '{:+7.2f}'):>7s} {f(minp, '{:8.4f}'):>8s} {f(maxshift, '{:12.2f}'):>12s} "
                  f"{np.median(its):6.0f} {f(itr, '{:8.2f}'):>8s} {np.median(wall) / 60:8.1f} "
                  f"{np.median(prior_s):7.1f} {np.median(ms):7.1f}")
        for r in all_rows:
            if r["id"] == gid and r["start"] != "standard" and "min_p_param" in r:
                print(f"    {r['start']}: smallest p in {r['min_p_param']}, largest median shift "
                      f"in {r['max_shift_param']}; CPUs {', '.join(r['cpus'])}")
        print(f"    standard CPUs: {', '.join(next(r['cpus'] for r in all_rows if r['id'] == gid and r['start'] == 'standard'))}")
        print()
    if mw_all:
        m = len(mw_all)
        pv = np.array([x[3] for x in mw_all])
        print(f"Mann-Whitney over all {m} (galaxy, start, parameter) tests: {np.sum(pv < 0.05)} "
              f"with p < 0.05 (expected {0.05 * m:.0f} by chance); smallest p {pv.min():.2g} "
              f"-> Bonferroni {min(1, pv.min() * m):.2g}")
        for gid, name, q, v in sorted(mw_all, key=lambda x: x[3])[:5]:
            print(f"   {gid} {name} {q}: p {v:.2g} (Bonferroni {min(1, v * m):.2g})")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(all_rows, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
