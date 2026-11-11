"""Standard NSS (26-d, logmass through its prior inverse CDF) against NSS with mass
integrated out (mass_marginal.py), from 20-seed Blanca layout runs.

Run directories: blanca_layout_<job>_<id>_k<k>_L<L>_<prior>[_ps<N>][_mm] anywhere under the
given roots; a directory's fit summaries (<id>_seed*.json) and its draws.npz
(blanca/posterior_draws.py) may sit under different roots, matched by directory name.
Directories with the same (galaxy, k, start, mass coordinate) are merged (reruns of seeds).
Job .out files (nss_fullfit_layout_<job>.out) give each job's CPU model.

Per galaxy and setting:
  logZ      mean +- std over seeds, R = std / mean quoted error, and the offset from the
            standard k=130 run in units of the combined standard error of the two means
  R-hat     rank-normalized (analyze_rhat.py), median and worst parameter, all fits kept,
            and inside the main redshift solution
  weights   number of redshift solutions, largest std of a solution weight across seeds
            (criterion <= 0.10)
  iters     median iterations; wall: median seconds per fit on the most common CPU type,
            and cost relative to the standard run on that same CPU type

  python compare_massmarg.py <root> [<root> ...]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_rhat import MIN_IN, find_solutions, rhat_rank

PAT = re.compile(r"blanca_layout_(\d+)_(\d+)_k(\d+)_L(\d+)_([a-z]+)(?:_ps([0-9e.]+))?(_mm)?$")


def cpu_of_jobs(roots):
    cpu = {}
    for r in roots:
        for f in glob.glob(os.path.join(r, "**", "nss_fullfit_layout_*.out"), recursive=True):
            job = re.search(r"_(\d+)\.out$", f).group(1)
            m = re.search(r"CPU:\s+(.+?);", open(f, errors="ignore").read())
            if m:
                c = m.group(1)
                for key, short in [("EPYC 9354", "EPYC9354"), ("5418Y", "Gold5418Y"),
                                   ("6130", "Gold6130"), ("6150", "Gold6150"), ("6230", "Gold6230"),
                                   ("E5-2680", "E5-2680v4"), ("8592", "Plat8592"), ("7713", "EPYC7713")]:
                    if key in c:
                        c = short
                        break
                cpu[job] = c
    return cpu


def collect(roots):
    groups = defaultdict(lambda: {"json": [], "draws": [], "jobs": []})
    for r in roots:
        for d in glob.glob(os.path.join(r, "**", "blanca_layout_*"), recursive=True):
            m = PAT.search(os.path.basename(d))
            if not m or not os.path.isdir(d):
                continue
            job, gal, k, L, prior, ps, mm = m.groups()
            key = (gal, int(k), f"ps{ps}" if ps else "std", "mm" if mm else "cdf")
            g = groups[key]
            js = sorted(glob.glob(os.path.join(d, f"{gal}_seed*.json")))
            if js:
                g["json"] += [(job, f) for f in js]
            if os.path.exists(os.path.join(d, "draws.npz")):
                g["draws"].append(os.path.join(d, "draws.npz"))
    return groups


def rhat_stats(files):
    D, names = [], None
    for f in files:
        z = np.load(f)
        D.append(z["draws"].astype(float))
        names = [str(n) for n in z["names"]]
    D = np.concatenate(D)                       # (fits, N, dim)
    M, N, P = D.shape
    r_all = np.array([rhat_rank([D[i, :, p] for i in range(M)])[2] for p in range(P)])
    Z = D[:, :, names.index("zred")]
    edges, meds = find_solutions(Z)
    lab = np.digitize(Z, edges[1:-1])
    W = np.stack([(lab == s).mean(1) for s in range(len(meds))], axis=1)
    main = int(np.argmax(W.mean(0)))
    keep = [i for i in range(M) if (lab[i] == main).sum() >= MIN_IN]
    r_main = np.array([rhat_rank([D[i, lab[i] == main, p] for i in keep])[2] for p in range(P)]) \
        if len(keep) >= 2 else np.full(P, np.nan)
    return dict(M=M, r_all=r_all, r_main=r_main, nsol=len(meds),
                wstd=float(W.std(axis=0, ddof=1).max()) if M > 1 else np.nan, names=names)


def main():
    roots = sys.argv[1:]
    groups = collect(roots)
    cpu = cpu_of_jobs(roots)
    gals = sorted({k[0] for k in groups if any(kk[0] == k[0] and kk[3] == "mm" for kk in groups)})
    ref_key = lambda g: (g, 130, "ps1e6", "cdf")
    rows = {}
    for key, g in groups.items():
        if key[0] not in gals or not g["json"]:
            continue
        S = [json.load(open(f)) for _, f in g["json"]]
        lz = np.array([s["logZ"] for s in S])
        r = dict(n=len(S), lz=lz, lze=np.mean([s["logZ_err"] for s in S]),
                 iters=np.median([s["iterations"] for s in S]),
                 wall=[(cpu.get(job, "?"), s["wall_s"]) for (job, _), s in zip(g["json"], S)])
        if g["draws"]:
            r.update(rhat_stats(g["draws"]))
        rows[key] = r

    mm_ks = sorted({k[1] for k in groups if k[3] == "mm"})
    order = [(130, "ps1e6", "cdf"), (260, "std", "cdf")] + [(k, "ps1e6", "mm") for k in mm_ks]
    lab = {(130, "ps1e6", "cdf"): "standard k=130", (260, "std", "cdf"): "standard k=260 (no ps)"}
    lab.update({(k, "ps1e6", "mm"): f"mass-marg k={k}" for k in mm_ks})
    summary = defaultdict(list)
    for gal in gals:
        ref = rows.get(ref_key(gal))
        print(f"== {gal}")
        print(f"   {'setting':24s} {'n':>3s} {'logZ mean +- std':>18s} {'R':>4s} {'dZ/SE':>6s} | "
              f"{'Rhat med':>8s} {'worst':>6s} | {'main med':>8s} {'worst':>6s} | {'#sol':>4s} "
              f"{'w std':>5s} | {'iters':>5s} {'wall s (CPU)':>18s} {'cost':>5s}")
        for o in order:
            key = (gal,) + o
            if key not in rows:
                continue
            r = rows[key]
            R = r["lz"].std(ddof=1) / r["lze"]
            if ref is not None and o != order[0]:
                se = np.sqrt(r["lz"].var(ddof=1) / r["n"] + ref["lz"].var(ddof=1) / ref["n"])
                dz = (r["lz"].mean() - ref["lz"].mean()) / se
            else:
                dz = np.nan
            # wall on the CPU type this setting and the reference share most
            cpus = [c for c, _ in r["wall"]]
            common = max(set(cpus), key=cpus.count)
            w = np.median([t for c, t in r["wall"] if c == common])
            cost = np.nan
            if ref is not None:
                rw = [t for c, t in ref["wall"] if c == common]
                if rw:
                    cost = w / np.median(rw)
            rh = f"{np.median(r['r_all']):8.3f} {r['r_all'].max():6.3f} | {np.nanmedian(r['r_main']):8.3f} " \
                 f"{np.nanmax(r['r_main']):6.3f} | {r['nsol']:4d} {r['wstd']:5.2f}" if "r_all" in r \
                else f"{'-':>8s} {'-':>6s} | {'-':>8s} {'-':>6s} | {'-':>4s} {'-':>5s}"
            print(f"   {lab[o]:24s} {r['n']:3d} {r['lz'].mean():9.2f} +- {r['lz'].std(ddof=1):5.2f} "
                  f"{R:4.1f} {dz:+6.1f} | {rh} | {r['iters']:5.0f} {w:7.0f} ({common:9s}) {cost:5.2f}")
            summary[o].append(dict(gal=gal, R=R, dz=dz, cost=cost, iters=r["iters"],
                                   rmed=np.median(r["r_all"]) if "r_all" in r else np.nan,
                                   rworst=r["r_all"].max() if "r_all" in r else np.nan,
                                   mmed=np.nanmedian(r["r_main"]) if "r_main" in r else np.nan,
                                   mworst=np.nanmax(r["r_main"]) if "r_main" in r else np.nan,
                                   wstd=r.get("wstd", np.nan), nsol=r.get("nsol", 0),
                                   pass_all=bool("r_all" in r and r["r_all"].max() < 1.01)))
        print()
    print("== Summary over galaxies (median; [worst galaxy])")
    print(f"   {'setting':24s} {'gal':>3s} | {'Rhat med':>14s} {'Rhat worst':>14s} | {'main worst':>14s} | "
          f"{'w std (multi-sol)':>18s} {'<=0.10':>6s} | {'R':>10s} {'|dZ/SE|':>10s} | {'iters':>5s} {'cost':>12s}")
    for o in order:
        v = summary.get(o)
        if not v:
            continue
        f = lambda k: np.array([x[k] for x in v], float)
        multi = [x for x in v if x["nsol"] > 1]
        ws = np.array([x["wstd"] for x in multi], float)
        print(f"   {lab[o]:24s} {len(v):3d} | {np.nanmedian(f('rmed')):6.3f} [{np.nanmax(f('rmed')):5.3f}] "
              f"{np.nanmedian(f('rworst')):6.3f} [{np.nanmax(f('rworst')):5.3f}] | "
              f"{np.nanmedian(f('mworst')):6.3f} [{np.nanmax(f('mworst')):5.3f}] | "
              f"{np.nanmedian(ws) if ws.size else np.nan:6.3f} [{np.nanmax(ws) if ws.size else np.nan:5.2f}] "
              f"({len(multi):2d}) {np.mean(ws <= 0.10) if ws.size else np.nan:6.2f} | "
              f"{np.nanmedian(f('R')):4.1f} [{np.nanmax(f('R')):4.1f}] "
              f"{np.nanmedian(np.abs(f('dz'))):4.1f} [{np.nanmax(np.abs(f('dz'))):4.1f}] | "
              f"{np.nanmedian(f('iters')):5.0f} {np.nanmedian(f('cost')):5.2f} [{np.nanmax(f('cost')):4.2f}]")


if __name__ == "__main__":
    main()
