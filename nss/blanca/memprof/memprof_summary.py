"""Summarise a memprof.sbatch run: per fit and per variant, peak memory (private = what the
Slurm cgroup charges per process; RSS = what /usr/bin/time reports), where the peak happens,
memory malloc holds free, and speed; plus bitwise checks that allocator settings change nothing.

usage: python memprof_summary.py OUTDIR
"""
import glob
import json
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore", "All-NaN")

out = sys.argv[1]
GB = 1e9
rows = []
for fn in sorted(glob.glob(os.path.join(out, "*.mem.json"))):
    m = json.load(open(fn))
    label = m["label"]                                   # <variant>_<id>_seed<s>
    variant, gid, seed = re.match(r"(.+)_(\d+)_seed(\d+)$", label).groups()
    fit = glob.glob(os.path.join(out, f"{gid}_{label}.json"))
    s = json.load(open(fit[0])) if fit else {}
    log = open(os.path.join(out, label + ".log"), errors="ignore").read()
    tm = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
    held = max([u.get("heap_free_held", 0) for _, _, u, *_ in m["marks"]] or [0])
    ps = s.get("prior_start", {}).get("eval_s", np.nan)
    it = s.get("iterations", np.nan)
    rows.append(dict(
        variant=variant, gid=gid, seed=int(seed), status=m["status"],
        peak_priv=m["peak_private"] / GB, peak_phase=m["peak_private_phase"],
        hwm=m["final"].get("VmHWM", 0) / GB, time_rss=int(tm.group(1)) / 1e6 if tm else np.nan,
        file=m["final"].get("RssFile", 0) / GB, held=held / GB,
        phases={k: v["private"] / GB for k, v in m["phase_peaks"].items()},
        wall=s.get("wall_s", np.nan), prior_s=ps, iters=it, compile_s=s.get("compile_s", np.nan),
        ms_per_it=1e3 * (s.get("wall_s", np.nan) - ps) / it if it == it else np.nan,
        logZ=s.get("logZ", np.nan)))

print(f"{'variant':12s} {'id':>8s} {'seed':>4s} {'peak priv':>9s} {'at phase':34s} {'HWM':>5s} "
      f"{'time -v':>7s} {'file':>5s} {'held':>5s} {'prior s':>7s} {'iters':>5s} {'ms/it':>6s} "
      f"{'logZ':>8s}")
for r in sorted(rows, key=lambda r: (r["gid"], r["seed"], r["variant"])):
    print(f"{r['variant']:12s} {r['gid']:>8s} {r['seed']:4d} {r['peak_priv']:9.3f} "
          f"{r['peak_phase'][:34]:34s} {r['hwm']:5.2f} {r['time_rss']:7.2f} {r['file']:5.2f} "
          f"{r['held']:5.2f} {r['prior_s']:7.1f} {r['iters']:5d} {r['ms_per_it']:6.0f} "
          f"{r['logZ']:8.2f}  {'' if r['status'] == 'ok' else r['status']}")

print("\nper variant (median over fits): peak private / HWM (GB), prior-start s, ms per iteration")
by = defaultdict(list)
for r in rows:
    by[r["variant"]].append(r)
for v, rs in sorted(by.items()):
    med = lambda k: np.nanmedian([r[k] for r in rs])
    print(f"  {v:12s} n={len(rs):2d}  private {med('peak_priv'):.3f}  HWM {med('hwm'):.3f}  "
          f"time -v {med('time_rss'):.3f}  file {med('file'):.3f}  held max {med('held'):.3f}  "
          f"prior {med('prior_s'):.1f} s  {med('ms_per_it'):.0f} ms/it")

print("\nprivate-memory peak by phase (GB, median over fits)")
phases = []
for r in rows:
    for k in r["phases"]:
        if k not in phases:
            phases.append(k)
print(f"  {'phase':38s} " + " ".join(f"{v:>12s}" for v in sorted(by)))
for ph in phases:
    vals = [np.nanmedian([r["phases"].get(ph, np.nan) for r in by[v]]) for v in sorted(by)]
    print(f"  {ph[:38]:38s} " + " ".join(f"{x:12.3f}" for x in vals))

# each variant against the reference (nofix; base in the first job), same galaxy and seed:
# bitwise identity of the fit outputs, and paired speed and memory
KEYS = ("u", "theta", "logl", "w", "logz_draws")
ref = "nofix" if "nofix" in by else "base"
idx = {(r["variant"], r["gid"], r["seed"]): r for r in rows}


def npz(v, gid, seed):
    return os.path.join(out, f"{gid}_{v}_{gid}_seed{seed}.npz")


print(f"\nagainst {ref}, same galaxy and seed: time per NSS iteration / prior start / compile ratios, peak memory, "
      f"fit outputs")
print(f"  {'variant':8s} {'id':>8s} {'seed':>4s} {'ms/it':>6s} {'prior':>6s} {'compile':>7s} "
      f"{'peak priv':>9s} {'(ref)':>6s} {'time -v':>7s} {'(ref)':>6s}  outputs")
summary = defaultdict(list)
for (v, gid, seed), r in sorted(idx.items()):
    if v == ref or (ref, gid, seed) not in idx:
        continue
    q = idx[(ref, gid, seed)]
    fa, fb = npz(ref, gid, seed), npz(v, gid, seed)
    if os.path.exists(fa) and os.path.exists(fb):
        A, B = np.load(fa), np.load(fb)
        if all(A[k].shape == B[k].shape and np.array_equal(A[k], B[k], equal_nan=True) for k in KEYS):
            verdict = "IDENTICAL"
        else:
            verdict = (f"different: iterations {q['iters']} vs {r['iters']}, "
                       f"logZ {r['logZ'] - q['logZ']:+.3f}")
    else:
        verdict = "(npz missing)"
    lr = r["ms_per_it"] / q["ms_per_it"]            # per iteration: iterations may differ
    pr, cr = r["prior_s"] / q["prior_s"], r["compile_s"] / q["compile_s"]
    summary[v].append((lr, pr, cr, r["peak_priv"], q["peak_priv"], r["time_rss"], q["time_rss"],
                       verdict == "IDENTICAL"))
    print(f"  {v:8s} {gid:>8s} {seed:4d} {lr:6.3f} {pr:6.3f} {cr:7.3f} {r['peak_priv']:9.3f} "
          f"{q['peak_priv']:6.3f} {r['time_rss']:7.3f} {q['time_rss']:6.3f}  {verdict}")
print(f"\n  median over galaxies and seeds (ratios to {ref}; memory in GB)")
for v, xs in sorted(summary.items()):
    x = np.array(xs, dtype=float)
    print(f"  {v:8s} n={len(x)}  ms/it x{np.median(x[:, 0]):.3f} (range {x[:, 0].min():.3f}-"
          f"{x[:, 0].max():.3f})  prior x{np.median(x[:, 1]):.3f}  compile x{np.median(x[:, 2]):.3f}  "
          f"peak private {np.median(x[:, 3]):.3f} vs {np.median(x[:, 4]):.3f}  time -v "
          f"{np.median(x[:, 5]):.3f} vs {np.median(x[:, 6]):.3f}  identical {int(x[:, 7].sum())}/{len(x)}")

# the first job's second pair (both = b5000 + glibc)
for a, b in [("b5000", "both")]:
    for fa in sorted(glob.glob(os.path.join(out, f"*_{a}_*_seed*.npz"))):
        fb = fa.replace(f"_{a}_", f"_{b}_")
        if os.path.exists(fb):
            A, B = np.load(fa), np.load(fb)
            same = all(np.array_equal(A[k], B[k], equal_nan=True) for k in KEYS)
            print(f"  {os.path.basename(fa)} vs {b}: {'IDENTICAL' if same else 'DIFFERENT'}")
