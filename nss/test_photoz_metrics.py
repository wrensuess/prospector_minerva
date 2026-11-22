"""Synthetic checks of photoz_metrics.py with known answers (no survey data).

    python test_photoz_metrics.py        -> ends with RESULT: PASS or RESULT: FAIL
"""
import os
import sys
import tempfile

import numpy as np

import photoz_metrics as pm

FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILS.append(name)


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


# 1. perfect estimates
zs = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 6.0, 8.0, 10.0, 12.0])
m = pm.metrics(zs, zs.copy())
check("perfect: f_out 0, nmad 0, bias 0", m["f_out"] == 0 and m["nmad"] == 0 and m["bias"] == 0
      and m["n_missing"] == 0 and m["N"] == 9)

# 2. outlier cut: |dz| 0.149 is not an outlier, 0.151 is (both signs)
zs = np.array([1.0, 1.0, 1.0, 1.0])
ze = zs + 2 * np.array([0.149, 0.151, -0.149, -0.151])
m = pm.metrics(zs, ze, nboot=0)
check("cut: 0.149 not outlier, 0.151 outlier", m["n_out"] == 2 and close(m["f_out"], 0.5))

# 3. missing values (-1, 0, nan, inf) count as outliers, are reported, and are left out of NMAD
zs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
ze = np.array([1.0, 2.0, 3.0, -1.0, np.nan, 0.0])
m = pm.metrics(zs, ze, nboot=0)
check("missing counted", m["n_missing"] == 3 and m["n_out"] == 3 and close(m["f_out"], 0.5)
      and m["f_out_valid"] == 0 and m["nmad"] == 0, str(m))

# 4. NMAD of a known set: dz = (0, .01, .02, .03, .10) -> median .02, |dev| (.02,.01,0,.01,.08)
#    -> median .01 -> NMAD .0148; bias .02
dz = np.array([0.0, 0.01, 0.02, 0.03, 0.10])
zs = np.full(5, 2.0)
m = pm.metrics(zs, zs + dz * 3, nboot=0)
check("nmad known value", close(m["nmad"], 0.0148, 1e-12) and close(m["bias"], 0.02, 1e-12),
      f"{m['nmad']} {m['bias']}")

# 5. Gaussian: NMAD recovers sigma; f_out matches the normal tail; binomial error formula
rng = np.random.default_rng(1)
n = 200000
zs = rng.uniform(0.2, 10, n)
sig = 0.05
ze = zs + rng.normal(0, sig, n) * (1 + zs)
m = pm.metrics(zs, ze, nboot=0)
from math import erf, sqrt
p_true = 1 - erf(0.15 / sig / sqrt(2))
check("gaussian nmad ~ sigma", abs(m["nmad"] / sig - 1) < 0.015,
      f"{m['nmad']:.5f}")
check("gaussian f_out ~ 2-sided tail", abs(m["f_out"] - p_true) < 4 * sqrt(p_true / n),
      f"{m['f_out']:.5f} vs {p_true:.5f}")
check("binomial error", close(m["f_out_err"], sqrt(m["f_out"] * (1 - m["f_out"]) / n), 1e-15))

# 6. alternative NMAD differs only through the (1+z) placement
zs = np.array([0.5, 1.0, 4.0, 9.0])
ze = zs + np.array([0.03, -0.02, 0.2, 0.0])
m = pm.metrics(zs, ze, nboot=0)
raw = ze - zs
alt = 1.48 * np.median(np.abs(raw - np.median(raw)) / (1 + zs))
d = raw / (1 + zs)
std = 1.48 * np.median(np.abs(d - np.median(d)))
check("nmad vs nmad_alt formulas", close(m["nmad_alt"], alt) and close(m["nmad"], std)
      and not close(alt, std))

# 7. weights: integer weights = row duplication for f_out, bias, nmad (odd totals)
zs = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
ze = np.array([1.05, 2.9, 3.1, 5.0, 2.0])
w = np.array([3, 1, 2, 1, 2], float)          # total 9 (odd -> medians are unambiguous)
mw = pm.metrics(zs, ze, w, nboot=0)
md = pm.metrics(np.repeat(zs, w.astype(int)), np.repeat(ze, w.astype(int)), nboot=0)
check("weights = duplication (f_out, bias, nmad)",
      close(mw["f_out"], md["f_out"]) and close(mw["bias"], md["bias"])
      and close(mw["nmad"], md["nmad"]), f"{mw['f_out']} {md['f_out']}")
check("weighted n_eff", close(mw["n_eff"], w.sum() ** 2 / (w ** 2).sum()))
# equal weights reproduce the unweighted f_out and its error
mu = pm.metrics(zs, ze, nboot=0)
me = pm.metrics(zs, ze, np.full(5, 2.5), nboot=0)
check("equal weights = unweighted f_out and error", close(mu["f_out"], me["f_out"])
      and close(mu["f_out_err"], me["f_out_err"]))

# 8. stratified re-weighting gives the parent-pool outlier rate
#    parent: 1000 at low z with 2% outliers, 100 at high z with 30%; sample 100 + 100
lo = np.r_[np.full(2, 1.0), np.zeros(98)]
hi = np.r_[np.full(30, 1.0), np.zeros(70)]
zs = np.r_[np.full(100, 1.0), np.full(100, 8.0)]
ze = np.where(np.r_[lo, hi] > 0, zs * 3 + 1, zs)
w = np.r_[np.full(100, 10.0), np.full(100, 1.0)]
m = pm.metrics(zs, ze, w, nboot=0)
check("stratified weights -> parent rate", close(m["f_out"], (20 + 30) / 1100), f"{m['f_out']}")
var = (1000 / 1100) ** 2 * 0.02 * 0.98 / 100 + (100 / 1100) ** 2 * 0.3 * 0.7 / 100
ms = pm.metrics(zs, ze, w, nboot=0, strata=(zs > 5).astype(int))
check("weighted error with strata = textbook stratified error",
      close(ms["f_out_err"], sqrt(var), 1e-12), f"{ms['f_out_err']:.6f} vs {sqrt(var):.6f}")
check("weighted error without strata >= stratified, within 5%",
      sqrt(var) <= m["f_out_err"] < 1.05 * sqrt(var), f"{m['f_out_err']:.5f}")

# 9. bins: labels, edges (left-closed), >11 separate, ALL row
t = {"z_spec": np.array([0.2, 6.99, 7.0, 8.99, 9.0, 10.99, 11.0, 14.0]),
     "a": np.array([0.2, 6.99, 7.0, 8.99, 9.0, 10.99, 11.0, -1.0]),
     "mag": np.array([23.0, 24.0, 25.9, 26.0, 27.5, 28.0, 30.0, np.nan])}
rows = pm.binned(t, "z_spec", {"A": "a"}, by="z", nboot=0)
nb = {r["bin"]: r["N"] for r in rows}
check("z bins 7-9 / 9-11 / >11 separate, left-closed",
      nb["5.5-7"] == 1 and nb["7-9"] == 2 and nb["9-11"] == 2 and nb[">11"] == 2
      and nb["ALL"] == 8, str(nb))
r11 = [r for r in rows if r["bin"] == ">11"][0]
check(">11 bin: the -1 estimate is a missing outlier", r11["n_missing"] == 1 and r11["n_out"] == 1)
rows = pm.binned(t, "z_spec", {"A": "a"}, by="mag", mag="mag", nboot=0)
nb = {r["bin"]: r["N"] for r in rows}
check("mag bins incl. missing magnitude", nb == {"<24": 1, "24-26": 2, "26-27": 1, "27-28": 1,
                                                 ">28": 2, "mag?": 1, "ALL": 8}, str(nb))

# 10. bootstrap error is reproducible
zs = rng.uniform(0.2, 6, 500)
ze = zs + rng.normal(0, 0.03, 500) * (1 + zs)
a, b = pm.metrics(zs, ze), pm.metrics(zs, ze)
check("bootstrap nmad_err deterministic and plausible", a["nmad_err"] == b["nmad_err"]
      and 0.3 < a["nmad_err"] / (a["nmad"] / np.sqrt(500)) < 3, f"{a['nmad_err']:.5f}")

# 11. invalid inputs refused
for bad, args in (("z_spec <= 0", (np.array([0.0, 1.0]), np.array([1.0, 1.0]))),
                  ("negative weight", (np.array([1.0, 1.0]), np.array([1.0, 1.0]),
                                       np.array([1.0, -1.0])))):
    try:
        pm.metrics(*args)
        check(f"refuses {bad}", False)
    except ValueError:
        check(f"refuses {bad}", True)

# 12. CLI end to end on a temporary CSV (tables, CSV output, plots)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "t.csv")
    with open(p, "w") as fh:
        fh.write("z_spec,good,bad,mag,w,keep\n")
        for i in range(60):
            z = 0.3 + 0.2 * i
            fh.write(f"{z},{z * 1.01},{z * 2 if i % 3 == 0 else -1},{23 + i * 0.1},{1 + i % 2},"
                     f"{i % 5 != 0}\n")
    rc = pm.main([p, "--est", "G=good", "--est", "B=bad", "--mag", "mag", "--weight", "w",
                  "--where", "keep", "--csv", os.path.join(d, "o"), "--plots",
                  os.path.join(d, "o")])
    made = sorted(os.listdir(d))
    want = ["o_bins.png", "o_bins_weighted.png", "o_cdf.png", "o_mag_raw.csv",
            "o_mag_weighted.csv", "o_z_raw.csv", "o_z_weighted.csv", "o_zz.png"]
    check("CLI writes tables and plots", rc == 0 and all(w_ in made for w_ in want), str(made))
    import csv
    rows = list(csv.DictReader(open(os.path.join(d, "o_z_raw.csv"))))
    allb = [r for r in rows if r["bin"] == "ALL" and r["estimate"] == "B"][0]
    check("CLI --where keeps 48 rows; B all outliers", allb["N"] == "48"
          and float(allb["f_out"]) == 1.0, str(allb["N"]))

print(f"\n{len(FAILS)} failure(s)" + (f": {FAILS}" if FAILS else ""))
print("RESULT: " + ("PASS" if not FAILS else "FAIL"))
sys.exit(1 if FAILS else 0)
