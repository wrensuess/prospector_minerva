"""Find galaxies with more than one redshift solution in a production Prospector catalogue,
from the 7 redshift quantiles in its quant_*.h5 file (levels 0.1, 2.3, 15.9, 50, 84.1, 97.7,
99.9%: postprocess_parrot.percents).

A second solution shows up as a large gap between two neighbouring quantiles, with the
quantiles holding the median clumped on one side. The gap's position brackets the second
solution's weight:
  gap at 0.1-2.3% or 97.7-99.9%     ->  weight 0.1-2.3%    ("weak")
  gap at 2.3-15.9% or 84.1-97.7%    ->  weight 2.3-15.9%   ("moderate")
  gap at 15.9-50% or 50-84.1%       ->  weight 15.9-50%    ("strong")
Rules (dz in units of 1 + z50):
  split   the largest gap > GAP (0.3: the same +-0.3(1+z) window as the NSS analyses), and
          the quantiles on the median's side span < half that gap (one clump, not a smear)
  broad   not split, but q2.3-q97.7 spans > BROAD (1.0): poorly constrained, not two solutions
  single  otherwise
The quantiles come from one production dynesty run per galaxy, whose solution weights are
themselves noisy, and 7 quantiles only bracket a weight. Use this to choose test galaxies
and to estimate how common second solutions are, not as a final answer.

  python select_bimodal.py --quant quant_*.h5 --sps *_SPScatalog_*.fits --phot *SUPER_CATALOG*.fits \
      --out <private dir>/bimodal_cosmos.csv
"""
import argparse

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table

LEVELS = np.array([0.1, 2.3, 15.9, 50, 84.1, 97.7, 99.9])
GAP, BROAD = 0.3, 1.0
CLASS_OF_GAP = {0: "weak", 5: "weak", 1: "moderate", 4: "moderate", 2: "strong", 3: "strong"}


def classify(q):
    """q: (n, 7) redshift quantiles -> class, gap index, secondary-side, normalized gap."""
    n = len(q)
    scale = 1 + q[:, 3]
    gaps = np.diff(q, axis=1) / scale[:, None]                      # (n, 6)
    i = np.argmax(gaps, axis=1)
    g = gaps[np.arange(n), i]
    # the median (index 3) is on the high side when the gap is below it (i < 3)
    lo_side = i < 3                                                   # secondary solution is low-z
    span_main = np.where(lo_side, q[:, 6] - q[np.arange(n), i + 1],
                         q[np.arange(n), i] - q[:, 0]) / scale
    # the main side's span over the quantiles it holds, excluding a farther tail on that side:
    # use only the quantiles between the gap and the median's +-2.3/97.7 neighbour
    core_hi = np.where(lo_side, np.maximum(q[:, 5], q[np.arange(n), i + 1]), q[np.arange(n), i])
    core_lo = np.where(lo_side, q[np.arange(n), i + 1], np.minimum(q[:, 1], q[np.arange(n), i]))
    span_core = (core_hi - core_lo) / scale
    split = (g > GAP) & (span_core < 0.5 * g)
    broad = ~split & ((q[:, 5] - q[:, 1]) / scale > BROAD)
    cls = np.where(split, np.vectorize(CLASS_OF_GAP.get)(i), np.where(broad, "broad", "single"))
    return cls, i, lo_side, g, span_main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True)
    ap.add_argument("--sps", required=True)
    ap.add_argument("--phot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with h5py.File(a.quant, "r") as f:
        oid = f["objid"][:]
        q = f["zred"][:].astype(float)
        zml = f["zred_ml"][:].astype(float)
    sps = Table(fits.getdata(a.sps, 1))
    ph = fits.getdata(a.phot, 1)
    pidx = {v: k for k, v in enumerate(ph["id"])}
    j = np.array([pidx[v] for v in oid])
    use = (ph["use_phot"][j] == 1) & (ph["flag_star"][j] == 0) & np.all(np.isfinite(q), axis=1)
    f444 = ph["f_f444w"][j].astype(float)
    e444 = ph["e_f444w"][j].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = 28.9 - 2.5 * np.log10(f444)                           # fluxes in 10 nJy
    zspec = ph["z_spec"][j].astype(float)

    oid, q, zml, mag, snr, zspec = oid[use], q[use], zml[use], mag[use], (f444 / e444)[use], zspec[use]
    cls, gi, lo_side, g, _ = classify(q)
    n = len(oid)
    print(f"{n} galaxies with use_phot = 1, flag_star = 0 and finite quantiles\n")

    order = ["single", "weak", "moderate", "strong", "broad"]
    label = {"single": "one solution", "weak": "2nd 0.1-2.3%", "moderate": "2nd 2.3-15.9%",
             "strong": "2nd 15.9-50%", "broad": "broad, unsplit"}
    print("== Share of galaxies by class, overall and by F444W magnitude")
    bins = [(-99, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 99)]
    print(f"   {'F444W':>9s} {'N':>7s} " + " ".join(f"{label[c]:>15s}" for c in order))
    for lo, hi in [(-99, 99)] + bins:
        m = (mag >= lo) & (mag < hi)
        name = "all" if lo == -99 and hi == 99 else (f"<{hi}" if lo == -99 else (f">{lo}" if hi == 99 else f"{lo}-{hi}"))
        print(f"   {name:>9s} {m.sum():7d} " + " ".join(f"{np.mean(cls[m] == c):15.3f}" for c in order))
    print("\n== Share by main-solution redshift (z50), F444W < 27")
    zb = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 20)]
    for lo, hi in zb:
        m = (q[:, 3] >= lo) & (q[:, 3] < hi) & (mag < 27)
        print(f"   z {lo:>2d}-{hi:<2d} {m.sum():7d} " + " ".join(f"{np.mean(cls[m] == c):15.3f}" for c in order))

    print("\n== Galaxies with a spectroscopic redshift and a second solution: where is z_spec?")
    has = np.isfinite(zspec) & (zspec > 0)
    for c in ["weak", "moderate", "strong"]:
        m = has & (cls == c)
        if not m.any():
            continue
        near_main = np.abs(zspec[m] - q[m, 3]) < 0.3 * (1 + q[m, 3])
        print(f"   {label[c]:>15s}: {m.sum():5d} with z_spec; in the main solution "
              f"{near_main.mean():.2f}, elsewhere {1 - near_main.mean():.2f}")

    t = Table()
    t["id"] = oid
    t["class"] = cls
    t["f444w_mag"] = np.round(mag, 2)
    t["snr_f444w"] = np.round(snr, 1)
    t["z_spec"] = zspec
    t["z_ml"] = np.round(zml, 3)
    for k, lv in enumerate(LEVELS):
        t[f"z_{lv:g}"] = np.round(q[:, k], 3)
    t["gap_at"] = [f"{LEVELS[i]:g}-{LEVELS[i + 1]:g}" for i in gi]
    t["second_is_low_z"] = lo_side
    t["gap_dz"] = np.round(g, 3)
    t.write(a.out, overwrite=True)
    print(f"\nwrote {a.out} ({n} rows)")


if __name__ == "__main__":
    main()
