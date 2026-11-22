"""Frozen photo-z metrics for the standing spec-z benchmark (PRODUCTION_PLAN task Z1).

Generic: a table with a spectroscopic redshift column and any number of photo-z estimate
columns (NSS z50, eazy z_phot, production z_50, ...), plus optional per-row weight, magnitude
and flag columns. No survey paths or galaxy data live here.

Definitions (fixed; change them only with a version bump of METRICS_VERSION):
  dz_i      = (z_est_i - z_spec_i) / (1 + z_spec_i)
  missing   = estimate not finite or <= 0 (eazy writes -1 where it did not fit). A missing
              estimate is COUNTED AS AN OUTLIER in f_out and reported in n_missing; it is left
              out of NMAD and bias (it has no dz). Nothing is dropped silently.
  outlier   = missing, or |dz| > OUTLIER_CUT (0.15)
  f_out     = sum(w * outlier) / sum(w)                      (w = 1 when unweighted)
  f_out_err = binomial: sqrt(p (1 - p) / N) unweighted; weighted, the linearised
              (ratio-estimator) error sqrt(sum w^2 (o - p)^2) / sum w, which equals
              sqrt(p (1 - p) / N) for equal weights. With `strata` (one label per row; for a
              stratified sample weighted back to a parent pool) the error is computed within
              strata: sqrt(sum_h sum_(i in h) (u_i - mean_h u)^2) / sum w with u = w (o - p),
              which for constant weights per stratum is the textbook stratified error
              sqrt(sum_h W_h^2 p_h (1 - p_h) / n_h) / sum_h W_h
  nmad      = 1.48 * median(|dz - median(dz)|)  over non-missing rows, dz already divided by
              (1 + z_spec) (the eazy / Brammer+08 convention; what analyze_specz1k.py used).
              Weighted: weighted medians (lower weighted median, see wquantile).
  nmad_err  = standard deviation of nmad over NBOOT bootstrap resamples (fixed seed, so the
              numbers are reproducible)
  bias      = median(dz) over non-missing rows (weighted median when weighted)
  nmad_alt  = 1.48 * median(|(z_est - z_spec) - median(z_est - z_spec)| / (1 + z_spec)), the
              other reading of the formula, reported for reference only

Bins (left-closed, right-open): Z_BINS in z_spec, with 7-9, 9-11 and > 11 always separate;
MAG_BINS in the magnitude column (F444W AB).

Library use:
    from photoz_metrics import metrics, binned, load_table
    t = load_table("benchmark.csv")
    m = metrics(t["z_spec"], t["eazy_z"])                       # dict
    rows = binned(t, "z_spec", {"eazy": "eazy_z"}, by="z")      # list of dicts

Command line:
    python photoz_metrics.py TABLE --est NSS=z50 --est eazy=eazy_z [--zspec z_spec]
        [--mag mag444] [--weight w [--strata s]] [--where clean] [--where-not spec_suspect]
        [--csv out_prefix] [--plots out_prefix] [--title text]
TABLE is .csv or .fits. --where / --where-not keep rows whose column is true / false
(several allowed, combined with AND); the rows removed are counted in the printout.
"""
import argparse
import math
import sys

import numpy as np

METRICS_VERSION = "1.0 (2026-11-16)"
OUTLIER_CUT = 0.15
Z_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.5),
          (5.5, 7.0), (7.0, 9.0), (9.0, 11.0), (11.0, np.inf)]
MAG_BINS = [(-np.inf, 24.0), (24.0, 26.0), (26.0, 27.0), (27.0, 28.0), (28.0, np.inf)]
LINE_LOCI = (1.34, 0.75)   # (1+z_phot)/(1+z_spec) of [OIII]5007 <-> [OII]3727 confusion
NBOOT = 200
SEED = 12345


# ----------------------------------------------------------------------------- core
def wquantile(x, w, q):
    """Weighted quantile: smallest x whose cumulative weight fraction is >= q (the 'lower'
    weighted quantile). With equal weights and q = 0.5 this is NOT numpy's median for even N
    (numpy averages the two middle values), so for unweighted data we call np.median instead
    (see _median)."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    o = np.argsort(x, kind="mergesort")
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, q * c[-1], side="left")]


def _median(x, w=None):
    if len(x) == 0:
        return np.nan
    if w is None:
        return float(np.median(x))
    return float(wquantile(x, w, 0.5))


def _nmad(dz, w=None):
    if len(dz) == 0:
        return np.nan
    return 1.48 * _median(np.abs(dz - _median(dz, w)), w)


def metrics(z_spec, z_est, weight=None, nboot=NBOOT, seed=SEED, strata=None):
    """Photo-z statistics of one estimate. Returns a plain dict:
    N, n_missing, n_out (outliers incl. missing), f_out, f_out_err, f_out_valid (outlier
    fraction among non-missing only), nmad, nmad_err, nmad_alt, bias, sum_w, n_eff.
    Weighted when `weight` is given (rows with weight 0 still count in N)."""
    zs = np.asarray(z_spec, float)
    ze = np.asarray(z_est, float)
    if zs.shape != ze.shape:
        raise ValueError("z_spec and z_est differ in length")
    if np.any(~np.isfinite(zs)) or np.any(zs <= 0):
        raise ValueError("z_spec must be finite and > 0 on every row given")
    w = None if weight is None else np.asarray(weight, float)
    if w is not None and (w.shape != zs.shape or np.any(~np.isfinite(w)) or np.any(w < 0)):
        raise ValueError("weights must be finite, >= 0 and one per row")
    n = len(zs)
    out = dict(N=n, n_missing=0, n_out=0, f_out=np.nan, f_out_err=np.nan, f_out_valid=np.nan,
               nmad=np.nan, nmad_err=np.nan, nmad_alt=np.nan, bias=np.nan,
               sum_w=float(n if w is None else w.sum()), n_eff=float(n))
    if n == 0:
        return out
    valid = np.isfinite(ze) & (ze > 0)
    dz = np.where(valid, (ze - zs) / (1 + zs), np.nan)
    o = (~valid) | (np.abs(np.where(valid, dz, 0.0)) > OUTLIER_CUT)
    ww = np.ones(n) if w is None else w
    sw = ww.sum()
    out["n_missing"] = int((~valid).sum())
    out["n_out"] = int(o.sum())
    if sw > 0:
        p = float((ww * o).sum() / sw)
        out["f_out"] = p
        if w is None:
            out["f_out_err"] = math.sqrt(p * (1 - p) / n)
        else:
            u = ww * (o - p)
            if strata is None:
                ss = (u ** 2).sum()
            else:
                st = np.asarray(strata)
                ss = 0.0
                for h in np.unique(st):
                    uh = u[st == h]
                    ss += ((uh - uh.mean()) ** 2).sum()
            out["f_out_err"] = float(np.sqrt(ss) / sw)
            out["n_eff"] = float(sw ** 2 / (ww ** 2).sum())
    v = valid & (ww > 0)
    if v.any():
        out["f_out_valid"] = float((ww[v] * o[v]).sum() / ww[v].sum())
        wv = None if w is None else w[v]
        out["nmad"] = _nmad(dz[v], wv)
        out["bias"] = _median(dz[v], wv)
        raw = ze[v] - zs[v]
        out["nmad_alt"] = 1.48 * _median(np.abs(raw - _median(raw, wv)) / (1 + zs[v]), wv)
        if nboot and v.sum() > 1:
            rng = np.random.default_rng(seed)
            d, k = dz[v], v.sum()
            b = np.empty(nboot)
            for i in range(nboot):
                j = rng.integers(0, k, k)
                b[i] = _nmad(d[j], None if wv is None else wv[j])
            out["nmad_err"] = float(np.std(b))
    return out


def bin_masks(values, bins):
    """[(label, mask)] for left-closed bins; labels like '7-9', '<24', '>11'."""
    v = np.asarray(values, float)
    res = []
    for lo, hi in bins:
        if np.isinf(lo):
            lab = f"<{hi:g}"
        elif np.isinf(hi):
            lab = f">{lo:g}"
        else:
            lab = f"{lo:g}-{hi:g}"
        res.append((lab, (v >= lo) & (v < hi)))
    return res


def _col(table, name):
    return np.asarray(table[name])


def binned(table, zspec, estimates, by="z", mag=None, weight=None, bins=None, nboot=NBOOT,
           strata=None):
    """Rows (dicts) of metrics per bin and per estimate, plus an 'ALL' row per estimate.
    estimates: {label: column}. by = 'z' (Z_BINS on zspec) or 'mag' (MAG_BINS on `mag`).
    Rows with a missing magnitude go into an extra bin 'mag?' when by='mag'."""
    zs = _col(table, zspec).astype(float)
    w = None if weight is None else _col(table, weight).astype(float)
    st = None if strata is None else _col(table, strata)
    if by == "z":
        masks = bin_masks(zs, bins or Z_BINS)
    elif by == "mag":
        if mag is None:
            raise ValueError("by='mag' needs a magnitude column")
        m = _col(table, mag).astype(float)
        masks = bin_masks(m, bins or MAG_BINS)
        bad = ~np.isfinite(m)
        if bad.any():
            masks.append(("mag?", bad))
    else:
        raise ValueError("by must be 'z' or 'mag'")
    masks.append(("ALL", np.ones(len(zs), bool)))
    rows = []
    for lab, col in estimates.items():
        ze = _col(table, col).astype(float)
        for blab, mk in masks:
            r = metrics(zs[mk], ze[mk], None if w is None else w[mk], nboot=nboot,
                        strata=None if st is None else st[mk])
            rows.append(dict(estimate=lab, by=by, bin=blab, **r))
    return rows


# ----------------------------------------------------------------------------- I/O
def load_table(path):
    """CSV (pandas) or FITS (astropy) -> dict of numpy arrays (column -> array)."""
    if path.lower().endswith((".fits", ".fit", ".fits.gz")):
        from astropy.table import Table
        t = Table.read(path)
        return {c: np.asarray(t[c]) for c in t.colnames if t[c].ndim == 1}
    import pandas as pd
    d = pd.read_csv(path)
    return {c: d[c].to_numpy() for c in d.columns}


def _truth(a):
    a = np.asarray(a)
    if a.dtype.kind in "biuf":
        return np.nan_to_num(a.astype(float), nan=0.0) != 0
    return np.isin(np.char.lower(a.astype(str)), ["true", "1", "t", "yes", "y"])


def select(table, where=(), where_not=()):
    """Keep rows with every `where` column true and every `where_not` column false."""
    n = len(next(iter(table.values())))
    m = np.ones(n, bool)
    for c in where:
        m &= _truth(table[c])
    for c in where_not:
        m &= ~_truth(table[c])
    return {k: np.asarray(v)[m] for k, v in table.items()}, m


def format_rows(rows, title=None):
    """Fixed-width text table: one line per (bin, estimate)."""
    lines = []
    if title:
        lines.append(title)
    hdr = (f"{'by':3s} {'bin':>8s} {'estimate':>10s} {'N':>6s} {'n_eff':>7s} {'miss':>5s} "
           f"{'n_out':>6s} {'f_out%':>7s} {'+-':>5s} {'NMAD':>7s} {'+-':>6s} {'bias':>8s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    order = []
    for r in rows:
        if r["bin"] not in order:
            order.append(r["bin"])
    for b in order:
        for r in [x for x in rows if x["bin"] == b]:
            f = lambda v, s, k=1: ("nan" if not np.isfinite(v) else f"{k * v:{s}}")
            lines.append(f"{r['by']:3s} {b:>8s} {r['estimate']:>10s} {r['N']:6d} "
                         f"{r['n_eff']:7.1f} {r['n_missing']:5d} {r['n_out']:6d} "
                         f"{f(r['f_out'], '7.2f', 100):>7s} {f(r['f_out_err'], '5.2f', 100):>5s} "
                         f"{f(r['nmad'], '7.4f'):>7s} {f(r['nmad_err'], '6.4f'):>6s} "
                         f"{f(r['bias'], '+8.4f'):>8s}")
    return "\n".join(lines)


def write_csv(rows, path):
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


# ----------------------------------------------------------------------------- plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_zz(table, zspec, estimates, path, title=None, zmax=15.0):
    """z_est vs z_spec per estimate on log(1+z) axes: 1:1, the +-0.15(1+z) outlier lines and
    the x1.34 / x0.75 line-confusion loci. Missing estimates are drawn at the bottom edge
    (red triangles) and counted in the legend."""
    plt = _plt()
    zs = _col(table, zspec).astype(float)
    k = len(estimates)
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k, 4.8), squeeze=False)
    lo, hi = 0.0, np.log10(1 + zmax)
    g = np.linspace(lo, hi, 50)
    for ax, (lab, col) in zip(axes[0], estimates.items()):
        ze = _col(table, col).astype(float)
        ok = np.isfinite(ze) & (ze > 0)
        m = metrics(zs, ze, nboot=0)
        x = np.log10(1 + zs)
        y = np.log10(1 + np.where(ok, ze, np.nan))
        out = ok & (np.abs((ze - zs) / (1 + zs)) > OUTLIER_CUT)
        ax.scatter(x[ok & ~out], y[ok & ~out], s=4, c="k", alpha=0.5, lw=0, label="within 0.15")
        ax.scatter(x[out], y[out], s=6, c="C1", alpha=0.8, lw=0, label="outlier")
        if (~ok).any():
            ax.scatter(x[~ok], np.full((~ok).sum(), lo + 0.01), marker="v", s=12, c="r",
                       label=f"missing ({(~ok).sum()})")
        ax.plot(g, g, "b-", lw=0.8)
        for f in (1 + OUTLIER_CUT, 1 - OUTLIER_CUT):
            ax.plot(g, g + np.log10(f), "b--", lw=0.6)
        for f, ls in zip(LINE_LOCI, (":", "-.")):
            ax.plot(g, g + np.log10(f), color="g", ls=ls, lw=0.8,
                    label=f"(1+z) x{f:g}")
        ticks = [0, 0.5, 1, 2, 3, 5, 7, 9, 11, 15]
        ax.set_xticks(np.log10(1 + np.array(ticks)))
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticks(np.log10(1 + np.array(ticks)))
        ax.set_yticklabels([str(t) for t in ticks])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("z_spec")
        ax.set_ylabel(f"z ({lab})")
        ax.set_title(f"{lab}: N={m['N']}, f_out={100 * m['f_out']:.1f}%, "
                     f"NMAD={m['nmad']:.3f}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_binned(table, zspec, estimates, path, mag=None, weight=None, title=None):
    """Outlier fraction and NMAD against z bin (and magnitude bin if `mag`), all estimates
    on one panel each, with error bars (binomial / bootstrap)."""
    plt = _plt()
    groups = [("z", None)] + ([("mag", mag)] if mag else [])
    fig, axes = plt.subplots(2, len(groups), figsize=(6.5 * len(groups), 7), squeeze=False)
    k = len(estimates)
    for j, (by, mcol) in enumerate(groups):
        rows = binned(table, zspec, estimates, by=by, mag=mcol, weight=weight)
        bins = [r["bin"] for r in rows if r["estimate"] == list(estimates)[0]
                and r["bin"] not in ("ALL", "mag?")]
        x = np.arange(len(bins))
        for i, lab in enumerate(estimates):
            rr = {r["bin"]: r for r in rows if r["estimate"] == lab}
            dx = (i - (k - 1) / 2) * 0.12
            f = np.array([rr[b]["f_out"] for b in bins]) * 100
            fe = np.array([rr[b]["f_out_err"] for b in bins]) * 100
            nm = np.array([rr[b]["nmad"] for b in bins])
            ne = np.array([rr[b]["nmad_err"] for b in bins])
            axes[0, j].errorbar(x + dx, f, fe, fmt="o-", ms=4, capsize=2, label=lab)
            axes[1, j].errorbar(x + dx, nm, ne, fmt="o-", ms=4, capsize=2, label=lab)
        n = [rr[b]["N"] for b in bins]
        for a in axes[:, j]:
            a.set_xticks(x)
            a.set_xticklabels([f"{b}\n({nn})" for b, nn in zip(bins, n)], fontsize=7)
            a.grid(alpha=0.3)
        axes[0, j].set_ylabel("outlier fraction (%)")
        axes[1, j].set_ylabel("NMAD")
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("z_spec bin (N)" if by == "z" else "F444W bin (N)")
        axes[0, j].legend(fontsize=8)
    if title:
        fig.suptitle(title + (" (weighted)" if weight else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cdf(table, zspec, estimates, path, title=None):
    """Cumulative distribution of |dz|/(1+z_spec) per estimate; missing estimates count as
    |dz| = infinity (so the curves end below 1 by the missing fraction)."""
    plt = _plt()
    zs = _col(table, zspec).astype(float)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for lab, col in estimates.items():
        ze = _col(table, col).astype(float)
        ok = np.isfinite(ze) & (ze > 0)
        a = np.where(ok, np.abs(ze - zs) / (1 + zs), np.inf)
        s = np.sort(a)
        y = np.arange(1, len(s) + 1) / len(s)
        fin = np.isfinite(s)
        ax.step(np.clip(s[fin], 1e-4, None), y[fin], where="post", label=lab)
    ax.axvline(OUTLIER_CUT, color="k", ls="--", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("|dz|/(1+z_spec)")
    ax.set_ylabel("cumulative fraction")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("table")
    ap.add_argument("--zspec", default="z_spec")
    ap.add_argument("--est", action="append", required=True, help="LABEL=COLUMN (repeat)")
    ap.add_argument("--mag", default=None)
    ap.add_argument("--weight", default=None)
    ap.add_argument("--strata", default=None,
                    help="stratum label column for the weighted f_out error")
    ap.add_argument("--where", action="append", default=[])
    ap.add_argument("--where-not", action="append", default=[])
    ap.add_argument("--csv", default=None, help="write <prefix>_z.csv and <prefix>_mag.csv")
    ap.add_argument("--plots", default=None, help="write <prefix>_zz/_bins/_cdf.png")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    est = dict(e.split("=", 1) for e in a.est)
    t0 = load_table(a.table)
    t, m = select(t0, a.where, a.where_not)
    zs = np.asarray(t[a.zspec], float)
    bad = ~(np.isfinite(zs) & (zs > 0))
    if bad.any():
        print(f"# {bad.sum()} rows without a valid z_spec removed", file=sys.stderr)
        t = {k: v[~bad] for k, v in t.items()}
    title = a.title or a.table
    print(f"# photoz_metrics {METRICS_VERSION}: {a.table}; {m.sum()} of {len(m)} rows kept "
          f"(where={a.where}, where-not={a.where_not}); outlier |dz|/(1+z) > {OUTLIER_CUT}; "
          f"missing estimates (not finite or <= 0) count as outliers")
    for wt in ([None] + ([a.weight] if a.weight else [])):
        tag = "weighted" if wt else "raw"
        sk = a.strata if wt else None
        rz = binned(t, a.zspec, est, by="z", weight=wt, strata=sk)
        print(format_rows(rz, f"\n## by z_spec ({tag})"))
        rm = None
        if a.mag:
            rm = binned(t, a.zspec, est, by="mag", mag=a.mag, weight=wt, strata=sk)
            print(format_rows(rm, f"\n## by magnitude {a.mag} ({tag})"))
        if a.csv:
            write_csv(rz, f"{a.csv}_z_{tag}.csv")
            if rm:
                write_csv(rm, f"{a.csv}_mag_{tag}.csv")
    if a.plots:
        plot_zz(t, a.zspec, est, f"{a.plots}_zz.png", title)
        plot_binned(t, a.zspec, est, f"{a.plots}_bins.png", mag=a.mag, title=title)
        if a.weight:
            plot_binned(t, a.zspec, est, f"{a.plots}_bins_weighted.png", mag=a.mag,
                        weight=a.weight, title=title)
        plot_cdf(t, a.zspec, est, f"{a.plots}_cdf.png", title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
