"""Per-seed posteriors of every parameter, mass-marginal NSS at several inner-step settings,
from 20-seed Blanca layout runs (draws.npz from blanca/posterior_draws.py).

Output (multi-page PDF):
  page 1    summary over galaxies: each run's median offset from the pooled median, and each
            run's 16-84% width relative to the pooled width, per parameter and setting
  pages 2+  one page per galaxy, every panel split into one strip per setting:
            p(z) vs z per seed, with each redshift solution's weight (mean +- std over seeds);
            the SFH (median SFR vs lookback time per seed) within each solution holding >= 2%
            of the pooled draws; and each seed's normalized posterior histogram for every
            other parameter. Dashed / grey = all runs of all settings pooled. R-hat is
            rank-normalized over seeds (red if >= 1.01).

  python plot_k_compare.py <runs root> <out.pdf> [k ...]      (default k: 32 52 78)

Each page is also written as a PNG next to the PDF (<out>_summary.png, <out>_<id>.png).
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_rhat import find_solutions, rhat_rank
sys.path.insert(0, os.environ.get("EFSPS_PARENT", "/Users/wren/Projects/MINERVA/efsps-photfit"))
from efsps_photfit.sfh import sfh_quantities  # noqa: E402

PAT = re.compile(r"blanca_layout_\d+_(\d+)_k(\d+)_L\d+_[a-z]+(?:_ps[0-9e.]+)?_mm$")
LOG10 = {"fagn", "agn_tau", "duste_gamma"}          # shown as log10 (positive, span decades)
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]


def load(root, ks):
    runs = {}
    for d in glob.glob(os.path.join(root, "**", "blanca_layout_*_mm"), recursive=True):
        m = PAT.search(os.path.basename(d))
        f = os.path.join(d, "draws.npz")
        if not m or int(m.group(2)) not in ks or not os.path.exists(f):
            continue
        z = np.load(f)
        key = (m.group(1), int(m.group(2)))
        D, s = z["draws"].astype(float), z["seeds"]
        names = [str(n) for n in z["names"]]
        if key in runs:                              # merge reruns of other seeds
            D0, s0, _ = runs[key]
            D, s = np.concatenate([D0, D]), np.concatenate([s0, s])
        runs[key] = (D, s, names)
    return runs


def transform(D, names):
    D = D.copy()
    for p, n in enumerate(names):
        if n in LOG10:
            D[..., p] = np.log10(np.clip(D[..., p], 1e-12, None))
    return D


def label(n):
    return f"log10 {n}" if n in LOG10 else n


TGRID = np.logspace(6, np.log10(1.4e10), 160)      # lookback time grid [yr]


def sfr_on_grid(z, logmass, ratios):
    """SFR(lookback time) on TGRID for each draw (constant within each age bin, 0 before the
    Big Bang), using the efsps age bins and Prospector's ratio -> mass convention."""
    q = sfh_quantities(z, logmass, ratios)
    t0, t1 = q["t0"], q["t1"]
    sfr = q["mass_bins"] / (t1 - t0)
    inbin = (TGRID[None, :, None] >= t0[:, None, :]) & (TGRID[None, :, None] < t1[:, None, :])
    return (inbin * sfr[:, None, :]).sum(-1)


def hist_strips(fig, cell, ks, Ds, p, edges, href, rhat, title, logy=False):
    nk = len(ks)
    inner = cell.subgridspec(nk, 1, hspace=0)
    axs, top = [], 0
    for j, k in enumerate(ks):
        ax = fig.add_subplot(inner[j], sharex=axs[0] if axs else None)
        axs.append(ax)
        D = Ds[k][:, :, p]
        for i in range(D.shape[0]):
            h, _ = np.histogram(D[i], edges, density=True)
            ax.stairs(h, edges, color=COLORS[j], lw=0.6, alpha=0.35)
            top = max(top, h.max())
        ax.stairs(href, edges, color="k", lw=0.9, ls="--")
        r = rhat[k][p]
        ax.text(0.99, 0.93, f"k={k}  R̂ {r:.3f}", transform=ax.transAxes, ha="right",
                va="top", fontsize=7, color="red" if r >= 1.01 else "k")
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        if j < nk - 1:
            ax.tick_params(labelbottom=False)
    for ax in axs:
        ax.set_ylim(0, top * 1.12)
    axs[0].set_title(title, fontsize=9, loc="left", pad=2)
    axs[-1].set_xlim(edges[0], edges[-1])
    return axs


def galaxy_page(pdf, out, gal, ks, runs, per, names):
    """Row 1: p(z) (two columns) and the SFH of each redshift solution; then a histogram
    panel for every other parameter (the SFR ratios are shown as SFHs instead)."""
    P = len(names)
    iz, im = names.index("zred"), names.index("logmass")
    ir = [names.index(f"logsfr_ratios_{i}") for i in range(1, 10)]
    raw = {k: runs[(gal, k)][0] for k in ks}
    Ds = {k: transform(raw[k], names) for k in ks}
    rhat = {k: per[(gal, k)]["rhat"] for k in ks}
    pool = np.concatenate([Ds[k].reshape(-1, P) for k in ks])
    nk = len(ks)

    # redshift solutions from all runs pooled; weights per seed
    zpool = np.concatenate([raw[k][:, :, iz] for k in ks])
    zedges, zmeds = find_solutions(zpool)
    wpool = np.array([np.mean((zpool >= lo) & (zpool < hi)) for lo, hi in zip(zedges[:-1], zedges[1:])])
    W = {k: np.stack([((raw[k][:, :, iz] >= lo) & (raw[k][:, :, iz] < hi)).mean(1)
                      for lo, hi in zip(zedges[:-1], zedges[1:])], axis=1) for k in ks}
    show = [s for s in np.argsort(-wpool) if wpool[s] >= 0.02][:3]
    show = sorted(show, key=lambda s: zmeds[s])

    others = [p for p in range(P) if p != iz and p not in ir]
    ncol = 5
    nrow = 1 + int(np.ceil((len(others) + 1) / ncol))
    fig = plt.figure(figsize=(18, 3.4 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.34, wspace=0.18,
                             left=0.03, right=0.99, top=0.955, bottom=0.025)

    # ---- p(z) vs z, linear, like the EAZY plots
    zmax = zpool.max() + 0.3
    xe = np.arange(np.log10(1.0), np.log10(1 + zmax) + 0.002, 0.002)  # bins ~0.5% in (1+z)
    ze = 10 ** xe - 1
    zc = 0.5 * (ze[1:] + ze[:-1])
    dz = np.diff(ze)
    inner = outer[0, 0:2].subgridspec(nk, 1, hspace=0)
    href = np.histogram(zpool, ze)[0] / (zpool.size * dz)
    axs, top = [], 0
    for j, k in enumerate(ks):
        ax = fig.add_subplot(inner[j], sharex=axs[0] if axs else None)
        axs.append(ax)
        Z = raw[k][:, :, iz]
        for i in range(Z.shape[0]):
            h = np.histogram(Z[i], ze)[0] / (Z.shape[1] * dz)
            ax.plot(zc, h, color=COLORS[j], lw=0.7, alpha=0.4)
            top = max(top, h.max())
        ax.plot(zc, href, color="k", lw=0.9, ls="--")
        for e in zedges[1:-1]:
            ax.axvline(e, color="0.5", lw=0.6, ls=":")
        txt = "   ".join(f"z≈{zmeds[s]:.2f}: {W[k][:, s].mean():.3f}±{W[k][:, s].std(ddof=1):.3f}"
                         for s in range(len(zmeds)))
        r = rhat[k][iz]
        ax.text(0.99, 0.93, f"k={k}  R̂ {r:.3f}\nweights {txt}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="red" if r >= 1.01 else "k",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        if j < nk - 1:
            ax.tick_params(labelbottom=False)
    for ax in axs:
        ax.set_ylim(0, top * 1.15)
    axs[-1].set_xlim(0, zmax)
    axs[-1].set_xlabel("z", fontsize=8, labelpad=1)
    axs[0].set_title("p(z)   (dotted: edges between solutions; weights = mean ± std over seeds)",
                     fontsize=9, loc="left", pad=2)

    # ---- SFH of each redshift solution
    rng = np.random.default_rng(1)
    for c, s in enumerate(show):
        lo, hi = zedges[s], zedges[s + 1]
        med, pooled = {}, []
        for k in ks:
            med[k] = []
            for i in range(raw[k].shape[0]):
                d = raw[k][i]
                d = d[(d[:, iz] >= lo) & (d[:, iz] < hi)]
                if len(d) < 100:
                    continue
                sfr = sfr_on_grid(d[:, iz], d[:, im], d[:, ir])
                with np.errstate(divide="ignore"):
                    med[k].append(np.log10(np.median(sfr, axis=0)))
                pooled.append(sfr[rng.choice(len(sfr), min(300, len(sfr)), replace=False)])
        pooled = np.concatenate(pooled)
        with np.errstate(divide="ignore"):
            p16, p50, p84 = np.log10(np.percentile(pooled, [16, 50, 84], axis=0))
        tmax = TGRID[np.isfinite(p50)].max()
        fin = np.isfinite(p16) & np.isfinite(p84)
        ylo, yhi = np.nanmin(p16[fin]) - 0.3, np.nanmax(p84[fin]) + 0.3
        inner = outer[0, 2 + c].subgridspec(nk, 1, hspace=0)
        axs = []
        for j, k in enumerate(ks):
            ax = fig.add_subplot(inner[j], sharex=axs[0] if axs else None)
            axs.append(ax)
            ax.fill_between(TGRID / 1e9, p16, p84, color="0.85", step="post", lw=0)
            ax.plot(TGRID / 1e9, p50, color="k", lw=0.9, ls="--", drawstyle="steps-post")
            for m in med[k]:
                ax.plot(TGRID / 1e9, m, color=COLORS[j], lw=0.7, alpha=0.45, drawstyle="steps-post")
            ax.text(0.03, 0.08, f"k={k}: {len(med[k])} seeds", transform=ax.transAxes, fontsize=7)
            ax.set_xscale("log")
            ax.set_ylim(ylo, yhi)
            ax.tick_params(labelsize=7)
            if j < nk - 1:
                ax.tick_params(labelbottom=False)
        axs[-1].set_xlim(1e-3, tmax / 1e9 * 1.02)
        axs[-1].set_xlabel("lookback time [Gyr]", fontsize=8, labelpad=1)
        axs[1].set_ylabel("log SFR [M☉/yr]", fontsize=8)
        axs[0].set_title(f"SFH, solution z≈{zmeds[s]:.2f} (weight {wpool[s]:.2f})", fontsize=9,
                         loc="left", pad=2)

    # ---- every other parameter
    for n, p in enumerate(others):
        v = pool[:, p]
        lo, hi = np.percentile(v, [0.1, 99.9])
        pad = 0.03 * (hi - lo + 1e-12)
        edges = np.linspace(lo - pad, hi + pad, 50)
        href = np.histogram(v, edges, density=True)[0]
        hist_strips(fig, outer[1 + n // ncol, n % ncol], ks, Ds, p, edges, href, rhat, label(names[p]))

    n = len(others)
    leg = fig.add_subplot(outer[1 + n // ncol, n % ncol])
    leg.axis("off")
    for j, k in enumerate(ks):
        leg.plot([], [], color=COLORS[j], lw=1, label=f"k={k}: one line per seed ({raw[k].shape[0]} seeds)")
    leg.plot([], [], color="k", ls="--", lw=0.9, label="all runs of all settings pooled")
    leg.legend(loc="upper left", fontsize=9, frameon=False)
    leg.text(0, 0.0, "Histograms: each seed's posterior, normalized.\n"
             "SFH: each seed's median SFR(t) within that\nredshift solution; grey = pooled 16–84%.\n"
             "R̂ (rank-normalized, all seeds) red if ≥ 1.01.", transform=leg.transAxes, fontsize=8.5)
    fig.suptitle(f"{gal}: posteriors from mass-marginal NSS at k = "
                 + ", ".join(str(k) for k in ks) + " inner steps", fontsize=12, y=0.99)
    pdf.savefig(fig)
    fig.savefig(out.replace(".pdf", f"_{gal}.png"), dpi=100)
    plt.close(fig)


def main():
    root, out = sys.argv[1], sys.argv[2]
    ks = [int(k) for k in sys.argv[3:]] or [32, 52, 78]
    runs = load(root, ks)
    gals = sorted({g for g, k in runs if all((g, kk) in runs for kk in ks)})
    names = next(iter(runs.values()))[2]
    P = len(names)

    per = {}                                         # (gal, k) -> dict of per-seed stats
    for gal in gals:
        Ds = {k: transform(runs[(gal, k)][0], names) for k in ks}
        pool = np.concatenate([Ds[k].reshape(-1, P) for k in ks])
        q16, q50, q84 = np.percentile(pool, [16, 50, 84], axis=0)
        half = np.maximum((q84 - q16) / 2, 1e-12)
        for k in ks:
            D = Ds[k]
            p16, p50, p84 = np.percentile(D, [16, 50, 84], axis=1)      # (seeds, P)
            r = np.array([rhat_rank([D[i, :, p] for i in range(D.shape[0])])[2] for p in range(P)])
            per[(gal, k)] = dict(p16=p16, p50=p50, p84=p84, rhat=r, seeds=runs[(gal, k)][1],
                                 off=(p50 - q50) / half, wid=(p84 - p16) / (q84 - q16 + 1e-12),
                                 pool=(q16, q50, q84))

    with PdfPages(out) as pdf:
        # summary page
        fig, axes = plt.subplots(1, 2, figsize=(15, 11), sharey=True)
        y = np.arange(P)
        dy = np.linspace(-0.25, 0.25, len(ks))
        for j, k in enumerate(ks):
            off = np.concatenate([per[(g, k)]["off"] for g in gals])       # (runs, P)
            wid = np.concatenate([per[(g, k)]["wid"] for g in gals])
            jit = np.random.default_rng(0).uniform(-0.06, 0.06, off.shape)
            for ax, v in ((axes[0], off), (axes[1], wid)):
                ax.scatter(v.ravel(), (y[None, :] + dy[j] + jit).ravel(), s=3, alpha=0.25,
                           color=COLORS[j], label=f"k={k}" if ax is axes[0] else None)
                lo, hi = np.percentile(v, [2.5, 97.5], axis=0)
                ax.hlines(y + dy[j], lo, hi, color=COLORS[j], lw=2.5)
        axes[0].axvline(0, color="k", lw=0.8)
        axes[1].axvline(1, color="k", lw=0.8)
        for x in (-0.25, 0.25):
            axes[0].axvline(x, color="k", lw=0.5, ls=":")
        axes[0].set_xlim(-1.2, 1.2)
        axes[1].set_xlim(0.3, 1.7)
        axes[0].set_yticks(y, [label(n) for n in names])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("run median − pooled median  [pooled (p84 − p16)/2]")
        axes[1].set_xlabel("run (p84 − p16) / pooled (p84 − p16)")
        axes[0].legend(loc="lower left", markerscale=4)
        fig.suptitle(f"Mass-marginal NSS, {len(gals)} galaxies × 20 seeds: dots = single runs, "
                     "bars = central 95% of runs (pooled = all runs of all settings)", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(out.replace(".pdf", "_summary.png"), dpi=110)
        plt.close(fig)

        for gal in gals:
            galaxy_page(pdf, out, gal, ks, runs, per, names)
    print("wrote", out, "galaxies:", len(gals))


if __name__ == "__main__":
    main()
