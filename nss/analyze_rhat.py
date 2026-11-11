"""Rank-normalized Gelman-Rubin R-hat (Vehtari et al. 2021) from the 20-seed posterior draws,
with every run kept, including runs that weight redshift solutions differently.

Input: run directories holding draws.npz (blanca/posterior_draws.py: draws (fits, N, dim),
N equal-weight draws per fit resampled from its weighted dead points, names, logZ, ...).
Each fit plays the role of one chain. The draws within a fit are independent resamples, so
splitting chains in half (as for MCMC) adds nothing and is not done.

Per parameter:
  bulk    R-hat of the rank-normalized draws: all draws of all fits ranked together, ranks
          mapped to normal scores. Insensitive to skew, heavy tails and prior edges.
  folded  the same for |theta - pooled median|: do the fits agree on the width/tails?
  R-hat   max(bulk, folded). Accept < 1.01 (STATUS.md section 5); warn above 1.04
          (scatter twice the accepted level: sqrt(1 + 0.28^2)).

Redshift solutions: the main solution is zref +- 0.3 (1 + zref), zref = the pooled median
of the fits' redshift draws that lie in the densest solution. Per fit, the weight of the
main solution; across fits, its mean and scatter (this is where runs disagree on how much
the secondary solutions count). Then R-hat again inside the main solution only, from the
fits with at least MIN_IN draws there: does the main solution converge on its own?

  python analyze_rhat.py <dir> [<dir> ...]       (e.g. results/blanca_layout_*_k130_*)
"""
import os
import re
import sys

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

OK, WARN, MIN_IN = 1.01, 1.04, 200


def rhat_basic(x):
    """Classic R-hat for a list of 1-d chains (unequal lengths allowed: pooled-within form)."""
    n = np.array([len(c) for c in x], float)
    means = np.array([c.mean() for c in x])
    W = np.sum([(len(c) - 1) * c.var(ddof=1) for c in x]) / np.sum(n - 1)
    nbar = n.mean()
    B = nbar * means.var(ddof=1)
    var_plus = (nbar - 1) / nbar * W + B / nbar
    return float(np.sqrt(var_plus / W)) if W > 0 else np.nan


def rank_normal(x):
    """Rank all draws of all chains together and map to normal scores (Vehtari+2021 eq. 14)."""
    flat = np.concatenate(x)
    z = ndtri((rankdata(flat) - 0.375) / (flat.size + 0.25))
    out, i = [], 0
    for c in x:
        out.append(z[i:i + len(c)])
        i += len(c)
    return out


def rhat_rank(x):
    """Rank-normalized R-hat: max of bulk and folded."""
    bulk = rhat_basic(rank_normal(x))
    med = np.median(np.concatenate(x))
    folded = rhat_basic(rank_normal([np.abs(c - med) for c in x]))
    return bulk, folded, max(bulk, folded)


def main_solution(z):
    """zref: median of the pooled redshift draws in the densest +-0.3(1+z) window."""
    zs = np.sort(z.ravel())
    best, zref = -1, np.median(zs)
    for zc in np.quantile(zs, np.linspace(0.01, 0.99, 99)):
        n = np.sum(np.abs(zs - zc) < 0.3 * (1 + zc))
        if n > best:
            best, zref = n, zc
    sel = zs[np.abs(zs - zref) < 0.3 * (1 + zref)]
    return float(np.median(sel))


def find_solutions(z, min_weight=0.005, depth=0.2):
    """Redshift solutions in pooled draws: peaks of a smoothed density in log10(1+z),
    split at minima lower than `depth` x the smaller neighbouring peak. Returns the edges
    (in z) between solutions and each solution's pooled median; solutions holding less
    than `min_weight` of the pooled draws are merged into their neighbour."""
    x = np.log10(1 + z.ravel())
    bins = np.arange(x.min() - 0.02, x.max() + 0.025, 0.005)
    h, e = np.histogram(x, bins)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    s = np.convolve(h, k / k.sum(), mode="same")
    c = 0.5 * (e[1:] + e[:-1])
    peaks = [i for i in range(1, len(s) - 1) if s[i] >= s[i - 1] and s[i] > s[i + 1]]
    cuts = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        m = a + int(np.argmin(s[a:b + 1]))
        if s[m] < depth * min(s[a], s[b]):
            cuts.append(c[m])
    while True:
        edges = np.concatenate([[-np.inf], cuts, [np.inf]])
        w = np.array([np.mean((x >= lo) & (x < hi)) for lo, hi in zip(edges[:-1], edges[1:])])
        if len(cuts) == 0 or w.min() >= min_weight:
            break
        j = int(np.argmin(w))                        # merge the lightest into a neighbour
        cuts.pop(min(j, len(cuts) - 1) if j > 0 else 0)
    edges_z = 10 ** edges - 1
    meds = [float(np.median(z.ravel()[(x >= lo) & (x < hi)]))
            for lo, hi in zip(edges[:-1], edges[1:])]
    return edges_z, meds


def solution_report(D, names):
    """Per-solution weights across fits, and R-hat inside each solution."""
    zc = names.index("zred")
    Z = D[:, :, zc]
    edges, meds = find_solutions(Z)
    M = D.shape[0]
    lab = np.digitize(Z, edges[1:-1])                                 # (fits, N)
    W = np.stack([(lab == s).mean(1) for s in range(len(meds))], axis=1)  # (fits, sol)
    out = []
    print(f"   redshift solutions (pooled medians): {', '.join(f'{m:.2f}' for m in meds)}")
    for s, m in enumerate(meds):
        keep = [i for i in range(M) if (lab[i] == s).sum() >= MIN_IN]
        if len(keep) >= 2:
            rr = np.array([rhat_rank([D[i, lab[i] == s, p] for i in keep])[2]
                           for p in range(D.shape[2])])
            rtxt = f"R-hat inside: median {np.median(rr):.3f}, worst {rr.max():.3f} " \
                   f"({names[int(np.argmax(rr))]}) from {len(keep)} fits"
        else:
            rr, rtxt = None, "too few fits with draws there for R-hat"
        print(f"     z~{m:<6.2f} weight mean {W[:, s].mean():.3f} std {W[:, s].std(ddof=1):.3f} "
              f"range {W[:, s].min():.3f}-{W[:, s].max():.3f}; {rtxt}")
        out.append(dict(z=m, w=W[:, s], rhat=rr))
    return out, W


def analyse(d):
    f = os.path.join(d, "draws.npz")
    if not os.path.exists(f):
        print(f"{d}: no draws.npz, skipped")
        return None
    z = np.load(f)
    D = z["draws"].astype(float)                     # (fits, N, dim)
    names = [str(n) for n in z["names"]]
    M, N, P = D.shape
    m = re.search(r"blanca_layout_\d+_(\d+)_k(\d+)_L(\d+)_([a-z]+)(?:_ps(\w+))?$",
                  os.path.basename(os.path.normpath(d)))
    label = f"{m.group(1)} k={m.group(2)} {('ps' + m.group(5)) if m.group(5) else 'std'}" if m else d

    zc = names.index("zred")
    zref = main_solution(D[:, :, zc])
    inmain = np.abs(D[:, :, zc] - zref) < 0.3 * (1 + zref)          # (fits, N)
    wmain = inmain.mean(1)

    allr = np.array([rhat_rank([D[i, :, p] for i in range(M)]) for p in range(P)])
    keep = [i for i in range(M) if inmain[i].sum() >= MIN_IN]
    if len(keep) >= 2:
        mainr = np.array([rhat_rank([D[i, inmain[i], p] for i in keep]) for p in range(P)])
    else:
        mainr = np.full((P, 3), np.nan)

    print(f"== {label}: {M} fits x {N} draws; main solution z = {zref:.2f}")
    print(f"   main-solution weight per fit: mean {wmain.mean():.3f}, std {wmain.std(ddof=1):.3f}, "
          f"range {wmain.min():.3f}-{wmain.max():.3f}; fits with >1% elsewhere: "
          f"{np.sum(wmain < 0.99)}/{M}")
    print(f"   {'parameter':26s} {'bulk':>6s} {'folded':>6s} {'R-hat':>6s} | "
          f"{'main solution only (' + str(len(keep)) + ' fits)':>30s}")
    for p in range(P):
        fl = lambda v: "*" if v > WARN else ("+" if v > OK else " ")
        print(f"   {names[p]:26s} {allr[p, 0]:6.3f} {allr[p, 1]:6.3f} {allr[p, 2]:6.3f}{fl(allr[p, 2])}| "
              f"{mainr[p, 0]:6.3f} {mainr[p, 1]:6.3f} {mainr[p, 2]:6.3f}{fl(mainr[p, 2])}")
    sols, W = solution_report(D, names)
    res = dict(label=label, zref=zref, w_mean=wmain.mean(), w_std=wmain.std(ddof=1),
               r_all=allr[:, 2], r_main=mainr[:, 2], names=names, sols=sols,
               wstd_max=float(W.std(axis=0, ddof=1).max()))
    print(f"   all fits: median R-hat {np.median(allr[:, 2]):.3f}, worst {allr[:, 2].max():.3f} "
          f"({names[int(np.argmax(allr[:, 2]))]}), share < {OK}: {np.mean(allr[:, 2] < OK):.2f}, "
          f"share > {WARN}: {np.mean(allr[:, 2] > WARN):.2f}")
    if np.isfinite(mainr).all():
        print(f"   main solution: median R-hat {np.median(mainr[:, 2]):.3f}, worst "
              f"{mainr[:, 2].max():.3f}, share < {OK}: {np.mean(mainr[:, 2] < OK):.2f}")
    print("   (+ above 1.01; * above 1.04)\n")
    return res


def main():
    res = [r for r in (analyse(d) for d in sys.argv[1:]) if r is not None]
    if len(res) > 1:
        print("== Summary (R-hat over all fits, every parameter)")
        print(f"   {'run':26s} {'w_main':>6s} {'w_std':>6s} | {'median':>6s} {'worst':>6s} "
              f"{'<1.01':>6s} {'>1.04':>6s} | {'main-only median':>16s} {'worst':>6s} | "
              f"{'#sol':>4s} {'max w std':>9s}")
        for r in res:
            print(f"   {r['label']:26s} {r['w_mean']:6.3f} {r['w_std']:6.3f} | "
                  f"{np.median(r['r_all']):6.3f} {r['r_all'].max():6.3f} "
                  f"{np.mean(r['r_all'] < OK):6.2f} {np.mean(r['r_all'] > WARN):6.2f} | "
                  f"{np.median(r['r_main']):16.3f} {np.max(r['r_main']):6.3f} | "
                  f"{len(r['sols']):4d} {r['wstd_max']:9.3f}")


if __name__ == "__main__":
    main()
