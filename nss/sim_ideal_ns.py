"""Seed-to-seed scatter of a PERFECT nested sampler with the NSS settings (the floor).

Toy: d-dim Gaussian prior N(0, a I), likelihood exp(-|x|^2 / 2). Points are drawn exactly
from the constrained prior (through the prior volume X(r) = P(chi2_d < r^2 / a)), so all
scatter between seeds comes from nested sampling itself: random volumes and finite samples.
Bookkeeping as in nss_jax.py: delete num_delete worst per iteration (live counts n, n-1, ...),
new points inside the worst deleted point's contour, stop when logZ_live - logZ < -3,
add the final live points; logZ and its quoted error from 100 Monte Carlo volume draws;
posterior quantiles from the mean log weights.

Reports, over seeds: S50/S16/S84 (std of each quantile / true posterior half-width) for a
symmetric coordinate x1 and a skewed one, r^2; R = std(logZ) / mean quoted error; and the
logZ bias against the exact value. Compare with analyze_convergence.py on real runs.

  python sim_ideal_ns.py [--seeds 200] [--a 8 15] [--n-live 1000] [--num-delete 100]
"""
import argparse

import numpy as np
from scipy.special import gammaincinv, gammaln
from scipy.stats import chi2, norm


def wquant(x, w, qs=(0.16, 0.5, 0.84)):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    c /= c[-1]
    return np.interp(qs, c, x[i])


def run(rng, d, a, n_live, num_delete, term=-3.0, n_mc=100):
    X = rng.uniform(size=n_live)                     # live prior volumes (iid U(0,1))
    dead_X, nlive_at = [], []                        # live count at each deletion
    logx_est, logz_est = 0.0, -np.inf

    def logL_of(Xv):
        r2 = 2 * a * gammaincinv(d / 2, Xv)
        return -0.5 * r2, r2

    while True:
        X.sort()
        dl = X[-num_delete:][::-1]                   # worst first (largest volume)
        dead_X.append(dl)
        nlive_at.append(n_live - np.arange(num_delete))
        lx = logx_est - np.cumsum(1.0 / (n_live - np.arange(num_delete)))
        ll, _ = logL_of(dl)
        dx = np.exp(np.concatenate([[logx_est], lx[:-1]])) - np.exp(lx)
        logz_est = np.logaddexp(logz_est, np.log(np.sum(np.exp(ll) * dx)))
        logx_est = lx[-1]
        Xstar = dl[-1]
        X = np.concatenate([X[:-num_delete], rng.uniform(0, Xstar, size=num_delete)])
        ll_live, _ = logL_of(X)
        logz_live = np.log(np.mean(np.exp(ll_live))) + logx_est
        if logz_live - logz_est < term:
            break
    X.sort()
    dead_X.append(X[::-1])
    nlive_at.append(n_live - np.arange(n_live))
    Xd = np.concatenate(dead_X)
    n_j = np.concatenate(nlive_at).astype(float)
    ll, r2 = logL_of(Xd)

    # Monte Carlo volumes: log t_j = log(U)/n_j
    logt = np.log(rng.uniform(size=(n_mc, len(n_j)))) / n_j
    logx = np.cumsum(logt, axis=1)
    xprev = np.concatenate([np.ones((n_mc, 1)), np.exp(logx[:, :-1])], axis=1)
    logw = np.log(xprev - np.exp(logx)) + ll
    logz = np.logaddexp.reduce(logw, axis=1)
    lw = logw.mean(0)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    # positions: radius from the volume, direction uniform
    g = rng.normal(size=(len(Xd), d))
    x1 = np.sqrt(r2) * g[:, 0] / np.linalg.norm(g, axis=1)
    return dict(logz=logz.mean(), err=logz.std(), x1=wquant(x1, w), r2=wquant(r2, w),
                ess=1 / np.sum(w ** 2), iters=len(dead_X) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--d", type=int, default=26)
    ap.add_argument("--a", type=float, nargs="+", default=[8.0, 15.0])
    ap.add_argument("--n-live", type=int, nargs="+", default=[1000])
    ap.add_argument("--num-delete", type=int, nargs="+", default=[100])
    o = ap.parse_args()
    d = o.d
    print(f"Ideal nested sampling, d={d}, {o.seeds} seeds per row\n")
    print(f"{'a':>5s} {'H':>5s} {'n_live':>6s} {'del':>4s} {'iter':>5s} {'ESS':>6s} | "
          f"{'x1 S50':>6s} {'S16':>5s} {'S84':>5s} | {'r2 S50':>6s} {'S16':>5s} {'S84':>5s} | "
          f"{'R':>5s} {'bias/err':>8s}")
    for a in o.a:
        s2 = a / (1 + a)                             # posterior variance per coordinate
        H = 0.5 * d * (np.log(1 + a) - a / (1 + a))
        logz_true = -0.5 * d * np.log(1 + a)
        hw_x1 = s2 ** 0.5 * norm.ppf(0.84)
        q = chi2.ppf([0.16, 0.84], d) * s2
        hw_r2 = 0.5 * (q[1] - q[0])
        for n_live in o.n_live:
            for nd in o.num_delete:
                rng = np.random.default_rng(12345)
                rs = [run(rng, d, a, n_live, nd) for _ in range(o.seeds)]
                X1 = np.array([r["x1"] for r in rs])
                R2 = np.array([r["r2"] for r in rs])
                lz = np.array([r["logz"] for r in rs])
                er = np.array([r["err"] for r in rs])
                S1 = X1.std(0, ddof=1) / hw_x1
                S2 = R2.std(0, ddof=1) / hw_r2
                print(f"{a:5.1f} {H:5.1f} {n_live:6d} {nd:4d} "
                      f"{np.median([r['iters'] for r in rs]):5.0f} "
                      f"{np.median([r['ess'] for r in rs]):6.0f} | {S1[1]:6.3f} {S1[0]:5.3f} "
                      f"{S1[2]:5.3f} | {S2[1]:6.3f} {S2[0]:5.3f} {S2[2]:5.3f} | "
                      f"{lz.std(ddof=1) / er.mean():5.2f} {(lz.mean() - logz_true) / er.mean():8.2f}",
                      flush=True)


if __name__ == "__main__":
    main()
