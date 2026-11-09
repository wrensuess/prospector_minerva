"""Known-answer test of the prior start in nss_jax.run_nss_core.

Toys on the 10-d unit cube, likelihoods normalised so the true evidence is Z = 1 (logZ = 0;
truncation by the cube is < 1e-6):
  single    one Gaussian, sigma 0.06 at u = 0.3 (bookkeeping only).
  moderate  equal-weight mixture: broad sigma 0.06 at 0.3 and narrow sigma 0.04 at 0.7.
            The narrow mode's share of the live volume dips to ~3% (~6 of 200 points).
  severe    same with narrow sigma 0.035: its share dips to ~0.5% (~1 of 200 points) near
            2e6 prior draws, like the z ~ 6 solution of MINERVA 1152437.
In both mixtures the true posterior weight of each mode is 0.5.

For each toy and start (standard, and prior starts of N draws) it runs many seeds and
reports: mean logZ and its standard error, the scatter of logZ vs the mean quoted error,
the narrow-mode weight, and how many runs lost the narrow mode (weight < 0.05).
PASS requires, for every toy and start: |mean logZ| < 3 standard errors, and the quoted
error not too small (scatter / quoted < 2); mode weights are reported, not judged here.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nss_jax import run_nss_core

D = 10
MU1, MU2, S1 = 0.3, 0.7, 0.06
TOYS = {"single": None, "moderate": 0.04, "severe": 0.035}


def make_loglike(s2):
    c1 = -D * np.log(np.sqrt(2 * np.pi) * S1)

    def broad(u):
        return -0.5 * jnp.sum((u - MU1) ** 2) / S1 ** 2 + c1

    if s2 is None:
        return broad, None
    c2 = -D * np.log(np.sqrt(2 * np.pi) * s2)

    def narrow(u):
        return -0.5 * jnp.sum((u - MU2) ** 2) / s2 ** 2 + c2

    def loglike(u):
        return jnp.logaddexp(broad(u), narrow(u)) + np.log(0.5)

    def is_narrow(u):
        a = -0.5 * np.sum((u - MU1) ** 2, 1) / S1 ** 2 + c1
        b = -0.5 * np.sum((u - MU2) ** 2, 1) / s2 ** 2 + c2
        return b > a

    return loglike, is_narrow


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--toys", nargs="+", default=list(TOYS))
    p.add_argument("--starts", type=float, nargs="+", default=[0, 1e5, 1e7],
                   help="0 = standard start")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--n-live", type=int, default=200)
    p.add_argument("--num-delete", type=int, default=20)
    p.add_argument("--num-inner-steps", type=int, default=3 * D)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows, ok_all = [], True
    for toy in args.toys:
        loglike, is_narrow = make_loglike(TOYS[toy])
        for n0 in args.starts:
            n0 = int(n0) if n0 > args.n_live else None
            lz, lze, wn, its, wall = [], [], [], [], []
            for s in range(args.seeds):
                t0 = time.time()
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    res = run_nss_core(loglike, D, args.n_live, args.num_delete,
                                       args.num_inner_steps, s, n_prior_start=n0,
                                       keep_discards=2000, log_every=10 ** 9)
                lz.append(float(np.mean(res["logz_draws"])))
                lze.append(float(np.std(res["logz_draws"])))
                its.append(res["iterations"])
                wall.append(time.time() - t0)
                if is_narrow is not None:
                    wn.append(float(res["w"][is_narrow(res["u"])].sum()))
            lz, lze = np.array(lz), np.array(lze)
            se = lz.std(ddof=1) / np.sqrt(len(lz))
            ratio = lz.std(ddof=1) / lze.mean()
            ok = abs(lz.mean()) < 3 * se and ratio < 2.0
            ok_all &= ok
            r = dict(toy=toy, start=n0 or "standard", seeds=args.seeds,
                     logZ_mean=lz.mean(), logZ_se=se, logZ_std=lz.std(ddof=1),
                     quoted_err=lze.mean(), scatter_over_quoted=ratio,
                     iterations_median=float(np.median(its)), wall_median_s=float(np.median(wall)),
                     ok=bool(ok))
            if wn:
                wn = np.array(wn)
                r.update(narrow_weight_mean=wn.mean(), narrow_weight_std=wn.std(ddof=1),
                         narrow_lost=int(np.sum(wn < 0.05)))
            rows.append(r)
            msg = (f"{toy:9s} start {str(r['start']):>9s}: logZ {r['logZ_mean']:+.3f} +- "
                   f"{se:.3f} (true 0)  scatter/quoted {ratio:.2f}  it {r['iterations_median']:.0f}"
                   f"  {r['wall_median_s']:.1f} s/run")
            if wn is not None and len(wn):
                msg += (f"  narrow weight {r['narrow_weight_mean']:.3f} +- "
                        f"{r['narrow_weight_std']:.3f} (true 0.5), lost in {r['narrow_lost']}"
                        f"/{args.seeds}")
            print(msg + ("" if ok else "   <-- FAIL"), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1, default=float)
    print("RESULT:", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
