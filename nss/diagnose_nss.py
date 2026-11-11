"""Phase 3 convergence diagnostics: NSS (as nss_jax.py: prior start, tabulated mass prior,
n_live / num_delete / inner steps) run with an instrumented copy of blackjax's update
strategy, which records what each inner slice chain does. The sampling is the same as
blackjax's default (update_with_mcmc_take_last); only extra information is returned.

Recorded per NSS iteration, for each of the num_delete new points:
  start / checkpoint positions   the chain's start point (a random survivor) and its
                                 position after CKPTS slice steps (u-space)
  d2                             squared whitened distance from the start, in the live
                                 covariance metric the kernel uses. For a new point
                                 independent of its start, E[d2] = 2 d, so r = d2/(2d) -> 1.
  ins                            insertion index: rank of the new point's log L among the
                                 survivors, in [0, 1]. Uniform if new points are fair draws
                                 from the constrained prior (Fowlie, Handley & Su 2020).
  start_rank                     the same rank for the chain's start point; corr(start_rank,
                                 ins) > 0 means the chain kept a memory of its start's L.
  shrinks, expansions, accepted  slice statistics per chain (sum / mean over its steps)
  live z                         redshift-coordinate u of every live point (for redshift-
                                 solution counts and cross-solution moves)

  python diagnose_nss.py --catalog minerva_uds_test9.fits --id 1192179 --out <private dir>
  python diagnose_nss.py --toy gauss|banana --out <dir>          (known-answer checks)
  python diagnose_nss.py --analyse <file.npz> [--z-solutions 0.31 1.58 2.8]
"""
import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)
import blackjax
from scipy.stats import kstest, rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPTS = (1, 2, 4, 8, 16, 32, 65, 130, 260, 520)


def diag_update_strategy(ckpts):
    """update_with_mcmc_take_last (blackjax 1.7.1, ns/from_mcmc.py), plus diagnostics."""

    def strategy(constrained_mcmc_step_fn, num_mcmc_steps, num_delete):
        ck = np.array([c for c in ckpts if c <= num_mcmc_steps]) - 1

        def update_function(rng_key, state, loglikelihood_0, **step_parameters):
            choice_key, sample_key = random.split(rng_key)
            particles = state.particles
            weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
            weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
            start_idx = random.choice(choice_key, len(weights), shape=(num_delete,),
                                      p=weights / weights.sum(), replace=True)
            start_state = jax.tree.map(lambda x: x[start_idx], particles)

            def step(key, s):
                return constrained_mcmc_step_fn(key, s, loglikelihood_0=loglikelihood_0,
                                                **step_parameters)

            def mcmc_kernel(key, s0):
                keys = random.split(key, num_mcmc_steps)

                def body(s, k):
                    ns, info = step(k, s)
                    return ns, (ns.position, info.num_shrink, info.num_expansions,
                                info.is_accepted)

                final, (traj, nshr, nexp, acc) = jax.lax.scan(body, s0, keys)
                return final, (traj[ck], nshr.sum(), nexp.sum(), acc.mean())

            new_particles, (ckpos, nshr, nexp, acc) = jax.vmap(mcmc_kernel)(
                random.split(sample_key, num_delete), start_state)
            new_particles = new_particles._replace(
                loglikelihood_birth=jnp.full_like(new_particles.loglikelihood_birth,
                                                  loglikelihood_0))
            L = step_parameters["covariance_factor"]
            d = L.shape[0]
            diff = ckpos - start_state.position[:, None, :]
            y = jax.scipy.linalg.solve_triangular(L, diff.reshape(-1, d).T, lower=True)
            d2 = jnp.sum(y ** 2, axis=0).reshape(num_delete, -1)
            surv = particles.loglikelihood > loglikelihood_0
            nsurv = surv.sum()
            below = lambda ll: jnp.sum((particles.loglikelihood[None, :] < ll[:, None])
                                       & surv[None, :], axis=1) / nsurv
            info = dict(start_pos=start_state.position, ck_pos=ckpos, d2=d2,
                        ins=below(new_particles.loglikelihood),
                        start_rank=below(start_state.loglikelihood),
                        shrinks=nshr, expansions=nexp, accepted=acc,
                        L0=loglikelihood_0, nsurv=nsurv)
            return new_particles, info

        return update_function

    return strategy


def toy_loglike(kind, d):
    s = 0.01                                     # posterior width in the unit cube

    def gauss(u):
        y = (u - 0.5) / s
        return -0.5 * jnp.sum(y ** 2)

    def banana(u):
        y = (u - 0.5) / s
        y1 = y[0] / 3.0
        y2 = y[1] - 0.6 * (y[0] ** 2 / 9.0 - 1.0) * 3.0
        return -0.5 * (y1 ** 2 + y2 ** 2 + jnp.sum(y[2:] ** 2))

    return {"gauss": gauss, "banana": banana}[kind]


def run(loglike, d, n_live, num_delete, k, seed, n_prior_start, zi, termination=-3.0,
        max_seconds=np.inf, logprior=None, loglike_u=None, to_x=None, d_u=None):
    """logprior / to_x / loglike_u: sampling coordinates other than the unit cube
    (linear_mass.LinearMass, mass_marginal.MassMarginal): prior draws are made in the
    standard d_u-dimensional cube (default d) with loglike_u, then mapped with to_x."""
    d_u = d if d_u is None else d_u
    from nss_jax import prior_start

    if logprior is None:
        def logprior(u):
            return jnp.where(jnp.all((u >= 0.0) & (u <= 1.0)), 0.0, -jnp.inf)
    loglike_u = loglike if loglike_u is None else loglike_u
    to_x = (lambda u: u) if to_x is None else to_x

    algo = blackjax.nss(logprior_fn=logprior, loglikelihood_fn=loglike, num_inner_steps=k,
                        num_delete=num_delete, update_strategy=diag_update_strategy(CKPTS))
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    if n_prior_start and n_prior_start > n_live:
        bf = jax.jit(jax.vmap(loglike_u)).lower(jnp.zeros((20000, d_u))).compile()
        ps = prior_start(bf, d_u, n_live, int(n_prior_start), k0, 20000, 0)
        live0 = jnp.asarray(to_x(ps["live_u"]))
        print(f"  prior start: {ps['n_total']} draws, threshold log L {ps['threshold']:.2f}",
              flush=True)
    else:
        live0 = jnp.asarray(to_x(np.asarray(jax.random.uniform(k0, (n_live, d_u),
                                                               dtype=jnp.float64))))
    state = jax.jit(algo.init)(live0)
    step = jax.jit(algo.step).lower(key, state).compile()
    rec = {n: [] for n in ("start_pos", "ck_pos", "d2", "ins", "start_rank", "shrinks",
                           "expansions", "accepted", "L0", "nsurv", "live_z", "maxL",
                           "cf_bad", "cov_eig_min", "cov_eig_max")}
    it, t0 = 0, time.time()
    while not (state.integrator.logZ_live - state.integrator.logZ < termination):
        key, sub = jax.random.split(key)
        state, info = step(sub, state)
        u = info.update_info
        for n in ("start_pos", "ck_pos"):
            rec[n].append(np.asarray(u[n], np.float32))
        for n in ("d2", "ins", "start_rank", "shrinks", "expansions", "accepted", "L0", "nsurv"):
            rec[n].append(np.asarray(u[n]))
        if zi is not None:
            rec["live_z"].append(np.asarray(state.particles.position[:, zi], np.float32))
        rec["maxL"].append(float(state.particles.loglikelihood.max()))
        cf = np.asarray(state.inner_kernel_params["covariance_factor"])
        rec["cf_bad"].append(not np.all(np.isfinite(cf)))
        ev = np.linalg.eigvalsh(np.cov(np.asarray(state.particles.position).T))
        rec["cov_eig_min"].append(float(ev[0]))
        rec["cov_eig_max"].append(float(ev[-1]))
        it += 1
        if it % 25 == 0:
            print(f"  it {it:5d}  logZ(NSS phase) {float(state.integrator.logZ):9.2f}  max logL "
                  f"{rec['maxL'][-1]:9.2f}  {time.time() - t0:7.1f} s", flush=True)
        if time.time() - t0 > max_seconds:
            break
    out = {n: np.stack(v) for n, v in rec.items() if v}
    out.update(ckpts=np.array([c for c in CKPTS if c <= k]), d=d, n_live=n_live,
               num_delete=num_delete, k=k, seed=seed, iterations=it, wall=time.time() - t0,
               logZ_nss=float(state.integrator.logZ))
    return out


def u_to_z_table(model, zi, n=4001):
    """zred as a function of its own unit-cube coordinate (checked independent of the rest)."""
    rng = np.random.default_rng(0)
    uz = np.linspace(0, 1, n)
    u = rng.uniform(size=(n, model.n_dim))
    u[:, zi] = uz
    z1 = model.prior_transform(u)[:, zi]
    u[:, [j for j in range(model.n_dim) if j != zi]] = rng.uniform(size=(n, model.n_dim - 1))
    z2 = model.prior_transform(u)[:, zi]
    assert np.allclose(z1, z2), "zred depends on other coordinates"
    return uz, z1


def analyse(f, z_solutions=None, names=None):
    r = dict(np.load(f, allow_pickle=True))
    d, ck, it = int(r["d"]), r["ckpts"], int(r["iterations"])
    names = list(r["names"]) if "names" in r else [f"x{i}" for i in range(d)]
    stages = np.array_split(np.arange(it), 4)
    print(f"{f}\n  {it} iterations, d={d}, n_live={int(r['n_live'])}, num_delete="
          f"{int(r['num_delete'])}, k={int(r['k'])}; stages = quarters of the run\n")

    print("  A. Distance travelled: r(k) = mean d2 / 2d  (1 = independent of the start point)")
    print("     stage   " + " ".join(f"k={c:<4d}" for c in ck))
    for s, idx in enumerate(stages):
        rr = r["d2"][idx].mean(axis=(0, 1)) / (2 * d)
        print(f"     {s + 1}       " + " ".join(f"{v:6.2f}" for v in rr))

    print("\n  B. Insertion index (uniform = fair draws) and memory of the start's likelihood")
    print("     stage  mean(ins) KS p    corr(start rank, new rank)   shrinks/step  "
          "expansions/step  accepted")
    for s, idx in enumerate(stages):
        ins = r["ins"][idx].ravel()
        rho = np.mean([spearmanr(r["start_rank"][i], r["ins"][i])[0] for i in idx])
        k = int(r["k"])
        print(f"     {s + 1}      {ins.mean():6.3f}  {kstest(ins, 'uniform').pvalue:7.1e}   "
              f"{rho:+6.3f}                      {r['shrinks'][idx].mean() / k:6.2f}        "
              f"{r['expansions'][idx].mean() / k:6.2f}         {r['accepted'][idx].mean():.3f}")

    print("\n  C. Memory per parameter: Spearman corr(start, position after k steps), mean over "
          "iterations (0 = forgotten; +-0.01 noise)")
    print(f"     {'parameter':26s} " + " ".join(f"k={c:<4d}" for c in ck) + "   | at k_max by stage")
    def zrank(x):                                  # rank-standardize along axis 1 (chains)
        q = rankdata(x, axis=1)
        q = q - q.mean(axis=1, keepdims=True)
        return q / np.sqrt((q ** 2).sum(axis=1, keepdims=True))

    S = zrank(r["start_pos"])                      # (it, chains, d)
    rho = np.stack([np.sum(S * zrank(r["ck_pos"][:, :, j]), axis=1)
                    for j in range(len(ck))], axis=1)                    # (it, ck, d)
    for p in range(d):
        bys = [np.nanmean(rho[idx, -1, p]) for idx in stages]
        print(f"     {names[p]:26s} " + " ".join(f"{np.nanmean(rho[:, j, p]):+6.2f}" for j in
                                                   range(len(ck))) +
              "   | " + " ".join(f"{v:+5.2f}" for v in bys))

    if z_solutions is not None and "live_z" in r:
        uz, zz = r["u_to_z"]
        zl = np.interp(r["live_z"], uz, zz)
        zs = np.array(z_solutions)
        lab = lambda z: np.argmin(np.abs(z[..., None] - zs) / (1 + zs), axis=-1)
        zi = int(r["zi"])
        print(f"\n  D. Redshift solutions {list(zs)}: live points per solution, and chains that "
              "end in a different solution from where they started")
        print("     iteration " + " ".join(f"n(z~{z:g})" for z in zs) + "   cross-solution moves")
        for i in list(range(0, it, max(1, it // 15))) + [it - 1]:
            n = np.bincount(lab(zl[i]), minlength=len(zs))
            sz = lab(np.interp(r["start_pos"][i, :, zi], uz, zz))
            ez = lab(np.interp(r["ck_pos"][i, :, -1, zi], uz, zz))
            print(f"     {i:9d} " + " ".join(f"{v:8d}" for v in n) +
                  f"   {np.mean(sz != ez):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog")
    ap.add_argument("--id")
    ap.add_argument("--toy", choices=["gauss", "banana"])
    ap.add_argument("--d", type=int, default=26)
    ap.add_argument("--n-live", type=int, default=1000)
    ap.add_argument("--num-delete", type=int, default=100)
    ap.add_argument("--k", type=int, default=130)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-prior-start", type=float, default=1e6)
    ap.add_argument("--max-seconds", type=float, default=np.inf)
    ap.add_argument("--out", default=".")
    ap.add_argument("--analyse")
    ap.add_argument("--z-solutions", type=float, nargs="+")
    ap.add_argument("--mass-coord", choices=["cdf", "linear", "marginal"], default="cdf",
                    help="logmass: prior inverse CDF (standard), linear with the mass prior as "
                         "a density (linear_mass.py), or integrated out (mass_marginal.py)")
    a = ap.parse_args()
    if a.analyse:
        analyse(a.analyse, a.z_solutions)
        return
    os.makedirs(a.out, exist_ok=True)
    if a.toy:
        loglike, d, zi, tag, names, extra = toy_loglike(a.toy, a.d), a.d, None, a.toy, None, {}
    else:
        from nss_efsps import build
        from photometry import read_photometry
        from jax_model import JaxLogLike
        from tabulated_mass_prior import use_tabulated_mass_prior
        phot = read_photometry(a.catalog, [a.id])[0]
        model = build(phot)
        use_tabulated_mass_prior(model)
        loglike, d, tag, names = JaxLogLike(model), model.n_dim, a.id, list(model.free)
        zi = model.col["zred"]
        extra = dict(u_to_z=np.array(u_to_z_table(model, zi)), zi=zi)
    kw = {}
    if a.mass_coord == "linear":
        from linear_mass import LinearMass
        lm = LinearMass(model)
        kw = dict(logprior=lm.logprior, loglike_u=loglike, to_x=lm.x_from_u)
        loglike = lm.loglike
        tag += "_linmass"
    elif a.mass_coord == "marginal":
        from mass_marginal import MassMarginal
        mm = MassMarginal(model)
        kw = dict(logprior=mm.logprior, loglike_u=lambda u: mm.loglike(mm._x_from_u(u)[0]),
                  to_x=lambda U: mm.x_from_u(U)[0], d_u=model.n_dim)
        loglike, d = mm.loglike, mm.n_dim
        names = [names[c] for c in mm.keep]
        zi = mm.x_z
        extra["zi"] = zi
        tag += "_massmarg"
    print(f"[{tag}] diagnostics run: d={d}, n_live={a.n_live}, num_delete={a.num_delete}, "
          f"k={a.k}, seed={a.seed}, prior start {a.n_prior_start:g}, mass coordinate "
          f"{a.mass_coord}", flush=True)
    out = run(loglike, d, a.n_live, a.num_delete, a.k, a.seed, a.n_prior_start, zi,
              max_seconds=a.max_seconds, **kw)
    if names:
        out["names"] = np.array(names)
    out.update(extra)
    f = os.path.join(a.out, f"diag_{tag}_k{a.k}_L{a.n_live}_del{a.num_delete}_s{a.seed}.npz")
    np.savez_compressed(f, **out)
    print(f"[{tag}] {out['iterations']} iterations, {out['wall']:.0f} s -> {f}\n", flush=True)
    analyse(f)


if __name__ == "__main__":
    main()
