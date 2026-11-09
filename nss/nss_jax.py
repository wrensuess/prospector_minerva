"""nss_efsps.py with the pure-JAX likelihood (jax_model.JaxLogLike) in place of the numpy
code behind jax.pure_callback. Same sampler, same loop, same summary (nss_efsps.summarise),
so the outputs compare directly with the numpy-callback runs.

The model-row counters of the callback runs are not available inside jit; the summary
reports wall time, and compile time separately.

Prior start (--n-prior-start N, N > n_live; DEFAULT N = 1e6, use 0 for the standard start
of n_live random prior draws): draw N points from the prior, delete the
worst N - n_live one at a time WITHOUT replacement (no MCMC), and start NSS from the best
n_live. This is ordinary nested sampling with a live set that shrinks from N to n_live.
The bookkeeping is split in two:
- discard phase: live counts N, N-1, ..., n_live+1 are known exactly. The handover
  contour is the last deleted point, the (n_live+1)-th best of N, so the volume left is
  X_h ~ Beta(n_live + 1, N - n_live), sampled exactly per Monte Carlo draw. The discards'
  own evidence uses the expected volumes.
- NSS phase: a standard run inside the constrained prior L > L* (its live points are iid
  uniform there). blackjax's weights for it are relative to that volume, and are scaled
  by X_h.
Termination uses the NSS-phase evidence alone, which is stricter than the total
(the total also includes the discard phase).
Only the best `keep_discards` discarded points keep their positions; the summary records
the posterior weight of the rest, which must be negligible.
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
import blackjax
from blackjax.ns.utils import finalise, log_weights

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nss_efsps import build, summarise
from efsps_photfit import config as C
from photometry import read_photometry
from jax_model import JaxLogLike
from tabulated_mass_prior import use_tabulated_mass_prior


class _NoCounts:
    n_calls = n_rows = n_rows_outside = None
    seconds = None


def _log1mexp(x):
    """log(1 - exp(x)) for x < 0."""
    return np.where(x > -0.693, np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


def prior_start(loglike_batch, d, n_live, n_total, key, batch=20000, keep_discards=10000):
    """Evaluate n_total prior draws; return the best n_live as the first live set, plus
    the bookkeeping of the n_total - n_live points deleted without replacement."""
    n_batches = -(-n_total // batch)
    all_l = np.empty(n_total)
    m_keep = n_live + keep_discards
    buf_u, buf_l = np.empty((0, d)), np.empty(0)
    t0 = time.time()
    for b in range(n_batches):
        m = min(batch, n_total - b * batch)
        u = jax.random.uniform(jax.random.fold_in(key, b), (batch, d), dtype=jnp.float64)
        lb = np.asarray(loglike_batch(u))[:m]
        u = np.asarray(u)[:m]
        all_l[b * batch:b * batch + m] = lb
        cu, cl = np.concatenate([buf_u, u]), np.concatenate([buf_l, lb])
        if cl.size > m_keep:
            idx = np.argpartition(-cl, m_keep - 1)[:m_keep]
            cu, cl = cu[idx], cl[idx]
        buf_u, buf_l = cu, cl
    eval_s = time.time() - t0
    order = np.argsort(buf_l)[::-1]
    buf_u, buf_l = buf_u[order], buf_l[order]

    # Discard phase: the k-th deletion (k = 1..N-n_live) happens with N-k+1 live points.
    # Expected log X after it is -sum_{j<=k} 1/(N-j+1); the shell is X_{k-1} - X_k, as in
    # blackjax's integrator. Done in chunks to keep memory at ~one copy of the N values.
    n_disc = n_total - n_live
    n_kept = min(keep_discards, n_disc)
    all_l.sort()                                          # ascending = deletion order
    logz_disc = logw_dropped = -np.inf
    kept_logw_asc = np.empty(n_kept)
    logx_prev, chunk = 0.0, 1_000_000
    for a in range(0, n_disc, chunk):
        b = min(a + chunk, n_disc)
        n_i = (n_total - np.arange(a, b)).astype(float)
        logx = logx_prev - np.cumsum(1.0 / n_i)
        prev = np.concatenate([[logx_prev], logx[:-1]])
        logw = all_l[a:b] + prev + _log1mexp(-1.0 / n_i)
        logw = np.where(np.isfinite(logw), logw, -np.inf)
        logz_disc = np.logaddexp(logz_disc, np.logaddexp.reduce(logw))
        cut = n_disc - n_kept                             # first index whose position is kept
        if a < cut:
            logw_dropped = np.logaddexp(logw_dropped, np.logaddexp.reduce(logw[:max(cut - a, 0)]))
        if b > cut:
            lo = max(cut, a)
            kept_logw_asc[lo - cut:b - cut] = logw[lo - a:]
        logx_prev = logx[-1]
    logx_end = logx_prev
    kept_l = buf_l[n_live:n_live + n_kept]                # descending = last deletions first
    kept_logw = kept_logw_asc[::-1].copy()
    return dict(live_u=buf_u[:n_live], threshold=float(buf_l[n_live - 1]),
                kept_u=buf_u[n_live:n_live + n_kept], kept_l=kept_l, kept_logw=kept_logw,
                logz_disc=float(logz_disc), logw_dropped=float(logw_dropped),
                logx_expected=float(logx_end), eval_s=eval_s,
                n_total=n_total, n_live=n_live)


def run_nss_jax(model, n_live, num_delete, num_inner_steps, seed, termination=-3.0,
                max_seconds=np.inf, log_every=50, n_prior_start=None, prior_batch=20000,
                keep_discards=10000):
    return run_nss_core(JaxLogLike(model), model.n_dim, n_live, num_delete, num_inner_steps,
                        seed, termination, max_seconds, log_every, n_prior_start,
                        prior_batch, keep_discards)


def run_nss_core(loglike, d, n_live, num_delete, num_inner_steps, seed, termination=-3.0,
                 max_seconds=np.inf, log_every=50, n_prior_start=None, prior_batch=20000,
                 keep_discards=10000):
    """NSS on the unit cube for a single-row log-likelihood u -> log L (pure JAX)."""

    def logprior(u):
        return jnp.where(jnp.all((u >= 0.0) & (u <= 1.0)), 0.0, -jnp.inf)

    algo = blackjax.nss(logprior_fn=logprior, loglikelihood_fn=loglike,
                        num_inner_steps=num_inner_steps, num_delete=num_delete)
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    ps = None
    t_c = time.time()
    if n_prior_start is not None and n_prior_start > n_live:
        batch_fn = jax.jit(jax.vmap(loglike)).lower(
            jnp.zeros((prior_batch, d), jnp.float64)).compile()
        compile_prior_s = time.time() - t_c
        ps = prior_start(batch_fn, d, n_live, int(n_prior_start), k0, prior_batch,
                         keep_discards)
        print(f"  prior start: {ps['n_total']} draws in {ps['eval_s']:.1f} s; threshold "
              f"log L {ps['threshold']:.2f}; E[log X] at handover {ps['logx_expected']:.3f}",
              flush=True)
        t_c = time.time()
        live0 = jnp.asarray(ps["live_u"])
    else:
        compile_prior_s = 0.0
        live0 = jax.random.uniform(k0, (n_live, d), dtype=jnp.float64)
    state = jax.jit(algo.init)(live0)
    step = jax.jit(algo.step).lower(key, state).compile()
    compile_s = time.time() - t_c + compile_prior_s

    dead, it, t0 = [], 0, time.time()
    timed_out = False
    while not (state.integrator.logZ_live - state.integrator.logZ < termination):
        key, sub = jax.random.split(key)
        state, info = step(sub, state)
        dead.append(info)
        it += 1
        if it % log_every == 0:
            print(f"  it {it:6d}  logZ {float(state.integrator.logZ):9.2f}  "
                  f"max logL {float(state.particles.loglikelihood.max()):9.2f}  "
                  f"{time.time() - t0:7.1f} s", flush=True)
        if time.time() - t0 > max_seconds:
            timed_out = True
            break
    jax.block_until_ready(state)
    wall = time.time() - t0

    final = finalise(state, dead)
    key, kw = jax.random.split(key)
    logw = np.asarray(log_weights(kw, final, shape=100))
    u_out = np.asarray(final.particles.position)
    logl_out = np.asarray(final.particles.loglikelihood)
    extra = {}
    if ps is not None:
        # Scale the NSS phase by the handover volume: the contour of the last deleted
        # point, the (n_live+1)-th best of N, i.e. the (n_live+1)-th smallest of N
        # uniform volumes: X_h ~ Beta(n_live + 1, N - n_live).
        # E[log X_h] = -sum_{k=n_live+1}^{N} 1/k, matching the discard-phase sum.
        key, kb = jax.random.split(key)
        log_xh = np.log(np.asarray(jax.random.beta(
            kb, n_live + 1, ps["n_total"] - n_live, shape=(logw.shape[1],), dtype=jnp.float64)))
        logw = logw + log_xh[None, :]
        logz_nss = np.asarray(jax.scipy.special.logsumexp(logw, axis=0))
        logz_draws = np.logaddexp(logz_nss, ps["logz_disc"])
        kept = np.repeat(ps["kept_logw"][:, None], logw.shape[1], axis=1)
        logw = np.concatenate([kept, logw])
        u_out = np.concatenate([ps["kept_u"], u_out])
        logl_out = np.concatenate([ps["kept_l"], logl_out])
        z_mean = float(np.mean(logz_draws))
        extra = {"prior_start": {
            "n_prior_start": ps["n_total"], "threshold_logL": ps["threshold"],
            "logX_handover_expected": ps["logx_expected"],
            "logX_handover_draws_mean": float(log_xh.mean()),
            "logX_handover_draws_std": float(log_xh.std()),
            "logZ_discard_phase": ps["logz_disc"],
            "n_discards_with_positions": int(ps["kept_u"].shape[0]),
            "log10_weight_fraction_dropped_discards":
                float((ps["logw_dropped"] - z_mean) / np.log(10)),
            "eval_s": ps["eval_s"]}}
        wall += ps["eval_s"]
    else:
        logz_draws = np.asarray(jax.scipy.special.logsumexp(logw, axis=0))
    lw = logw.mean(axis=1)
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    return dict(u=u_out, logl=logl_out, w=w, logz_draws=logz_draws,
                iterations=it, wall_s=wall, compile_s=compile_s, timed_out=timed_out,
                cb=_NoCounts(), **extra)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--catalog", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="jax")
    p.add_argument("--n-live", type=int, default=500)
    p.add_argument("--num-delete", type=int, default=50)
    p.add_argument("--num-inner-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--termination", type=float, default=-3.0)
    p.add_argument("--fix-z", type=float, default=None)
    p.add_argument("--z-window", type=float, nargs=2, default=None)
    p.add_argument("--max-seconds", type=float, default=np.inf)
    p.add_argument("--mass-prior", choices=["exact", "tabulated"], default="exact",
                   help="tabulated: tabulated_mass_prior.py (no mass function per call)")
    p.add_argument("--n-prior-start", type=float, default=1e6,
                   help="start from the best n_live of this many prior draws (see module doc). "
                        "Default 1e6, adopted 2026-11-09 after the Blanca test on 9 MINERVA "
                        "galaxies; 0 (or any value <= n_live) gives the standard start")
    p.add_argument("--keep-discards", type=int, default=10000,
                   help="discarded prior draws whose positions are kept for the posterior")
    args = p.parse_args()
    args.n_prior_start = int(args.n_prior_start)

    phot = read_photometry(args.catalog, [args.id])[0]
    model = build(phot, args.fix_z, args.z_window)
    if args.mass_prior == "tabulated":
        use_tabulated_mass_prior(model)
    if args.num_inner_steps is None:
        args.num_inner_steps = 2 * model.n_dim
    print(f"[{phot.id}] JAX model: {model.n_dim} free params, {int(model.valid.sum())} bands, "
          f"n_live={args.n_live}, num_delete={args.num_delete}, "
          f"num_inner_steps={args.num_inner_steps}, seed={args.seed}, mass prior {args.mass_prior}, "
          f"prior start {args.n_prior_start}, "
          f"emulator {C.net_for_md5(model.emu.md5)} ({model.emu.md5})",
          flush=True)
    res = run_nss_jax(model, args.n_live, args.num_delete, args.num_inner_steps, args.seed,
                      args.termination, args.max_seconds, n_prior_start=args.n_prior_start,
                      keep_discards=args.keep_discards)
    summary, theta = summarise(model, res, phot, args, args.tag)
    summary["settings"]["n_prior_start"] = args.n_prior_start
    if "prior_start" in res:
        summary["prior_start"] = res["prior_start"]
    summary["likelihood"] = "pure JAX (jax_model.JaxLogLike)"
    summary["compile_s"] = res["compile_s"]
    summary["mass_prior"] = args.mass_prior
    summary["catalog"] = os.path.abspath(args.catalog)
    summary["emulator"] = {"file": os.path.basename(model.emu.path), "md5": model.emu.md5,
                           "net": C.net_for_md5(model.emu.md5)}
    summary["photometry"] = {k: phot.meta.get(k) for k in ("dropped_bands", "masked_negative",
                                                         "mu_source", "z_spec")}
    summary["bands_used"] = [b for b, ok in zip(phot.bands, phot.valid) if ok]
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"{phot.id}_{args.tag}")
    np.savez_compressed(stem + ".npz", u=res["u"], theta=theta, logl=res["logl"], w=res["w"],
                        logz_draws=res["logz_draws"], names=np.array(model.free))
    with open(stem + ".json", "w") as fh:
        json.dump(summary, fh, indent=1)
    s = summary
    print(f"[{phot.id}] done: logZ {s['logZ']:.2f} +- {s['logZ_err']:.2f}, max logL "
          f"{s['max_logL']:.2f} (chi2 {s['chi2_at_max']:.1f}), ESS {s['ess']:.0f}, "
          f"{s['iterations']} iterations, wall {s['wall_s']:.0f} s + compile "
          f"{s['compile_s']:.0f} s")
    if "z_weight_by_bin" in s:
        print(f"[{phot.id}] z weight by bin: {s['z_weight_by_bin']}; z at max logL "
              f"{s['z_at_max_logL']:.3f}")


if __name__ == "__main__":
    main()
