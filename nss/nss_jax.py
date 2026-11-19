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
import os
import resource
import sys
import time

_T_PROCESS0 = time.time()        # for the summary's process_elapsed_s (from the imports on)

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
from efsps_photfit.catalog import CatalogRefusal, _norm_id
from photometry import read_photometry
from jax_model import JaxLogLike
from tabulated_mass_prior import use_tabulated_mass_prior
from provenance import provenance, write_json_atomic
import malloc_tools

#: Fit status in the summary JSON ("status"). Only "ok" is a finished fit whose posterior is
#: complete; pipeline_status.py treats "bad_input" as final (never retried) and anything
#: other than "ok" as not done; make_catalog_nss.py writes non-"ok" fits as masked rows.
#:   ok         terminated normally, logZ / logZ_err / ESS finite, ESS > 0
#:   timed_out  stopped by --max-seconds (the summary's timed_out is true)
#:   capped     stopped by --max-iterations
#:   nonfinite  the termination criterion became NaN (e.g. every live point has
#:              log L = -inf), or logZ, logZ_err or ESS is not finite, or ESS = 0
#:   bad_input  deterministic input failure (id not in the catalogue, no usable band);
#:              written by main() with no fit and no logZ
STATUSES = ("ok", "timed_out", "capped", "nonfinite", "bad_input")


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
                keep_discards=10000, mass_coord="cdf", dust_prior=None, max_iterations=None):
    if dust_prior is not None and mass_coord != "marginal":
        raise ValueError("the dust prior is implemented for --mass-coord marginal only")
    if mass_coord == "cdf":
        ll = JaxLogLike(model)
        res = run_nss_core(ll, model.n_dim, n_live, num_delete,
                           num_inner_steps, seed, termination, max_seconds, log_every,
                           n_prior_start, prior_batch, keep_discards,
                           max_iterations=max_iterations)
        res["emu_dtype"] = getattr(ll, "emu_dtype", "float64")
        return res
    if mass_coord != "marginal":
        raise ValueError(f"unknown mass coordinate {mass_coord!r}")
    # mass integrated out of the likelihood (mass_marginal.py): 25 sampled parameters;
    # prior draws in the standard 26-d cube, mapped to the sampled coordinates
    from mass_marginal import MassMarginal
    mm = MassMarginal(model, dust_prior=dust_prior)
    res = run_nss_core(mm.loglike, mm.n_dim, n_live, num_delete, num_inner_steps, seed,
                       termination, max_seconds, log_every, n_prior_start, prior_batch,
                       keep_discards, logprior=mm.logprior, d_u=model.n_dim,
                       loglike_u=lambda u: mm.loglike(mm._x_from_u(u)[0]),
                       to_x=lambda U: mm.x_from_u(U)[0], max_iterations=max_iterations)
    res["emu_dtype"] = getattr(mm.ll, "emu_dtype", "float64")
    res["mass_marginal"] = mm
    if res["status"] == "nonfinite":
        return res                     # no usable weights: no mass draws (summary only)
    # the mass posterior: one draw of logmass from L(m, x) q(m; x) per sample
    x = jnp.asarray(res["u"])
    draw = jax.jit(jax.vmap(mm.sample_logmass))
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), 7), x.shape[0])
    m = np.concatenate([np.asarray(draw(keys[i:i + 5000], x[i:i + 5000]))
                        for i in range(0, x.shape[0], 5000)])
    res["theta"] = np.concatenate([np.asarray(mm.batch_theta(x[i:i + 5000], jnp.asarray(m[i:i + 5000])))
                                   for i in range(0, x.shape[0], 5000)])
    res["logmass_draws"] = m
    return res


def run_nss_core(loglike, d, n_live, num_delete, num_inner_steps, seed, termination=-3.0,
                 max_seconds=np.inf, log_every=50, n_prior_start=None, prior_batch=20000,
                 keep_discards=10000, logprior=None, d_u=None, loglike_u=None, to_x=None,
                 max_iterations=None):
    """NSS on the unit cube for a single-row log-likelihood u -> log L (pure JAX).

    Stops when log Z_live - log Z < termination (status "ok"), or, earlier, when that
    criterion is NaN (status "nonfinite": e.g. every live point at log L = -inf, where it is
    -inf - -inf; the plain comparison is False for NaN and never ended), after
    max_iterations iterations ("capped"; None = no cap) or after max_seconds of sampling
    ("timed_out"; the prior start is not counted). A run that ends with non-finite logZ
    draws is "nonfinite" too. The returned dict has "status"; "timed_out" is kept.

    Other sampling coordinates (mass_marginal.MassMarginal): logprior is their prior
    log-density; prior draws (standard start and prior start) are made in the standard
    d_u-dimensional cube, ranked with loglike_u, and mapped with to_x. With all four left
    at None this is the standard run, unchanged."""
    if logprior is None:
        def logprior(u):
            return jnp.where(jnp.all((u >= 0.0) & (u <= 1.0)), 0.0, -jnp.inf)
    other = to_x is not None
    d_u = d if d_u is None else d_u
    loglike_u = loglike if loglike_u is None else loglike_u

    algo = blackjax.nss(logprior_fn=logprior, loglikelihood_fn=loglike,
                        num_inner_steps=num_inner_steps, num_delete=num_delete)
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    ps = None
    t_c = time.time()
    if n_prior_start is not None and n_prior_start > n_live:
        batch_fn = jax.jit(jax.vmap(loglike_u)).lower(
            jnp.zeros((prior_batch, d_u), jnp.float64)).compile()
        compile_prior_s = time.time() - t_c
        ps = prior_start(batch_fn, d_u, n_live, int(n_prior_start), k0, prior_batch,
                         keep_discards)
        if other:
            ps["live_u"] = to_x(ps["live_u"])
            ps["kept_u"] = to_x(ps["kept_u"]) if len(ps["kept_u"]) else np.empty((0, d))
        malloc_tools.release_free_memory()       # memory only (malloc_tools.py)
        print(f"  prior start: {ps['n_total']} draws in {ps['eval_s']:.1f} s; threshold "
              f"log L {ps['threshold']:.2f}; E[log X] at handover {ps['logx_expected']:.3f}",
              flush=True)
        t_c = time.time()
        live0 = jnp.asarray(ps["live_u"])
    else:
        compile_prior_s = 0.0
        live0 = jax.random.uniform(k0, (n_live, d_u), dtype=jnp.float64)
        if other:
            live0 = jnp.asarray(to_x(np.asarray(live0)))
    state = jax.jit(algo.init)(live0)
    step = jax.jit(algo.step).lower(key, state).compile()
    compile_s = time.time() - t_c + compile_prior_s
    # the compile's ~0.8 GB, freed but kept by glibc (memory only; malloc_tools.py)
    malloc_tools.release_free_memory()

    dead, it, t0 = [], 0, time.time()
    timed_out = False
    status = "ok"
    while True:
        # the same float64 difference and comparison as before, in Python, so NaN is caught
        # (at the start log Z = -inf and the criterion is +inf, which is fine)
        crit = float(state.integrator.logZ_live - state.integrator.logZ)
        if np.isnan(crit):
            status = "nonfinite"
            print(f"  it {it:6d}  termination criterion is NaN (log Z_live "
                  f"{float(state.integrator.logZ_live)}, log Z {float(state.integrator.logZ)}): "
                  "stopping, status nonfinite", flush=True)
            break
        if crit < termination:
            break
        if max_iterations is not None and it >= max_iterations:
            status = "capped"
            print(f"  it {it:6d}  iteration cap reached: stopping, status capped", flush=True)
            break
        key, sub = jax.random.split(key)
        state, info = step(sub, state)
        # keep only the dead particles: info.update_info holds every slice step's brackets
        # for every chain (~2 MB per iteration at 52 steps), which nothing uses, and which
        # pushed a 20-fit layout job past its memory limit (Blanca job 29439310, 1152437)
        dead.append(info._replace(update_info=None))
        it += 1
        if it % log_every == 0:
            print(f"  it {it:6d}  logZ {float(state.integrator.logZ):9.2f}  "
                  f"max logL {float(state.particles.loglikelihood.max()):9.2f}  "
                  f"{time.time() - t0:7.1f} s", flush=True)
        if time.time() - t0 > max_seconds:
            timed_out = True
            status = "timed_out"
            print(f"  it {it:6d}  --max-seconds {max_seconds:g} reached: stopping, status "
                  "timed_out", flush=True)
            break
    jax.block_until_ready(state)
    wall = time.time() - t0
    # the arrays after the loop (log_weights, mass draws, theta) get their own mmaps, which go
    # back to the system when freed, instead of growing the heap (memory only; malloc_tools.py)
    malloc_tools.mmap_large_allocations()

    final = finalise(state, dead, update_info=False)
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
    with np.errstate(invalid="ignore"):            # all -inf gives NaN weights: "nonfinite"
        lw -= lw.max()
        w = np.exp(lw)
        w /= w.sum()
    if not (np.all(np.isfinite(logz_draws)) and np.all(np.isfinite(w))):
        status = "nonfinite"
    return dict(u=u_out, logl=logl_out, w=w, logz_draws=logz_draws,
                iterations=it, wall_s=wall, compile_s=compile_s, timed_out=timed_out,
                status=status, cb=_NoCounts(), **extra)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--catalog", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="jax")
    # defaults = the production setting validated on Blanca (STATUS.md section 5)
    p.add_argument("--n-live", type=int, default=1000)
    p.add_argument("--num-delete", type=int, default=100)
    p.add_argument("--num-inner-steps", type=int, default=None,
                   help="slice steps per new point. Default 32 with --mass-coord marginal "
                        "(chosen 2026-11-11 from 20-seed Blanca tests at 16-78 on 21 galaxies), "
                        "130 with --mass-coord cdf (the earlier standard)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--termination", type=float, default=-3.0)
    p.add_argument("--fix-z", type=float, default=None)
    p.add_argument("--z-window", type=float, nargs=2, default=None)
    p.add_argument("--max-seconds", type=float, default=5400.0,
                   help="stop sampling after this many seconds (prior start not counted); the "
                        "fit gets status timed_out and no postprocessing. Default 5400 (90 min; "
                        "production fits take ~6-9 min on one Blanca core)")
    p.add_argument("--max-iterations", type=int, default=5000,
                   help="stop after this many NSS iterations; the fit gets status capped and no "
                        "postprocessing. Default 5000 (production fits take ~60-400)")
    p.add_argument("--mass-prior", choices=["exact", "tabulated"], default="tabulated",
                   help="tabulated (default): tabulated_mass_prior.py, no mass function per call; "
                        "exact: the mass function evaluated per call")
    p.add_argument("--n-prior-start", type=float, default=1e6,
                   help="start from the best n_live of this many prior draws (see module doc). "
                        "Default 1e6, adopted 2026-11-09 after the Blanca test on 9 MINERVA "
                        "galaxies; 0 (or any value <= n_live) gives the standard start")
    p.add_argument("--keep-discards", type=int, default=10000,
                   help="discarded prior draws whose positions are kept for the posterior")
    p.add_argument("--mass-coord", choices=["cdf", "marginal"], default="marginal",
                   help="marginal (default, adopted 2026-11-11): integrate mass out of the "
                        "likelihood (mass_marginal.py; needs --mass-prior tabulated); cdf: sample "
                        "logmass through its prior inverse CDF (the earlier standard)")
    p.add_argument("--fagn-prior", choices=["flat1_s025", "loguniform"], default="flat1_s025",
                   help="flat1_s025 (default, production since 2026-11-11; nss_priors.py): flat "
                        "in log10 fagn to -1, then a 0.25-dex half-Gaussian tail; loguniform: "
                        "efsps's original log-uniform on [1e-5, 3]")
    p.add_argument("--dust-prior", choices=["none", "mass", "sfr", "mass_sfr"], default="mass",
                   help="soft-ceiling dust2 prior (dust_prior.py) against the low-z dusty-dwarf "
                        "trap: today's flat dust2 prior below a ceiling, a half-Gaussian penalty "
                        "(sigma 0.2) above. Default mass (adopted 2026-11-11): ceiling A_V = 1 mag "
                        "below log M 8, rising to 4 (no change) at log M 9. none = no dust prior. "
                        "Needs --mass-coord marginal")
    p.add_argument("--no-full-npz", action="store_true",
                   help="don't write <id>_<tag>.npz (all weighted samples, ~14 MB); use with "
                        "--save-draws for large test runs where only the draws are copied back")
    p.add_argument("--save-draws", type=int, default=0,
                   help="also write <id>_<tag>_draws.npz: this many equal-weight posterior draws "
                        "of theta (float32), for small transfers from Blanca (0 = off)")
    p.add_argument("--no-postprocess", action="store_true",
                   help="don't write <id>_<tag>_post.h5 (postprocess_nss.py: percentiles, SFH, "
                        "stellar mass, SFRs, model photometry, 1000 draws with their likelihood)")
    p.add_argument("--include-miri", action="store_true",
                   help="fit MIRI bands too. Off by default: MIRI is fitted only when "
                        "explicitly asked for")
    args = p.parse_args()
    args.n_prior_start = int(args.n_prior_start)

    # deterministic input failures: recorded once as status bad_input (exit code 0, so job
    # scripts judge by the file, and pipeline_status.py never retries it); anything else
    # propagates (non-zero exit, no json)
    try:
        phot = read_photometry(args.catalog, [args.id], include_miri=args.include_miri)[0]
    except CatalogRefusal as e:
        if not _per_object_refusal(e):
            raise                      # the catalogue itself is unreadable: not this galaxy's
        return write_bad_input(args, str(e))
    try:
        model = build(phot, args.fix_z, args.z_window, fagn_prior=args.fagn_prior)
    except ValueError as e:
        if str(e) != NO_VALID_PHOTOMETRY:
            raise
        return write_bad_input(args, f"{args.catalog}: object {phot.id}: {e}")
    if args.mass_prior == "tabulated":
        use_tabulated_mass_prior(model)
    if args.num_inner_steps is None:
        args.num_inner_steps = 32 if args.mass_coord == "marginal" else 130
    print(f"[{phot.id}] JAX model: {model.n_dim} free params, {int(model.valid.sum())} bands, "
          f"n_live={args.n_live}, num_delete={args.num_delete}, "
          f"num_inner_steps={args.num_inner_steps}, seed={args.seed}, mass prior {args.mass_prior}, "
          f"prior start {args.n_prior_start}, mass coordinate {args.mass_coord}, "
          f"fagn prior {args.fagn_prior}, dust prior {args.dust_prior}, "
          f"emulator {C.net_for_md5(model.emu.md5)} ({model.emu.md5})",
          flush=True)
    if args.mass_coord == "marginal" and args.mass_prior != "tabulated":
        raise SystemExit("--mass-coord marginal needs --mass-prior tabulated")
    res = run_nss_jax(model, args.n_live, args.num_delete, args.num_inner_steps, args.seed,
                      args.termination, args.max_seconds, n_prior_start=args.n_prior_start,
                      keep_discards=args.keep_discards, mass_coord=args.mass_coord,
                      dust_prior=None if args.dust_prior == "none" else args.dust_prior,
                      max_iterations=args.max_iterations)
    # (with --mass-coord cdf, pass --dust-prior none: the dust prior lives in the mass integral)
    summary, theta = summarise(model, res, phot, args, args.tag)
    status = res["status"]
    if status != "nonfinite" and not (np.isfinite(summary["logZ"]) and np.isfinite(summary["logZ_err"])
                                      and np.isfinite(summary["ess"]) and summary["ess"] > 0):
        status = "nonfinite"
    summary = {"status": status, **summary}
    summary["settings"]["n_prior_start"] = args.n_prior_start
    summary["settings"]["max_iterations"] = args.max_iterations
    summary["settings"]["max_seconds"] = args.max_seconds
    if "prior_start" in res:
        summary["prior_start"] = res["prior_start"]
    summary["likelihood"] = ("pure JAX (jax_model.JaxLogLike)" if args.mass_coord == "cdf" else
                             "pure JAX, mass integrated out (mass_marginal.MassMarginal); "
                             "logl/max_logL are the mass-marginal L', logZ is the same evidence")
    summary["mass_coord"] = args.mass_coord
    summary["compile_s"] = res["compile_s"]
    summary["mass_prior"] = args.mass_prior
    summary["malloc_tuning"] = malloc_tools.ENABLED
    summary["xla_flags"] = os.environ.get("XLA_FLAGS", "")
    summary["fagn_prior"] = args.fagn_prior
    summary["dust_prior"] = args.dust_prior
    if args.dust_prior != "none":
        summary["dust_prior_cfg"] = res["mass_marginal"].dust_cfg
    summary["nss_prior_overrides"] = {k: [v[0], list(v[1])]
                                      for k, v in model.nss_prior_overrides.items()}
    summary["catalog"] = os.path.abspath(args.catalog)
    summary["emulator"] = {"file": os.path.basename(model.emu.path), "md5": model.emu.md5,
                           "net": C.net_for_md5(model.emu.md5)}
    summary["emu_dtype"] = res["emu_dtype"]
    summary["provenance"] = provenance(catalog=args.catalog, emulator=summary["emulator"])
    summary["photometry"] = {k: phot.meta.get(k) for k in ("dropped_bands", "masked_negative",
                                                         "mu_source", "z_spec",
                                                         "excluded_miri", "include_miri")}
    summary["bands_used"] = [b for b, ok in zip(phot.bands, phot.valid) if ok]
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"{phot.id}_{args.tag}")

    def save_full_npz():
        np.savez_compressed(stem + ".npz", u=res["u"], theta=theta, logl=res["logl"], w=res["w"],
                            logz_draws=res["logz_draws"], names=np.array(model.free))

    if not args.no_full_npz and theta is not None:       # theta None: "nonfinite", no weights
        save_full_npz()
    if args.save_draws > 0 and theta is not None:
        w = res["w"] / res["w"].sum()
        pick = np.random.default_rng(args.seed).choice(w.size, args.save_draws, p=w)
        np.savez_compressed(stem + "_draws.npz", draws=theta[pick].astype(np.float32),
                            names=np.array(model.free), logZ=summary["logZ"],
                            logZ_err=summary["logZ_err"])
    # postprocessing BEFORE the json: job scripts take the json as "this fit is finished".
    # Only for status ok: a timed-out, capped or non-finite posterior is incomplete.
    if not args.no_postprocess and status != "ok":
        summary["postprocess"] = {"skipped": f"fit status {status}: the posterior is incomplete"}
        print(f"[{phot.id}] status {status}: postprocessing skipped", flush=True)
    elif not args.no_postprocess:
        from postprocess_nss import postprocess
        try:
            summary["postprocess"] = postprocess(model, phot, theta, res["w"], summary,
                                                 stem + "_post.h5", seed=args.seed)
            print(f"[{phot.id}] postprocessed in {summary['postprocess']['postprocess_s']:.1f} s "
                  f"-> {summary['postprocess']['file']}", flush=True)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[{phot.id}] POSTPROCESSING FAILED: {e!r}\n{tb}", flush=True)
            summary["postprocess"] = {"error": repr(e), "traceback": tb}
            if args.no_full_npz:
                # keep the posterior, so postprocess_nss.py can be rerun on it standalone
                save_full_npz()
                summary["postprocess"]["full_npz_saved_for_rerun"] = True
    summary["process_elapsed_s"] = time.time() - _T_PROCESS0
    summary["peak_rss_mb"] = peak_rss_mb()
    # atomically (tmp + fsync + replace): job scripts take a valid json as "this fit is
    # finished", so a fit killed while writing it must not leave a truncated one behind.
    # Non-finite numbers are written as null (JSON has no NaN).
    write_json_atomic(stem + ".json", summary)
    s = summary

    def num(x, fmt):
        return "n/a" if x is None or not np.isfinite(x) else format(x, fmt)

    print(f"[{phot.id}] done, status {status}: logZ {num(s['logZ'], '.2f')} +- "
          f"{num(s['logZ_err'], '.2f')}, max logL {num(s['max_logL'], '.2f')} (chi2 "
          f"{num(s['chi2_at_max'], '.1f')}), ESS {num(s['ess'], '.0f')}, "
          f"{s['iterations']} iterations, wall {s['wall_s']:.0f} s + compile "
          f"{s['compile_s']:.0f} s")
    if "z_weight_by_bin" in s:
        print(f"[{phot.id}] z weight by bin: {s['z_weight_by_bin']}; z at max logL "
              f"{s['z_at_max_logL']:.3f}")


#: FitModel's message when no band of the object is usable (efsps_photfit/model.py)
NO_VALID_PHOTOMETRY = "no valid photometry for this object."


def _per_object_refusal(e):
    """True for the CatalogRefusals about this one object (id absent, no valid
    magnification), which no retry can fix; False for refusals of the whole catalogue (missing
    error column, wrong unit, unknown file type), which must fail loudly instead."""
    m = str(e)
    return m.endswith("not found.") or "has no valid magnification" in m or "has invalid mu=" in m


def peak_rss_mb():
    """Peak resident set size of this process in MB (ru_maxrss is bytes on macOS, kB on
    Linux). On macOS RSS overstates memory in use (freed pages stay counted)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10


def write_bad_input(args, reason):
    """<out>/<id>_<tag>.json = {"status": "bad_input", "reason", "id", "tag", "catalog",
    "provenance"}, with no logZ; returns from main() (exit code 0)."""
    from nss_efsps import EMULATOR
    os.makedirs(args.out, exist_ok=True)
    gid = _norm_id(args.id)
    out = {"status": "bad_input", "reason": reason, "id": gid, "tag": args.tag,
           "catalog": os.path.abspath(args.catalog),
           "provenance": provenance(catalog=args.catalog, emulator={"path": EMULATOR}),
           "process_elapsed_s": time.time() - _T_PROCESS0}
    path = os.path.join(args.out, f"{gid}_{args.tag}.json")
    write_json_atomic(path, out)
    print(f"[{gid}] BAD INPUT (not retried): {reason} -> {path}", flush=True)


if __name__ == "__main__":
    main()
