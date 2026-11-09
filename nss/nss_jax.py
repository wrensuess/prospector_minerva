"""nss_efsps.py with the pure-JAX likelihood (jax_model.JaxLogLike) in place of the numpy
code behind jax.pure_callback. Same sampler, same loop, same summary (nss_efsps.summarise),
so the outputs compare directly with the numpy-callback runs.

The model-row counters of the callback runs are not available inside jit; the summary
reports wall time, and compile time separately.
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


def run_nss_jax(model, n_live, num_delete, num_inner_steps, seed, termination=-3.0,
                max_seconds=np.inf, log_every=50):
    d = model.n_dim
    loglike = JaxLogLike(model)

    def logprior(u):
        return jnp.where(jnp.all((u >= 0.0) & (u <= 1.0)), 0.0, -jnp.inf)

    algo = blackjax.nss(logprior_fn=logprior, loglikelihood_fn=loglike,
                        num_inner_steps=num_inner_steps, num_delete=num_delete)
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    t_c = time.time()
    state = jax.jit(algo.init)(jax.random.uniform(k0, (n_live, d), dtype=jnp.float64))
    step = jax.jit(algo.step).lower(key, state).compile()
    compile_s = time.time() - t_c

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
    logz_draws = np.asarray(jax.scipy.special.logsumexp(logw, axis=0))
    lw = logw.mean(axis=1)
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    return dict(u=np.asarray(final.particles.position),
                logl=np.asarray(final.particles.loglikelihood), w=w, logz_draws=logz_draws,
                iterations=it, wall_s=wall, compile_s=compile_s, timed_out=timed_out,
                cb=_NoCounts())


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
    args = p.parse_args()

    phot = read_photometry(args.catalog, [args.id])[0]
    model = build(phot, args.fix_z, args.z_window)
    if args.mass_prior == "tabulated":
        use_tabulated_mass_prior(model)
    if args.num_inner_steps is None:
        args.num_inner_steps = 2 * model.n_dim
    print(f"[{phot.id}] JAX model: {model.n_dim} free params, {int(model.valid.sum())} bands, "
          f"n_live={args.n_live}, num_delete={args.num_delete}, "
          f"num_inner_steps={args.num_inner_steps}, seed={args.seed}, mass prior {args.mass_prior}, "
          f"emulator {C.net_for_md5(model.emu.md5)} ({model.emu.md5})",
          flush=True)
    res = run_nss_jax(model, args.n_live, args.num_delete, args.num_inner_steps, args.seed,
                      args.termination, args.max_seconds)
    summary, theta = summarise(model, res, phot, args, args.tag)
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
