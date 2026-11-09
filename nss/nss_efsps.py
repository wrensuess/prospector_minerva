"""Prototype: fit one galaxy with the efsps-photfit model, sampled by blackjax nested
slice sampling (NSS; Yallup, Kroupa & Handley 2026, arXiv:2601.23252).

The efsps numpy code is used UNCHANGED through jax.pure_callback. blackjax samples the
unit cube u with a uniform prior and log-likelihood

    L(u) = FitModel.loglike(FitModel.prior_transform(u)),

so the Prospector-beta prior is exactly efsps's. Every batched call into numpy is
counted, so the run reports the true number of model evaluations (including the extra
rows that vmapped slice loops evaluate while waiting for the slowest chain).

Outputs in --out: <id>_<tag>.npz (dead points, weights, logZ draws) and
<id>_<tag>.json (summary).
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

sys.path.insert(0, os.environ.get("EFSPS_PARENT", "/Users/wren/Projects/MINERVA/efsps-photfit"))
from efsps_photfit import config as C
from efsps_photfit.catalog import read_catalog
from efsps_photfit.emulator import PhotEmulator, EmulatorRefusal
from efsps_photfit.fit import weighted_quantiles
from efsps_photfit.model import FitModel, table_for_run

EMULATOR = os.path.join(os.environ.get("EFSPS_PARENT", "/Users/wren/Projects/MINERVA/efsps-photfit"),
                        "efsps_photfit/data/v4_n22000000_w256_d5_s0_parrot.npy")


class CallbackLikelihood:
    """Unit-cube rows -> efsps log-likelihood, in numpy batches. Counts the work."""

    def __init__(self, model):
        self.model = model
        self.n_calls = 0
        self.n_rows = 0          # rows handed to numpy (inside the cube)
        self.n_rows_outside = 0  # rows outside the cube: -inf without evaluation
        self.seconds = 0.0

    def _one(self, theta_row):
        try:
            return float(self.model.loglike(theta_row[None, :])[0])
        except EmulatorRefusal:
            return -np.inf

    def __call__(self, u):
        t0 = time.perf_counter()
        u = np.asarray(u, dtype=float)
        shape = u.shape[:-1]
        u = u.reshape(-1, u.shape[-1])
        out = np.full(u.shape[0], -np.inf)
        inside = np.all((u >= 0.0) & (u <= 1.0), axis=1)
        if inside.any():
            theta = self.model.prior_transform(u[inside])
            try:
                ll = self.model.loglike(theta)
            except EmulatorRefusal:
                ll = np.array([self._one(t) for t in theta])
            out[inside] = np.where(np.isfinite(ll), ll, -np.inf)
        self.n_calls += 1
        self.n_rows += int(inside.sum())
        self.n_rows_outside += int((~inside).sum())
        self.seconds += time.perf_counter() - t0
        return out.reshape(shape)


def build(phot, fix_z=None, z_window=None):
    emu = PhotEmulator(EMULATOR, bands=phot.bands, expected_md5=list(C.KNOWN_EMULATOR_MD5))
    table = table_for_run(C.PARAMETER_TABLE, fix_z, z_window)
    return FitModel(emu, phot.bands, phot.flux, phot.err, phot.mu, table=table)


def run_nss(model, n_live, num_delete, num_inner_steps, seed, termination=-3.0,
            max_seconds=np.inf, log_every=50):
    cb = CallbackLikelihood(model)
    d = model.n_dim

    def loglike(u):
        return jax.pure_callback(cb, jax.ShapeDtypeStruct(u.shape[:-1], jnp.float64), u,
                                 vmap_method="broadcast_all")

    def logprior(u):
        return jnp.where(jnp.all((u >= 0.0) & (u <= 1.0)), 0.0, -jnp.inf)

    algo = blackjax.nss(logprior_fn=logprior, loglikelihood_fn=loglike,
                        num_inner_steps=num_inner_steps, num_delete=num_delete)
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    state = algo.init(jax.random.uniform(k0, (n_live, d), dtype=jnp.float64))
    step = jax.jit(algo.step)

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
                  f"rows {cb.n_rows:>10d}  {time.time() - t0:7.0f} s", flush=True)
        if time.time() - t0 > max_seconds:
            timed_out = True
            break
    wall = time.time() - t0

    final = finalise(state, dead)
    key, kw = jax.random.split(key)
    logw = np.asarray(log_weights(kw, final, shape=100))         # (N, 100)
    logz_draws = np.asarray(jax.scipy.special.logsumexp(logw, axis=0))
    lw = logw.mean(axis=1)
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    u = np.asarray(final.particles.position)
    logl = np.asarray(final.particles.loglikelihood)
    return dict(u=u, logl=logl, w=w, logz_draws=logz_draws, iterations=it, wall_s=wall,
                timed_out=timed_out, cb=cb)


def summarise(model, res, phot, args, tag):
    theta = model.prior_transform(res["u"])
    w = res["w"]
    keep = w > 1e-8 * w.max()
    names = model.free
    ess = 1.0 / np.sum(w ** 2)
    out = {
        "id": phot.id, "tag": tag, "n_dim": model.n_dim, "n_bands_used": int(model.valid.sum()),
        "z_catalog": phot.z_catalog, "mu": phot.mu,
        "settings": {"n_live": args.n_live, "num_delete": args.num_delete,
                     "num_inner_steps": args.num_inner_steps, "seed": args.seed,
                     "fix_z": args.fix_z, "z_window": args.z_window,
                     "termination": args.termination},
        "logZ": float(np.mean(res["logz_draws"])), "logZ_err": float(np.std(res["logz_draws"])),
        "max_logL": float(res["logl"].max()),
        "chi2_at_max": None,
        "ess": float(ess), "n_dead": int(len(w)), "iterations": res["iterations"],
        "wall_s": res["wall_s"], "timed_out": res["timed_out"],
        "callback_calls": res["cb"].n_calls, "model_rows_evaluated": res["cb"].n_rows,
        "rows_outside_cube": res["cb"].n_rows_outside,
        "seconds_in_numpy_model": res["cb"].seconds,
        "percentiles": {n: weighted_quantiles(theta[keep, i], w[keep]) for i, n in enumerate(names)},
    }
    best = theta[np.argmax(res["logl"])]
    fm, sig_emu, _ = model.model_fluxes(best[None, :])
    v = model.valid
    var = model._var_obs + sig_emu[0, v] ** 2
    out["chi2_at_max"] = float(np.sum((model._fv - fm[0, v]) ** 2 / var))
    if "zred" in model.col:
        z = theta[:, model.col["zred"]]
        edges = np.array([0.0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15.1])
        frac = np.array([w[(z >= a) & (z < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
        out["z_weight_by_bin"] = {f"{a:g}-{b:g}": float(f) for a, b, f in
                                  zip(edges[:-1], edges[1:], frac) if f > 1e-4}
        out["z_at_max_logL"] = float(best[model.col["zred"]])
    return out, theta


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--catalog", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="nss")
    p.add_argument("--n-live", type=int, default=500)
    p.add_argument("--num-delete", type=int, default=50)
    p.add_argument("--num-inner-steps", type=int, default=None,
                   help="slice steps per new point (default 2 x n_dim, blackjax's guidance)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--termination", type=float, default=-3.0,
                   help="stop when log(Z_live) - log(Z) < this")
    p.add_argument("--fix-z", type=float, default=None)
    p.add_argument("--z-window", type=float, nargs=2, default=None)
    p.add_argument("--max-seconds", type=float, default=np.inf)
    args = p.parse_args()

    phot = read_catalog(args.catalog, ids=[args.id])[0]
    model = build(phot, args.fix_z, args.z_window)
    if args.num_inner_steps is None:
        args.num_inner_steps = 2 * model.n_dim
    print(f"[{phot.id}] {model.n_dim} free params, {int(model.valid.sum())} bands, mu={phot.mu:.3f}, "
          f"n_live={args.n_live}, num_delete={args.num_delete}, "
          f"num_inner_steps={args.num_inner_steps}, z_window={args.z_window}, fix_z={args.fix_z}",
          flush=True)
    res = run_nss(model, args.n_live, args.num_delete, args.num_inner_steps, args.seed,
                  args.termination, args.max_seconds)
    summary, theta = summarise(model, res, phot, args, args.tag)
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"{phot.id}_{args.tag}")
    np.savez_compressed(stem + ".npz", u=res["u"], theta=theta, logl=res["logl"], w=res["w"],
                        logz_draws=res["logz_draws"], names=np.array(model.free))
    with open(stem + ".json", "w") as fh:
        json.dump(summary, fh, indent=1)
    s = summary
    print(f"[{phot.id}] done: logZ {s['logZ']:.2f} +- {s['logZ_err']:.2f}, max logL "
          f"{s['max_logL']:.2f} (chi2 {s['chi2_at_max']:.1f}), ESS {s['ess']:.0f}, "
          f"{s['model_rows_evaluated']} model rows in {s['callback_calls']} calls, "
          f"wall {s['wall_s']:.0f} s ({s['seconds_in_numpy_model']:.0f} s in numpy model)")
    if "z_weight_by_bin" in s:
        print(f"[{phot.id}] z weight by bin: {s['z_weight_by_bin']}; z at max logL "
              f"{s['z_at_max_logL']:.3f}")


if __name__ == "__main__":
    main()
