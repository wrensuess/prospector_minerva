"""Where does the wall time of the pure-JAX NSS fit of 58858 go?

Same settings as results/58858_jax_free_L1000 (L=1000, num_delete=100, 52 inner steps,
seed 0), first --n-it iterations.

A. COUNTS. A zero-returning pure_callback counts every batched likelihood call (rows per
   call, rows inside the cube) and the per-chain slice info gives the evaluations each
   chain actually needed: 2 (initial bracket ends) + expansions + shrinks. Lanes computed
   minus lanes needed = lockstep waste (vmapped while loops run every chain until the
   slowest finishes).
B. IN CONTEXT. The real jitted NSS step, timed, with one component evaluated a SECOND
   time on an input that equals the original at run time but that the compiler cannot
   prove equal (u * one, one = where(u[0] > -1e300, 1, 2)); its value enters the result
   as + where(extra > 1e300, 1, 0) = +0. The trajectory is therefore bit-identical
   (checked: final logZ) and (variant - baseline) = that component's cost inside the
   real step. Baseline - full likelihood = everything that is not the likelihood.
C. MICRO. Each piece jitted+vmapped alone at batch 100 (= num_delete, the batch NSS
   hands the likelihood), median of interleaved repeats; CPU time / wall = cores used.
   (Misleading inside the step; B is the measurement that counts.)
Variants fix_twopoint / fix_tables swap in the candidate mass inverse CDFs of
mass_ppf_candidates.py (same likelihood otherwise; final logZ shows whether the
trajectory stayed identical).

ONE CORE. Production fits get one core each. On Linux XLA sizes its CPU thread pool from
the process's CPU affinity; the XLA_FLAGS commonly used to cap it
(--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1) are ignored by jax
0.6.1 and 0.10.2 (tested). So run this in a one-core allocation: blanca/profile_nss.sbatch.
The header prints the affinity (must be ONE PHYSICAL core: 1 CPU, or the 2 hyperthreads of
one core on a hyperthreaded node with --hint=nomultithread) and the process's OS thread
count; every timing line prints CPU/wall (~1.00 on one core without hyperthreads). On macOS there is no affinity, so Mac
numbers are multi-threaded and do not stand in for one core.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nss_efsps import build
from photometry import read_photometry
from jax_model import JaxLogLike
import jax_prior as J

N_LIVE, N_DELETE, N_INNER, SEED = 1000, 100, 52, 0


def cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import subprocess
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip() or None
    except OSError:
        return None


def physical_cores(cpus):
    """Distinct physical cores among these logical CPUs (Linux sysfs; None elsewhere)."""
    if cpus is None:
        return None
    sibs = set()
    for c in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as fh:
                sibs.add(fh.read().strip())
        except OSError:
            return None
    return len(sibs)


def os_threads():
    """OS threads in this process (Linux /proc; None elsewhere)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except OSError:
        return None


def logprior(u):
    return jnp.where(jnp.all((u >= 0.0) & (u <= 1.0)), 0.0, -jnp.inf)


def apply_fix(ll, name):
    import mass_ppf_candidates as MC
    if name == "fix_twopoint":
        ll.pt.mass_ppf = MC.mass_ppf_twopoint(ll.pt)
    elif name == "fix_tables":
        ll.pt.mass_ppf = MC.mass_ppf_tables_twopoint(ll.pt)[0]
    return ll


def make_loglike(ll, extra=None, counter=None):
    pt = ll.pt
    cz, cm = pt.col["zred"], pt.col["logmass"]

    def f(u):
        inside = jnp.all((u >= 0.0) & (u <= 1.0))
        uc = jnp.clip(u, 0.0, 1.0)
        th = pt(uc)
        val = ll.loglike_theta(th)
        out = jnp.where(inside & jnp.isfinite(val), val, -jnp.inf)
        if extra is not None:
            one = jnp.where(u[0] > -1e300, 1.0, 2.0)
            if extra == "loglike":
                e = ll.loglike_theta(pt(uc * one))
            elif extra == "prior":
                e = jnp.sum(pt(uc * one))
            elif extra == "mass":
                e = pt.mass_ppf(uc[cm] * one, th[cz] * one)
            elif extra == "mass_pdf":
                e = jnp.sum(J.mass_function(th[cz] * one, pt._m_grid, pt._m_const_phi,
                                            pt._m_bounds))
            elif extra == "ratios":
                e = jnp.sum(pt.expected_ratios(th[cz] * one, th[cm] * one))
            elif extra == "network":
                e = jnp.sum(ll.f_net(th * one))
            else:
                raise ValueError(extra)
            out = out + jnp.where(e > 1e300, 1.0, 0.0)
        if counter is not None:
            out = out + jax.pure_callback(counter, jax.ShapeDtypeStruct(u.shape[:-1], u.dtype),
                                          u, vmap_method="broadcast_all")
        return out

    return f


class Counter:
    def __init__(self):
        self.calls = self.rows = self.inside = 0

    def __call__(self, u):
        u = np.asarray(u)
        r = u.reshape(-1, u.shape[-1])
        self.calls += 1
        self.rows += r.shape[0]
        self.inside += int(np.all((r >= 0) & (r <= 1), axis=1).sum())
        return np.zeros(u.shape[:-1])


def run(loglike_fn, d, n_it, collect=False):
    algo = blackjax.nss(logprior_fn=logprior, loglikelihood_fn=loglike_fn,
                        num_inner_steps=N_INNER, num_delete=N_DELETE)
    key = jax.random.PRNGKey(SEED)
    key, k0 = jax.random.split(key)
    state = jax.jit(algo.init)(jax.random.uniform(k0, (N_LIVE, d), dtype=jnp.float64))
    t = time.perf_counter()
    step = jax.jit(algo.step).lower(key, state).compile()
    compile_s = time.perf_counter() - t
    infos = []
    jax.block_until_ready(state)
    t0, c0 = time.perf_counter(), time.process_time()
    for _ in range(n_it):
        key, sub = jax.random.split(key)
        state, info = step(sub, state)
        if collect:
            ui = info.update_info
            infos.append((np.asarray(ui.num_expansions), np.asarray(ui.num_shrink)))
    jax.block_until_ready(state)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    return dict(wall=wall, cpu=cpu, compile=compile_s, logZ=float(state.integrator.logZ),
                infos=infos)


def micro(ll, model, rng, n=100, reps=400):
    pt = ll.pt
    cz, cm = pt.col["zred"], pt.col["logmass"]
    dead = np.load(os.path.join(HERE, "results/58858_jax_free_L1000.npz"))
    u = jnp.asarray(dead["u"][rng.choice(dead["u"].shape[0], n, replace=False)])
    th = jnp.asarray(model.prior_transform(np.asarray(u)))
    z, m = th[:, cz], th[:, cm]
    pdf = jax.vmap(lambda zz: J.mass_function(zz, pt._m_grid, pt._m_const_phi,
                                              pt._m_bounds))(z)
    V = jax.vmap
    fns = {
        "full log L(u)": (jax.jit(V(ll)), (u,)),
        "  prior transform": (jax.jit(V(pt)), (u,)),
        "    z inverse CDF": (jax.jit(V(pt.z_ppf)), (u[:, cz],)),
        "    mass inverse CDF (all)": (jax.jit(V(pt.mass_ppf)), (u[:, cm], z)),
        "      mass function on grid": (jax.jit(V(lambda zz: J.mass_function(
            zz, pt._m_grid, pt._m_const_phi, pt._m_bounds))), (z,)),
        "      sequential cumsum": (jax.jit(V(lambda p: J._cum_trapz(p, pt._m_h))), (pdf,)),
        "    expected SFH ratios": (jax.jit(V(pt.expected_ratios)), (z, m)),
        "  network f_net(theta)": (jax.jit(V(ll.f_net)), (th,)),
        "  network + likelihood": (jax.jit(V(ll.loglike_theta)), (th,)),
    }
    for f, a in fns.values():
        jax.block_until_ready(f(*a))
    times = {k: [] for k in fns}
    cpus = {k: [] for k in fns}
    for _ in range(reps):
        for k, (f, a) in fns.items():
            t, c = time.perf_counter(), time.process_time()
            jax.block_until_ready(f(*a))
            times[k].append(time.perf_counter() - t)
            cpus[k].append(time.process_time() - c)
    return {k: dict(us_per_call=1e6 * float(np.median(times[k])),
                    us_per_row=1e6 * float(np.median(times[k])) / n,
                    cores=float(np.sum(cpus[k]) / np.sum(times[k]))) for k in fns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-it", type=int, default=60)
    ap.add_argument("--variants", nargs="*",
                    default=["baseline", "loglike", "prior", "mass", "mass_pdf", "network",
                             "fix_twopoint", "fix_tables", "baseline"])
    ap.add_argument("--skip-micro", action="store_true")
    ap.add_argument("--catalog", default=os.path.join(HERE, "uncover_test.csv"))
    ap.add_argument("--id", default="58858")
    ap.add_argument("--skip-counts", action="store_true",
                    help="skip section A (identical on every machine: 46564 calls in 60 iterations)")
    ap.add_argument("--out", default=os.path.join(HERE, "results/profile_58858.json"))
    args = ap.parse_args()
    phot = read_photometry(args.catalog, [args.id])[0]
    model = build(phot)
    ll = JaxLogLike(model)
    d = model.n_dim
    aff = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    jax.block_until_ready(ll.batch(jnp.full((4, d), 0.5)))      # start XLA's thread pools
    ncore = physical_cores(aff)
    out = {"n_it": args.n_it, "jax": jax.__version__, "blackjax": blackjax.__version__,
           "affinity": aff, "physical_cores": ncore, "cpu_count": os.cpu_count(),
           "cpu_model": cpu_model(),
           "os_threads_after_first_jit": os_threads(), "host": os.uname().nodename,
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "xla_flags": os.environ.get("XLA_FLAGS", ""),
           "emulator": {"file": os.path.basename(model.emu.path), "md5": model.emu.md5}}
    print(f"jax {jax.__version__}, blackjax {blackjax.__version__}; host {out['host']}, "
          f"CPU {out['cpu_model']}; affinity "
          f"{aff if aff is not None else 'n/a (not Linux)'} of {os.cpu_count()} CPUs "
          f"= {ncore if ncore is not None else '?'} physical core(s); "
          f"OS threads after the first jit: {out['os_threads_after_first_jit']}; "
          f"XLA_FLAGS={out['xla_flags']!r}; emulator {out['emulator']['file']} "
          f"({out['emulator']['md5']})", flush=True)
    if aff is not None and ncore != 1:
        print(f"WARNING: this process may run on {len(aff)} CPUs on {ncore} physical cores, "
              "so XLA will spread over them; these timings are NOT one-core timings.",
              flush=True)
    elif aff is not None and len(aff) > 1:
        print(f"note: {len(aff)} hyperthreads of ONE physical core (--hint=nomultithread on a "
              "hyperthreaded node); XLA will use both, as production fits would.", flush=True)

    if not args.skip_micro:
        print(f"== C. micro-benchmarks at batch {N_DELETE} (posterior-region u)", flush=True)
        mc = micro(ll, model, np.random.default_rng(0))
        for k, v in mc.items():
            print(f"  {k:30s} {v['us_per_call']:9.1f} us/call {v['us_per_row']:7.2f} us/row "
                  f"cores {v['cores']:.2f}", flush=True)
        out["micro"] = mc

    if args.skip_counts:
        c = {"calls": 46564 if args.id == "58858" else float("nan")}  # 58858: same everywhere
    else:
        print(f"\n== A. counts over the first {args.n_it} iterations", flush=True)
        cnt = Counter()
        r = run(make_loglike(ll, counter=cnt), d, args.n_it, collect=True)
        exps = np.stack([e for e, _ in r["infos"]])          # (n_it, n_delete, n_inner)
        shr = np.stack([s for _, s in r["infos"]])
        needed = int(np.sum(2 + exps + shr))
        out["counts"] = dict(calls=cnt.calls, lanes=cnt.rows, lanes_inside_cube=cnt.inside,
                             lanes_needed=needed, logZ=r["logZ"],
                             mean_expansions=float(exps.mean()), mean_shrinks=float(shr.mean()),
                             max_expansions=int(exps.max()), max_shrinks=int(shr.max()))
        c = out["counts"]
        print(f"  {cnt.calls} batched calls ({cnt.calls / args.n_it:.0f} per iteration, "
              f"{cnt.calls / (args.n_it * N_INNER):.1f} per inner step), {cnt.rows} lanes")
        print(f"  lanes inside the cube {cnt.inside} ({100 * cnt.inside / cnt.rows:.1f}%)")
        print(f"  lanes the chains needed (2 + expansions + shrinks) {needed} "
              f"({100 * needed / cnt.rows:.1f}% of computed); per chain-step mean expansions "
              f"{c['mean_expansions']:.2f}, shrinks {c['mean_shrinks']:.2f} "
              f"(max {c['max_expansions']}, {c['max_shrinks']})", flush=True)

    print(f"\n== B. in-context timing, {args.n_it} iterations per variant", flush=True)
    out["variants"] = []
    for v in args.variants:
        if v == "tabulated":
            from tabulated_mass_prior import use_tabulated_mass_prior
            r = run(make_loglike(JaxLogLike(use_tabulated_mass_prior(build(phot)))), d, args.n_it)
        elif v.startswith("fix_"):
            r = run(make_loglike(apply_fix(JaxLogLike(model), v)), d, args.n_it)
        else:
            r = run(make_loglike(ll, extra=None if v == "baseline" else v), d, args.n_it)
        out["variants"].append(dict(name=v, wall=r["wall"], cpu=r["cpu"], logZ=r["logZ"],
                                    compile=r["compile"]))
        print(f"  {v:10s} wall {r['wall']:7.1f} s  cpu/wall {r['cpu'] / r['wall']:.2f}  "
              f"logZ {r['logZ']!r}  (compile {r['compile']:.1f} s)", flush=True)
    base = np.mean([x["wall"] for x in out["variants"] if x["name"] == "baseline"])
    print(f"\n  baseline mean {base:.1f} s; cost of each component inside the step:")
    for x in out["variants"]:
        if x["name"].startswith("fix_") or x["name"] == "tabulated":
            print(f"    {x['name']:10s} {x['wall'] - base:+7.1f} s  "
                  f"({base / x['wall']:.2f}x faster than baseline)")
        elif x["name"] != "baseline":
            dlt = x["wall"] - base
            print(f"    {x['name']:10s} {dlt:7.1f} s  ({100 * dlt / base:5.1f}% of baseline), "
                  f"{1e6 * dlt / c['calls']:7.1f} us per batched call")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
