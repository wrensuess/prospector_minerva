"""Tests for linear_mass.py: the linear logmass coordinate must give the SAME prior.

1. density vs inverse CDF: integrating exp(log_mass_prior) from lo up to m(u) = mass_ppf(u, z)
   gives back u, at z across every piece of the mass function (both sides of 0.2, 3, 4, 12,
   integers between), and the density integrates to 1;
2. same point: a standard-cube prior draw u and its linear-mass image x = x_from_u(u) give the
   same theta, and every x has a finite log prior;
3. same likelihood: LinearMass.loglike(x) = JaxLogLike(u), by redshift bin;
4. NSS with a prior density: a toy with a steep, tail-like prior on one coordinate (like the
   mass function) and a known evidence, run with blackjax.nss(logprior_fn=density) over seeds,
   against the same toy in inverse-CDF coordinates. logZ must agree with the truth.

  EFSPS_PARENT=... python test_linear_mass.py --catalog minerva_uds_test9.fits --id 1197855
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ZS = [0.01, 0.1, 0.2, 0.2001, 0.5, 1.237, 2.5, 2.9999, 3.0, 3.0001, 3.3, 3.5, 3.9999, 4.0,
      4.0001, 4.5, 5.0, 5.8, 6.0, 7.3, 9.99, 11.5, 11.9999, 12.0, 12.5, 14.9]


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def test_density(lm, results):
    pt = lm.pt
    grid = np.asarray(pt._m_grid)
    fine = np.unique(np.concatenate([grid, np.linspace(grid[0], grid[-1], 200001)]))
    lp = jax.jit(jax.vmap(lm.log_mass_prior, in_axes=(0, None)))
    ppf = jax.jit(jax.vmap(pt.mass_ppf, in_axes=(0, None)))
    u = np.concatenate([np.linspace(0.001, 0.999, 999), 1 - np.geomspace(1e-3, 1e-10, 50)])
    worst_norm, worst_u, worst_rel = 0.0, 0.0, 0.0
    for z in ZS:
        dens = np.exp(np.asarray(lp(jnp.asarray(fine), z)))
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(fine))])
        worst_norm = max(worst_norm, abs(cdf[-1] - 1))
        m = np.asarray(ppf(jnp.asarray(u), z))
        uu = np.interp(m, fine, cdf)
        worst_u = max(worst_u, np.max(np.abs(uu - u)))
        # the tail, relative to 1 - u (where galaxies sit): integrate DOWN from the top, so
        # the test keeps the precision of 1 - u instead of subtracting from 1
        seg = 0.5 * (dens[1:] + dens[:-1]) * np.diff(fine)
        sf = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]]) / cdf[-1]
        tail = u > 0.999
        sfu = np.interp(m[tail], fine, sf)
        worst_rel = max(worst_rel, np.max(np.abs(sfu - (1 - u[tail])) / (1 - u[tail])))
    results.append(check("density integrates to 1", worst_norm < 1e-9, f"max |int - 1| = {worst_norm:.1e}"))
    results.append(check("CDF of density at mass_ppf(u) = u", worst_u < 1e-8,
                         f"max |u' - u| = {worst_u:.1e} over {len(ZS)} z values"))
    results.append(check("same in the tail, relative to 1 - u (down to 1e-10)", worst_rel < 1e-4,
                         f"max relative error of 1 - u = {worst_rel:.1e}"))


def test_same_point(lm, ll, model, results, n=100000, seed=3):
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(n, model.n_dim))
    x = lm.x_from_u(u)
    th_u = np.asarray(ll.pt.batch(jnp.asarray(u)))
    th_x = np.asarray(lm.batch_theta(jnp.asarray(x)))
    d = np.abs(th_u - th_x)
    im = lm.im
    results.append(check("same theta from u and x_from_u(u)", d.max() < 1e-8,
                         f"max |dtheta| = {d.max():.1e} (logmass {d[:, im].max():.1e})"))
    lpx = np.asarray(lm.batch_logprior(jnp.asarray(x)))
    results.append(check("finite log prior at every prior draw", np.all(np.isfinite(lpx)),
                         f"{np.sum(~np.isfinite(lpx))} non-finite of {n}"))
    lu = np.asarray(ll.batch(jnp.asarray(u[:20000])))
    lx = np.asarray(lm.batch_loglike(jnp.asarray(x[:20000])))
    z = th_u[:20000, model.col["zred"]] if "zred" in model.col else np.zeros(20000)
    edges = [0, 0.5, 1, 2, 3, 4, 6, 8, 12, 15.1]
    worst = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        s = (z >= a) & (z < b) & np.isfinite(lu)
        if s.any():
            r = np.max(np.abs(lx[s] - lu[s]) / np.maximum(1.0, np.abs(lu[s])))
            worst = max(worst, r)
            print(f"      z {a:4g}-{b:<4g}: {s.sum():5d} rows, max relative |dlogL| {r:.1e}")
    results.append(check("same log L", worst < 1e-9 and np.array_equal(np.isfinite(lu), np.isfinite(lx)),
                         f"max relative |dlogL| = {worst:.1e}"))


def toy_evidence(results, seeds=30, a=12.0):
    """3-d toy: x0 has prior density a e^{a x0}/(e^a - 1) on [0,1] (steep, mass-function-like
    tail), x1, x2 uniform; likelihood Gaussian at (0.93, 0.5, 0.5)."""
    import blackjax
    mu, sd = jnp.array([0.93, 0.5, 0.5]), jnp.array([0.01, 0.05, 0.05])

    def loglike(x):
        return -0.5 * jnp.sum(((x - mu) / sd) ** 2)

    def inside(x):
        return jnp.all((x >= 0.0) & (x <= 1.0))

    def logprior_density(x):
        lp = np.log(a) + a * x[0] - np.log(np.expm1(a))
        return jnp.where(inside(x), lp, -jnp.inf)

    def ppf0(u):                                  # inverse CDF of the x0 prior
        return jnp.log1p(u * jnp.expm1(a)) / a

    def loglike_u(u):
        x = u.at[0].set(ppf0(u[0]))
        return jnp.where(inside(u), loglike(x), -jnp.inf)

    def logprior_u(u):
        return jnp.where(inside(u), 0.0, -jnp.inf)

    # truth: product of 1-d integrals
    g = np.linspace(0, 1, 2000001)
    p0 = a * np.exp(a * g - np.log(np.expm1(a)))
    I0 = np.trapezoid(p0 * np.exp(-0.5 * ((g - 0.93) / 0.01) ** 2), g)
    I1 = np.trapezoid(np.exp(-0.5 * ((g - 0.5) / 0.05) ** 2), g)
    truth = np.log(I0) + 2 * np.log(I1)

    def run(lp_fn, ll_fn, start, seed):
        algo = blackjax.nss(logprior_fn=lp_fn, loglikelihood_fn=ll_fn, num_inner_steps=15,
                            num_delete=20)
        key = jax.random.PRNGKey(seed)
        key, k0 = jax.random.split(key)
        st = jax.jit(algo.init)(start(k0))
        step = jax.jit(algo.step)
        while not (st.integrator.logZ_live - st.integrator.logZ < -5):
            key, s = jax.random.split(key)
            st, _ = step(s, st)
        return float(jnp.logaddexp(st.integrator.logZ, st.integrator.logZ_live))

    n = 400
    start_x = lambda k: (lambda u: u.at[:, 0].set(ppf0(u[:, 0])))(jax.random.uniform(k, (n, 3)))
    start_u = lambda k: jax.random.uniform(k, (n, 3))
    zx = np.array([run(logprior_density, lambda x: jnp.where(inside(x), loglike(x), -jnp.inf),
                       start_x, s) for s in range(seeds)])
    zu = np.array([run(logprior_u, loglike_u, start_u, s) for s in range(seeds)])
    for lab, z in [("density coordinates", zx), ("inverse-CDF coordinates", zu)]:
        t = (z.mean() - truth) / (z.std(ddof=1) / np.sqrt(seeds))
        results.append(check(f"toy logZ, {lab}", abs(t) < 3,
                             f"{z.mean():.3f} +- {z.std(ddof=1) / np.sqrt(seeds):.3f} vs truth "
                             f"{truth:.3f} (t = {t:+.1f}; run scatter {z.std(ddof=1):.3f})"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="minerva_uds_test9.fits")
    ap.add_argument("--id", default="1197855")
    ap.add_argument("--skip-toy", action="store_true")
    a = ap.parse_args()
    from nss_efsps import build
    from photometry import read_photometry
    from jax_model import JaxLogLike
    from tabulated_mass_prior import use_tabulated_mass_prior
    from linear_mass import LinearMass
    model = build(read_photometry(a.catalog, [a.id])[0])
    use_tabulated_mass_prior(model)
    lm, ll = LinearMass(model), JaxLogLike(model)
    results = []
    print(f"[{a.id}] linear logmass coordinate, bounds {lm.lo}-{lm.hi}")
    test_density(lm, results)
    test_same_point(lm, ll, model, results)
    if not a.skip_toy:
        toy_evidence(results)
    print("\nRESULT:", "PASS" if all(results) else "FAIL")


if __name__ == "__main__":
    main()
