"""Tests for mass_marginal.py (mass integrated out of the likelihood; the same prior).

1. same point: a standard 26-d prior draw u -> (x, m) -> theta(x, m) equals the standard
   transform theta(u), including the clipped SFH ratios;
2. densities: p(Z | m) and the ratio offset density match 1 / (d ppf / du) of the standard
   transforms (finite differences), and each integrates to 1;
3. integral accuracy: log g(x) and log L'(x) against a brute-force 0.0005-dex mass grid over
   the whole prior range, at prior draws and at late live points of a real run;
4. cost: time per likelihood row, against the standard 26-d likelihood.

  EFSPS_PARENT=... python test_mass_marginal.py --catalog minerva_uds_test9.fits --id 1197855 \
      [--live <diag_*.npz from diagnose_nss.py>]
"""
import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def brute(mm, x, n=11001, chunk=1101):
    """log g and log integral L q on a uniform n-point grid (trapezoid)."""
    grid = jnp.linspace(mm.m_lo, mm.m_hi, n)
    w = jnp.log(mm._trap_w(grid))
    lg, li, lqpk = [], [], []
    f_q = jax.jit(lambda xx, g: mm.log_q(xx, g))
    f_l = jax.jit(lambda xx, g: mm.log_like_m(xx, g))
    for xx in x:
        xx = jnp.asarray(xx)
        q = jnp.concatenate([f_q(xx, grid[i:i + chunk]) for i in range(0, n, chunk)])
        l = jnp.concatenate([f_l(xx, grid[i:i + chunk]) for i in range(0, n, chunk)])
        lg.append(float(jax.scipy.special.logsumexp(q + w)))
        li.append(float(jax.scipy.special.logsumexp(q + l + w)))
        k = int(jnp.nanargmax(q + l))
        lqpk.append(float(q[k] - jnp.max(q)))       # prior density at the peak, vs its own max
    lg, li = np.array(lg), np.array(li)
    brute.log_q_at_peak = np.array(lqpk)
    return lg, li - lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="minerva_uds_test9.fits")
    ap.add_argument("--id", default="1197855")
    ap.add_argument("--live", default=None, help="diag npz: late live points to test at")
    ap.add_argument("--n-acc", type=int, default=100)
    a = ap.parse_args()
    from nss_efsps import build
    from photometry import read_photometry
    from tabulated_mass_prior import use_tabulated_mass_prior
    from mass_marginal import MassMarginal
    from jax_prior import truncnorm_ppf, truncated_t2_ppf
    model = build(read_photometry(a.catalog, [a.id])[0])
    use_tabulated_mass_prior(model)
    mm = MassMarginal(model)
    pt = mm.pt
    res = []
    print(f"[{a.id}] mass integrated out: {mm.n_dim} sampled parameters; coarse grid "
          f"{mm.mc.size} x {float(mm.mc[1] - mm.mc[0]):.4f} dex, fine grid {mm.n_fine}")

    # 1. same point
    rng = np.random.default_rng(5)
    u = rng.uniform(size=(50000, model.n_dim))
    x, m = mm.x_from_u(u)
    th_u = np.asarray(pt.batch(jnp.asarray(u)))
    th_x = np.asarray(mm.batch_theta(jnp.asarray(x), jnp.asarray(m)))
    d = np.abs(th_u - th_x)
    rc = np.asarray(mm.c_r)
    other = np.setdiff1d(np.arange(model.n_dim), rc)
    res.append(check("same theta from u and (x, m)", d[:, other].max() < 1e-9 and d[:, rc].max() < 1e-5,
                     f"max |dtheta| = {d[:, other].max():.1e} (SFH ratios {d[:, rc].max():.1e}: the "
                     f"expected-ratio table); x inside the cube: {np.all((x >= 0) & (x <= 1))}"))
    lp = np.asarray(mm.batch_logprior(jnp.asarray(x[:5000])))
    res.append(check("finite marginal prior at prior draws", np.all(np.isfinite(lp)),
                     f"{np.sum(~np.isfinite(lp))} of 5000 non-finite"))

    # 1b. the tabulated expected SFH ratios against the current (exact) chain
    zz = np.concatenate([np.asarray(pt.batch(jnp.asarray(u[:2000])))[:, model.col["zred"]],
                         [0.01, 0.15, 0.1501, 1.0, 3.0, 6.0, 9.99, 10.0, 12.0, 15.0]])
    mgrid = jnp.concatenate([jnp.linspace(mm.m_lo, mm.m_hi, 551),
                             jnp.array([8.999, 9.0, 9.001, 11.999, 12.0, 12.001])])
    ex = jax.jit(jax.vmap(lambda z0: jax.vmap(lambda m0: pt.expected_ratios(z0, m0)[mm.r_j])(mgrid)))
    tb = jax.jit(jax.vmap(lambda z0: mm.exp_ratios_tab(mm.log10_age(z0), mgrid)))
    de = np.abs(np.asarray(ex(jnp.asarray(zz))) - np.asarray(tb(jnp.asarray(zz))))
    res.append(check("tabulated expected ratios vs the current chain", de.max() < 1e-5,
                     f"max |d| = {de.max():.1e} dex over {zz.size} z x {mgrid.size} masses "
                     f"(incl. the kinks at logM 9 and 12 and the z_eff clamps)"))

    # 2. densities vs 1 / (d ppf / du)
    us = np.linspace(0.002, 0.998, 499)
    worst_z, worst_t, worst_nz, worst_nt = 0.0, 0.0, 0.0, 0.0
    for mval in (8.0, 9.3, 10.5, 11.7):
        loc = float(jnp.interp(mval, pt._mm_x, pt._mm_p50))
        sig = float(jnp.interp(mval, pt._mm_x, pt._mm_p84) - jnp.interp(mval, pt._mm_x, pt._mm_p16))
        lo, hi = mm.zsol_lo, mm.zsol_hi
        h = 1e-6
        zz = np.asarray(truncnorm_ppf(jnp.asarray(us), loc, sig, lo, hi))
        dz = (np.asarray(truncnorm_ppf(jnp.asarray(us + h), loc, sig, lo, hi)) -
              np.asarray(truncnorm_ppf(jnp.asarray(us - h), loc, sig, lo, hi))) / (2 * h)
        a_ = jax.scipy.special.ndtr((lo - loc) / sig); b_ = jax.scipy.special.ndtr((hi - loc) / sig)
        pdf = np.exp(-0.5 * ((zz - loc) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig * float(b_ - a_))
        worst_z = max(worst_z, np.max(np.abs(pdf * dz - 1)))
        g = np.linspace(lo, hi, 200001)
        pg = np.exp(-0.5 * ((g - loc) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig * float(b_ - a_))
        worst_nz = max(worst_nz, abs(np.trapezoid(pg, g) - 1))
    oo = np.asarray(truncated_t2_ppf(jnp.asarray(us), mm.hw, mm.ts))
    do = (np.asarray(truncated_t2_ppf(jnp.asarray(us + h), mm.hw, mm.ts)) -
          np.asarray(truncated_t2_ppf(jnp.asarray(us - h), mm.hw, mm.ts))) / (2 * h)
    pdf_t = np.exp(mm._log_t2_norm - 1.5 * np.log(2 * mm.ts ** 2 + oo ** 2))
    worst_t = np.max(np.abs(pdf_t * do - 1))
    g = np.linspace(-mm.hw, mm.hw, 400001)
    worst_nt = abs(np.trapezoid(np.exp(mm._log_t2_norm - 1.5 * np.log(2 * mm.ts ** 2 + g ** 2)), g) - 1)
    res.append(check("p(Z | m) = 1 / (dZ/du)", worst_z < 1e-5, f"max relative error {worst_z:.1e}"))
    res.append(check("p(Z | m) integrates to 1", worst_nz < 1e-8, f"{worst_nz:.1e}"))
    res.append(check("ratio offset density = 1 / (do/du)", worst_t < 1e-5,
                     f"max relative error {worst_t:.1e}"))
    res.append(check("ratio offset density integrates to 1", worst_nt < 1e-8, f"{worst_nt:.1e}"))

    # 3. integral accuracy against brute force
    sets = [("prior draws", x[:a.n_acc])]
    if a.live:
        r = np.load(a.live)
        it = int(r["iterations"])
        U = np.concatenate([r["start_pos"][j] for j in (int(0.5 * it), int(0.8 * it), it - 2)])
        U = U[rng.choice(len(U), min(a.n_acc, len(U)), replace=False)].astype(float)
        sets.append(("late live points", mm.x_from_u(U)[0]))
    for lab, xs in sets:
        lp, ll = (np.asarray(v) for v in mm.batch_both(jnp.asarray(xs)))
        bg, bl = brute(mm, xs)
        # accuracy where it can matter: within 1000 nats of the best fit (a run's threshold
        # passes far above anything lower in its first iteration), and not deep in the mass
        # prior's tail (prior density at the peak below e^-100 of its maximum), where the
        # tabulated prior's piecewise-linear density zigzags on its 0.01-dex nodes and
        # neither this scheme nor the 0.0005-dex brute force is converged
        tail = brute.log_q_at_peak < -100
        rel = np.isfinite(bl) & (bl > np.nanmax(bl) - 1000) & ~tail
        if tail.any():
            dt = np.abs(ll[tail] - bl[tail])
            dt = dt[np.isfinite(dt)]
            print(f"      {tail.sum()} points with the mass prior below e^-100 of its peak at the "
                  f"likelihood peak (deep tail): max |d| {dt.max() if dt.size else 0:.2f} nats "
                  f"(reported, not tested)")
        dg, dl = np.abs(lp - bg), np.abs(ll[rel] - bl[rel])
        res.append(check(f"log g vs brute force ({lab})", dg.max() < 1e-3,
                         f"max |d| {dg.max():.1e} nats, median {np.median(dg):.1e}"))
        res.append(check(f"log L' vs brute force ({lab}, within 1000 nats of the best)",
                         dl.max() < 1e-2 and np.median(dl) < 1e-3,
                         f"{rel.sum()} points: max |d| {dl.max():.1e} nats, median "
                         f"{np.median(dl):.1e}; log L' range {np.min(bl[rel]):.1f}..{np.max(bl[rel]):.1f}"))
        far = ~rel & np.isfinite(bl)
        if far.any():
            fin = np.isfinite(ll[far])
            print(f"      {far.sum()} points more than 1000 nats below the best (brute log L' "
                  f"{bl[far].min():.0f}..{bl[far].max():.0f}): fast value finite for {fin.sum()}, "
                  f"max |d| there {np.max(np.abs(ll[far][fin] - bl[far][fin])) if fin.any() else 0:.1e}")
        bad = ~np.isfinite(ll) & np.isfinite(bl)
        res.append(check(f"fast -inf only far below the best ({lab})",
                         not bad.any() or bl[bad].max() < np.nanmax(bl) - 1000,
                         f"{bad.sum()} points with fast -inf, brute log L' up to "
                         f"{bl[bad].max() if bad.any() else float('nan'):.0f}"))

    # 4. cost per row
    from jax_model import JaxLogLike
    ll26 = JaxLogLike(model)
    for nb in (100, 1000):
        ub, xb = jnp.asarray(u[:nb]), jnp.asarray(x[:nb])
        f26 = jax.jit(jax.vmap(ll26)); f25 = mm.batch_both
        jax.block_until_ready(f26(ub)); jax.block_until_ready(f25(xb))
        t0 = time.time(); [jax.block_until_ready(f26(ub)) for _ in range(5)]; t26 = (time.time() - t0) / 5 / nb
        t0 = time.time(); [jax.block_until_ready(f25(xb)) for _ in range(5)]; t25 = (time.time() - t0) / 5 / nb
        print(f"      batch {nb:5d}: standard {t26 * 1e6:7.1f} us/row, mass-marginal (prior + like) "
              f"{t25 * 1e6:7.1f} us/row, ratio {t25 / t26:.1f}x  (multi-threaded Mac)")
    print("\nRESULT:", "PASS" if all(res) else "FAIL")


if __name__ == "__main__":
    main()
