"""Tests for the soft-ceiling dust prior (dust_prior.py, MassMarginal(dust_prior=...)).

1. log10 SFR_100 per unit formed mass: JAX port vs efsps sfh.sfh_quantities (numpy), by z.
2. ln w: JAX vs numpy, all three kinds, random (dust2, M, S) including the ramps' kinks.
3. With a ceiling at the top of the dust2 range (no penalty anywhere), loglike is identical to
   the run without the prior (bitwise), at prior draws.
4. Mass integral with the penalty against brute force (0.0005-dex grid over the whole range), at
   prior draws and at constructed trap-like points (z ~ 0.15-0.3, dust2 1-2.5, low mass): the
   error the penalty ADDS (fast and brute force, each with minus without the penalty); the existing
   integral error without the penalty is reported alongside (STATUS.md section 6, item A1).
5. ln C = ln E_prior[w] for each kind from prior draws (the global constant in logZ), and the
   fraction of prior volume penalised.

  EFSPS_PARENT=... python test_dust_prior.py [--catalog minerva_uds_test9.fits --id 1197855]
Ends "RESULT: PASS" or "RESULT: FAIL ...".
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dust_prior as DPR

FAILS = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="minerva_uds_test9.fits")
    ap.add_argument("--id", default="1197855")
    ap.add_argument("--n-acc", type=int, default=60)
    ap.add_argument("--n-prior", type=int, default=200000)
    a = ap.parse_args()
    rng = np.random.default_rng(3)

    from photometry import read_photometry
    from nss_efsps import build
    from tabulated_mass_prior import use_tabulated_mass_prior
    from mass_marginal import MassMarginal
    model = use_tabulated_mass_prior(build(read_photometry(a.catalog, [a.id])[0]))
    pt_age = MassMarginal(model).pt.age

    # 1. SFR_100 per unit mass
    print("1. log10 SFR_100 / M_formed: JAX vs efsps numpy")
    zz = np.concatenate([10 ** rng.uniform(-2, np.log10(15), 20000), [0.01, 0.1, 0.2, 1, 3, 6, 10, 15]])
    rr = rng.uniform(-4.5, 4.5, (zz.size, 9))
    rr[:2000] = rng.standard_t(2, (2000, 9)) * 0.3
    ref = DPR.log10_sfr100_per_mass_np(zz, rr)
    f = jax.jit(jax.vmap(lambda z, r: DPR.log10_sfr100_per_mass_jax(pt_age(z), r)))
    got = np.asarray(f(jnp.asarray(zz), jnp.asarray(rr)))
    d = np.abs(got - ref)
    worst = []
    for lo, hi in [(0.0, 0.1), (0.1, 0.5), (0.5, 2), (2, 5), (5, 15.1)]:
        s = (zz >= lo) & (zz < hi)
        worst.append(f"z {lo}-{hi}: {d[s].max():.1e}")
    check("SFR_100 per mass", np.all(np.isfinite(got)) and d.max() < 1e-9, "max |d| dex " + "; ".join(worst))

    # 2. ln w
    print("2. ln w: JAX vs numpy")
    n = 200000
    dd = rng.uniform(0, 4, n)
    MM = np.concatenate([rng.uniform(5.5, 12.5, n - 6), [8.0, 10.0, 7.999, 10.001, 9.0, 9.0]])
    SS = np.concatenate([rng.uniform(-5, 3.5, n - 6), [-1.0, 0.0, -1.001, 0.001, -0.5, -0.5]])
    for k in DPR.KINDS:
        r_np = DPR.lnw_np(dd, k, MM, SS)
        r_jx = np.asarray(jax.jit(lambda d_, m_, s_: DPR.lnw_jax(d_, k, m_, s_))(dd, MM, SS))
        check(f"ln w {k}", np.max(np.abs(r_np - r_jx)) < 1e-12 and np.all(r_jx <= 0),
              f"max |d| {np.max(np.abs(r_np - r_jx)):.1e}; penalised fraction of these points {np.mean(r_jx < 0):.3f}")

    # 3. no-penalty ceiling == no prior
    print("3. ceiling at the top of the range == no dust prior")
    mm0 = MassMarginal(model)
    u = rng.uniform(size=(3000, model.n_dim))
    x, m = mm0.x_from_u(u)
    l0 = np.asarray(mm0.batch_loglike(jnp.asarray(x)))
    for k in DPR.KINDS:
        mmk = MassMarginal(model, dust_prior=k, dust_cfg=dict(d_lo=4.0, d_hi=4.0))
        lk = np.asarray(mmk.batch_loglike(jnp.asarray(x)))
        same = np.array_equal(l0, lk) or np.array_equal(np.nan_to_num(l0, neginf=-1e300), np.nan_to_num(lk, neginf=-1e300))
        check(f"no-penalty {k} bitwise", same, f"max |d| {np.nanmax(np.abs(np.where(np.isfinite(l0), l0 - lk, 0))):.1e}")

    # 4. integral accuracy
    print("4. mass integral with the penalty vs brute force")
    grid = jnp.linspace(mm0.m_lo, mm0.m_hi, 11001)
    wlog = jnp.log(mm0._trap_w(grid))
    xs_prior = x[:a.n_acc]
    xt = x[a.n_acc:2 * a.n_acc].copy()                         # trap-like points
    from efsps_photfit.emulator import REF_LOGMASS
    zcol = mm0.x_z
    zq = np.asarray([float(mm0.pt.z_ppf(jnp.asarray(v))) for v in np.linspace(0, 1, 20001)])
    target_z = rng.uniform(0.15, 0.3, a.n_acc)
    xt[:, zcol] = np.interp(target_z, zq, np.linspace(0, 1, 20001))
    kd = [k for k, c, kind, lo, hi, args in mm0.simple if c == model.col["dust2"]][0]
    xt[:, kd] = rng.uniform(1.0, 2.5, a.n_acc) / 4.0           # dust2 uniform on [0, 4]
    for kind in DPR.KINDS:
        mmk = MassMarginal(model, dust_prior=kind)
        f_q = jax.jit(lambda xx, g: mmk.log_q(xx, g))
        f_l = jax.jit(lambda xx, g: mmk.log_like_m(xx, g))
        f_w = jax.jit(lambda xx, g: mmk.dust_lnw(mmk.theta(xx, jnp.asarray(REF_LOGMASS)), mmk._z(xx), g))
        for lab, xs in [("prior draws", xs_prior), ("trap-like", xt)]:
            ll_fast = np.asarray(mmk.batch_loglike(jnp.asarray(xs)))
            ll_nod = np.asarray(mm0.batch_loglike(jnp.asarray(xs)))
            err, err0, dpen = [], [], []
            for xx, lf, l0_ in zip(xs, ll_fast, ll_nod):
                xx = jnp.asarray(xx)
                q = f_q(xx, grid); l = f_l(xx, grid); w = f_w(xx, grid)
                lg = float(jax.scipy.special.logsumexp(q + wlog))
                lb = float(jax.scipy.special.logsumexp(q + l + w + wlog)) - lg
                lb0 = float(jax.scipy.special.logsumexp(q + l + wlog)) - lg
                if np.isfinite(lb) and lb0 > np.max(ll_nod) - 1000:
                    err.append(abs((lf - l0_) - (lb - lb0)))      # error added by the penalty
                    err0.append(abs(l0_ - lb0))                    # existing integral error
                    dpen.append(lf - l0_)
            err, err0, dpen = np.array(err), np.array(err0), np.array(dpen)
            check(f"{kind}, {lab}", len(err) > 0 and err.max() < 1e-3,
                  f"{len(err)} points within 1000 nats; error ADDED by the penalty max {err.max() if len(err) else np.nan:.1e} nats "
                  f"(existing integral error without it, max {err0.max() if len(err0) else np.nan:.1e}); "
                  f"penalty applied (median, min) {np.median(dpen) if len(dpen) else np.nan:.2f}, {dpen.min() if len(dpen) else np.nan:.2f}")

    # 5. ln C
    print("5. ln C = ln E_prior[w] (global constant in logZ), from standard prior draws")
    th = model.prior_transform(rng.uniform(size=(a.n_prior, model.n_dim)))
    zc, mc, dc = th[:, model.col["zred"]], th[:, model.col["logmass"]], th[:, model.col["dust2"]]
    rc = th[:, [model.col[nm] for nm in model.ratio_names]]
    sc = mc + DPR.log10_sfr100_per_mass_np(zc, rc)
    for k in DPR.KINDS:
        w = np.exp(DPR.lnw_np(dc, k, mc, sc))
        se = w.std() / np.sqrt(w.size) / w.mean()
        print(f"   {k:9s} ln C = {np.log(w.mean()):+.4f} (+-{se:.4f}); prior volume penalised > 2 nats "
              f"{np.mean(w < np.exp(-2)):.3f}, any {np.mean(w < 1):.3f}")
    print("\nRESULT:", "PASS" if not FAILS else "FAIL " + ", ".join(FAILS))


if __name__ == "__main__":
    main()
