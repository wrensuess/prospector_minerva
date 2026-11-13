"""Tests for mass_marginal.py (mass integrated out of the likelihood; the same prior).

1. same point: a standard 26-d prior draw u -> (x, m) -> theta(x, m) equals the standard
   transform theta(u), including the clipped SFH ratios;
2. densities: p(Z | m) and the ratio offset density match 1 / (d ppf / du) of the standard
   transforms (finite differences), and each integrates to 1;
3. integral accuracy: log g(x) and log L'(x) against a brute-force 0.0005-dex mass grid over
   the whole prior range, at prior draws and at late live points of a real run; and at prior
   draws where L q peaks at a prior bound (7.0 or 12.5), against the brute force refined
   geometrically toward both bounds; and for the same galaxy made faint (errors inflated to
   total S/N ~5 and ~3), at points where its posterior lies, including the masses drawn by
   sample_logmass against the brute-force mass distribution at fixed x;
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


def brute(mm, x, n=11001, chunk=1101, edge_refine=False):
    """log g and log integral L q on a uniform n-point grid (trapezoid). edge_refine adds 4000
    geometric points within 0.05 dex of each bound (down to 1e-12 dex): a peak at a bound can be
    far narrower than the uniform step."""
    grid = jnp.linspace(mm.m_lo, mm.m_hi, n)
    w = jnp.log(mm._trap_w(grid))
    if edge_refine:
        geo = np.logspace(-12, np.log10(0.05), 4000)
        g = np.unique(np.concatenate([np.asarray(grid), mm.m_lo + geo, mm.m_hi - geo]))
        h = np.diff(g)
        wn = np.zeros(g.size)
        wn[:-1] += h / 2
        wn[1:] += h / 2
        grid, w, n = jnp.asarray(g), jnp.log(jnp.asarray(wn)), g.size
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


def brute_cdf(mm, xx, n=11001, chunk=1101):
    """CDF of L q in m at one x, on the uniform grid refined toward both bounds (as brute)."""
    geo = np.logspace(-12, np.log10(0.05), 4000)
    g = np.unique(np.concatenate([np.linspace(mm.m_lo, mm.m_hi, n), mm.m_lo + geo, mm.m_hi - geo]))
    f_q = jax.jit(lambda x_, g_: mm.log_q(x_, g_))
    f_l = jax.jit(lambda x_, g_: mm.log_like_m(x_, g_))
    xx = jnp.asarray(xx)
    t = np.concatenate([np.asarray(f_q(xx, jnp.asarray(g[i:i + chunk])) + f_l(xx, jnp.asarray(g[i:i + chunk])))
                        for i in range(0, g.size, chunk)])
    p = np.exp(t - np.nanmax(t))
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(g))])
    return g, c / c[-1]


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
        # prior's tail (prior density at the peak below e^-100 of its maximum). There the
        # prior's log slope is steep, and the 41-point fine grid aliases against the kinks of
        # its piecewise-linear density (every 0.01 dex): up to ~0.3 nats. The brute force, with
        # 20 points per prior cell, is converged; no real posterior is affected (STATUS.md
        # section 6, item A1)
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

    # 3b. L q peaking at a prior bound because the likelihood alone peaks beyond it (L and L q
    # both peak at the same end of the coarse grid): the fine grid starts at the bound and spans
    # EDGE_NATS of log(L q) from its analytic slope there. (Not tested here: L q peaking at m_lo
    # while L peaks far above, beyond where q ends, so the integral comes from the low-mass
    # plateau where the model flux is negligible. The fine window misses it, and the integral
    # is -inf; all such points seen so far are > 1000 nats below the best: STATUS.md item A1.)
    xe = x[:20000]
    f_c = jax.jit(jax.vmap(lambda xx: jnp.stack([mm.log_q(xx, mm.mc), mm.log_like_m(xx, mm.mc)])))
    tc = np.concatenate([np.asarray(f_c(jnp.asarray(xe[i:i + 2000]))) for i in range(0, len(xe), 2000)])
    tl = tc[:, 1]
    tc = tc[:, 0] + tc[:, 1]
    tc = np.where(np.isfinite(tc), tc, -np.inf)
    kmax, kl = np.argmax(tc, axis=1), np.argmax(np.where(np.isfinite(tl), tl, -np.inf), axis=1)
    top = tc[np.arange(len(tc)), kmax]
    pinned = np.isfinite(top) & (kmax == kl) & ((kmax == 0) | (kmax == mm.mc.size - 1))
    xs = xe[pinned][:a.n_acc]
    if len(xs):
        ll = np.asarray(mm.batch_loglike(jnp.asarray(xs)))
        _, bl = brute(mm, xs, edge_refine=True)
        d = np.abs(ll - bl)
        # as in 3: the deep mass-prior tail (at 12.5 the cutoff falls ~9 nats per 0.01-dex prior
        # cell) is reported, not tested
        tail = brute.log_q_at_peak < -100
        if tail.any():
            dt = d[tail]
            print(f"      {tail.sum()} of these in the deep mass-prior tail: max |d| "
                  f"{dt[np.isfinite(dt)].max() if np.isfinite(dt).any() else np.inf:.2f} nats, median "
                  f"{np.median(dt):.1e} (reported, not tested)")
        t = ~tail
        res.append(check("log L' vs refined brute force (L q peaking at a prior bound)",
                         t.any() and np.all(np.isfinite(ll[t])) and d[t].max() < 2e-2,
                         f"{t.sum()} points ({np.sum(t & (kmax[pinned][:a.n_acc] == 0))} at m_lo) of "
                         f"{pinned.sum()} in {len(xe)} prior draws: max |d| "
                         f"{d[t].max() if t.any() else np.nan:.1e} nats, median "
                         f"{np.median(d[t]) if t.any() else np.nan:.1e}; log L' "
                         f"{bl[t].min() if t.any() else np.nan:.0f}..{bl[t].max() if t.any() else np.nan:.0f}"))
    else:
        print(f"      no prior draw of {len(xe)} has L q peaking at a prior bound (not tested)")

    # 3c. faint galaxies: the same photometry, errors inflated to total S/N ~5 and ~3 (total
    # S/N = sqrt(sum f^2 / var), with the error floor). As m drops, the model flux goes to zero
    # and L levels off only ~(S/N)^2/2 nats below its peak, while the mass function rises: much
    # of the integral can lie below the fine window. It is covered by the coarse grid. Tested at
    # points where the faint galaxy's posterior lies (prior draws resampled by L').
    v = model.valid
    fv, ev = model.f_obs[v], model.e_obs[v]
    floor = float(model.lk["flux_floor_frac"])
    snr_k = lambda k: float(np.sqrt(np.sum(fv ** 2 / ((k * ev) ** 2 + (floor * fv) ** 2))))
    xt = x[:50000]
    for target in (5.0, 3.0):
        k_lo, k_hi = 1.0, 1e8
        for _ in range(200):
            k_mid = np.sqrt(k_lo * k_hi)
            k_lo, k_hi = (k_mid, k_hi) if snr_k(k_mid) > target else (k_lo, k_mid)
        mod_f = build(read_photometry(a.catalog, [a.id])[0])
        use_tabulated_mass_prior(mod_f)
        mod_f._var_obs = (k_hi * ev) ** 2 + (floor * fv) ** 2
        mmf = MassMarginal(mod_f)
        llf = np.asarray(mmf.batch_loglike(jnp.asarray(xt)))
        w = np.exp(np.where(np.isfinite(llf), llf - np.nanmax(llf), -np.inf))
        w /= w.sum()
        sel = np.unique(rng.choice(len(xt), 4000, p=w))[:a.n_acc]
        xs = xt[sel]

        def parts(xc):
            z, exp_c, log_g, lq_c = mmf._prior_part(xc)
            mf, wf, tot, rest = mmf._fine(xc, z, exp_c, lq_c)
            return jax.scipy.special.logsumexp(tot + jnp.log(wf)) - log_g
        win = np.asarray(jax.jit(jax.vmap(parts))(jnp.asarray(xs)))     # the window alone
        _, bl = brute(mmf, xs, edge_refine=True)
        d, d_win = np.abs(llf[sel] - bl), np.abs(win - bl)
        out = 1.0 - np.exp(win - llf[sel])                                 # fraction outside it
        res.append(check(f"log L' vs refined brute force (faint: total S/N {snr_k(k_hi):.1f}, "
                         f"errors x{k_hi:.0f})",
                         np.all(np.isfinite(llf[sel])) and d.max() < 1e-2 and np.median(d) < 2e-3,
                         f"{len(xs)} posterior points: max |d| {d.max():.1e} nats, median "
                         f"{np.median(d):.1e}. Integral outside the window: median "
                         f"{np.median(out):.2f}, max {out.max():.2f}; the window alone would be off "
                         f"by up to {d_win.max():.2f} nats (median {np.median(d_win):.1e})"))
        # masses drawn by sample_logmass vs the brute-force distribution at fixed x, at the 4
        # points with the most integral outside the window (KS distance, 4000 draws each)
        ks = []
        for i in np.argsort(-out)[:4]:
            g, cdf = brute_cdf(mmf, xs[i])
            keys = jax.random.split(jax.random.PRNGKey(int(i)), 4000)
            ms = np.sort(np.asarray(jax.vmap(mmf.sample_logmass, in_axes=(0, None))(keys, jnp.asarray(xs[i]))))
            c_true = np.interp(ms, g, cdf)
            e = np.arange(1, ms.size + 1) / ms.size
            ks.append(max(np.max(np.abs(e - c_true)), np.max(np.abs(e - 1.0 / ms.size - c_true))))
        res.append(check(f"sampled logmass vs brute force (faint, S/N {snr_k(k_hi):.1f})",
                         max(ks) < 0.04,
                         f"KS distance at the 4 points with the most integral outside the window: "
                         f"{', '.join(f'{v_:.3f}' for v_ in ks)} (4000 draws each; 95% noise 0.021)"))

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
