"""Tests for nss_priors.py (the flat1_s025 fagn prior).

1. ppf against a numerical inversion of the CDF of logpdf (dense trapezoid grid).
2. ppf against an independent closed form (ndtri instead of erfinv).
3. NumPy against JAX ppf.
4. Prior masses: P(fagn > 0.1) = 0.0726, P(fagn > 0.32) ~ 0.003, analytic and by sampling.
5. On a built model: the numpy transform, JaxPriorTransform and MassMarginal.theta all
   give fagn from the new prior, the other 25 parameters unchanged from efsps's
   log-uniform model, and "loguniform" still reproduces efsps exactly.

Run from nss/:  python test_nss_priors.py   (ends "RESULT: PASS" or "RESULT: FAIL")
"""
import os
import sys

import numpy as np
from scipy.special import erf, erfc, erfinv

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nss_priors as NP

FAILS = []


def check(ok, what):
    if not ok:
        FAILS.append(what)
    return "ok" if ok else "FAIL"


A, B = np.log10(1e-5), np.log10(3.0)
X0, SIG = NP.FAGN_PRIORS["flat1_s025"][1]

# ---- 1. numerical CDF inversion
x = np.linspace(A, B, 2_000_001)
pdf = np.exp(NP.logflat_gausstail_logpdf_np(x, A, B, X0, SIG))
cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))])
print(f"1. integral of pdf over [a, b] = {cdf[-1]:.10f} [{check(abs(cdf[-1] - 1) < 1e-9, 'norm')}]")
u = np.concatenate([np.linspace(0, 1, 100001), np.random.default_rng(0).random(100000),
                    1 - np.logspace(-12, -1, 200)])
x_num = np.interp(u, cdf / cdf[-1], x)
x_ppf = NP.logflat_gausstail_ppf_np(u, A, B, X0, SIG)
d = np.abs(x_num - x_ppf)
# the grid's own resolution near the top (where pdf ~ 0) limits the numerical inverse
core = u < 1 - 1e-7
print(f"   max |ppf - numerical inverse|: {d[core].max():.2e} dex (u < 1-1e-7) "
      f"[{check(d[core].max() < 1e-6, 'ppf vs numerical')}], {d.max():.2e} anywhere")

# ---- 2. the prospector reference (FastFlatGaussTail.unit_transform, erfinv form)
m_flat, m_tail, norm = NP._masses(A, B, X0, SIG)
need = np.clip(u * norm - m_flat, 0.0, m_tail)
arg = np.clip(need / (SIG * np.sqrt(np.pi / 2.0)), 0.0, 1.0 - 1e-16)
alt = np.clip(np.where(u <= m_flat / norm, A + u * norm, X0 + SIG * np.sqrt(2.0) * erfinv(arg)), A, B)
d2 = np.abs(alt - x_ppf)
mid = u < 1 - 1e-6
print(f"2. max |prospector erfinv form - this ppf|: {d2[mid].max():.2e} dex for u < 1-1e-6 "
      f"[{check(d2[mid].max() < 1e-9, 'vs prospector reference')}]; {d2.max():.2e} anywhere "
      f"(the erfinv form loses precision at u -> 1)")
# exact round trip in the tail: the analytic prior mass above x_ppf must equal (1-u)*norm
tail = (u > m_flat / norm) & (u < 1)
xt = x_ppf[tail]
above = SIG * np.sqrt(np.pi / 2) * (erfc((xt - X0) / (SIG * np.sqrt(2))) - erfc((B - X0) / (SIG * np.sqrt(2))))
rel = np.abs(above / ((1 - u[tail]) * norm) - 1)
print(f"   round trip, tail: max rel |mass above x / ((1-u) norm) - 1| = {rel.max():.2e} "
      f"[{check(rel.max() < 1e-9, 'tail round trip')}] over {tail.sum()} u, up to 1-u = {1 - u[tail].max():.0e}")

# ---- 3. numpy vs JAX
xj = np.asarray(jax.jit(lambda q: NP.logflat_gausstail_ppf_jax(q, A, B, X0, SIG))(jnp.asarray(u)))
d3 = np.abs(xj - x_ppf)
print(f"3. max |numpy - JAX| ppf: {d3.max():.2e} dex [{check(d3.max() < 1e-12, 'numpy vs jax')}]")

# ---- 4. masses
p01 = m_tail / norm
p032 = (SIG * np.sqrt(np.pi / 2) * (erf((B - X0) / (SIG * np.sqrt(2)))
                                     - erf((np.log10(0.32) - X0) / (SIG * np.sqrt(2))))) / norm
us = np.random.default_rng(1).random(2_000_000)
xs = NP.logflat_gausstail_ppf_np(us, A, B, X0, SIG)
print(f"4. P(fagn > 0.1) analytic {p01:.4f}, sampled {np.mean(xs > -1):.4f} "
      f"[{check(abs(p01 - 0.0726) < 5e-4 and abs(np.mean(xs > -1) - p01) < 1e-3, 'P>0.1')}]; "
      f"P(fagn > 0.32) analytic {p032:.4f}, sampled {np.mean(xs > np.log10(0.32)):.4f}")

# ---- 5. on a built model
from photometry import read_photometry
from nss_efsps import build
from jax_prior import JaxPriorTransform
from tabulated_mass_prior import use_tabulated_mass_prior
from mass_marginal import MassMarginal

phot = read_photometry(os.path.join(HERE, "uncover_test.csv"), ["58858"])[0]
m_new = use_tabulated_mass_prior(build(phot))
m_old = use_tabulated_mass_prior(build(phot, fagn_prior="loguniform"))
c = m_new.col["fagn"]
uu = np.random.default_rng(2).random((20000, m_new.n_dim))
th_new, th_old = m_new.prior_transform(uu), m_old.prior_transform(uu)
others = [j for j in range(m_new.n_dim) if j != c]
lo, hi = m_new.bounds["fagn"]
print(f"5. model fagn bounds {lo:g}..{hi:g}; overrides {m_new.nss_prior_overrides}")
ok = np.array_equal(th_new[:, others], th_old[:, others])
print(f"   numpy: other 25 parameters bitwise unchanged [{check(ok, 'others unchanged')}]")
ref = 10 ** NP.logflat_gausstail_ppf_np(uu[:, c], np.log10(lo), np.log10(hi), X0, SIG)
ok = np.allclose(th_new[:, c], ref, rtol=1e-14, atol=0)
print(f"   numpy: fagn = new prior [{check(ok, 'numpy fagn')}]; P(fagn>0.1) = "
      f"{np.mean(th_new[:, c] > 0.1):.4f} (old prior {np.mean(th_old[:, c] > 0.1):.4f})")
for lab, mod, th in [("new", m_new, th_new), ("loguniform", m_old, th_old)]:
    pt = JaxPriorTransform(mod)
    tj = np.asarray(pt.batch(jnp.asarray(uu)))
    df = np.abs(tj[:, c] - th[:, c]) / th[:, c]
    print(f"   JAX transform ({lab}): max rel |d fagn| {df.max():.2e} "
          f"[{check(df.max() < 1e-12, 'jax fagn ' + lab)}]")
mm = MassMarginal(m_new)
x, lm = mm.x_from_u(uu[:2000])
tm = np.asarray(jax.jit(jax.vmap(mm.theta))(jnp.asarray(x), jnp.asarray(lm)))
df = np.abs(tm[:, c] - th_new[:2000, c]) / th_new[:2000, c]
print(f"   MassMarginal.theta: max rel |d fagn| {df.max():.2e} [{check(df.max() < 1e-12, 'marginal fagn')}]")

print("\nRESULT:", "PASS" if not FAILS else "FAIL " + ", ".join(FAILS))
