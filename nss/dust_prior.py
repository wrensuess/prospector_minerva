"""Soft-ceiling dust2 prior against the low-z "dusty dwarf" trap (design: 2026-11-11, Wren).

Production fits put some z > 5.5 galaxies at z ~ 0.2 as log M ~ 6-7.7 dwarfs with dust2 1.2-3.4,
far above any real galaxy at that mass (secure-z production fits: max dust2 0.22 at 10^7.5-8, 0.63
at 10^8-8.5). The goal is ONLY to remove such solutions, so the prior stays exactly today's flat
dust2 prior (uniform on [0, 4]) below a ceiling, and is multiplied by a penalty above it:

    ln w(dust2; M, S) = 0                                  dust2 <= d_max(M, S)
                      = -(dust2 - d_max)^2 / (2 s^2)       dust2 >  d_max

    ramp r(x) = clip((x - x1) / (x2 - x1), 0, 1)
    d_max = d_lo + (d_hi - d_lo) * r_M(M)                       kind "mass"
          = d_lo + (d_hi - d_lo) * r_S(S)                       kind "sfr"
          = d_lo + (d_hi - d_lo) * max(r_M(M), r_S(S))          kind "mass_sfr"

M = log10 formed mass (efsps logmass), S = log10 SFR averaged over the last 100 Myr [Msun/yr]
(as efsps sfh.sfh_quantities computes it).

ADOPTED 2026-11-11 (Wren): kind "mass", d_lo = A_V of 1 mag = 1/1.086 = 0.921 (dust2 is the
diffuse tau_V), ramp log M 8 -> 9, d_hi 4 (= the top of the dust2 range: no penalty above 10^9),
tail s 0.2. So: flat below dust2 0.92 at log M < 8, ceiling rising linearly to 4 at log M 9,
today's prior unchanged above 10^9. Chosen on 48 test galaxies (Blanca, 576 fits, plus
reweighting): mean weight at z_spec for 13 trap galaxies 0.65 -> 0.79, and no non-trap galaxy's
posterior moves beyond seed noise. The "sfr" and "mass_sfr" kinds (SFR ramp -1 -> 0) are kept for
reference; they let low-mass dusty solutions with log SFR -1..0 through.

The joint prior is pi'(theta) = pi(theta) w(theta) / C with ONE global constant C = E_pi[w], not
renormalised per (M, S): solutions below the ceiling keep exactly their current relative prior
weight. In NSS, w goes into the likelihood (mass_marginal.MassMarginal: inside the mass integral,
since it depends on mass), so prior draws and logprior are unchanged, and the reported logZ is
ln C + the evidence under pi'. ln C is the same for every galaxy (estimated in test_dust_prior.py).
"""
import numpy as np

KINDS = ("mass", "sfr", "mass_sfr")
AV_LOW_MASS = 1.0                  # mag: the ceiling below log M 8, as A_V = 1.086 dust2
DEFAULT_KIND = "mass"
CEILING = dict(d_lo=AV_LOW_MASS / 1.086, d_hi=4.0, s=0.2, M1=8.0, M2=9.0, S1=-1.0, S2=0.0)
#: the setting of the dust48 Blanca test (2026-11-11), for reproducing it
CEILING_DUST48 = dict(d_lo=0.8, d_hi=4.0, s=0.2, M1=8.0, M2=10.0, S1=-1.0, S2=0.0)
T_SFR = 1e8          # SFR averaging time [yr]
FIRST_RIGHT = np.log10(30.0e6)     # efsps age bins (sfh.agebins_batch)
LAST_BIN = 0.9


# ------------------------------------------------------------------ numpy reference
def ceiling_np(kind, M, S, cfg=CEILING):
    rM = np.clip((np.asarray(M, float) - cfg["M1"]) / (cfg["M2"] - cfg["M1"]), 0.0, 1.0)
    rS = np.clip((np.asarray(S, float) - cfg["S1"]) / (cfg["S2"] - cfg["S1"]), 0.0, 1.0)
    r = {"mass": rM + 0.0 * rS, "sfr": rS + 0.0 * rM, "mass_sfr": np.maximum(rM, rS)}[kind]
    return cfg["d_lo"] + (cfg["d_hi"] - cfg["d_lo"]) * r


def lnw_np(dust2, kind, M, S, cfg=CEILING):
    x = np.asarray(dust2, float) - ceiling_np(kind, M, S, cfg)
    return np.where(x > 0, -0.5 * (x / cfg["s"]) ** 2, 0.0)


def log10_sfr100_per_mass_np(z, log_ratios):
    """log10(SFR_100 / formed mass) [1/yr] from efsps's own sfh_quantities (logmass = 0)."""
    from efsps_photfit import sfh as S
    z = np.atleast_1d(np.asarray(z, float))
    q = S.sfh_quantities(z, np.zeros(z.size), np.atleast_2d(log_ratios))
    return np.log10(q["sfr100"])


# ------------------------------------------------------------------ JAX
def ceiling_jax(kind, M, S, cfg=CEILING):
    import jax.numpy as jnp
    rM = jnp.clip((M - cfg["M1"]) / (cfg["M2"] - cfg["M1"]), 0.0, 1.0)
    rS = jnp.clip((S - cfg["S1"]) / (cfg["S2"] - cfg["S1"]), 0.0, 1.0)
    if kind == "mass":
        r = rM + 0.0 * rS
    elif kind == "sfr":
        r = rS + 0.0 * rM
    elif kind == "mass_sfr":
        r = jnp.maximum(rM, rS)
    else:
        raise ValueError(f"unknown dust prior kind {kind!r}")
    return cfg["d_lo"] + (cfg["d_hi"] - cfg["d_lo"]) * r


def lnw_jax(dust2, kind, M, S, cfg=CEILING):
    import jax.numpy as jnp
    x = dust2 - ceiling_jax(kind, M, S, cfg)
    return jnp.where(x > 0, -0.5 * (x / cfg["s"]) ** 2, 0.0)


def log10_sfr100_per_mass_jax(age_gyr, log_ratios):
    """JAX port of sfh.sfh_quantities' sfr100 / 10^logmass for one row: age of the universe at
    z [Gyr] (JaxPriorTransform.age), 9 log SFR ratios -> log10(SFR_100 / formed mass)."""
    import jax.numpy as jnp
    n_bins = log_ratios.shape[-1] + 1
    last_right = jnp.log10(age_gyr) + 9.0
    last_left = last_right + np.log10(LAST_BIN)
    frac = np.linspace(0.0, 1.0, n_bins - 1)[1:-1]
    inner = FIRST_RIGHT + frac * (last_left - FIRST_RIGHT)
    edges = jnp.concatenate([jnp.zeros(1), jnp.array([FIRST_RIGHT]), inner,
                             jnp.atleast_1d(last_left), jnp.atleast_1d(last_right)])
    t0 = 10.0 ** edges[:-1]      # the first edge is log 0, i.e. 1 yr, as in numpy's agebins
    t1 = 10.0 ** edges[1:]
    dt = t1 - t0
    log_mult = -log_ratios + jnp.log10(dt[1:]) - jnp.log10(dt[:-1])
    log_w = jnp.concatenate([jnp.zeros(1), jnp.cumsum(log_mult)])
    wgt = 10.0 ** (log_w - jnp.max(log_w))
    fr = wgt / jnp.sum(wgt)
    overlap = jnp.clip(jnp.minimum(t1, T_SFR) - t0, 0.0, None)
    return jnp.log10(jnp.sum(fr / dt * overlap) / T_SFR)
