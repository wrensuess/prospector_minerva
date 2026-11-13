"""Priors that differ from efsps-photfit's config table, applied on the NSS side.

efsps-photfit is Joel's repo and stays unmodified (commit 6d5ab6d), so prior changes are
applied to a built FitModel here:

    model = build(phot)                    # nss_efsps.build applies them by default
    apply_nss_priors(model)                # or by hand; idempotent

`apply_nss_priors` records the changes in `model.nss_prior_overrides` (read by
jax_prior.JaxPriorTransform and so by mass_marginal.MassMarginal) and patches the numpy
reference transform `model._prior_transform_block`, so the JAX-vs-numpy numerics tests
check the new prior too.

Current overrides
-----------------
fagn: "flat1_s025" (adopted for production by Wren, 2026-11-11; the same prior as
`FastFlatGaussTail(-5, log10 3, -1, 0.25)` in the prospector branch fagn-prior). On
x = log10 fagn over [a, b] = [log10 1e-5, log10 3] (the efsps bounds), the density is
flat for x <= x0 = -1 and exp(-(x - x0)^2 / (2 sig^2)) above, sig = 0.25 dex, normalised
over [a, b]. P(fagn > 0.1) = 7.26%, P(fagn > 0.32) = 0.3%.
It replaces efsps's log-uniform fagn on [1e-5, 3]. Never use a half-Gaussian in LINEAR
fagn: sampled in log space it favours AGN (HANDOFF.md section 6.4).

Joint prior (z, mass), matched to the production dynesty fits (Wren, 2026-11-13). Production
is params_prosp_parrot_AGNfree.py on prospector_minerva dev_curc: PhiSFH(zred 1e-3..20,
mass 6..12.5, const_phi=True).
  z_prior "uniform": z uniform over the model's z bounds. PhiSFH.unit_transform draws z from
      FastUniform; it loads the n(z) table but never uses it. The emulator limits z to
      [0.01, 15], where production allows [0.001, 20]. "nz" = efsps's number-density p(z)
      (the table follows const_phi: pdf_of_z_l20 if True, else pdf_of_z_l20t18).
  const_phi True: Leja+20 mass function at every z, clamped to its z = 0.2 and z = 3 forms
      (efsps's default was False: Leja+20 blended into Tacchella+18 above z = 3).
  mass_min 6.0: floor of the mass prior (efsps: 7.0). The mass grid keeps 0.01-dex steps.

duste priors, "production" (Wren, 2026-11-13), replace efsps's uniform / log-uniform:
  duste_qpah  TruncNormal(2, 2) on production's [0, 7] -> [0.47, 4.58] (the emulator's range)
  duste_umin  TruncNormal(1, 10) on [0.1, 25]
  duste_gamma TruncNormal(-2, 1) in log10 gamma on production's [-4, 0], i.e. gamma 1e-4..1
              (efsps had 1e-3..0.15; the emulator takes gamma linearly on [0, 1])
Science bounds are intersected with the emulator's trained range, as efsps's resolve does.
"""
from dataclasses import replace

import numpy as np
from scipy.special import erf, ndtr, ndtri

#: name -> (kind, args). args for "logflat_gausstail" are (x0, sig) in log10 units; the
#: range [a, b] is log10 of the parameter's own bounds in the model.
FAGN_PRIORS = {
    "flat1_s025": ("logflat_gausstail", (-1.0, 0.25)),
    "loguniform": None,                      # efsps default (no override)
}
DEFAULT_FAGN_PRIOR = "flat1_s025"

#: duste priors: name -> {parameter: (kind, args, science bounds)}; None = efsps's rows.
#: "logtruncnormal": truncated normal in log10 of the value, args (mu, sig) in log10 units.
DUSTE_PRIORS = {
    "production": {
        "duste_qpah": ("truncnormal", (2.0, 2.0), (0.0, 7.0)),
        "duste_umin": ("truncnormal", (1.0, 10.0), (0.1, 25.0)),
        "duste_gamma": ("logtruncnormal", (-2.0, 1.0), (1e-4, 1.0)),
    },
    "efsps": None,
}
DEFAULT_DUSTE_PRIOR = "production"

#: joint z / mass prior settings (module docstring)
Z_PRIORS = ("uniform", "nz")
DEFAULT_Z_PRIOR = "uniform"
DEFAULT_CONST_PHI = True
DEFAULT_MASS_MIN = 6.0
MASS_GRID_DX = 0.01            # efsps: 551 points over 7-12.5


class UniformRedshiftPrior:
    """z uniform on [lo, hi], with the interface of efsps priors.RedshiftPrior (z_nodes and
    pdf_nodes, which jax_prior.JaxPriorTransform inverts): a constant density on two nodes,
    whose piecewise-linear inverse CDF is exactly lo + u (hi - lo)."""

    def __init__(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)
        if not self.lo < self.hi:
            raise ValueError(f"empty redshift range [{lo}, {hi}]")
        self.z_nodes = np.array([self.lo, self.hi])
        self.pdf_nodes = np.full(2, 1.0 / (self.hi - self.lo))

    def pdf(self, z):
        z = np.asarray(z, dtype=float)
        return np.where((z >= self.lo) & (z <= self.hi), self.pdf_nodes[0], 0.0)

    def ppf(self, u):
        from efsps_photfit.priors import inverse_cdf_piecewise_linear
        return inverse_cdf_piecewise_linear(u, self.z_nodes, self.pdf_nodes)


# ------------------------------------------------------------------ flat + Gaussian tail
def _masses(a, b, x0, sig):
    m_flat = x0 - a
    m_tail = sig * np.sqrt(np.pi / 2.0) * erf((b - x0) / (sig * np.sqrt(2.0)))
    return m_flat, m_tail, m_flat + m_tail


def logflat_gausstail_ppf_np(u, a, b, x0, sig):
    """Inverse CDF in x = log10(value).

    The same function as FastFlatGaussTail.unit_transform (agrees to ~1e-9 dex;
    test_nss_priors.py), written in complement form for the tail: with R = (1-u)*norm the
    prior mass above x (1-u is exact for u > 0.5, and the tail starts at u = 0.93),
        x = x0 - sig * ndtri(ndtr(-(b-x0)/sig) + R/(sig*sqrt(2 pi))).
    Every quantity stays small near u = 1, so rounding is not amplified there (the
    erfinv form loses ~1e-9 dex at u ~ 1, enough for JAX and scipy to disagree)."""
    if not (a <= x0 <= b) or b <= a or sig <= 0:
        raise ValueError("need a <= x0 <= b, a < b and sig > 0")
    u = np.asarray(u, dtype=float)
    m_flat, m_tail, norm = _masses(a, b, x0, sig)
    p_top = float(ndtr(-(b - x0) / sig))
    x_flat = a + u * norm
    above = np.clip((1.0 - u) * norm, 0.0, m_tail)
    x_tail = x0 - sig * ndtri(p_top + above / (sig * np.sqrt(2.0 * np.pi)))
    return np.clip(np.where(u <= m_flat / norm, x_flat, x_tail), a, b)


def logflat_gausstail_logpdf_np(x, a, b, x0, sig):
    """log density in x = log10(value), normalised over [a, b]."""
    x = np.asarray(x, dtype=float)
    _, _, norm = _masses(a, b, x0, sig)
    lp = np.where(x <= x0, 0.0, -0.5 * ((x - x0) / sig) ** 2) - np.log(norm)
    return np.where((x >= a) & (x <= b), lp, -np.inf)


def logflat_gausstail_ppf_jax(u, a, b, x0, sig):
    """JAX version of logflat_gausstail_ppf_np (a, b, x0, sig are Python floats)."""
    import jax.numpy as jnp
    from jax.scipy.special import ndtri as jndtri
    m_flat, m_tail, norm = _masses(a, b, x0, sig)
    p_top = float(ndtr(-(b - x0) / sig))
    x_flat = a + u * norm
    above = jnp.clip((1.0 - u) * norm, 0.0, m_tail)
    x_tail = x0 - sig * jndtri(p_top + above / (sig * np.sqrt(2.0 * np.pi)))
    return jnp.clip(jnp.where(u <= m_flat / norm, x_flat, x_tail), a, b)


# ------------------------------------------------------------------ applying to a model
def _truncnorm_ppf_np(u, mu, sig, lo, hi):
    """efsps priors.truncnorm_ppf, the same operations (jax_prior.truncnorm_ppf mirrors it)."""
    a, b = ndtr((lo - mu) / sig), ndtr((hi - mu) / sig)
    return np.clip(mu + sig * ndtri(a + u * (b - a)), lo, hi)


def ppf_value_np(kind, args, u, lo, hi):
    """Parameter value (linear units) for an override kind; lo, hi are the model bounds."""
    if kind == "logflat_gausstail":
        return 10.0 ** logflat_gausstail_ppf_np(u, np.log10(lo), np.log10(hi), *args)
    if kind == "truncnormal":
        return _truncnorm_ppf_np(u, args[0], args[1], lo, hi)
    if kind == "logtruncnormal":
        return np.clip(10.0 ** _truncnorm_ppf_np(u, args[0], args[1], np.log10(lo),
                                                 np.log10(hi)), lo, hi)
    raise ValueError(f"unknown NSS prior kind {kind!r}")


def ppf_value_jax(kind, args, u, lo, hi):
    if kind == "logflat_gausstail":
        return 10.0 ** logflat_gausstail_ppf_jax(u, float(np.log10(lo)), float(np.log10(hi)),
                                                 *args)
    if kind == "logtruncnormal":
        import jax.numpy as jnp
        from jax_prior import truncnorm_ppf
        return jnp.clip(10.0 ** truncnorm_ppf(u, args[0], args[1], float(np.log10(lo)),
                                              float(np.log10(hi))), lo, hi)
    raise ValueError(f"unknown NSS prior kind {kind!r}")


def _set_bounds(model, name, lo, hi):
    """New bounds for a free parameter, everywhere the FitModel keeps them."""
    model.bounds[name] = (float(lo), float(hi))
    model.resolved.bounds[name] = (float(lo), float(hi))
    model.rows[name] = replace(model.rows[name], bounds=(float(lo), float(hi)))


def _emulator_range(model, name):
    lim = dict(zip(model.emu.parameters, model.emu.limits))
    return tuple(float(v) for v in lim[name]) if name in lim else (-np.inf, np.inf)


def apply_nss_priors(model, fagn_prior=DEFAULT_FAGN_PRIOR, z_prior=DEFAULT_Z_PRIOR,
                     const_phi=DEFAULT_CONST_PHI, mass_min=DEFAULT_MASS_MIN,
                     duste_prior=DEFAULT_DUSTE_PRIOR):
    """Apply the NSS-side prior changes to a freshly built FitModel, in place; returns it.

    Must be called before use_tabulated_mass_prior and before JaxPriorTransform /
    JaxLogLike / MassMarginal are built from the model. Calling it twice with the same
    settings is a no-op; changing them after the first call is refused (the patched numpy
    transform would be stale).

    model.nss_prior_overrides: {name: (kind, args)} for every parameter whose prior differs
    from the efsps config table; the 1-D kinds are read by jax_prior.JaxPriorTransform (zred
    and logmass entries are provenance only: their transforms come from model.zprior and
    model.mprior). model.nss_prior_names: the settings by name."""
    from efsps_photfit import priors as P
    if fagn_prior not in FAGN_PRIORS:
        raise ValueError(f"fagn_prior must be one of {sorted(FAGN_PRIORS)}")
    if duste_prior not in DUSTE_PRIORS:
        raise ValueError(f"duste_prior must be one of {sorted(DUSTE_PRIORS)}")
    if z_prior not in Z_PRIORS:
        raise ValueError(f"z_prior must be one of {Z_PRIORS}")
    names = {"fagn": fagn_prior, "duste": duste_prior, "z": z_prior,
             "const_phi": bool(const_phi), "mass_min": float(mass_min)}
    prev = getattr(model, "nss_prior_names", None)
    if prev is not None:
        if prev != names:
            raise RuntimeError("NSS prior overrides already applied with different settings")
        return model
    if hasattr(model.mprior, "cum_table"):
        raise RuntimeError("apply_nss_priors must come before use_tabulated_mass_prior")

    # 1-D priors: (kind, args), plus new bounds where the science bounds change
    over, new_bounds = {}, {}
    if FAGN_PRIORS[fagn_prior] is not None and "fagn" in model.col:
        kind, args = FAGN_PRIORS[fagn_prior]
        lo, _hi = model.bounds["fagn"]
        if model.rows["fagn"].kind != "loguniform" or lo <= 0:
            raise ValueError(f"fagn: override {kind} expects a positive log-uniform row")
        over["fagn"] = (kind, args)
    for name, (kind, args, (slo, shi)) in (DUSTE_PRIORS[duste_prior] or {}).items():
        if name not in model.col:
            continue
        elo, ehi = _emulator_range(model, name)
        lo, hi = max(slo, elo), min(shi, ehi)
        if kind.startswith("log") and lo <= 0:
            raise ValueError(f"{name}: {kind} needs a positive lower bound")
        over[name] = (kind, tuple(float(a) for a in args))
        new_bounds[name] = (lo, hi)
    for name, (lo, hi) in new_bounds.items():
        _set_bounds(model, name, lo, hi)

    # joint prior: mass floor and mass function, then the z prior
    m_lo, m_hi = float(mass_min), float(model.bounds["logmass"][1])
    n_grid = int(round((m_hi - m_lo) / MASS_GRID_DX)) + 1
    model.nzsfh["const_phi"] = bool(const_phi)
    model.nzsfh["mass_grid_points"] = n_grid
    _set_bounds(model, "logmass", m_lo, m_hi)
    model.mprior = P.MassPrior(m_lo, m_hi, const_phi=bool(const_phi), n_grid=n_grid)
    over["logmass"] = ("mass_function_" + ("leja20" if const_phi else "leja20_tacchella18"),
                       (m_lo, m_hi))
    if "zred" in model.col:
        zlo, zhi = model.bounds["zred"]
        if z_prior == "uniform":
            model.zprior = UniformRedshiftPrior(zlo, zhi)
        else:
            model.zprior = P.RedshiftPrior(zlo, zhi, const_phi=bool(const_phi))
        over["zred"] = (z_prior if z_prior == "uniform" else
                        "nz_" + ("l20" if const_phi else "l20t18"), (zlo, zhi))

    model.nss_prior_overrides = over
    model.nss_prior_names = names
    one_d = {k: v for k, v in over.items() if k not in ("zred", "logmass")}
    if one_d:
        orig = model._prior_transform_block

        def _block(u):
            th = orig(u)
            u2 = np.atleast_2d(np.asarray(u, float))
            for name, (kind, args) in one_d.items():
                c = model.col[name]
                lo, hi = model.bounds[name]
                th[:, c] = ppf_value_np(kind, args, u2[:, c], lo, hi)
            return th

        model._prior_transform_block = _block
    return model
