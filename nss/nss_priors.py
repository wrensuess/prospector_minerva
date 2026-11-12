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
"""
import numpy as np
from scipy.special import erf, ndtr, ndtri

#: name -> (kind, args). args for "logflat_gausstail" are (x0, sig) in log10 units; the
#: range [a, b] is log10 of the parameter's own bounds in the model.
FAGN_PRIORS = {
    "flat1_s025": ("logflat_gausstail", (-1.0, 0.25)),
    "loguniform": None,                      # efsps default (no override)
}
DEFAULT_FAGN_PRIOR = "flat1_s025"


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
def ppf_value_np(kind, args, u, lo, hi):
    """Parameter value (linear units) for an override kind; lo, hi are the model bounds."""
    if kind == "logflat_gausstail":
        return 10.0 ** logflat_gausstail_ppf_np(u, np.log10(lo), np.log10(hi), *args)
    raise ValueError(f"unknown NSS prior kind {kind!r}")


def ppf_value_jax(kind, args, u, lo, hi):
    if kind == "logflat_gausstail":
        return 10.0 ** logflat_gausstail_ppf_jax(u, float(np.log10(lo)), float(np.log10(hi)),
                                                 *args)
    raise ValueError(f"unknown NSS prior kind {kind!r}")


def apply_nss_priors(model, fagn_prior=DEFAULT_FAGN_PRIOR):
    """Apply the NSS-side prior overrides to a built FitModel, in place; returns it.

    Must be called before JaxPriorTransform / JaxLogLike / MassMarginal are built from the
    model. Calling it twice with the same setting is a no-op; changing the setting after
    the first call is refused (the patched numpy transform would be stale)."""
    if fagn_prior not in FAGN_PRIORS:
        raise ValueError(f"fagn_prior must be one of {sorted(FAGN_PRIORS)}")
    over = {}
    if FAGN_PRIORS[fagn_prior] is not None and "fagn" in model.col:
        over["fagn"] = FAGN_PRIORS[fagn_prior]
    prev = getattr(model, "nss_prior_overrides", None)
    if prev is not None:
        if prev != over:
            raise RuntimeError("NSS prior overrides already applied with different settings")
        return model
    for name, (kind, _) in over.items():
        lo, _hi = model.bounds[name]
        if model.rows[name].kind != "loguniform" or lo <= 0:
            raise ValueError(f"{name}: override {kind} expects a positive log-uniform row")
    model.nss_prior_overrides = over
    model.nss_prior_names = {"fagn": fagn_prior}

    if over:
        orig = model._prior_transform_block

        def _block(u):
            th = orig(u)
            u2 = np.atleast_2d(np.asarray(u, float))
            for name, (kind, args) in over.items():
                c = model.col[name]
                lo, hi = model.bounds[name]
                th[:, c] = ppf_value_np(kind, args, u2[:, c], lo, hi)
            return th

        model._prior_transform_block = _block
    return model
