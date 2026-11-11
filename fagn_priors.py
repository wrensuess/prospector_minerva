"""Candidate priors on log_fagn that down-weight large AGN fractions.

Motivation: with log_fagn ~ U(-5, log10(3)), prospector sometimes puts galaxies at
low photo-z with log_fagn > 0 instead of at the (correct) higher z with little AGN.
Observed AGN fractions rarely exceed fagn ~ 0.1-0.4, so these priors keep the
low-fagn range flat and suppress the high-fagn end.

FastFlatGaussTail is flat on [a, x0] and falls off as a half-Gaussian of width
`sig` above x0, truncated at b. Special cases:
  - x0 = b                -> plain uniform U(a, b) (hard cut at b)
  - x0 = a                -> half-Gaussian peaked at a
Nested sampling only uses unit_transform (the inverse CDF), so that has to be
exact; __call__ (log pdf) is used for prior_product / postprocessing.
"""
import numpy as np
from scipy.special import erf, erfinv
from prospect.models.priors import Prior


class FastFlatGaussTail(Prior):

    prior_params = ['a', 'b', 'x0', 'sig']

    def __init__(self, a=-5.0, b=np.log10(3.0), x0=-1.0, sig=0.5, parnames=[], name=''):
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}
        self.name = name

        if not (a <= x0 <= b) or b <= a:
            raise ValueError('need a <= x0 <= b and a < b')
        if sig <= 0:
            raise ValueError('sig must be > 0')
        self.a, self.b, self.x0, self.sig = float(a), float(b), float(x0), float(sig)

        # unnormalized masses of the flat part (density 1) and the tail
        self.m_flat = self.x0 - self.a
        self.m_tail = self.sig * np.sqrt(np.pi / 2.0) * erf((self.b - self.x0) / (self.sig * np.sqrt(2.0)))
        self.norm = self.m_flat + self.m_tail
        self.u_break = self.m_flat / self.norm
        self.lognorm = np.log(self.norm)

    def __len__(self):
        return 1

    def _logpdf(self, x):
        x = np.asarray(x, dtype=float)
        lp = np.where(x <= self.x0, 0.0, -0.5 * ((x - self.x0) / self.sig)**2) - self.lognorm
        return np.where((x >= self.a) & (x <= self.b), lp, -np.inf)

    def __call__(self, x):
        out = self._logpdf(x)
        return float(out) if np.ndim(out) == 0 else list(out)

    def scale(self):
        return 0.5 * (self.b - self.a)

    def loc(self):
        return 0.5 * (self.a + self.b)

    def unit_transform(self, u):
        u = np.asarray(u, dtype=float)
        # flat part
        x_flat = self.a + u * self.norm
        # tail: mass above x0 needed = (u*norm - m_flat); invert sig*sqrt(pi/2)*erf(t/(sig*sqrt2))
        need = np.clip(u * self.norm - self.m_flat, 0.0, self.m_tail)
        arg = np.clip(need / (self.sig * np.sqrt(np.pi / 2.0)), 0.0, 1.0 - 1e-16)
        x_tail = self.x0 + self.sig * np.sqrt(2.0) * erfinv(arg)
        x = np.where(u <= self.u_break, x_flat, x_tail)
        x = np.clip(x, self.a, self.b)
        return float(x) if np.ndim(x) == 0 else x

    def sample(self):
        return self.unit_transform(np.random.rand())


LOG3 = np.log10(3.0)

# name -> (description, prior factory). All share the production range [-5, log10 3]
# unless the point of the prior is a hard upper cut.
def candidate_priors():
    from prospect.models import priors
    return {
        'base':       ('U(-5, log 3): production',
                       priors.FastUniform(a=-5.0, b=LOG3)),
        'cut04':      ('U(-5, log 0.4): hard cut at fagn = 0.4',
                       priors.FastUniform(a=-5.0, b=np.log10(0.4))),
        'flat1_s05':  ('flat to -1, half-Gaussian tail sig = 0.5 dex',
                       FastFlatGaussTail(a=-5.0, b=LOG3, x0=-1.0, sig=0.5)),
        'flat1_s025': ('flat to -1, half-Gaussian tail sig = 0.25 dex',
                       FastFlatGaussTail(a=-5.0, b=LOG3, x0=-1.0, sig=0.25)),
        'flat1_s01':  ('flat to -1, half-Gaussian tail sig = 0.1 dex (harsh)',
                       FastFlatGaussTail(a=-5.0, b=LOG3, x0=-1.0, sig=0.1)),
        'flat05_s01': ('flat to -0.5, half-Gaussian tail sig = 0.1 dex (free up to fagn ~ 0.3)',
                       FastFlatGaussTail(a=-5.0, b=LOG3, x0=-0.5, sig=0.1)),
        'hg_s2':      ('half-Gaussian in log_fagn peaked at -5, sig = 2 dex',
                       FastFlatGaussTail(a=-5.0, b=LOG3, x0=-5.0, sig=2.0)),
    }
