import sys, os
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from prospect.models import priors, transforms
from prospect.models import priors_beta as PZ
from scipy.special import erfc
from scipy.integrate import cumulative_trapezoid

class SmoothAGNFracPrior:
    def __init__(self, mini=-5.0, maxi=0.0,
                 center=-0.3, sigma=0.1):

        self.mini = mini
        self.maxi = maxi
        self.center = center
        self.sigma = sigma
        self.flat_end = center - 3.0 * sigma

        # Normalized CDF for unit_transform and sampling
        self.grid = np.linspace(mini, maxi, 10001)
        pdf = self._weight(self.grid)

        self.norm = np.trapz(pdf, self.grid)

        cdf = cumulative_trapezoid(
            pdf, self.grid, initial=0.0
        ) / self.norm

        self.cdf = cdf

    def _weight(self, x):
        x = np.asarray(x)

        w = 0.5 * erfc(
            (x - self.center) / (np.sqrt(2) * self.sigma)
        )

        w = np.where(x <= self.flat_end, 1.0, w)
        w = np.where((x < self.mini) | (x >= self.maxi),
                     0.0, w)
        return w

    def __call__(self, x, **kwargs):
        return self.logpdf(x)

    def logpdf(self, x):
        w = self._weight(x)
        with np.errstate(divide="ignore"):
            return np.log(w) - np.log(self.norm)

    def unit_transform(self, u):
        return np.interp(u, self.cdf, self.grid)

    def sample(self, nsample=None, **kwargs):
        if nsample is None:
            u = np.random.uniform()
        else:
            u = np.random.uniform(size=nsample)

        return self.unit_transform(u)

    def bounds(self, **kwargs):
        return (self.mini, self.maxi)

    def range(self, **kwargs):
        return (self.mini, self.maxi)

def params_parrot_phisfh(obs=None, **extra):
    '''mass + metallicity + sfh priors from p-beta
    '''

    fit_order = ['nzsfh',
                 'dust2',
                 'dust_index',
                 'dust1_fraction',
                 'log_fagn',
                 'log_agn_tau',
                 'gas_logz',
                 'duste_qpah',
                 'duste_umin',
                 'log_duste_gamma']

    # -------------
    # MODEL_PARAMS
    model_params = {}

    # --- BASIC PARAMETERS ---
    model_params['nzsfh'] = {'N': 9, 'isfree': True, 'init': np.array([0.5,8,0.0,0,0,0,0,0,0]),
                             'prior': PZ.PhiSFH(zred_mini=1e-3, zred_maxi=20.0,
                                                mass_mini=6.0, mass_maxi=12.5,
                                                z_mini=-1.98, z_maxi=0.19,
                                                logsfr_ratio_mini=-5.0, logsfr_ratio_maxi=5.0,
                                                logsfr_ratio_tscale=0.3, nbins_sfh=7,
                                                const_phi=True)}
    model_params['zred'] = {'N': 1, 'isfree': False,
                            'depends_on': transforms.nzsfh_to_zred,
                            'init': 0.5,
                            'prior': priors.FastUniform(a=1e-3, b=20.0)}

    model_params['logmass'] = {'N': 1, 'isfree': False,
                              'depends_on': transforms.nzsfh_to_logmass,
                              'init': 8.0,
                              'units': 'Msun',
                              'prior': priors.FastUniform(a=6.0, b=12.5)}

    model_params['logzsol'] = {'N': 1, 'isfree': False,
                               'depends_on': transforms.nzsfh_to_logzsol,
                               'init': -0.5,
                               'units': r'$\log (Z/Z_\odot)$',
                               'prior': priors.FastUniform(a=-1.98, b=0.19)}

    # --- SFH ---
    model_params['logsfr_ratios'] = {'N': 6, 'isfree': False,
                                     'depends_on': transforms.nzsfh_to_logsfr_ratios,
                                     'init': 0.0,
                                     'prior': priors.FastTruncatedEvenStudentTFreeDeg2(hw=np.ones(6)*5.0, sig=np.ones(6)*0.3)}

    # --- Dust Absorption ---
    model_params['dust1_fraction'] = {'N': 1, 'isfree': True, 'init': 1.0,
                                      'prior': priors.FastTruncatedNormal(a=0.0, b=2.0, mu=1.0, sig=0.3)}

    model_params['dust2'] = {'N': 1, 'isfree': True, 'init': 0.0,
                             'units': '',
                             'prior': priors.FastTruncatedNormal(a=0.0, b=4.0, mu=0.3, sig=1.0)}

    model_params['dust_index'] = {'N': 1, 'isfree': True, 'init': 0.7,
                                  'units': '',
                                  'prior': priors.FastUniform(a=-1.0, b=0.4)}

    # --- Nebular Emission ---
    model_params['gas_logz'] = {'N': 1, 'isfree': True, 'init': -0.5,
                                'units': r'log Z/Z_\odot',
                                'prior': priors.FastUniform(a=-2.0, b=0.5)}

    # --- AGN dust ---
    #model_params['log_fagn'] = {'N': 1, 'isfree': True, 'init': -7.0e-5,
    #                            'prior': priors.FastUniform(a=-5.0, b=np.log10(3.0))}
    #model_params['log_fagn'] = {'N': 1, 'isfree': True, 'init': -5.0,
    #                            'prior': priors.FastUniform(a=-5.01, b=-4.99)}
    ### fixed, not good?
    #model_params['log_fagn'] = {'N': 1, 'isfree': False, 'init': -5.0,
    #                            'prior': priors.FastUniform(a=-5.0, b=np.log10(3.0))}
    ### truncated 
    model_params['log_fagn'] = {'N': 1, 'isfree': True, 'init': -1.0,
                                'prior': SmoothAGNFracPrior(mini=-5.0,maxi=0.0,center=-0.3,sigma=0.1)}

    model_params['log_agn_tau'] = {'N': 1, 'isfree': True, 'init': np.log10(20.0),
                                   'prior': priors.FastUniform(a=np.log10(5.0), b=np.log10(150.0))}

    # --- Dust Emission ---
    model_params['duste_qpah'] = {'N':1, 'isfree':True, 'init': 2.0,
                                  'prior': priors.FastTruncatedNormal(a=0.0, b=7.0, mu=2.0, sig=2.0)}

    model_params['duste_umin'] = {'N':1, 'isfree':True, 'init': 1.0,
                                  'prior': priors.FastTruncatedNormal(a=0.1, b=25.0, mu=1.0, sig=10.0)}

    model_params['log_duste_gamma'] = {'N':1, 'isfree':True, 'init': -2.0,
                                       'prior': priors.FastTruncatedNormal(a=-4.0, b=0.0, mu=-2.0, sig=1.0)}

    #---- Units ----
    model_params['peraa'] = {'N': 1, 'isfree': False, 'init': False}

    model_params['mass_units'] = {'N': 1, 'isfree': False, 'init': 'mformed'}

    extra = [k for k in model_params.keys() if k not in fit_order]
    fit_order = fit_order + list(extra)

    tparams = {}
    for i in fit_order:
        tparams[i] = model_params[i]
    for i in list(model_params.keys()):
        if i not in fit_order:
            tparams[i] = model_params[i]
    model_params = tparams

    return (model_params, fit_order)


def params_parrot_phisfhfixzred(obs=None, **extra):
    '''mass + metallicity + sfh priors from p-beta
    redshift fixed to obs['zspec']
    '''

    fit_order = ['nzsfh',
                 'dust2',
                 'dust_index',
                 'dust1_fraction',
                 'log_fagn',
                 'log_agn_tau',
                 'gas_logz',
                 'duste_qpah',
                 'duste_umin',
                 'log_duste_gamma']

    # -------------
    # MODEL_PARAMS
    model_params = {}

    # --- BASIC PARAMETERS ---
    model_params['nzsfh'] = {'N': 9, 'isfree': True, 'init': np.array([0.5,8,0.0,0,0,0,0,0,0]),
                             'prior': PZ.PhiSFHfixZred(zred=obs['zspec'],
                                               mass_mini=6.0, mass_maxi=12.5,
                                               z_mini=-1.98, z_maxi=0.19,
                                               logsfr_ratio_mini=-5.0, logsfr_ratio_maxi=5.0,
                                               logsfr_ratio_tscale=0.3, nbins_sfh=7,
                                               const_phi=True)}
    model_params['zred'] = {'N': 1, 'isfree': False,
                            'depends_on': transforms.nzsfh_to_zred,
                            'init': 0.5,
                            'prior': priors.FastUniform(a=1e-3, b=20.0)}

    model_params['logmass'] = {'N': 1, 'isfree': False,
                              'depends_on': transforms.nzsfh_to_logmass,
                              'init': 8.0,
                              'units': 'Msun',
                              'prior': priors.FastUniform(a=6.0, b=12.5)}

    model_params['logzsol'] = {'N': 1, 'isfree': False,
                               'depends_on': transforms.nzsfh_to_logzsol,
                               'init': -0.5,
                               'units': r'$\log (Z/Z_\odot)$',
                               'prior': priors.FastUniform(a=-1.98, b=0.19)}

    # --- SFH ---
    model_params['logsfr_ratios'] = {'N': 6, 'isfree': False,
                                     'depends_on': transforms.nzsfh_to_logsfr_ratios,
                                     'init': 0.0,
                                     'prior': priors.FastTruncatedEvenStudentTFreeDeg2(hw=np.ones(6)*5.0, sig=np.ones(6)*0.3)}

    # --- Dust Absorption ---
    model_params['dust1_fraction'] = {'N': 1, 'isfree': True, 'init': 1.0,
                                      'prior': priors.FastTruncatedNormal(a=0.0, b=2.0, mu=1.0, sig=0.3)}

    model_params['dust2'] = {'N': 1, 'isfree': True, 'init': 0.0,
                             'units': '',
                             'prior': priors.FastTruncatedNormal(a=0.0, b=4.0, mu=0.3, sig=1.0)}

    model_params['dust_index'] = {'N': 1, 'isfree': True, 'init': 0.7,
                                  'units': '',
                                  'prior': priors.FastUniform(a=-1.0, b=0.4)}

    # --- Nebular Emission ---
    model_params['gas_logz'] = {'N': 1, 'isfree': True, 'init': -0.5,
                                'units': r'log Z/Z_\odot',
                                'prior': priors.FastUniform(a=-2.0, b=0.5)}

    # --- AGN dust ---
    model_params['log_fagn'] = {'N': 1, 'isfree': True, 'init': -7.0e-5,
                                'prior': priors.FastUniform(a=-5.0, b=np.log10(3.0))}

    model_params['log_agn_tau'] = {'N': 1, 'isfree': True, 'init': np.log10(20.0),
                                   'prior': priors.FastUniform(a=np.log10(5.0), b=np.log10(150.0))}

    # --- Dust Emission ---
    model_params['duste_qpah'] = {'N':1, 'isfree':True, 'init': 2.0,
                                  'prior': priors.FastTruncatedNormal(a=0.0, b=7.0, mu=2.0, sig=2.0)}

    model_params['duste_umin'] = {'N':1, 'isfree':True, 'init': 1.0,
                                  'prior': priors.FastTruncatedNormal(a=0.1, b=25.0, mu=1.0, sig=10.0)}

    model_params['log_duste_gamma'] = {'N':1, 'isfree':True, 'init': -2.0,
                                       'prior': priors.FastTruncatedNormal(a=-4.0, b=0.0, mu=-2.0, sig=1.0)}

    #---- Units ----
    model_params['peraa'] = {'N': 1, 'isfree': False, 'init': False}

    model_params['mass_units'] = {'N': 1, 'isfree': False, 'init': 'mformed'}

    extra = [k for k in model_params.keys() if k not in fit_order]
    fit_order = fit_order + list(extra)

    tparams = {}
    for i in fit_order:
        tparams[i] = model_params[i]
    for i in list(model_params.keys()):
        if i not in fit_order:
            tparams[i] = model_params[i]
    model_params = tparams

    return (model_params, fit_order)


def params_parrot_alpha(obs=None, **extra):
    '''p-alpha priors
    '''

    fit_order = ['zred', 'logmass', 'logzsol',
                 'logsfr_ratios',
                 'dust2',
                 'dust_index',
                 'dust1_fraction',
                 'log_fagn',
                 'log_agn_tau',
                 'gas_logz',
                 'duste_qpah',
                 'duste_umin',
                 'log_duste_gamma']

    # -------------
    # MODEL_PARAMS
    model_params = {}

    # --- BASIC PARAMETERS ---
    model_params['zred'] = {'N': 1, 'isfree': True,
                            'init': 0.5,
                            'prior': priors.FastUniform(a=1e-3, b=20.0)}

    model_params['logmass'] = {'N': 1, 'isfree': True,
                               'init': 8.0,
                               'units': 'Msun',
                               'prior': priors.FastUniform(a=6.0, b=12.5)}

    model_params['logzsol'] = {'N': 1, 'isfree': True,
                               'init': -0.5,
                               'units': r'$\log (Z/Z_\odot)$',
                               'prior': priors.FastUniform(a=-1.98, b=0.19)}

    # --- SFH ---
    model_params['logsfr_ratios'] = {'N': 6, 'isfree': True, 'init': np.zeros(6),
                                     'prior': priors.FastTruncatedEvenStudentTFreeDeg2(hw=np.ones(6)*5.0, sig=np.ones(6)*0.3)}

    # --- Dust Absorption ---
    model_params['dust1_fraction'] = {'N': 1, 'isfree': True, 'init': 1.0,
                                      'prior': priors.FastTruncatedNormal(a=0.0, b=2.0, mu=1.0, sig=0.3)}

    model_params['dust2'] = {'N': 1, 'isfree': True, 'init': 0.0,
                             'units': '',
                             'prior': priors.FastTruncatedNormal(a=0.0, b=4.0, mu=0.3, sig=1.0)}

    model_params['dust_index'] = {'N': 1, 'isfree': True, 'init': 0.7,
                                  'units': '',
                                  'prior': priors.FastUniform(a=-1.0, b=0.4)}

    # --- Nebular Emission ---
    model_params['gas_logz'] = {'N': 1, 'isfree': True, 'init': -0.5,
                                'units': r'log Z/Z_\odot',
                                'prior': priors.FastUniform(a=-2.0, b=0.5)}

    # --- AGN dust ---
    model_params['log_fagn'] = {'N': 1, 'isfree': True, 'init': -7.0e-5,
                                'prior': priors.FastUniform(a=-5.0, b=np.log10(3.0))}

    model_params['log_agn_tau'] = {'N': 1, 'isfree': True, 'init': np.log10(20.0),
                                   'prior': priors.FastUniform(a=np.log10(5.0), b=np.log10(150.0))}

    # --- Dust Emission ---
    model_params['duste_qpah'] = {'N':1, 'isfree':True, 'init': 2.0,
                                  'prior': priors.FastTruncatedNormal(a=0.0, b=7.0, mu=2.0, sig=2.0)}

    model_params['duste_umin'] = {'N':1, 'isfree':True, 'init': 1.0,
                                  'prior': priors.FastTruncatedNormal(a=0.1, b=25.0, mu=1.0, sig=10.0)}

    model_params['log_duste_gamma'] = {'N':1, 'isfree':True, 'init': -2.0,
                                       'prior': priors.FastTruncatedNormal(a=-4.0, b=0.0, mu=-2.0, sig=1.0)}

    #---- Units ----
    model_params['peraa'] = {'N': 1, 'isfree': False, 'init': False}

    model_params['mass_units'] = {'N': 1, 'isfree': False, 'init': 'mformed'}

    extra = [k for k in model_params.keys() if k not in fit_order]
    fit_order = fit_order + list(extra)

    tparams = {}
    for i in fit_order:
        tparams[i] = model_params[i]
    for i in list(model_params.keys()):
        if i not in fit_order:
            tparams[i] = model_params[i]
    model_params = tparams

    return (model_params, fit_order)
