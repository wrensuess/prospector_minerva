'''Faster drop-in for prospect's priors_beta.PhiSFH prior transform.

PhiSFH.unit_transform (called on every likelihood evaluation in nested sampling)
spends most of its time in astropy -- cosmo.age(z) and two cosmo.lookback_time()
calls, each a numerical integral -- and in building a scipy interp1d for the mass
draw. PhiSFHFast runs the same algorithm (code copied from prospect 1.4.0
priors_beta.py) with:
  - lookback time and age of the universe read from a dense precomputed table for
    the same cosmology (agrees with astropy to ~1e-9 relative or better, below the
    tolerance of astropy's own quadrature),
  - cosmo.lookback_time(15), which is a constant, computed once,
  - np.interp in place of interp1d (the same linear interpolation).
The prior density (__call__) and everything else are inherited unchanged.
'''
import numpy as np
from astropy import units as u
from scipy.integrate import cumulative_simpson
from prospect.models import priors
from prospect.models import priors_beta as PZ

cosmo = PZ.cosmo

# lookback time on a fine redshift grid, integrated once (Simpson's rule)
_zgrid = np.linspace(0., 30., 1000001)
_lookback_gyr = cosmo.hubble_time.to(u.Gyr).value * cumulative_simpson(
    cosmo.lookback_time_integrand(_zgrid), x=_zgrid, initial=0.)
_age0_gyr = cosmo.age(0).value
_tuniv15_yr = cosmo.lookback_time(15).to(u.yr).value  # onset of star formation (z~15)


def age_gyr(z):
    '''cosmo.age(z).value'''
    return _age0_gyr - np.interp(z, _zgrid, _lookback_gyr)


def lookback_time_yr(z):
    '''cosmo.lookback_time(z).to(u.yr).value'''
    return np.interp(z, _zgrid, _lookback_gyr) * 1e9


def z_to_agebins_rescale(zstart, nbins_sfh=7, amin=7.1295):
    '''priors_beta.z_to_agebins_rescale, with tabulated cosmology'''

    agelims = np.zeros(nbins_sfh+1)
    agelims[0] = lookback_time_yr(zstart) # shift the start of the agebin
    tuniv = _tuniv15_yr # cap at z~15, for the onset of star formation
    tbinmax = tuniv - (tuniv-agelims[0]) * 0.10
    agelims[-2] = tbinmax
    agelims[-1] = tuniv

    if zstart <= 3.0:
        agelims[1] = agelims[0] + 3e7 # 1st bin is 30 Myr wide
        agelims[2] = agelims[1] + 1e8 # 2nd bin is 100 Myr wide
        i_age = 3
        nbins = len(agelims)-3
    else:
        agelims[1] = agelims[0] + 10**amin
        i_age = 2
        nbins = len(agelims)-2

    if agelims[0] == 0:
        with np.errstate(invalid='ignore', divide='ignore'):
            agelims = np.log10(agelims[:i_age]).tolist()[:-1] + np.squeeze(np.linspace(np.log10(agelims[i_age-1]),np.log10(tbinmax),nbins)).tolist() + [np.log10(tuniv)]
            agelims[0] = 0

    else:
        agelims = np.log10(agelims[:i_age]).tolist()[:-1] + np.squeeze(np.linspace(np.log10(agelims[i_age-1]),np.log10(tbinmax),nbins)).tolist() + [np.log10(tuniv)]

    agebins = np.array([agelims[:-1], agelims[1:]]).T
    return 10**agebins


def expe_logsfr_ratios(this_z, this_m, logsfr_ratio_mini, logsfr_ratio_maxi,
                       nbins_sfh=7, amin=7.1295):
    '''priors_beta.expe_logsfr_ratios, with tabulated cosmology'''

    age_shifted = np.log10(age_gyr(this_z)) + PZ.delta_t_dex(this_m)
    age_shifted = 10**age_shifted

    zmin_thres = 0.15
    zmax_thres = 10
    if age_shifted < PZ.age[-1]:
        z_shifted = zmax_thres * 1
    elif age_shifted > PZ.age[0]:
        z_shifted = zmin_thres * 1
    else:
        z_shifted = PZ.f_age_z(age_shifted)
    if z_shifted > zmax_thres:
        z_shifted = zmax_thres * 1
    if z_shifted < zmin_thres:
        z_shifted = zmin_thres * 1

    agebins_shifted = z_to_agebins_rescale(zstart=z_shifted, nbins_sfh=nbins_sfh, amin=amin)

    nsfrbins = agebins_shifted.shape[0]
    sfr_shifted = np.zeros(nsfrbins)
    for i in range(nsfrbins):
        a = agebins_shifted[i,0]
        b = agebins_shifted[i,1]
        sfr_shifted[i] = PZ.spl_tl_sfrd.integral(a=a, b=b)/(b-a)

    logsfr_ratios_shifted = np.zeros(nsfrbins-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        for i in range(nsfrbins-1):
            logsfr_ratios_shifted[i] = np.log10(sfr_shifted[i]/sfr_shifted[i+1])
    logsfr_ratios_shifted = np.clip(logsfr_ratios_shifted, logsfr_ratio_mini, logsfr_ratio_maxi)

    if not np.all(np.isfinite(logsfr_ratios_shifted)):
        # set nan accord. to its neighbor
        nan_idx = np.isnan(logsfr_ratios_shifted)
        finite_idx = np.min(np.where(nan_idx==True))-1
        neigh = logsfr_ratios_shifted[finite_idx]
        nan_idx = np.arange(6-finite_idx-1) + finite_idx + 1
        for i in range(len(nan_idx)):
            logsfr_ratios_shifted[nan_idx[i]] = neigh * 1.

    return logsfr_ratios_shifted


class PhiSFHFast(PZ.PhiSFH):
    '''PZ.PhiSFH with a faster unit_transform (see module docstring).'''

    def unit_transform(self, x, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)

        zred = self.zred_dist.unit_transform(x[0])

        cdf_mass = PZ.cdf_mass_func_at_z(z=zred, logm=self.mgrid, const_phi=self.params['const_phi'], bounds=[self.params['mass_mini'], self.params['mass_maxi']])
        mass = np.interp(x[1], cdf_mass, self.mgrid)

        met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                              mu=PZ.loc_massmet(mass), sig=PZ.scale_massmet(mass))
        met = met_dist.unit_transform(x[2])

        # sfh = sfrd
        logsfr_ratios = expe_logsfr_ratios(this_z=zred, this_m=mass, nbins_sfh=self.params['nbins_sfh'],
                                           logsfr_ratio_mini=self.params['logsfr_ratio_mini'],
                                           logsfr_ratio_maxi=self.params['logsfr_ratio_maxi'])

        logsfr_ratios_ppf = np.zeros_like(logsfr_ratios)
        for i in range(len(logsfr_ratios_ppf)):
            logsfr_ratios_ppf[i] = self.logsfr_ratios_dist.unit_transform(x[3+i]) + logsfr_ratios[i]
        logsfr_ratios_ppf = np.clip(logsfr_ratios_ppf, a_min=self.params['logsfr_ratio_mini'], a_max=self.params['logsfr_ratio_maxi'])
        return np.concatenate([np.atleast_1d(zred), np.atleast_1d(mass),
                               np.atleast_1d(met), np.atleast_1d(logsfr_ratios_ppf)])
