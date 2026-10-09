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
  - np.interp in place of interp1d (the same linear interpolation), for the mass draw
    and for the age -> redshift lookup,
  - the SFRD integral over each age bin as differences of the spline's antiderivative,
    evaluated in one call (instead of one FITPACK integral per bin),
  - the six SFR-ratio Student-t transforms in one vectorized call,
  - the mass-function CDF tabulated instead of evaluated on every call:
      0.2 < z <= 3 (Leja+20, whose parameters vary smoothly with z): prospect's own CDF
        at redshift nodes dz_mass apart, interpolated linearly in z. This is the one
        approximation (see MassCDFTable);
      z <= 0.2, and z > 3 with const_phi: prospect holds the mass function at its
        z = 0.2 / z = 3 form (priors_beta.draw_at_z), so the CDF is the end node, exactly;
      z > 3 without const_phi: the fixed Leja+20(z=3) and Tacchella+18 curves are stored
        and blended with prospect's cos^2 weights, then normalised as prospect does
        (exact to rounding).
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

# PZ.f_age_z = interp1d(age, z_age): the same table in increasing age, for np.interp
_age_asc = PZ.age[::-1].copy()
_z_age_asc = PZ.z_age[::-1].copy()

# antiderivative of the SFRD spline, so that integral(a, b) = F(b) - F(a)
_sfrd_cumulative = PZ.spl_tl_sfrd.antiderivative()


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
        z_shifted = np.interp(age_shifted, _age_asc, _z_age_asc)
    if z_shifted > zmax_thres:
        z_shifted = zmax_thres * 1
    if z_shifted < zmin_thres:
        z_shifted = zmin_thres * 1

    agebins_shifted = z_to_agebins_rescale(zstart=z_shifted, nbins_sfh=nbins_sfh, amin=amin)

    nsfrbins = agebins_shifted.shape[0]
    a = agebins_shifted[:,0]
    b = agebins_shifted[:,1]
    F = _sfrd_cumulative(agebins_shifted)
    sfr_shifted = (F[:,1] - F[:,0])/(b-a)

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


def _cdf_from_phi(phi, logm):
    '''The normalisation steps of PZ.pdf_mass_func_at_z + PZ.cdf_mass_func_at_z, in the
    same order, for a mass function phi already evaluated on logm.'''
    pdf_at_m = phi/np.trapz(phi, logm)
    cdf_of_m = np.cumsum(pdf_at_m)
    cdf_of_m /= max(cdf_of_m)
    cdf_of_m[cdf_of_m < 0] = 0
    cdf_of_m[0] = 0
    cdf_of_m[-1] = 1
    return cdf_of_m


class MassCDFTable:
    '''PZ.cdf_mass_func_at_z(z, logm, const_phi, bounds) without evaluating the mass
    function per call.

    0.2 < z <= 3: prospect's CDF at nodes z_j = 0.2, 0.2 + dz, ..., 3 (computed with
    prospect's own function), interpolated linearly in z between the two neighbouring
    nodes. Exact at the nodes; a mix of two CDFs is itself a monotone CDF. Below 0.2 (and
    above 3 with const_phi) prospect uses the z = 0.2 (z = 3) mass function, which is the
    end node, so those are exact.
    z > 3 without const_phi: the fixed curves (Leja+20 at z = 3, Tacchella+18 at z = 4..12)
    are stored and blended with prospect's cos^2 weights in prospect's operation order,
    then normalised as prospect does: equal to prospect's CDF.
    '''

    def __init__(self, logm, const_phi, bounds, dz=1e-3):
        self.logm, self.const_phi, self.bounds = logm, bool(const_phi), list(bounds)
        n = int(round((3.0 - 0.2) / dz)) + 1
        self.z_nodes = np.linspace(0.2, 3.0, n)
        with np.errstate(over='ignore', under='ignore'):
            self.rows = np.array([PZ.cdf_mass_func_at_z(z=z, logm=logm, const_phi=const_phi, bounds=bounds)
                                  for z in self.z_nodes])
            if not self.const_phi:
                inside = (logm >= bounds[0]) & (logm <= bounds[1])
                self.phi_l3 = np.where(inside, PZ.low_z_mass_func(z0=3, logm=logm), 0.)
                self.phi_t18 = {k: np.where(inside, PZ.high_z_mass_func_discreate(k, 10**logm), 0.)
                                for k in range(4, 13)}

    def __call__(self, z):
        if z > 3.0 and not self.const_phi:
            return self._high_z(z)
        if z <= 0.2:
            return self.rows[0]
        if z >= 3.0:
            return self.rows[-1]
        zc = z
        j = min(max(np.searchsorted(self.z_nodes, zc, side='right') - 1, 0), len(self.z_nodes) - 2)
        t = (zc - self.z_nodes[j]) / (self.z_nodes[j+1] - self.z_nodes[j])
        return (1. - t) * self.rows[j] + t * self.rows[j+1]

    def _high_z(self, z):
        '''PZ.mass_func_at_z for z > 3, const_phi False, from the stored curves'''
        if z < 4.0:
            w = PZ.cos_weight(z)
            phi = self.phi_l3*w[0] + self.phi_t18[4]*w[1]
        elif z <= 4.0:
            phi = self.phi_t18[4]
        elif z >= 12.0:
            phi = self.phi_t18[12]
        else:
            k = int(z)
            w = PZ.cos_weight(z, zrange=[k, k+1])
            phi = self.phi_t18[k]*w[0] + self.phi_t18[k+1]*w[1]
        return _cdf_from_phi(phi, self.logm)


_mass_cdf_tables = {}  # one table per (grid, const_phi, dz), shared by every model in a process


def mass_cdf_table(logm, const_phi, bounds, dz):
    key = (logm[0], logm[-1], len(logm), bool(const_phi), tuple(bounds), dz)
    if key not in _mass_cdf_tables:
        _mass_cdf_tables[key] = MassCDFTable(logm, const_phi, bounds, dz)
    return _mass_cdf_tables[key]


class PhiSFHFast(PZ.PhiSFH):
    '''PZ.PhiSFH with a faster unit_transform (see module docstring).
    dz_mass: node spacing of the tabulated mass-function CDF in 0.2 < z <= 3.'''

    dz_mass = 1e-3

    def cdf_mass(self, zred):
        '''PZ.cdf_mass_func_at_z on self.mgrid, from the tabulated CDF'''
        if self._mass_cdf is None:
            self._mass_cdf = mass_cdf_table(self.mgrid, self.params['const_phi'],
                                            [self.params['mass_mini'], self.params['mass_maxi']], self.dz_mass)
        return self._mass_cdf(zred)

    def update(self, **kwargs):
        super(PhiSFHFast, self).update(**kwargs)
        self._mass_cdf = None # the prior parameters may have changed

    def unit_transform(self, x, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)

        zred = self.zred_dist.unit_transform(x[0])

        cdf_mass = self.cdf_mass(zred)
        mass = np.interp(x[1], cdf_mass, self.mgrid)

        met_dist = priors.FastTruncatedNormal(a=self.params['z_mini'], b=self.params['z_maxi'],
                                              mu=PZ.loc_massmet(mass), sig=PZ.scale_massmet(mass))
        met = met_dist.unit_transform(x[2])

        # sfh = sfrd
        logsfr_ratios = expe_logsfr_ratios(this_z=zred, this_m=mass, nbins_sfh=self.params['nbins_sfh'],
                                           logsfr_ratio_mini=self.params['logsfr_ratio_mini'],
                                           logsfr_ratio_maxi=self.params['logsfr_ratio_maxi'])

        logsfr_ratios_ppf = self.logsfr_ratios_dist.unit_transform(x[3:3+len(logsfr_ratios)]) + logsfr_ratios
        logsfr_ratios_ppf = np.clip(logsfr_ratios_ppf, a_min=self.params['logsfr_ratio_mini'], a_max=self.params['logsfr_ratio_maxi'])
        return np.concatenate([np.atleast_1d(zred), np.atleast_1d(mass),
                               np.atleast_1d(met), np.atleast_1d(logsfr_ratios_ppf)])
