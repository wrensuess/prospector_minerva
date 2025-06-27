import time, sys, os
import numpy as np
import numpy.ma as ma

from astropy.table import Table, Column, MaskedColumn
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
from prospect.models.sedmodel import PolySpecModel

from collections import OrderedDict
from datetime import date
import argparse
import utils as ut_cwd
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh', help='phisfh, phisfhzfixed')
parser.add_argument('--catalog', type=str, default="UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits")
parser.add_argument('--dir_indiv', type=str, default='chains_parrot', help='input folder storing inidividual results')
parser.add_argument('--dir_collected', type=str, default='results', help='output folder storing unweighted chains and quantiles')
parser.add_argument('--basedir', type=str, default='../test/', help='base directory for all outputs')
parser.add_argument('--spsver', type=str, default='spsv0.0')
parser.add_argument('--ver', type=str, default='v0.0')
args = parser.parse_args()
print(args)

# # update these to depend on command line arguments
# ver = 'v3.0.2_LW_SUPER'
# sps_ver = 'spsv2.0'
# prior = 'phisfh'
# cat = Table.read('../phot_catalog/UNCOVER_{}_CATALOG.fits'.format(ver))
# fver = ver+'_'+sps_ver
# print('fver', fver)

# load in catalog
prior = args.prior
catalog_file = args.catalog
mdir = ut_cwd.get_dir("phot", args.basedir) 
cat = Table.read(mdir+catalog_file)

# decide what columns to keep from original phot catalog
# (this is mostly convenience -- can/should update as we see how folks are using the catalog)
# catalog might not have all of the below, but anything that is _not_ in the below list will be removed
keep_colnames = ['id', 'ra', 'dec', 
                 'id_DR1', #'match_radius_DR1',
                 'id_INT_v2', #'match_radius_INT_v2', 
                 'id_msa', 'id_alma', 
                 'use_aper', 'use_phot', 
                 'flag_kron', 
                 #'flag_nophot', 'flag_lowsnr', 'flag_star', 'flag_artifact', 'flag_nearbcg'
                ]
rm_cols = []
for i in range(len(cat.colnames)):
    if cat.colnames[i] not in keep_colnames:
        rm_cols.append(cat.colnames[i])
cat.remove_columns(rm_cols)

# metadata
today = date.today()
print("Today date is: ", today)
meta = OrderedDict()
meta['AUTHOR'] = 'Suess research group; suess@colorado.edu' # feel free to replace!
meta['CREATED'] = str(today)
cat.meta = meta

# load in collected results files
fspec = np.load(os.path.join(args.dir_collected, 'spec_{}'.format(args.prior)+'.npz'), allow_pickle=True)
fperc = np.load(os.path.join(args.dir_collected, 'quant_{}'.format(args.prior)+'.npz'), allow_pickle=True)

# unpack percentiles file
argsorted = np.argsort(fperc['objid'])
id_sort = fperc['objid'][argsorted]
dperc = {}
for key in fperc.files:
    dperc[key] = fperc[key][argsorted]

# unpack spectrum file
argsorted = np.argsort(fspec['objid'])
id_sort = fspec['objid'][argsorted]
# fspec.files.pop(-3) # rm eline_wave
dspec = {}
for key in fspec.files:
    if key not in ['chi2_parrot', 'parrot_mags']:
        dspec[key] = fspec[key][argsorted]

# make sure that these two dictionaries are line-matched
# (hopefully this doesn't break! wren added)
assert np.array_equal(dperc['objid'], dspec['objid'])

# map our finished objects onto the catalog
# (unrun IDs, including all use_phot=0 objects, will be filled with NaNs)
# first, get index in cat for the results files
mask = np.isin(cat['id'], dperc['objid'])
idx_finished = np.where(mask)[0]
assert np.array_equal(cat['id'][idx_finished], dperc['objid'])
print('there '+str(len(cat))+' total objects in the catalog and '+str(np.sum(cat['use_phot']==1))+
    ' with use_phot=1. '+str(len(dperc['objid']))+' have SED-fitting results.')

def fill_col(data, idx_finished=idx_finished):
    new_arr = np.ones(len(cat['id'])) + np.nan    
    new_arr[idx_finished] = np.copy(data)
    mask = np.ones_like(new_arr, dtype=bool)
    mask[idx_finished] = 0
    return new_arr, mask

# get indices for 16/50/84%
ii16 = len(dperc['zred'][0])//2 - 1
ii50 = len(dperc['zred'][0])//2
ii84 = len(dperc['zred'][0])//2 + 1

# add zspec
_data, _mask = fill_col(dperc['zred_spec'])
col_a = MaskedColumn(data=_data, name='z_spec', mask=_mask)
cat.add_columns([col_a])

# add max-likelihood redshift
_data, _mask = fill_col(dperc['zred_ml'], idx_finished=idx_finished)
col_a = MaskedColumn(name='z_ml', data=_data, mask=_mask)
cat.add_columns([col_a])

# and now, add the rest of the theta values!
thetas = ['zred', 'total_mass', 'stellar_mass', 
          'met', 'mwa', 'dust2', 'dust_index', 'dust1_fraction', 
          'log_fagn', 
          'sfr10', 'sfr30', 'sfr100', 
          'ssfr10', 'ssfr30', 'ssfr100', 
         ]
theta_colnames = ['z_16', 'z_50', 'z_84', 
                  'mtot_16', 'mtot_50', 'mtot_84', 
                  'mstar_16', 'mstar_50', 'mstar_84', 
                  'met_16', 'met_50', 'met_84',
                  'mwa_16', 'mwa_50', 'mwa_84', 
                  'dust2_16', 'dust2_50', 'dust2_84',
                  'dust_index_16', 'dust_index_50', 'dust_index_84', 
                  'dust1_fraction_16', 'dust1_fraction_50', 'dust1_fraction_84',
                  'logfagn_16', 'logfagn_50', 'logfagn_84',
                  'sfr10_16', 'sfr10_50', 'sfr10_84',
                  'sfr30_16', 'sfr30_50', 'sfr30_84',
                  'sfr100_16', 'sfr100_50', 'sfr100_84',
                  'ssfr10_16', 'ssfr10_50', 'ssfr10_84',
                  'ssfr30_16', 'ssfr30_50', 'ssfr30_84',
                  'ssfr100_16', 'ssfr100_50', 'ssfr100_84'
                 ]
theta_col_units = [None, None, None, 
                   'log Msol', 'log Msol', 'log Msol', 
                   'log Msol', 'log Msol', 'log Msol', 
                   'log Zsol', 'log Zsol', 'log Zsol',
                   u.Gyr, u.Gyr, u.Gyr,
                   None, None, None, None, None, None, None, None, None, None, None, None, 
                   u.solMass/u.yr, u.solMass/u.yr, u.solMass/u.yr, 
                   u.solMass/u.yr, u.solMass/u.yr, u.solMass/u.yr, 
                   u.solMass/u.yr, u.solMass/u.yr, u.solMass/u.yr, 
                   1/u.yr, 1/u.yr, 1/u.yr, 
                   1/u.yr, 1/u.yr, 1/u.yr, 
                   1/u.yr, 1/u.yr, 1/u.yr, 
                  ]
# expand out names for 16/50/84
dict_thetas = {}
dict_thetas['zred'] = ['z_16', 'z_50', 'z_84']
dict_thetas['total_mass'] = ['mtot_16', 'mtot_50', 'mtot_84']
dict_thetas['stellar_mass'] = ['mstar_16', 'mstar_50', 'mstar_84']
dict_thetas['met'] = ['met_16', 'met_50', 'met_84']
dict_thetas['mwa'] = ['mwa_16', 'mwa_50', 'mwa_84']
dict_thetas['dust2'] = ['dust2_16', 'dust2_50', 'dust2_84']
dict_thetas['dust_index'] = ['dust_index_16', 'dust_index_50', 'dust_index_84']
dict_thetas['dust1_fraction'] = ['dust1_fraction_16', 'dust1_fraction_50', 'dust1_fraction_84']
dict_thetas['log_fagn'] = ['logfagn_16', 'logfagn_50', 'logfagn_84']
dict_thetas['sfr10'] = ['sfr10_16', 'sfr10_50', 'sfr10_84']
dict_thetas['sfr30'] = ['sfr30_16', 'sfr30_50', 'sfr30_84']
dict_thetas['sfr100'] = ['sfr100_16', 'sfr100_50', 'sfr100_84']
dict_thetas['ssfr10'] = ['ssfr10_16', 'ssfr10_50', 'ssfr10_84']
dict_thetas['ssfr30'] = ['ssfr30_16', 'ssfr30_50', 'ssfr30_84']
dict_thetas['ssfr100'] = ['ssfr100_16', 'ssfr100_50', 'ssfr100_84']

# now fill them in
kk = 0
for t in thetas:
    for i_dict, ii in enumerate(np.array([ii16, ii50, ii84])):
        _data, _mask = fill_col(dperc[t][:,ii])
        col_a = MaskedColumn(name=dict_thetas[t][i_dict], data=_data, mask=_mask, unit=theta_col_units[kk])
        cat.add_columns([col_a])
        
        kk += 1
        
# now add rest-frame colors
thetas = ['rest_U', 'rest_V', 'rest_J', 'rest_u', 'rest_g', 'rest_i']
theta_colnames = ['rest_U_16', 'rest_U_50', 'rest_U_84',
                  'rest_V_16', 'rest_V_50', 'rest_V_84',
                  'rest_J_16', 'rest_J_50', 'rest_J_84',
                  'rest_u_16', 'rest_u_50', 'rest_u_84',
                  'rest_g_16', 'rest_g_50', 'rest_g_84',
                  'rest_i_16', 'rest_i_50', 'rest_i_84']
dict_thetas['rest_U'] = ['rest_U_16', 'rest_U_50', 'rest_U_84']
dict_thetas['rest_V'] = ['rest_V_16', 'rest_V_50', 'rest_V_84',]
dict_thetas['rest_J'] = ['rest_J_16', 'rest_J_50', 'rest_J_84',]
dict_thetas['rest_u'] = ['rest_u_16', 'rest_u_50', 'rest_u_84']
dict_thetas['rest_g'] = ['rest_g_16', 'rest_g_50', 'rest_g_84']
dict_thetas['rest_i'] = ['rest_i_16', 'rest_i_50', 'rest_i_84']
for i_t, t in enumerate(thetas):
    for i_dict, ii in enumerate(np.array([ii16, ii50, ii84])):
        _data, _mask = fill_col(dperc['rest_UVJugi'][:,i_t,ii])
        col_a = MaskedColumn(name=dict_thetas[t][i_dict], data=_data, mask=_mask, unit=u.ABmag)
        cat.add_columns([col_a])
thetas = ['UV', 'VJ', 'gi', 'ug']
theta_colnames = ['UV_16', 'UV_50', 'UV_84',
                  'VJ_16', 'VJ_50', 'VJ_84',
                  'gi_16', 'gi_50', 'gi_84',
                  'ug_16', 'ug_50', 'ug_84']
dict_thetas['UV'] = ['UV_16', 'UV_50', 'UV_84']
dict_thetas['VJ'] = ['VJ_16', 'VJ_50', 'VJ_84']
dict_thetas['gi'] = ['gi_16', 'gi_50', 'gi_84']
dict_thetas['ug'] = ['ug_16', 'ug_50', 'ug_84']
for i_t, t in enumerate(thetas):
    for i_dict, ii in enumerate(np.array([ii16, ii50, ii84])):
        _data, _mask = fill_col(dperc['rest_UVJugi_colors'][:,i_t,ii])
        col_a = MaskedColumn(name=dict_thetas[t][i_dict], data=_data, mask=_mask, unit=u.ABmag)
        cat.add_columns([col_a])

# rest sdss colors (for jenny, not made by standard pipeline)
# thetas = ['rest_g_sdss', 'rest_z_sdss']
# theta_colnames = ['rest_g_sdss_16', 'rest_g_sdss_50', 'rest_g_sdss_84',
#                   'rest_z_sdss_16', 'rest_sdss_50', 'rest_z_sdss_84']
# dict_thetas['rest_g_sdss'] = ['rest_g_sdss_16', 'rest_g_sdss_50', 'rest_g_sdss_84']
# dict_thetas['rest_z_sdss'] = ['rest_z_sdss_16', 'rest_z_sdss_50', 'rest_z_sdss_84']
# for i_t, t in enumerate(thetas):
#     for i_dict, ii in enumerate(np.array([ii16, ii50, ii84])):
#         _data, _mask = fill_col(dperc_gz['rest_gz'][:,i_t,ii])
#         col_a = MaskedColumn(name=dict_thetas[t][i_dict], data=_data, mask=_mask, unit=u.ABmag)
#         cat.add_columns([col_a])
    
# add chi^2
_data, _mask = fill_col(dspec['chi2_fsps'], idx_finished=idx_finished)
col_a = MaskedColumn(name='chi2', data=_data, mask=_mask)
cat.add_columns([col_a])

# add number of bands that were fit
_data, _mask = fill_col(dspec['nbands'], idx_finished=idx_finished)
col_a = MaskedColumn(name='nbands', data=_data, mask=_mask)
cat.add_columns([col_a])    

# save!
# make sure directory exists
if not os.path.exists('{}/sps_catalog'.format(args.basedir)):
    os.makedirs('{}/sps_catalog'.format(args.basedir))
    os.makedirs('{}/sps_catalog/ancillaries'.format(args.basedir))
    print("new sps catalog directory created:", '{}/sps_catalog'.format(args.basedir))
if 'fixed' in prior:
    fcat = '{}/sps_catalog/zspec_{}_{}_SPScatalog_{}.fits'.format(args.basedir, args.catalog.split('_')[0], args.ver, args.spsver)
else:
    fcat = '{}/sps_catalog/{}_{}_SPScatalog_{}.fits'.format(args.basedir, args.catalog.split('_')[0], args.ver, args.spsver)
cat.write(fcat, format='fits', overwrite=True)                    
print('SPS catalog saved to '+fcat)


''' now, MAP spectrum'''

# read in an example h5 file to get the effective wavelengths
fname = glob('{}/chains_parrot_{}_{}'.format(args.basedir, args.ver, args.spsver)+'/*.h5')[0]
res, obs, _ = reader.results_from(fname, dangerous=False)
weff = obs['wave_effective']

# load sps
def load_sps(zcontinuous=2, compute_vega_mags=False, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags)  # special to remove redshifting issue
    return sps
sps = load_sps()


if 'fixed' in prior:
    fnpz = '{}/sps_catalog/ancillaries/seds_map_zspec_{}_{}.npz'.format(args.basedir, args.ver, args.spsver)
else:
    fnpz = '{}/sps_catalog/ancillaries/seds_map_{}_{}.npz'.format(args.basedir, args.ver, args.spsver)

print('saving SEDs to '+fnpz)
np.savez(fnpz, objid=dspec['objid'], zred=dperc['zred_ml'],
         obsmags=dspec['obsmag'], obsunc=dspec['obsmag_unc'], 
         modmags=dspec['modmag_map'], modspec=dspec['modspec_map'], 
         wavspec=sps.wavelengths, weff=weff)
        


    