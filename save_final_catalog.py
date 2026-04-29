""" now, last step: make the actual final .fits file!"""

import warnings
warnings.filterwarnings('ignore')

import time, sys, os
import numpy as np
import numpy.ma as ma
import argparse
from astropy.table import Table, Column, MaskedColumn
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from datetime import date
import h5py
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh', help='phisfh, phisfhzfixed')
parser.add_argument('--catalog_path', type=str, default="UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits")
parser.add_argument('--dir_collected', type=str, default='results')
parser.add_argument('--sps_version', type=str, default='spsv1.0', help='SPS version used for the catalog')
args = parser.parse_args()

# read in photometric catalog
cat = Table.read(args.catalog_path)

# load in collected percentiles and spectra 
# should probably update to have a version number in the filename....
dperc = h5py.File('{}/quant_{}.h5'.format(args.dir_collected, args.prior), 'r')
dspec = h5py.File('{}/spec_{}.h5'.format(args.dir_collected, args.prior), 'r')

obj_perc = dperc['objid'][:]
obj_spec = dspec['objid'][:]

print(obj_perc.shape, obj_spec.shape)
print(obj_perc[:10])
print(obj_spec[:10])

print("same set?", set(obj_perc) == set(obj_spec))
print("n common", len(set(obj_perc) & set(obj_spec)))
print("only perc", sorted(set(obj_perc) - set(obj_spec))[:20])
print("only spec", sorted(set(obj_spec) - set(obj_perc))[:20])

np.where(dspec['objid'][:] == 0)

assert np.array_equal(dperc['objid'][:], dspec['objid'][:]) # update the [:,0] once i fix the save_perc.py file

# make a new table to hold the final catalog
keep_colnames = np.array(['id', 'ra', 'dec', 
                 'use_aper', 'use_phot',  'flag_kron', 
                 'n_bands_mb', 'flag_acs_coverage'
                ])
cat.keep_columns(keep_colnames)

# metadata
today = date.today()
print("Today date is: ", today)
meta = OrderedDict()
meta['AUTHOR'] = 'Suess research group; suess@colorado.edu' # feel free to replace!
meta['CREATED'] = str(today)
cat.meta = meta

# map our finished objects onto the catalog
# (unrun IDs, including all use_phot=0 objects, will be filled with NaNs)
# first, get index in cat for the results files
mask = np.isin(cat['id'], dspec['objid'][:])
idx_finished = np.where(mask)[0]
assert np.array_equal(cat['id'][idx_finished], dspec['objid'][:])
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
if not os.path.exists('{}/sps_catalog'.format(args.dir_collected)):
    os.makedirs('{}/sps_catalog'.format(args.dir_collected))
    os.makedirs('{}/sps_catalog/ancillaries'.format(args.dir_collected))
    print("new sps catalog directory created:", '{}/sps_catalog'.format(args.dir_collected))
if 'fixed' in args.prior:
    fcat = '{}/sps_catalog/zspec_{}_SPScatalog_{}.fits'.format(args.dir_collected, args.catalog_path.split('/')[-1][:-5], args.sps_version)
else:
    fcat = '{}/sps_catalog/{}_SPScatalog_{}.fits'.format(args.dir_collected, args.catalog_path.split('/')[-1][:-5], args.sps_version)
cat.write(fcat, format='fits', overwrite=True)                    
print('SPS catalog saved to '+fcat)

# cleanup
dperc.close()
dspec.close()


