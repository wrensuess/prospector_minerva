import os
import numpy as np
from astropy.table import Table
import utils as ut_cwd
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh', help='phisfh, phisfhzfixed')
parser.add_argument('--catalog', type=str, default="UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits")
parser.add_argument('--dir_indiv', type=str, default='chains_parrot', help='input folder storing inidividual results')
parser.add_argument('--dir_collected', type=str, default='results', help='output folder storing unweighted chains and quantiles')
parser.add_argument('--basedir', type=str, default='../test/', help='base directory for all outputs')
args = parser.parse_args()
print(args)

which_prior = args.prior
catalog_file = args.catalog

sname = os.path.join(args.dir_collected, f'chains_{args.prior}.h5')
print('will be saved to', sname)

perc_dir = args.dir_collected
chain_dir = args.dir_indiv

mdir = ut_cwd.photdir
cat = Table.read(mdir+catalog_file)
all_filternames = np.array([f[2:] for f in cat.dtype.names if f.startswith('f_f')])
print('all filters in catalog: ', all_filternames)
filter_dict = ut_cwd.filter_dictionary(all_filternames)
filts = list(filter_dict.keys())
filternames = list(filter_dict.values())
print('fitting filters: ', filternames)

all_files = sorted([f for f in os.listdir(chain_dir) if f.endswith(f'_spec_{which_prior}.npz')])
n_obj = len(all_files)

# Check shapes from first file
sample_file = os.path.join(chain_dir, all_files[0])
dat = np.load(sample_file, allow_pickle=True)
modspec_map_shape = dat['modspec_map'].shape
modmag_map_shape = dat['modmags_map'].shape
obs_fnu_shape = ut_cwd.get_fnu_maggies(idx=0, catalog=cat, filts=filts).shape
obs_enu_shape = ut_cwd.get_enu_maggies(idx=0, catalog=cat, filts=filts).shape

objid = np.empty(n_obj, dtype=np.int32)
chi2_fsps = np.empty(n_obj, dtype=np.float32)
nbands = np.empty(n_obj, dtype=np.int32)
modspec_map = np.empty((n_obj,) + modspec_map_shape, dtype=np.float32)
modmag_map = np.empty((n_obj,) + modmag_map_shape, dtype=np.float32)
obsmag = np.empty((n_obj,) + obs_fnu_shape, dtype=np.float32)
obsmag_unc = np.empty((n_obj,) + obs_enu_shape, dtype=np.float32)

def chi2(modmags, obsmags, obsunc):
    _obsunc = np.clip(obsunc, a_min=obsmags*0.05, a_max=None)
    return (((modmags-obsmags)/_obsunc)**2).sum()

for i, this_file in enumerate(all_files):
    mid = int(this_file.split('_')[1])
    dat = np.load(os.path.join(chain_dir, this_file), allow_pickle=True)
    modspec_map[i] = dat['modspec_map']
    modmag_map[i] = dat['modmags_map']

    _idx = np.where(cat['id']==mid)[0][0]
    obs_fnu = ut_cwd.get_fnu_maggies(idx=_idx, catalog=cat, filts=filts)
    obs_enu = ut_cwd.get_enu_maggies(idx=_idx, catalog=cat, filts=filts)
    obsmag[i] = obs_fnu
    obsmag_unc[i] = obs_enu

    # chi2
    phot_mask = (obs_enu > 0) & (np.isfinite(obs_fnu))
    _mask = np.ones_like(obs_fnu, dtype=bool)
    for k in range(len(obs_fnu)):
        if obs_enu[k] > 0:
            if obs_fnu[k] < 0 and obs_fnu[k] + 5*obs_enu[k] < 0:
                _mask[k] = False
    phot_mask &= _mask
    mask = phot_mask

    obsmags = obs_fnu[mask]
    obsunc = obs_enu[mask]
    nbands[i] = len(obsmags)
    fsps_mags = dat['modmags_map'][mask]
    chi2_fsps[i] = chi2(fsps_mags, obsmags, obsunc)
    objid[i] = mid

    if (i+1) % 100 == 0:
        print(i+1)

with h5py.File(sname, 'w') as h5f:
    h5f.create_dataset('objid', data=objid, compression='gzip', chunks=True)
    h5f.create_dataset('obsmag', data=obsmag, compression='gzip', chunks=(1,) + obs_fnu_shape)
    h5f.create_dataset('obsmag_unc', data=obsmag_unc, compression='gzip', chunks=(1,) + obs_enu_shape)
    h5f.create_dataset('modspec_map', data=modspec_map, compression='gzip', chunks=(1,) + modspec_map_shape)
    h5f.create_dataset('modmag_map', data=modmag_map, compression='gzip', chunks=(1,) + modmag_map_shape)
    h5f.create_dataset('chi2_fsps', data=chi2_fsps, compression='gzip', chunks=True)
    h5f.create_dataset('nbands', data=nbands, compression='gzip', chunks=True)

print('length:', n_obj)
print('saved to', sname)