'''creates *_perc_*.h5
ALL prospector model paramters are saved
'''
import os
import numpy as np
from astropy.table import Table
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh')
parser.add_argument('--catalog', type=str, default="UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits")
parser.add_argument('--indir', type=str, default='chains_parrot', help='input folder storing chains')
parser.add_argument('--outdir', type=str, default='results', help='output folder storing unweighted chains and quantiles')
args = parser.parse_args()
print(args)

which_prior = args.prior
catalog_file = args.catalog

foo = args.indir
if foo.endswith('/'):
    foo = foo[:-1]
sname = os.path.join(args.outdir, f'quant_{args.prior}_{foo}.h5')
print('will be saved to', sname)

perc_dir = args.indir
cat = Table.read('phot_catalog/' + catalog_file)

# Get list of files
all_files = sorted([f for f in os.listdir(perc_dir) if f.endswith(f'perc_{which_prior}.npz')])
n_obj = len(all_files)

# Check shapes from first file
sample_file = os.path.join(perc_dir, all_files[0])
dat = np.load(sample_file, allow_pickle=True)
perc = dat['percentiles'][()]
rest_UVJugi_shape = perc['rest_UVJugi'].shape
rest_UVJugi_map_shape = perc['rest_UVJugi_map'].shape
rest_UVJugi_colors_shape = perc['rest_UVJugi_colors'].shape
rest_UVJugi_colors_map_shape = perc['rest_UVJugi_colors_map'].shape
rest_gz_shape = perc['rest_gz'].shape
rest_gz_map_shape = perc['rest_gz_map'].shape
rest_gz_colors_shape = perc['rest_gz_colors'].shape
rest_gz_colors_map_shape = perc['rest_gz_colors_map'].shape
rest_NUVrJ_shape = perc['rest_NUVrJ'].shape
rest_NUVrJ_map_shape = perc['rest_NUVrJ_map'].shape
rest_NUVrJ_colors_shape = perc['rest_NUVrJ_colors'].shape
rest_NUVrJ_colors_map_shape = perc['rest_NUVrJ_colors_map'].shape

# Pre-allocate arrays
objid = np.empty(n_obj, dtype=np.int32)
zred_spec = np.empty(n_obj, dtype=np.float32)
zred_ml = np.empty(n_obj, dtype=np.float32)
zred = np.empty(n_obj, dtype=np.float32)
total_mass = np.empty(n_obj, dtype=np.float32)
stellar_mass = np.empty(n_obj, dtype=np.float32)
met = np.empty(n_obj, dtype=np.float32)
dust2 = np.empty(n_obj, dtype=np.float32)
dust_index = np.empty(n_obj, dtype=np.float32)
dust1_fraction = np.empty(n_obj, dtype=np.float32)
log_fagn = np.empty(n_obj, dtype=np.float32)
log_agn_tau = np.empty(n_obj, dtype=np.float32)
gas_logz = np.empty(n_obj, dtype=np.float32)
duste_qpah = np.empty(n_obj, dtype=np.float32)
duste_umin = np.empty(n_obj, dtype=np.float32)
log_duste_gamma = np.empty(n_obj, dtype=np.float32)
mwa = np.empty(n_obj, dtype=np.float32)
sfr10 = np.empty(n_obj, dtype=np.float32)
sfr30 = np.empty(n_obj, dtype=np.float32)
sfr100 = np.empty(n_obj, dtype=np.float32)
ssfr10 = np.empty(n_obj, dtype=np.float32)
ssfr30 = np.empty(n_obj, dtype=np.float32)
ssfr100 = np.empty(n_obj, dtype=np.float32)

rest_UVJugi = np.empty((n_obj,) + rest_UVJugi_shape, dtype=np.float32)
rest_UVJugi_map = np.empty((n_obj,) + rest_UVJugi_map_shape, dtype=np.float32)
rest_UVJugi_colors = np.empty((n_obj,) + rest_UVJugi_colors_shape, dtype=np.float32)
rest_UVJugi_colors_map = np.empty((n_obj,) + rest_UVJugi_colors_map_shape, dtype=np.float32)
rest_gz = np.empty((n_obj,) + rest_gz_shape, dtype=np.float32)
rest_gz_map = np.empty((n_obj,) + rest_gz_map_shape, dtype=np.float32)
rest_gz_colors = np.empty((n_obj,) + rest_gz_colors_shape, dtype=np.float32)
rest_gz_colors_map = np.empty((n_obj,) + rest_gz_colors_map_shape, dtype=np.float32)
rest_NUVrJ = np.empty((n_obj,) + rest_NUVrJ_shape, dtype=np.float32)
rest_NUVrJ_map = np.empty((n_obj,) + rest_NUVrJ_map_shape, dtype=np.float32)
rest_NUVrJ_colors = np.empty((n_obj,) + rest_NUVrJ_colors_shape, dtype=np.float32)
rest_NUVrJ_colors_map = np.empty((n_obj,) + rest_NUVrJ_colors_map_shape, dtype=np.float32)

for i, this_file in enumerate(all_files):
    mid = int(this_file.split('_')[1])
    ffnpz = os.path.join(perc_dir, this_file)
    dat = np.load(ffnpz, allow_pickle=True)
    perc = dat['percentiles'][()]
    objid[i] = mid
    zred[i] = perc['zred']
    total_mass[i] = perc['total_mass']
    stellar_mass[i] = perc['stellar_mass']
    met[i] = perc['logzsol']
    mwa[i] = perc['mwa']
    sfr10[i] = perc['sfr'][0]
    sfr30[i] = perc['sfr'][1]
    sfr100[i] = perc['sfr'][2]
    ssfr10[i] = perc['ssfr'][0]
    ssfr30[i] = perc['ssfr'][1]
    ssfr100[i] = perc['ssfr'][2]
    dust2[i] = perc['dust2']
    dust_index[i] = perc['dust_index']
    dust1_fraction[i] = perc['dust1_fraction']
    log_fagn[i] = perc['log_fagn']
    log_agn_tau[i] = perc['log_agn_tau']
    gas_logz[i] = perc['gas_logz']
    duste_qpah[i] = perc['duste_qpah']
    duste_umin[i] = perc['duste_umin']
    log_duste_gamma[i] = perc['log_duste_gamma']
    rest_UVJugi[i] = perc['rest_UVJugi']
    rest_UVJugi_map[i] = perc['rest_UVJugi_map']
    rest_UVJugi_colors[i] = perc['rest_UVJugi_colors']
    rest_UVJugi_colors_map[i] = perc['rest_UVJugi_colors_map']
    rest_gz[i] = perc['rest_gz']
    rest_gz_map[i] = perc['rest_gz_map']
    rest_gz_colors[i] = perc['rest_gz_colors']
    rest_gz_colors_map[i] = perc['rest_gz_colors_map']
    rest_NUVrJ[i] = perc['rest_NUVrJ']
    rest_NUVrJ_map[i] = perc['rest_NUVrJ_map']
    rest_NUVrJ_colors[i] = perc['rest_NUVrJ_colors']
    rest_NUVrJ_colors_map[i] = perc['rest_NUVrJ_colors_map']
    zred_ml[i] = dat['chain_ml'][0]
    idx_ftrue = np.where(cat['id'] == mid)[0][0]
    zred_spec[i] = cat['z_spec'][idx_ftrue]
    if (i+1) % 1000 == 0:
        print(f'Processed {i+1} files')

with h5py.File(sname, 'w') as h5f:
    h5f.create_dataset('objid', data=objid, compression='gzip', chunks=True)
    h5f.create_dataset('zred_spec', data=zred_spec, compression='gzip', chunks=True)
    h5f.create_dataset('zred', data=zred, compression='gzip', chunks=True)
    h5f.create_dataset('zred_ml', data=zred_ml, compression='gzip', chunks=True)
    h5f.create_dataset('total_mass', data=total_mass, compression='gzip', chunks=True)
    h5f.create_dataset('stellar_mass', data=stellar_mass, compression='gzip', chunks=True)
    h5f.create_dataset('met', data=met, compression='gzip', chunks=True)
    h5f.create_dataset('dust2', data=dust2, compression='gzip', chunks=True)
    h5f.create_dataset('dust_index', data=dust_index, compression='gzip', chunks=True)
    h5f.create_dataset('dust1_fraction', data=dust1_fraction, compression='gzip', chunks=True)
    h5f.create_dataset('log_fagn', data=log_fagn, compression='gzip', chunks=True)
    h5f.create_dataset('log_agn_tau', data=log_agn_tau, compression='gzip', chunks=True)
    h5f.create_dataset('gas_logz', data=gas_logz, compression='gzip', chunks=True)
    h5f.create_dataset('duste_qpah', data=duste_qpah, compression='gzip', chunks=True)
    h5f.create_dataset('duste_umin', data=duste_umin, compression='gzip', chunks=True)
    h5f.create_dataset('log_duste_gamma', data=log_duste_gamma, compression='gzip', chunks=True)
    h5f.create_dataset('mwa', data=mwa, compression='gzip', chunks=True)
    h5f.create_dataset('sfr10', data=sfr10, compression='gzip', chunks=True)
    h5f.create_dataset('sfr30', data=sfr30, compression='gzip', chunks=True)
    h5f.create_dataset('sfr100', data=sfr100, compression='gzip', chunks=True)
    h5f.create_dataset('ssfr10', data=ssfr10, compression='gzip', chunks=True)
    h5f.create_dataset('ssfr30', data=ssfr30, compression='gzip', chunks=True)
    h5f.create_dataset('ssfr100', data=ssfr100, compression='gzip', chunks=True)
    h5f.create_dataset('rest_UVJugi', data=rest_UVJugi, compression='gzip', chunks=(1,) + rest_UVJugi_shape)
    h5f.create_dataset('rest_UVJugi_map', data=rest_UVJugi_map, compression='gzip', chunks=(1,) + rest_UVJugi_map_shape)
    h5f.create_dataset('rest_UVJugi_colors', data=rest_UVJugi_colors, compression='gzip', chunks=(1,) + rest_UVJugi_colors_shape)
    h5f.create_dataset('rest_UVJugi_colors_map', data=rest_UVJugi_colors_map, compression='gzip', chunks=(1,) + rest_UVJugi_colors_map_shape)
    h5f.create_dataset('rest_gz', data=rest_gz, compression='gzip', chunks=(1,) + rest_gz_shape)
    h5f.create_dataset('rest_gz_map', data=rest_gz_map, compression='gzip', chunks=(1,) + rest_gz_map_shape)
    h5f.create_dataset('rest_gz_colors', data=rest_gz_colors, compression='gzip', chunks=(1,) + rest_gz_colors_shape)
    h5f.create_dataset('rest_gz_colors_map', data=rest_gz_colors_map, compression='gzip', chunks=(1,) + rest_gz_colors_map_shape)
    h5f.create_dataset('rest_NUVrJ', data=rest_NUVrJ, compression='gzip', chunks=(1,) + rest_NUVrJ_shape)
    h5f.create_dataset('rest_NUVrJ_map', data=rest_NUVrJ_map, compression='gzip', chunks=(1,) + rest_NUVrJ_map_shape)
    h5f.create_dataset('rest_NUVrJ_colors', data=rest_NUVrJ_colors, compression='gzip', chunks=(1,) + rest_NUVrJ_colors_shape)
    h5f.create_dataset('rest_NUVrJ_colors_map', data=rest_NUVrJ_colors_map, compression='gzip', chunks=(1,) + rest_NUVrJ_colors_map_shape)

print('length:', n_obj)
print('saved to', sname)