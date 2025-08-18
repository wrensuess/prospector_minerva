'''creates *_sfh_*.npz
ML chain is the last entry
'''
import os
import numpy as np
import argparse
import h5py 

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh', help='phisfh, phisfhzfixed')
parser.add_argument('--dir_indiv', type=str, default='chains_parrot', help='input folder storing chains')
parser.add_argument('--dir_collected', type=str, default='results', help='output folder storing unweighted chains and quantiles')
args = parser.parse_args()
print(args)

which_prior = args.prior

sname = os.path.join(args.dir_collected, 'sfh_{}'.format(args.prior)+'.npz')
print('sfhs will be saved to', sname)

all_files = sorted([f for f in os.listdir(args.dir_indiv) if f.endswith(f'unw_{}.npz'.format(which_prior))])
n_obj = len(all_files)

# percentiles for SFH
perc = np.array([0.1, 2.3, 15.9, 50, 84.1, 97.7, 99.9]) * 0.01

# Check shape from first file
sample_file = os.path.join(args.dir_indiv, all_files[0])
sample_chain = np.load(sample_file, allow_pickle=True)['chains'][()]
n_percentiles = len(perc)
n_bins = sample_chain['sfh'].shape[1]

# initialize arrays to hold all results
objid = np.empty(n_obj, dtype=np.int32)
agebins_max = np.empty(n_obj, dtype=np.float32)
sfh = np.empty((n_obj, n_percentiles, n_bins), dtype=np.float32)

# open files and parse results
for i, this_file in enumerate(all_files):
    mid = int(this_file.split('_')[1])
    _ffile = os.path.join(args.dir_indiv, this_file)
    dat = np.load(_ffile, allow_pickle=True)
    chains = dat['chains'][()]
    objid[i] = mid
    agebins_max[i] = chains['agebins_max']
    sfh[i] = np.quantile(chains['sfh'], perc, axis=0)
    if (i+1) % 10000 == 0:
        print(f'Processed {i+1} files')

with h5py.File(sname, 'w') as h5f:
    h5f.create_dataset('objid', data=objid, compression='gzip', chunks=True)
    h5f.create_dataset('agebins_max', data=agebins_max, compression='gzip', chunks=True)
    h5f.create_dataset('sfh', data=sfh, compression='gzip', chunks=(1, n_percentiles, n_bins))
    h5f.create_dataset('percentiles', data=perc)

print('length:', n_obj)
print('saved to', sname)