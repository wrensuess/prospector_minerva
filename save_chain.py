'''creates *_chains_*.npz
'''
import os
import numpy as np
import argparse
import h5py 

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh')
parser.add_argument('--dir_indiv', type=str, default='chains_parrot', help='input folder storing chains')
parser.add_argument('--dir_collected', type=str, default='results', help='output folder storing unweighted chains and quantiles')
args = parser.parse_args()
print(args)

which_prior = args.prior

keys = ['zred', 'total_mass', 'stellar_mass', 'logzsol', 'mwa',
        'sfr10', 'sfr30', 'sfr100',
        'ssfr10', 'ssfr30', 'ssfr100',
        'dust2', 'dust_index', 'dust1_fraction',
        'log_fagn', 'log_agn_tau', 'gas_logz',
        'duste_qpah', 'duste_umin', 'log_duste_gamma',
        'logsfr_ratios_1', 'logsfr_ratios_2', 'logsfr_ratios_3',
        'logsfr_ratios_4', 'logsfr_ratios_5', 'logsfr_ratios_6'
       ]

# make an output directory if it doesn't exist
if not os.path.exists(args.dir_collected):
    os.makedirs(args.dir_collected)
sname = os.path.join(args.dir_collected, 'chains_{}'.format(args.prior)+'.npz')
print('will be saved to', sname)

# get list of all the files
all_files = sorted([f for f in os.listdir(args.dir_indiv) if f.endswith(f'unw_{which_prior}.npz')])
n_obj = len(all_files)

# check the shape of one chain to get n_samples
sample_file = os.path.join(args.dir_indiv, all_files[0])
sample_chain = np.load(sample_file, allow_pickle=True)['chains'][()]
n_samples = sample_chain['zred'].shape[0]
n_params = len(keys)
objids = np.empty(n_obj, dtype=np.int32)

with h5py.File(sname, 'w') as h5f:
    h5f.create_dataset('theta_labels', data=np.array(keys, dtype='S'))
    
    # make array to hold all chains
    chains_ds = h5f.create_dataset(
        'chains',
        shape=(n_obj, n_samples, n_params),
        dtype=np.float32,
        compression='gzip',
        chunks=(1, n_samples, n_params))
    
    # open each file, extract chains, and store in the big array
    for i, this_file in enumerate(all_files):
        mid = int(this_file.split('_')[1])
        _ffile = os.path.join(args.dir_indiv, this_file)
        dat = np.load(_ffile, allow_pickle=True)
        chains = dat['chains'][()]
        chain_eqwt = np.stack([
            chains['zred'], chains['total_mass'], chains['stellar_mass'],
            chains['logzsol'], chains['mwa'],
            chains['sfr'][:,0], chains['sfr'][:,1], chains['sfr'][:,2],
            chains['ssfr'][:,0], chains['ssfr'][:,1], chains['ssfr'][:,2],
            chains['dust2'], chains['dust_index'], chains['dust1_fraction'],
            chains['log_fagn'], chains['log_agn_tau'], chains['gas_logz'],
            chains['duste_qpah'], chains['duste_umin'], chains['log_duste_gamma'],
            chains['logsfr_ratios_1'], chains['logsfr_ratios_2'], chains['logsfr_ratios_3'],
            chains['logsfr_ratios_4'], chains['logsfr_ratios_5'], chains['logsfr_ratios_6']
        ]).T
        chains_ds[i] = chain_eqwt
        objids[i] = mid
        if (i+1) % 5000 == 0:
            print(f'Processed {i+1} files')
    h5f.create_dataset('objid', data=objids)

print('Total objects:', len(objids))
print('Saved to', sname)
