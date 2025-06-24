import os, sys
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh', help='phisfh')
parser.add_argument('--dir_indiv', type=str, default='chains_parrot', help='input folder storing chains')
parser.add_argument('--dir_collected', type=str, default='results', help='output folder storing unweighted chains and quantiles')
args = parser.parse_args()
print(args)

which_prior = args.prior

objid_list = []
chains_all = []

all_files = os.listdir(args.dir_indiv)

keys = ['zred', 'total_mass']
for this_file in all_files:
    if this_file.endswith('unw_{}.npz'.format(which_prior)):
        _ini_file = os.path.join(args.dir_indiv, this_file)
        dat = np.load(_ini_file, allow_pickle=True)
        chains = dat['chains'][()]
        
        for t in list(chains.keys()):
            if 'logsfr_ratios' in t:
                keys.append(t)
        break

sname = os.path.join(args.dir_collected, 'chains_sfrr_{}'.format(args.prior)+'.npz')
print('will be saved to', sname)

cnt = 0
missed = []
for this_file in all_files:
    if this_file.endswith('unw_{}.npz'.format(which_prior)):
        mid = int(this_file.split('_')[1])
        _ffile = os.path.join(args.dir_indiv, this_file)
        
        dat = np.load(_ffile, allow_pickle=True)
        chains = dat['chains'][()]

        chain_eqwt = np.stack([chains['zred'], chains['total_mass'], 
                               chains['logsfr_ratios_1'], chains['logsfr_ratios_2'],
                               chains['logsfr_ratios_3'], chains['logsfr_ratios_4'],
                               chains['logsfr_ratios_5'], chains['logsfr_ratios_6']
                               ]).T

        chains_all.append(chain_eqwt)
        objid_list.append(mid)

        cnt += 1
        if cnt % 5000 == 0:
            print(cnt)
            np.savez(sname, objid=objid_list, chains=chains_all, theta_labels=keys)

print('done')

np.savez(sname, objid=objid_list, chains=chains_all, theta_labels=keys)

print('length:', len(objid_list))
print('saved to', sname+'\n')
