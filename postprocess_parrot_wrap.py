import os, sys
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Table

import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
from prospect.models.sedmodel import PolySpecModel
from prospect.models import priors_beta as PZ

import utils as ut_cwd
ddir = ut_cwd.data_dir('cwd')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh', help='phisfh')
parser.add_argument('--fit', type=str, default='fid')
parser.add_argument('--catalog', type=str, default="UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits")
parser.add_argument('--indir', type=str, default='chains', help='input folder storing chains')
parser.add_argument('--outdir', type=str, default='chains', help='output folder storing unweighted chains and quantiles')
parser.add_argument('--narr', type=int, default=70, help='divide the total number into xxx cores')
parser.add_argument('--iarr', type=int, default=0, help='run on the ith sub-array')
parser.add_argument('--ids_file', type=str, default='None', help='None: get ids from indir; else: read in from the file')
parser.add_argument('--free_gas_logu', type=int, default=0, help='0: False; 1: True')
args = parser.parse_args()
print(args)

import postprocess_parrot

import postprocess_parrot as pp

n_split_arr = args.narr
i_split_arr = args.iarr
catalog_file = args.catalog

run_params = {
'free_gas_logu':bool(args.free_gas_logu),
'verbose':True,
'debug': False,
'outdir': ddir+args.outdir,
'nofork': True,
# dynesty params
'dynesty': True,
'nested_maxbatch': None, # maximum number of dynamic patches
'nested_bound': 'multi', # bounding method
'nested_sample': 'rwalk', # sampling method
#'nested_walks': 50, # MC walks
'nested_nlive_batch': 400, # size of live point "batches"
'nested_nlive_init': 1600, # number of initial live points
'nested_weight_kwargs': {'pfrac': 1.0}, # weight posterior over evidence by 100%
'nested_dlogz_init': 0.01,
'nested_target_n_effective': 20000, #20000,
# 'nested_maxcall': 200,
# 'nested_maxcall_init': 200,
# Model info - not much of this is actually needed
'zcontinuous': 2,
'compute_vega_mags': False,
'initial_disp':0.1,
'interp_type': 'logarithmic',
'nbins_sfh': 7,
'sigma': 0.3,
'df': 2,
'agelims': [0.0,7.4772,8.0,8.5,9.0,9.5,9.8,10.0]
}

def build_model(obs=None, free_gas_logu=False, **extras):

    import params_prosp_fsps as pfile
    model_params, fit_order = pfile.params_fsps_phisfh(obs=obs, free_gas_logu=free_gas_logu)

    return PolySpecModel(model_params)

def build_sps_fsps(zcontinuous=2, compute_vega_mags=False, interp_type='logarithmic', **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags)
    return sps

_fdir = os.path.join(ddir, args.outdir)
flist_finished = ut_cwd.finished_id(indir=_fdir, prior=args.prior, dtype='_spec')

_indir = os.path.join(ddir, args.indir)
if args.ids_file == 'None':
    flist = ut_cwd.finished_id(indir=_indir, prior=args.prior, dtype='_mcmc')
else:
    flist = np.loadtxt(args.ids_file, dtype=int)
groups = np.array_split(flist, n_split_arr)

mod_fsps = build_model(**run_params)
mod_for_prior = None
sps = build_sps_fsps(**run_params)

for mid in groups[i_split_arr]:
    if mid not in flist_finished:
        print(mid)
        pp.run_all(objid=mid, fit=args.fit, prior=args.prior,
                   mod_fsps=mod_fsps, mod_for_prior=mod_for_prior, sps=sps,
                   input_folder=args.indir, output_folder=args.outdir, catalog_file=catalog_file)
