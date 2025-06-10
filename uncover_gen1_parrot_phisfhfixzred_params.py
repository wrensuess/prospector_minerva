import time, sys, os
import numpy as np
import numpy.ma as ma
from astropy.table import Table
from astropy import units as u

import sedpy
import prospect
from prospect.fitting import fit_model
from prospect.sources import FastStepBasis
from prospect.models.sedmodel import PolySpecModel
from prospect.likelihood import lnprobfn

import dynesty

import utils as ut_cwd

import emulator as Emu

pdir = ut_cwd.data_dir(data='pirate')
multiemul_file = os.path.join(pdir, "parrot_v4_obsphot_512n_5l_24s_00z24.npy")

'''TODO: update parser for good MINERVA defaults; make sure we match general phisfh_params '''
# - Parser with default arguments -
parser = prospect.prospect_args.get_parser()
parser.add_argument('--catalog', type=str, default="zspec_UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits")
parser.add_argument('--idx0', type=int, default=0,
                    help="Range of galaxies to fit, from idx0 to idx1-1; zero-index row number of the catalog.")
parser.add_argument('--idx1', type=int, default=1,
                    help="Range of galaxies to fit, from idx0 to idx1-1; zero-index row number of the catalog.")
parser.add_argument('--outdir', type=str, default='chains_parrot_zspec/', help="Output folder name.")
parser.add_argument('--dyn', type=int, default=0, 
                    help="If 0, std run; if 1, quick dynesty run; if 2, debug, max=1100")
args = parser.parse_args()
catalog_file = args.catalog

run_params = vars(args)
run_params.update({
'free_gas_logu': False, # parrot is trained with fixed gas_logu
'verbose': True,
'dyn': args.dyn,
'outdir': ut_cwd.data_dir('cwd')+args.outdir,
'nofork': True,
# dynesty params
'dynesty': True,
'nested_maxcall': None,
'nested_maxcall_init': None,
'nested_maxiter': None,
'nested_maxbatch': None, # maximum number of dynamic patches
'nested_bound': 'multi', # bounding method
'nested_sample': 'rwalk', # sampling method
#'nested_walks': 50, # MC walks
'nested_nlive_batch': 400, # size of live point "batches"
'nested_nlive_init': 1600, # number of initial live points
'nested_weight_kwargs': {'pfrac': 1.0}, # weight posterior over evidence by 100%
'nested_dlogz_init': 0.01,
'nested_target_n_effective': 20000,
# Model info - not much of this is actually needed
'zcontinuous': 2,
'compute_vega_mags': False,
'initial_disp':0.1,
'interp_type': 'logarithmic',
'nbins_sfh': 7,
'sigma': 0.3,
'df': 2,
'agelims': [0.0,7.4772,8.0,8.5,9.0,9.5,9.8,10.0]
})

if run_params['dyn'] == 1:
    # quick dynesty fits, for testing purpose
    run_params.update({
    'nested_nlive_init': 800,
    # 'nested_dlogz_init': 0.1,
    'nested_target_n_effective': 10000,
    'nested_maxcall': 500000,
    'nested_maxcall_init': 500000,
    })
if run_params['dyn'] == 2:
    # debug
    run_params.update({
    'nested_maxcall': 1100,
    'nested_maxcall_init': 1100,
    })
    
run_params["param_file"] = __file__

if not run_params['outdir'].endswith('/'):
    run_params['outdir'] = run_params['outdir'] + '/'
if not os.path.exists(run_params['outdir']):
    os.makedirs(run_params['outdir'])
    print("new directory created:", run_params['outdir'])
print(run_params)

mdir = ut_cwd.data_dir('gen1') + 'phot_catalog/'
cat = Table.read(mdir+catalog_file)

if 'f_alma' in cat.colnames:
    alma = True
else:
    alma = False
if 'f_f460m' in cat.colnames:
    mb = True
else:
    mb = False
filter_dict = ut_cwd.filter_dictionary(mb=mb, alma=alma)
filts = list(filter_dict.keys())
filternames = list(filter_dict.values())

def load_obs(idx=None, err_floor=0.05, **extras):
    '''
    idx: obj idx in the catalog
    '''

    from prospect.utils.obsutils import fix_obs

    flux = ut_cwd.get_fnu_maggies(idx, cat, filts)
    unc = ut_cwd.get_enu_maggies(idx, cat, filts)

    obs = {}
    obs["filters"] = sedpy.observate.load_filters(filternames)
    obs["wave_effective"] = np.array([f.wave_effective for f in obs["filters"]])
    obs["maggies"] = flux
    obs["maggies_unc"] = unc
    # define photometric mask
    # mask out fluxes with negative errors, and high-confidence negative flux
    phot_mask = (unc > 0) & (np.isfinite(flux))
    _mask = np.ones_like(unc, dtype=bool)
    for k in range(len(flux)):
        if unc[k] > 0:
            if flux[k] < 0 and flux[k] + 5*unc[k] < 0:
                _mask[k] = False
    phot_mask &= _mask
    obs['phot_mask'] = phot_mask
    # impose minimum error floor
    obs['maggies_unc'] = np.clip(obs['maggies_unc'], a_min=obs['maggies']*err_floor, a_max=None)

    obs["wavelength"] = None
    obs["spectrum"] = None
    obs['unc'] = None
    obs['mask'] = None

    # other useful info
    obs['objid'] = cat['id'][idx]
    obs['catalog'] = catalog_file

    obs = fix_obs(obs)

    return obs

def build_model(obs=None, emulfp=multiemul_file, **extras):

    import params_prosp_parrot as pfile
    model_params, fit_order = pfile.params_parrot_phisfhfixzred(obs=obs)

    return Emu.EmulatorBeta(model_params, fp=emulfp, obs=obs, param_order=fit_order)


def load_sps(**extras):

    return None


# ---------------- fit !
badobs_ids_list = []
for ifit in np.arange(args.idx0, args.idx1, 1):

    # run on the full catalog
    objid = cat['id'][ifit]
    print("\nFitting {}".format(objid))
    print("------------------\n")
    run_params['idx'] = ifit # choose a galaxy
    _can_fit = False
    try:
        obs = load_obs(**run_params)
        _can_fit = True
    except(AssertionError):
        # all NaNs, etc.
        _can_fit = False
        badobs_ids_list.append(objid)
        print('no phot')

    if _can_fit:
        obs['x_pixel'] = 0; obs['y_pixel'] = 0
        obs['ra'] = cat[ifit]['ra']; obs['dec'] = cat[ifit]['dec']
        ra = obs['ra']*u.deg
        dec = obs['dec']*u.deg
        
        obs['zspec'] = cat[ifit]['z_spec']

        model = build_model(obs=obs, **run_params)
        sps = load_sps(**run_params)
        
        print(obs)
        print(model)

        ts = time.strftime("%y%b%d-%H.%M", time.localtime())
        hfile = os.path.join(run_params['outdir'], "id_{0}_mcmc_phisfhzfixed.h5".format(objid))

        output = fit_model(obs, model, sps, **run_params)
        print('done in {0}s'.format(output["sampling"][1]))

        prospect.io.write_results.write_hdf5(hfile, run_params, model, obs,
                          output["sampling"][0], output["optimization"][0],
                          tsample=output["sampling"][1],
                          toptimize=output["optimization"][1],
                          sps=sps, write_model_params=False)
        try:
            hfile.close()
        except(AttributeError):
            pass

        print('Finished. Saved to {}'.format(hfile))
