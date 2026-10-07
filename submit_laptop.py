import os, sys, time, argparse
import numpy as np
from astropy.table import Table
import utils as ut_cwd

'''Run the whole pipeline (fit -> postprocess -> collected catalogs) locally, mirroring
what submit.py / run_postprocess_blanca.py / save_all.sh do on Blanca.

Data locations come from utils.py; on a laptop set them with environment variables, e.g.
    export MINERVA_EMULATOR_DIR=~/Projects/pirate/pirate/data/nn/
    export MINERVA_PHOTDIR=../test_prospector/phot_catalog/
'''

parser = argparse.ArgumentParser()
parser.add_argument('--field', type=str, default='UDS')
parser.add_argument('--ver', type=str, default='test', help="catalog is MINERVA-{field}_{ver}_CATALOG.fits")
parser.add_argument('--spsver', type=str, default='spsv0.0')
parser.add_argument('--outdir', type=str, default='../test_prospector/', help='base directory for all outputs')
parser.add_argument('--dyn', type=int, default=1,
                    help="real fits that we're using should be 0! 1 is faster fit for testing, 2 is fully debug mode")
parser.add_argument('--ids', type=int, nargs='*', default=None, help='catalog ids to fit (default: all)')
parser.add_argument('--steps', type=int, nargs='*', default=[1, 2, 3], help='which steps to run')
args = parser.parse_args()

if __name__ == '__main__':

    print("make sure youre in the prosp environment!! this script doesn't activate it for you :P")
    python = sys.executable + ' ' # run every step with the same python as this script

    outdir = os.path.abspath(args.outdir) + '/'
    catalog = 'MINERVA-{}_{}_CATALOG.fits'.format(args.field, args.ver)
    catalog_path = os.path.abspath(ut_cwd.photdir + catalog)
    chaindir = outdir + 'chains_parrot_{}_{}/'.format(args.ver, args.spsver) # where to put the chains
    postdir = outdir + 'postprocess_{}_{}/'.format(args.ver, args.spsver) # where postprocessing goes
    prior = 'phisfh'

    cat = Table.read(catalog_path)
    ids = np.array(cat['id']) if args.ids is None else np.array(args.ids)
    assert len(ids) <= 100 # let's not go crazy here -- obviously update later!

    for d in [chaindir, postdir + 'npz/', postdir + 'fail/', outdir + 'log/']:
        os.makedirs(d, exist_ok=True)

    ################################## step 1. sed fit ####################################

    if 1 in args.steps:
        # the fit script loops over every integer id in [idx0, idx1) and skips ids that are
        # not in the catalog, so fit one id per call to keep the loop short
        for objid in ids:
            _cmd = 'uncover_gen1_parrot_phisfh_params.py --catalog {} --idx0 {} --idx1 {} --outdir {} --dyn {}'.format(
                catalog, int(objid), int(objid) + 1, chaindir, args.dyn)
            print(_cmd)
            os.system(python + _cmd)

    ################################ step 2. post-processing ################################

    if 2 in args.steps:
        # absolute paths, so ddir is just a placeholder (as in run_postprocess_blanca.py)
        _cmd = "postprocess_parrot_wrap.py --prior {} --fit 'fid' --catalog {} --indir {} --outdir {} --narr 1 --iarr 0 --ids_file None --ddir / --faildir {}".format(
            prior, catalog_path, chaindir, postdir + 'npz/', postdir + 'fail/')
        print(_cmd)
        os.system(python + _cmd)

    ########################## step 3. parse individual results into summary files ##########################

    if 3 in args.steps:
        dirs = '--dir_indiv {} --dir_collected {}'.format(postdir + 'npz/', postdir)
        for _cmd in ['save_sfh.py {} --n_workers 1'.format(dirs),
                     'save_chain.py {} --n_workers 1'.format(dirs),
                     'save_spec.py {} --catalog_path {} --n_workers 1'.format(dirs, catalog_path),
                     'save_perc.py {} --catalog_path {} --n_workers 1'.format(dirs, catalog_path),
                     'save_final_catalog.py {} --catalog_path {} --sps_version {}'.format(dirs, catalog_path, args.spsver)]:
            print(_cmd)
            os.system(python + _cmd)
