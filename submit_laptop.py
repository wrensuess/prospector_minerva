import os, sys, time
import numpy as np
from astropy.table import Table
from utils import get_dir


if __name__ == '__main__':
    
    field = 'UDS'
    ver = 'test' #'v0.0_LW_Kf444w_SUPER'
    spsver = 'spsv0.0'
    outdir = '../test_prospector/' # make sure slash at end
    ncores = 1 # this might actually be the number of fits that each core does? test.
    chaindir = 'chains_parrot_{}_{}'.format(ver, spsver) # where to put the chains
    fast_dyn = 2 # real fits that we're using should be 0! 1 is faster fit for testing, 2 is fully debug mode

    ################################## step 1. sed fit ####################################
    
    print("make sure youre in the prosp environment!! this script doesn't activate it for you :P")

    catalog = 'MINERVA-{}_{}_CATALOG.fits'.format(field, ver)

    cat = Table.read(get_dir("phot", outdir)+catalog) # this can be updated as needed
    tot = np.arange(len(cat))
    assert len(tot) <= 5 # let's not go crazy here -- obviously update later!
    
    # divide the thing into cores
    groups = np.array_split(tot, ncores) # divide the total number into xxx cores

    # create output and log directories
    isExist = os.path.exists(outdir + chaindir)
    if not isExist:
        os.makedirs(outdir + chaindir)
        print("new output directory created:", outdir)
    logdir = outdir+'log/'
    isExist = os.path.exists(logdir)
    if not isExist:
        os.makedirs(logdir)
        print("new log directory created:", logdir)
   
    for igroup in range(len(groups)):
        idx0 = groups[igroup][0]
        idx1 = groups[igroup][-1] + 1 # +1 b/c id1 is not included when running the fit
        if 'zspec' in catalog:
            _cmd = 'uncover_gen1_parrot_phisfhzspec_params.py --catalog {} --idx0 {} --idx1 {} --outdir {}'.format(catalog, idx0, idx1, outdir)
        else:
            _cmd = 'uncover_gen1_parrot_phisfh_params.py --catalog {} --idx0 {} --idx1 {} --outdir {} --dyn {}'.format(catalog, idx0, idx1, outdir+chaindir, fast_dyn)
        if igroup == 0:
            print(_cmd)
        os.system('python '+_cmd)#, jobname='mb', log_dir=logdir, acc=acc, i=idx0, wtime=wtime, env='prosp-dev')
        time.sleep(0.05)


    ################################ step 2. post-prsocessing ################################

    wtime = 48
    sps = 'parrot'
    if 'zspec' in catalog:
        indir = 'chains_parrot_zspec_{}_{}'.format(ver, spsver)
        prior = 'phisfhzspec'
    else:
        indir = 'chains_parrot_{}_{}'.format(ver, spsver)
        prior = 'phisfh'

    run = 'std'
    acc = 'bc'

    ids_file  = 'None' # can also pass a .txt file that contains the ids of the sources that need to perform the post-prsocessing on

    ## have to be matched to that in postprocess_parrot_wrap.py
    # n_split_arr = 800 # number of cores
    n_split_arr = 1

    for i in range(n_split_arr):
        # _cmd = "postprocess_parrot_wrap.py --prior {} --fit 'fid' --catalog {} --indir {} --outdir {} --narr {} --iarr {} --ids_file {} --run {}".format(prior, catalog, indir, outdir, n_split_arr, i, ids_file, run)
        _cmd = "postprocess_parrot_wrap.py --prior {} --fit 'fid' --catalog {} --indir {} --outdir {} --narr {} --iarr {} --ids_file {} --ddir {}".format(prior, 
            catalog, indir, chaindir, n_split_arr, i, ids_file, outdir)
        if i == 0:
            print(_cmd)
        # run_params(_cmd, log_dir=logdir, acc=acc, i=i, jobname='p', wtime=wtime, env='prosp-dev')
        os.system('python '+_cmd)
        time.sleep(0.05)


    ########################## step 3. parse individual results into summary files ##########################
    
    # make sure directory exists
    if not os.path.exists('{}/post_parrot_{}_{}'.format(outdir, ver, spsver)):
        os.makedirs('{}/post_parrot_{}_{}'.format(outdir, ver, spsver))
        print("new log directory created:", logdir)

    # saves transformed chains (i.e., those published in the data release)
    _cmd = 'save_chain.py --dir_indiv {}/chains_parrot_{}_{} --dir_collected {}post_parrot_{}_{}/results'.format(outdir, ver, spsver, outdir, ver, spsver)
    print(_cmd)
    os.system('python '+_cmd) 
    # run_params(_cmd, jobname='chain', log_dir='log', acc='sc', i=0, wtime=10)

    # saves zred, total_mass, logsfr_ratios
    _cmd = 'save_chain_untrans.py --dir_indiv {}/chains_parrot_{}_{} --dir_collected {}post_parrot_{}_{}/results'.format(outdir, ver, spsver, outdir, ver, spsver)
    print(_cmd)
    # run_params(_cmd, jobname='chainu', log_dir='log', acc='sc', i=0, wtime=10)
    os.system('python '+_cmd) 
       
    _cmd = 'save_sfh.py --dir_indiv {}/chains_parrot_{}_{} --dir_collected {}post_parrot_{}_{}/results'.format(outdir, ver, spsver, outdir, ver, spsver)
    print(_cmd)
    # run_params(_cmd, jobname='sfh', log_dir='log', acc='sc', i=0, wtime=10)
    os.system('python '+_cmd)
    
    _cmd = 'save_spec.py --dir_indiv {}/chains_parrot_{}_{} --dir_collected {}post_parrot_{}_{}/results --catalog {} --basedir {}'.format(outdir, ver, spsver, outdir, ver, spsver, catalog, outdir)
    print(_cmd)
    # run_params(_cmd, jobname='spec', log_dir='log/', acc='sc', i=0, wtime=10)
    os.system('python '+_cmd)
    
    # saves percentiles
    _cmd = 'save_perc.py --dir_indiv {}/chains_parrot_{}_{} --dir_collected {}post_parrot_{}_{}/results --catalog {} --basedir {}'.format(outdir, ver, spsver, outdir, ver, spsver, catalog, outdir)
    print(_cmd)
    # run_params(_cmd, jobname='chainu', log_dir='log', acc='sc', i=0, wtime=10)
    os.system('python '+_cmd)
    
    # make final SPS catalog
    _cmd = 'make_final_sps_catalog.py --dir_indiv {}/chains_parrot_{}_{} --dir_collected {}post_parrot_{}_{}/results --catalog {} --basedir {} --ver {}'.format(outdir, ver, spsver, outdir, ver, spsver, catalog, outdir, ver)
    print(_cmd)
    # run_params(_cmd, jobname='chainu', log_dir='log', acc='sc', i=0, wtime=10)
    os.system('python '+_cmd)
