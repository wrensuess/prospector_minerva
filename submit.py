import os, sys, time
import numpy as np
from astropy.table import Table

def data_dir():

    ''' 
    TODO: update for our data storage directories
    '''
    dat_dirs = ['/projects/ikmi3774/']
    
    for _dir in dat_dirs:
        if os.path.isdir(_dir): return _dir

def run_params(pycmd, log_dir='log', acc='bc', i=0, jobname='p', wtime=48, env='prosp'):
    
    jname = '{}_{}'.format(jobname, i)
    
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())

    ''' 
    TODO: update slurm file for our supercomputer, whatever we end up using
    '''
    if acc == 'bc':
        txt_acc = '\n'.join(["#!/bin/bash -l",
                             "#SBATCH --account=ucb-general\n",
                             "#SBATCH --partition=amilan\n"])
        
    txt_acc += "#SBATCH --time={:d}:00:00\n".format(wtime)

    txt_2 = '\n'.join([
        "#SBATCH --nodes=1",
        "#SBATCH --job-name={}".format(jname[:16]),
        "#SBATCH --output={}/{}_{}.out".format(log_dir, jname, ts),
        "#SBATCH --error={}/{}_{}.err".format(log_dir, jname, ts),
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        'module load anaconda',
        "source activate {}".format(env),
        "",
        "cd /projects/ikmi3774/minerva_sps_gen1/stellar_pop_catalog_bb/prospector_minerva",
        "python {}".format(pycmd),
        "",
        'now=$(date +"%T")',
        'echo "end time ... $now"',
        ""])

    txt = txt_acc + txt_2

    f = open('_params.slurm','w')
    f.write(txt)
    f.close()
    os.system('sbatch _params.slurm')
    os.system('rm _params.slurm')
    return None


if __name__ == '__main__':
    
    field = 'UDS'
    ver = 'v0.01_LW_Kf444w_SUPER'
    spsver = 'spsv0.01'
    outdir = '../test_slurm/'
    chaindir = outdir+'chains_parrot_{}_{}'.format(ver, spsver)
    logdir = outdir+'log/{}'.format(outdir)
    fast_dyn = 2

    #ncores = len(tot)
    acc = 'bc' ### we do not have to use this specification, but useful if we use both alpine&blanca
    ncores = 5 #840 # number of cores to request
    wtime = int(24) #int(24*7) # time

    ################################## step 1. sed fit ####################################

    catalog = 'MINERVA-{}_{}_CATALOG.fits'.format(field, ver)

    cat = Table.read('../phot_catalog/' + catalog)
    tot = np.arange(len(cat))

    ''' TODO: what is nfiles_phot? how do we fit sub-portion of the phot catalog? '''
    tot = []
    for _id in cat['id'].data:
        #if _id not in nfiles_phot:### nfiles_photo is not clear
        tot.append(_id)
    tot = np.array(tot)
    print(tot)
    tot = tot - 1 # id to idx # this only works if using the full phot catalog

    groups = np.array_split(tot, ncores) # divide the total number into xxx cores

    isExist = os.path.exists(outdir+chaindir)
    if not isExist:
        os.makedirs(outdir)
        print("new output directory created:", outdir+chaindir)
    isExist = os.path.exists(logdir)
    if not isExist:
        os.makedirs(logdir)
        print("new log directory created:", logdir)
   
    for igroup in range(len(groups)):
        idx0 = groups[igroup][0]
        idx1 = groups[igroup][-1] + 1 # +1 b/c id1 is not included when running the fit
        if 'zspec' in catalog: ### not edited
            _cmd = 'uncover_gen1_parrot_phisfhzspec_params.py --catalog {} --idx0 {} --idx1 {} --outdir {}'.format(catalog, idx0, idx1, outdir)
        else:
            _cmd = 'uncover_gen1_parrot_phisfh_params.py --catalog {} --idx0 {} --idx1 {} --outdir {} --dyn {}'.format(catalog, idx0, idx1, outdir+chaindir, fast_dyn)
        if igroup == 0:
            print(_cmd)
        run_params(_cmd, jobname='bb', log_dir=logdir, acc=acc, i=idx0, wtime=wtime, env='prosp-dev')
        time.sleep(0.05)


    '''
    ################################ step 2. post-prsocessing ################################

    wtime = 48
    catalog = 'MINERVA-{}_{}_CATALOG.fits'.format(field, ver)

    sps = 'parrot'
    if 'zspec' in catalog:
        indir = 'chains_parrot_zspec_{}_{}'.format(ver, spsver)
        prior = 'phisfhzspec'
    else:
        indir = 'chains_parrot_{}_{}'.format(ver, spsver)
        prior = 'phisfh'

    run = 'std'
    
    outdir = indir
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print("new output directory created:", outdir)
    logdir = 'log/{}'.format(outdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        print("new log directory created:", logdir)

    acc = 'bc'

    ids_file  = 'None' # can also pass a .txt file that contains the ids of the sources that need to perform the post-prsocessing on

    ## have to be matched to that in postprocess_parrot_wrap.py
    n_split_arr = 800 # number of cores

    for i in range(n_split_arr):
        _cmd = "postprocess_parrot_wrap.py --prior {} --fit 'fid' --catalog {} --indir {} --outdir {} --narr {} --iarr {} --ids_file {} --run {}".format(prior, catalog, indir, outdir, n_split_arr, i, ids_file, run)
        if i == 0:
            print(_cmd)
        run_params(_cmd, log_dir=logdir, acc=acc, i=i, jobname='p', wtime=wtime, env='prosp-dev')
        time.sleep(0.05)


    ########################## step 3. parse individual results into summary files ##########################

    # saves transformed chains (i.e., those published in the data release)
    _cmd = 'save_chain.py --catalog UNCOVER_{}_CATALOG.fits --indir post_parrot_{}_{}'.format(ver, ver, spsver)
    print(_cmd)
    run_params(_cmd, jobname='chain', log_dir='log', acc='sc', i=0, wtime=10)

    # saves zred, total_mass, logsfr_ratios
    _cmd = 'save_chain_untrans.py --catalog UNCOVER_{}_CATALOG.fits --indir post_parrot_{}_{} --prior {}'.format(ver, ver, spsver, prior)
    print(_cmd)
    run_params(_cmd, jobname='chainu', log_dir='log', acc='sc', i=0, wtime=10)
    
    _cmd = 'save_sfh.py --catalog UNCOVER_{}_CATALOG.fits --indir post_parrot_{}_{}'.format(ver, ver, spsver)
    print(_cmd)
    run_params(_cmd, jobname='sfh', log_dir='log', acc='sc', i=0, wtime=10)
    
    _cmd = 'save_spec.py --catalog UNCOVER_{}_CATALOG.fits --chain_indir chains_parrot_{}_{} --perc_indir chains_parrot_{}_{} --outdir results'.format(ver, ver, spsver, ver, spsver)
    print(_cmd)
    run_params(_cmd, jobname='spec', log_dir='log/', acc='sc', i=0, wtime=10)
    '''
