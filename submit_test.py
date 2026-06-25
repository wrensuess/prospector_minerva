import os, sys, time
import numpy as np
from astropy.table import Table

import utils as ut_cwd
mdir = ut_cwd.photdir

def data_dir():

    ''' 
    TODO: update for our data storage directories
    '''
    dat_dirs = ['/projects/ikmi3774/']
    
    for _dir in dat_dirs:
        if os.path.isdir(_dir): return _dir

def run_params(task_dir='hoge', log_dir='hoge', acc='priority', jobname='p', wtime=24, env='prosp', njob=10):
    
    jname = '{}'.format(jobname)
    
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())

    if acc == 'preempt':
        txt_acc = '\n'.join(["#!/bin/bash -l",
                             "#SBATCH --account=blanca-casa\n",
                             "#SBATCH --partition=blanca\n",
                             "#SBATCH --qos=preemptable\n"])
    if acc == 'priority':
        txt_acc = '\n'.join(["#!/bin/bash -l",
                             "#SBATCH --account=blanca-casa\n",
                             "#SBATCH --partition=blanca-casa\n",
                             "#SBATCH --qos=blanca-casa\n"])                             
        
    txt_acc += "#SBATCH --time={:d}:00:00\n".format(wtime)

    txt_2 = '\n'.join([
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=2", #1 for main job +1 for loadbalancer
        "#SBATCH --job-name={}".format(jname),
        "#SBATCH --array=0-{}".format(int(njob-1)),
        "#SBATCH --output={}/{}_{}_%A_%a.out".format(log_dir, jname, ts),
        "#SBATCH --error={}/{}_{}_%A_%a.err".format(log_dir, jname, ts),
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        'echo "Running task ID: ${SLURM_ARRAY_TASK_ID}"',
        "",
        'module purge',
        'module load slurm/blanca', ### does "purge" purge the slurm? just in case
        'module load loadbalance',
        'module load anaconda',
        "source activate {}".format(env),
        "",
#        "export OMP_NUM_THREADS=1",
#        "export OPENBLAS_NUM_THREADS=1",
#        "export MKL_NUM_THREADS=1",
#        "export NUMEXPR_NUM_THREADS=1",
#        'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"',
#        "",
        "cd /projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/prospector_minerva",
        "mpirun lb {}/taskfile_${{SLURM_ARRAY_TASK_ID}}.txt".format(task_dir),
        "",
        'now=$(date +"%T")',
        'echo "end time ... $now"',
        ""])

    txt = txt_acc + txt_2

    f = open('_params_test.sh','w')
    f.write(txt)
    f.close()
    os.system('sbatch _params_test.sh')
    time.sleep(0.5)
    os.system('rm _params_test.sh')
    return None




if __name__ == '__main__':

    ### variables
    outdir = '/scratch/alpine/ikmi3774/slurm/'
    catdir = '../phot_catalog/'
    field = 'UDS'
    ver = 'n3.0_v1.2'
    spsver = 'spsbeta'
    fast_dyn = 0 #0:std run, 1:brief run, 2:debug

    cujobname = 'massive20000_oldjax'
    cathead = 'massiveid'
    chaindir = outdir+'chains_parrot_{}_{}_ACS+WEBB_Kf444w_SUPER_{}_massive20000_oldjax'.format(field, ver, spsver)
    logdir = outdir+'log_'+field+"_"+ver+"_massive20000_oldjax"
    taskdir = outdir+'task_lists_'+field+"_"+ver+"_massive20000_oldjax"

    acc = 'priority' ### 'priority' or 'preempt'
    #env = 'prosp'
    env = '/projects/kasu8993/software/anaconda/envs/prosp_oldjax'
    njobs = 10 #number of job array, max=1000
    wtime = int(24) #int(24*7) # time

    ################################## step 1. sed fit ####################################
    catalog = 'MINERVA-{}_{}_ACS+WEBB_Kf444w_SUPER_CATALOG.fits'.format(field, ver)

    isExist = os.path.exists(chaindir)
    if not isExist:
        os.makedirs(chaindir)
        print("new output directory created:", chaindir)
    isExist = os.path.exists(logdir)
    if not isExist:
        os.makedirs(logdir)
        print("new log directory created:", logdir)
    isExist = os.path.exists(taskdir)
    if not isExist:
        os.makedirs(taskdir)
        print("new task directory created:", taskdir)
    
    
    fitcatalog = catdir+cathead+'_MINERVA-'+field+'_'+ver+'_ACS+WEBB_Kf444w_SUPER_CATALOG.txt' #file includes your objects
    ids_fit = np.loadtxt(fitcatalog)
    ids_fit = ids_fit[:20]
    print('[INFO] Ngalaxies to fit in '+cathead+':',len(ids_fit)) #[Nfit]


    ids_fit_split = np.array_split(ids_fit, njobs) #split [Nfit] into [njobs] jobs
    #print(ids_fit_split[0])
    for j in range(njobs):
        taskfile = taskdir+"/taskfile_{}.txt".format(int(j))
        ids_fit_injobarray = ids_fit_split[j] #each list MUST include [Nfit]/[njobs]
        isExist = os.path.exists(taskfile)
        if isExist:
            os.system('rm '+taskfile)
        with open(taskfile, mode="w") as f:
            for k in range(len(ids_fit_injobarray)):
                _cmd = 'python uncover_gen1_parrot_phisfh_params.py --catalog {} --outdir {} --dyn {} --idx0 {} --idx1 {}'.format(catalog, chaindir, fast_dyn, int(ids_fit_injobarray[k]), int(ids_fit_injobarray[k]+1))
                f.write(_cmd+"\n")
    run_params(jobname=cujobname, task_dir=taskdir, log_dir=logdir, acc=acc, wtime=wtime, env=env, njob=njobs)
    time.sleep(0.05)

