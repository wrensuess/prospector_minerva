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

def run_params(task_dir='hoge', log_dir='hoge', acc='bc', jobname='p', wtime=24, env='prosp', njob=10):
    
    jname = '{}'.format(jobname)
    
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())

    if acc == 'bc':
        txt_acc = '\n'.join(["#!/bin/bash -l",
                             "#SBATCH --account=blanca-casa\n",
                             "#SBATCH --partition=blanca\n",
                             "#SBATCH --qos=preemptable\n"])
                             #"#SBATCH --account=ucb-general\n",
                             #"#SBATCH --partition=amilan\n"])
                             
        
    txt_acc += "#SBATCH --time={:d}:00:00\n".format(wtime)

    txt_2 = '\n'.join([
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=21", #it's better to use numbers which can split 200(=Nsource/Njobarray), +1 for loadbalancer
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
        "cd /projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/prospector_minerva",
        "mpirun lb {}/taskfile_${{SLURM_ARRAY_TASK_ID}}.txt".format(task_dir),
        "",
        'now=$(date +"%T")',
        'echo "end time ... $now"',
        ""])

    txt = txt_acc + txt_2

    f = open('_params1.sh','w')
    f.write(txt)
    f.close()
    os.system('sbatch _params1.sh')
    #os.system('rm _params1.sh')
    return None




if __name__ == '__main__':

    # check the number of arguments
    if len(sys.argv)!=3:
        print('Error! usage: python '+sys.argv[0]+' [cathead] [field] [ver]')
        sys.exit()
    cathead = sys.argv[1]
    field = sys.argv[2] 
    ver = sys.argv[3]
    
    #field = 'UDS'
    #ver = 'n2.2_m2.0_v1.0_LW_Kf444w_SUPER'
    #ver = 'n2.3_v1.1_LW_Kf444w_SUPER'
    spsver = 'spsbeta'
    #outdir = '../slurm/'
    outdir = '/scratch/alpine/ikmi3774/slurm/'
    catdir = '../phot_catalog/'
    chaindir = outdir+'chains_parrot_{}_{}'.format(ver, spsver)
    #chaindir = '/scratch/alpine/ikmi3774/slurm/chains_parrot_{}_{}'.format(ver, spsver)
    logdir = outdir+'log_'+field
    taskdir = outdir+'task_lists_'+field
    fast_dyn = 2#0 #0:std run, 1:brief run, 2:debug

    acc = 'bc' ### we do not have to use this specification, but useful if we use both alpine&blanca
    env = 'prosp'
    #ncores = len(tot)
    #ncores = 5 #840 # number of cores to request
    njobs = 1000 #number of job array, max=1000
    wtime = int(24) #int(24*7) # time

    ################################## step 1. sed fit ####################################
    catalog = 'MINERVA-{}_{}_CATALOG.fits'.format(field, ver)

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

    
    
    fitcatalog = catdir+cathead+'_MINERVA-'+field+'_'+ver+'_CATALOG.txt' #file includes use_photo=1 sources
    ids_fit = np.loadtxt(fitcatalog)
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
    run_params(jobname='bb', task_dir=taskdir, log_dir=logdir, acc=acc, wtime=wtime, env=env, njob=njobs)
    time.sleep(0.05)

