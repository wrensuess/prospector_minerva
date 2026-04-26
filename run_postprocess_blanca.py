# postprocess on blanca

import os, sys, time, glob
import numpy as np
from astropy.table import Table
from utils import get_dir 

# set up paths 
# change these as needed!
#catalog = '/projects/kasu8993/minerva/data/MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG.fits'
#indir = '/scratch/alpine/ikmi3774/slurm/chains_parrot_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_spsbeta/'
#outdir = '/scratch/alpine/kasu8993/postprocess_n2.2_m1.0_v1.0/'
#log_dir = '/scratch/alpine/kasu8993/postprocess_n2.2_m1.0_v1.0/logs/'

catalog = '/projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/phot_catalog/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_CATALOG.fits'
indir = '/scratch/alpine/ikmi3774/slurm/chains_parrot_UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_spsbeta/'
outdir = '/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/'
log_dir = '/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/logs/'
#outdir = '/scratch/alpine1/kasu8993/postprocess_UDS_n3.0_v1.2/'
#log_dir = '/scratch/alpine1/kasu8993/postprocess_UDS_n3.0_v1.2/logs/'

ddir = '/' # we're giving absolute paths so this is just a placeholder 
narr = 1; iarr = 0 # we're going to make our own ID lists
env = '/projects/kasu8993/software/anaconda/envs/prosp'
#env = 'prosp'
spsver = 'spsv0.0'
sps = 'parrot' 
prior = 'phisfh' # 'phisfhzfixed' if you're using zspec fits
#code_dir = '/projects/kasu8993/prospector_minerva/'
code_dir= '/projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/prospector_minerva/'

njobs = 1#1000 #number of job array, max=1000
wtime = int(24) # time
jname = 'postproc_{}'.format(prior)
ts = time.strftime("%y%b%d-%H.%M", time.localtime())

# first let's split up the finished IDs into njobs parts
ids = np.array([int(i.split('_')[1]) for i in os.listdir(indir)])
#ids = ids[:200]

last = 0#0

if last!=0:
    ids_split = np.array_split(ids, njobs)
    # save each chunk to a separate file
    os.makedirs(outdir+'id_files', exist_ok=True)
    for i, ids_chunk in enumerate(ids_split):
        np.savetxt(f'{outdir}/id_files/ids_postprocess_{i}.txt', ids_chunk, fmt='%d')

    # now create the command to submit
    _cmd = "python -u postprocess_parrot_wrap.py --prior {} --fit 'fid' --catalog {} --indir {} --outdir {} --narr {} --iarr {} --ids_file {} --ddir {}".format(prior, 
        catalog, indir, outdir+'npz/', narr, iarr, outdir+'id_files/ids_postprocess_${SLURM_ARRAY_TASK_ID}.txt', ddir)

    # and make our slurm file 
    txt_acc = '\n'.join(["#!/bin/bash -l",
                                 "#SBATCH --account=blanca-casa\n",
                                 "#SBATCH --partition=blanca\n",
                                 "#SBATCH --qos=preemptable\n",
                                 "#SBATCH --time={:d}:00:00\n".format(wtime),
                                 "#SBATCH --nodes=1",
                                 "#SBATCH --ntasks=1",
                                 "#SBATCH --job-name={}".format(jname),
                                 "#SBATCH --array=0-{}".format(int(njobs-1)),
                                 "#SBATCH --output={}/{}_{}_%A_%a.out".format(log_dir, jname, ts),
                                 "#SBATCH --error={}/{}_{}_%A_%a.err".format(log_dir, jname, ts),
                                 "",
                                 'now=$(date +"%T")',
                                 'echo "start time ... $now"',
                                 'echo "Running task ID: ${SLURM_ARRAY_TASK_ID}"',
                                 "",
                                 'module purge',
                                 'module load anaconda',
                                 "source activate {}".format(env),
                                 "",
                                 "cd {}".format(code_dir),
                                 _cmd,
                                 'now=$(date +"%T")',
                                 'echo "end time ... $now"',
                                 ""])

if last==0: 
    # this is added to adjust ids, for the last ~a few % of the sources
    ids_comp_ = glob.glob(outdir+"npz/*_spec_phisfh.h5")
    ids_comp = [float(a.split("/")[-1].split("_")[1]) for a in ids_comp_]
    ids_fit = ids[np.where(np.isin(ids, ids_comp)==False)[0]]
    ids = ids_fit
    print(len(ids))
    print(dfafa)
    np.savetxt(f'{outdir}/id_files/ids_postprocess_0.txt', ids, fmt='%d')
    
    _cmd = "python -u postprocess_parrot_wrap.py --prior {} --fit 'fid' --catalog {} --indir {} --outdir {} --narr {} --iarr {} --ids_file {} --ddir {}".format(prior, 
        catalog, indir, outdir+'npz/', narr, iarr, outdir+'id_files/ids_postprocess_0.txt', ddir)

    # and make our slurm file 
    txt_acc = '\n'.join(["#!/bin/bash -l",
                                 "#SBATCH --account=blanca-casa\n",
                                 "#SBATCH --partition=blanca\n",
                                 "#SBATCH --qos=preemptable\n",
                                 "#SBATCH --time={:d}:00:00\n".format(wtime),
                                 "#SBATCH --nodes=1",
                                 "#SBATCH --ntasks=1",
                                 "#SBATCH --job-name={}".format(jname),
                                 "#SBATCH --array=0",
                                 "#SBATCH --output={}/{}_{}_%A_%a.out".format(log_dir, jname, ts),
                                 "#SBATCH --error={}/{}_{}_%A_%a.err".format(log_dir, jname, ts),
                                 "",
                                 'now=$(date +"%T")',
                                 'echo "start time ... $now"',
                                 'echo "Running task ID: 0"',
                                 "",
                                 'module purge',
                                 'module load anaconda',
                                 "source activate {}".format(env),
                                 "",
                                 "cd {}".format(code_dir),
                                 _cmd,
                                 'now=$(date +"%T")',
                                 'echo "end time ... $now"',
                                 ""])
    

with open(outdir+'slurm_postprocess.sh', 'w') as f:
    f.write(txt_acc)

os.system('sbatch {}/slurm_postprocess.sh'.format(outdir))    
