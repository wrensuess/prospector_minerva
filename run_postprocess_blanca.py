# postprocess on blanca

import os, sys, time
import numpy as np
from astropy.table import Table
from utils import get_dir 

# set up paths 
# change these as needed!
catalog = '/projects/kasu8993/minerva/data/MINERVA-UDS_n2.2_m2.0_v1.0_LW_Kf444w_SUPER_CATALOG.fits'
indir = '/scratch/alpine/ikmi3774/slurm/tmp_test/chains_test/'
outdir = '/scratch/alpine/kasu8993/test_postprocess/'
log_dir = '/scratch/alpine/kasu8993/test_postprocess/logs/'
ddir = './' # we're giving absolute paths so this is just a placeholder 
narr = 0; iarr = 0 # we're going to make our own ID lists
env = '/projects/kasu8993/software/anaconda/envs/prosp'
spsver = 'spsv0.0'
sps = 'parrot' 
prior = 'phisfh' # 'phisfhzfixed' if you're using zspec fits
code_dir = '/projects/kasu8993/minerva/prospector_minerva/'

njobs = 20 #number of job array, max=1000
wtime = int(12) # time
jname = 'postproc_{}'.format(prior)
ts = time.strftime("%y%b%d-%H.%M", time.localtime())

# first let's split up the finished IDs into njobs parts
ids = np.array([int(i.split('_')[1]) for i in  os.listdir(indir)]) 
ids_split = np.array_split(ids, njobs)
# save each chunk to a separate file
os.makedirs(outdir+'id_files', exist_ok=True)
for i, ids_chunk in enumerate(ids_split):
    np.savetxt(f'{outdir}/id_files/ids_postprocess_{i}.txt', ids_chunk, fmt='%d')

# now create the command to submit
_cmd = "postprocess_parrot_wrap.py --prior {} --fit 'fid' --catalog {} --indir {} --outdir {} --narr {} --iarr {} --ids_file {} --ddir {}".format(prior, 
    catalog, indir, outdir, narr, iarr, outdir+'id_files/ids_postprocess_${SLURM_ARRAY_TASK_ID}', ddir)

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
with open(outdir+'slurm_postprocess.sh', 'w') as f:
    f.write(txt_acc)

os.system('sbatch slurm_postprocess.sh')    