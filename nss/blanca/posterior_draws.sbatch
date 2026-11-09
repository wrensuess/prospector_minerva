#!/bin/bash -l
# Run posterior_draws.py on every inner-steps run directory (one core, ~10-20 min of
# reading). Submit from /projects/$USER/nss_blanca/nss_test/blanca.
#SBATCH --account=blanca-casa
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=nss_draws
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

NSS_ROOT=/projects/$USER/nss_blanca/nss_test
module purge
module load anaconda
source activate /projects/$USER/software/anaconda/envs/nss
cd "$NSS_ROOT" || exit 1
python -u blanca/posterior_draws.py results/blanca_layout_*_k*_L*
