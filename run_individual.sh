#!/bin/bash -l
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ind
#SBATCH --output=./../log_chain/ind_%A_%a.out
#SBATCH --error=./../log_chain/ind_%A_%a.err

now=$(date +"%T")
echo "start time ... $now"

module purge
module load anaconda
source activate /projects/kasu8993/software/anaconda/envs/prosp

python uncover_gen1_parrot_phisfh_params.py --catalog MINERVA-EGS_n2.0_v1.3_ACS+WEBB_Kf444w_SUPER_CATALOG.fits --outdir /pl/active/suess-minerva/chains_parrot_EGS_n2.0_v1.3_ACS+WEBB_Kf444w_SUPER_spsbeta --dyn 0 --idx0 3415896 --idx1 3415897

now=$(date +"%T")
echo "end time ... $now"
