#!/bin/bash
#SBATCH --job-name=run_chan_job
#SBATCH --partition=blanca-casa
#SBATCH --qos=blanca-casa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --output=./../log_chain/%x_%j.out
#SBATCH --error=./../log_chain/%x_%j.err

module purge
module load slurm/blanca

python save_chain.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/" --n_workers=10
