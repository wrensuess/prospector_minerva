#!/bin/bash
#SBATCH --job-name=run_chan_job
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --array=0-459%20
#SBATCH --output=./../log_chain/%x_%j.out
#SBATCH --error=./../log_chain/%x_%j.err

module purge

CHUNK=500
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))

python save_chain.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2" --n_workers=1 --start $START --end $END --split_id ${START}_${END}
