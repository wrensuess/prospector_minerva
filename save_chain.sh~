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
module load slurm/blanca

#python save_sfh.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2" --n_workers=20

#python save_chain.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2" --n_workers=20

#python save_spec.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2" --catalog_path="/projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/phot_catalog/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_CATALOG.fits" --n_workers=20

#python save_perc.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2" --catalog_path="/projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/phot_catalog/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_CATALOG.fits" --n_workers=20

#python save_final_catalog.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2" --catalog_path="/projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/phot_catalog/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_CATALOG.fits"

CHUNK=500
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))

python save_chain.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2" --n_workers=20 --start $START --end $END --split_id ${START}_${END}
