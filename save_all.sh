#!/bin/bash
#SBATCH --job-name=run_chan_job
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --output=./../log_chain/%x_%j.out
#SBATCH --error=./../log_chain/%x_%j.err

module purge
module load slurm/blanca

python save_sfh.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2_massive20000/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2_massive10000_fagn1e5" --n_workers=20

### this is not good due to time out, run save_chain.sh+combine_chain.py first
python save_chain.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2_massive10000_fagn1e5/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2_massive10000_fagn1e5" --n_workers=20

python save_spec.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2_massive10000_fagn1e5/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2_massive10000_fagn1e5" --catalog_path="/projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/phot_catalog/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_CATALOG.fits" --n_workers=20

python save_perc.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2_massive10000_fagn1e5/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2_massive10000_fagn1e5" --catalog_path="/projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/phot_catalog/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_CATALOG.fits" --n_workers=20

python save_final_catalog.py --dir_indiv="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2_massive10000_fagn1e5/npz" --dir_collected="/scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2_massive10000_fagn1e5" --catalog_path="/projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/phot_catalog/MINERVA-UDS_n3.0_v1.2_ACS+WEBB_Kf444w_SUPER_CATALOG.fits"


