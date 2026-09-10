#!/bin/bash -l
#SBATCH --account=blanca-casa
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=21
#SBATCH --job-name=mpi_test

#SBATCH --output=./../log_chain/mpi_test_%j.out
#SBATCH --error=./../log_chain/mpi_test_%j.err

echo "===== START ====="
date
hostname

module purge
module load slurm/blanca
module load loadbalance
module load anaconda
source activate /projects/kasu8993/software/anaconda/envs/prosp

export SLURM_EXPORT_ENV=ALL

echo
echo "===== MODULES ====="
module list

echo
echo "===== MPI ====="
echo "which mpirun:"
which mpirun
echo
echo "mpirun version:"
mpirun --version

echo
echo "===== CURC LB MPI ====="
echo "CURC_LB_BIN = $CURC_LB_BIN"
echo "CURC LB mpirun = $CURC_LB_BIN/mpirun"
$CURC_LB_BIN/mpirun --version

echo
echo "===== LOADBALANCE ====="
which lb

echo
echo "===== PYTHON ====="
which python
python --version

echo
echo "===== SHARED MEMORY ====="
df -h /dev/shm
df -i /dev/shm

echo
echo "===== TMP ====="
df -h /tmp
df -i /tmp

echo
echo "===== SLURM ====="
echo "SLURM_JOB_ID       = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS       = $SLURM_NTASKS"

cd /projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/prospector_minerva

echo
echo "===== RUN ====="

TASKFILE=/scratch/alpine/ikmi3774/slurm/EGS3/task_lists_EGS_n2.0_v1.3/taskfile_0.txt

echo "taskfile = $TASKFILE"
wc -l "$TASKFILE"
head "$TASKFILE"

$CURC_LB_BIN/mpirun lb "$TASKFILE"

echo
echo "===== END ====="
date
