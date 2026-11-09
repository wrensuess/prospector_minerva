#!/bin/bash -l
# Does each fit in the production layout (21 tasks/node, CURC loadbalance) get ONE core?
#
# Same header and launcher as slurm_mpitest.sh / submit.py. Each of the 20 worker lines
# runs check_binding.py, which prints the process's CPU affinity, its OS thread count
# after a jitted workload, and CPU/wall. Two rounds: (a) the launcher exactly as
# production uses it, (b) with explicit core binding (OpenMPI syntax; check which MPI
# `mpirun --version` reports).
#
# What to look for: each line's "on N physical core(s)" must be 1.
# ALREADY MEASURED by the prospector-minerva session (2026-10-08, OpenMPI 4.1.1):
# - nodes without hyperthreading: (a) already binds each worker to one core;
# - hyperthreaded nodes (Xeon Gold 6130): (a) gives 20 fits 11 cores, unbound, and (b)
#   fails ("#cpus: 1"). The fix is --hint=nomultithread, which this script does NOT set,
#   so that it reproduces the problem:
#     sbatch --hint=nomultithread check_binding.sbatch     (to verify the fix)
#
#   sbatch check_binding.sbatch          (from nss_test/blanca/)
#
#SBATCH --account=blanca-casa
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=21
#SBATCH --job-name=check_binding
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ---- paths (see README.md)
NSS_ROOT=/projects/$USER/nss_blanca/nss_test
ENV=/projects/$USER/software/anaconda/envs/nss      # or .../envs/prosp to check production
SCRATCH_TASKS=${SLURM_SUBMIT_DIR}/check_binding_tasks_${SLURM_JOB_ID}.txt

module purge   # (slurm/blanca is loaded on the LOGIN node before sbatch; inside a job it only gives an Lmod error)
module load loadbalance
module load anaconda
source activate "$ENV"
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "===== $(date) on $(hostname), job $SLURM_JOB_ID, $SLURM_NTASKS tasks"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2); node CPUs $(nproc --all); job allowed $(nproc)"
$CURC_LB_BIN/mpirun --version | head -2

for i in $(seq 1 20); do echo "python -u $NSS_ROOT/blanca/check_binding.py"; done > "$SCRATCH_TASKS"

echo "===== (a) as in production: \$CURC_LB_BIN/mpirun lb"
$CURC_LB_BIN/mpirun lb "$SCRATCH_TASKS"

echo "===== (b) with core binding: --bind-to core --map-by core"
$CURC_LB_BIN/mpirun --bind-to core --map-by core lb "$SCRATCH_TASKS"

rm -f "$SCRATCH_TASKS"
echo "===== $(date) done"
