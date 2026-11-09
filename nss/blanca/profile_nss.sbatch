#!/bin/bash -l
# Profile the pure-JAX NSS fit of UNCOVER 58858 on ONE core, as production fits run.
#
# One task, one CPU, --hint=nomultithread: Slurm confines the job to ONE PHYSICAL core
# (on a hyperthreaded node, its two hyperthreads, as production fits get with the same
# hint), and XLA sizes its thread pool to that affinity. (XLA_FLAGS cannot do this: jax
# 0.6.1 and 0.10.2 ignore --xla_cpu_multi_thread_eigen / intra_op_parallelism_threads.)
#
# Runs on CASA's own Blanca nodes (as chain_jobs.sh does), not preemptable: a preempted
# profile would restart, and preemptable jobs land on mixed node types. Every variant runs
# in this one job, on one node, so the variants compare on the same CPU.
#
# Order: (1) the two numerics tests at reduced size (XLA on x86 may round differently
# from the Mac; stop if they fail), (2) the profile (README.md says what to look at).
#
#   sbatch profile_nss.sbatch            (from nss_test/blanca/; CASA's own nodes, full profile)
#
# On preemptable nodes (what production uses), command-line options override the lines
# below; use the short variant list so a preemption costs less (with --requeue a
# preempted job starts over):
#   sbatch --partition=blanca --account=blanca-casa --qos=preemptable --requeue \
#          --time=02:00:00 --export=ALL,SKIP_TESTS=0,PROFILE_ARGS="--skip-micro \
#          --variants baseline fix_twopoint fix_tables baseline" profile_nss.sbatch
#
# Environment knobs: SKIP_TESTS=1 skips the numerics tests (only on a CPU model they
# already passed on); PROFILE_ARGS replaces the profile's default arguments.
#
#SBATCH --job-name=nss_profile
#SBATCH --partition=blanca-casa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ---- paths (see README.md)
NSS_ROOT=/projects/$USER/nss_blanca/nss_test                    # copy of MINERVA/nss_test
export EFSPS_PARENT=/projects/$USER/nss_blanca/efsps-photfit    # copy of MINERVA/efsps-photfit
NSS_ENV=/projects/$USER/software/anaconda/envs/nss   # built from requirements_nss.txt

module purge   # (slurm/blanca is loaded on the LOGIN node before sbatch; inside a job it only gives an Lmod error)
module load anaconda
source activate "$NSS_ENV"

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "===== $(date) on $(hostname), job $SLURM_JOB_ID"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
echo "nproc: $(nproc); affinity: $(taskset -cp $$)"
python -c "import jax, blackjax, numpy, scipy; print('jax', jax.__version__, 'blackjax', blackjax.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"

cd "$NSS_ROOT" || exit 1
mkdir -p logs results

echo "partition $SLURM_JOB_PARTITION, qos ${SLURM_JOB_QOS:-?}, restart count ${SLURM_RESTART_COUNT:-0}"

if [ "${SKIP_TESTS:-0}" = "1" ]; then
    echo "===== numerics tests SKIPPED (SKIP_TESTS=1)"
else
echo "===== numerics tests (reduced size)"
python -u test_jax_prior.py --n 50000 --n-fixed 5000 > logs/test_jax_prior_blanca_${SLURM_JOB_ID}.log 2>&1
tail -1 logs/test_jax_prior_blanca_${SLURM_JOB_ID}.log
python -u test_jax_loglike.py --n 30000 --n-fixed 3000 > logs/test_jax_loglike_blanca_${SLURM_JOB_ID}.log 2>&1
tail -1 logs/test_jax_loglike_blanca_${SLURM_JOB_ID}.log
# the tabulated mass prior: vs the exact prior, then JAX vs numpy with it
python -u test_tabulated_mass_prior.py > logs/test_tabulated_mass_prior_blanca_${SLURM_JOB_ID}.log 2>&1
tail -1 logs/test_tabulated_mass_prior_blanca_${SLURM_JOB_ID}.log
python -u test_jax_prior.py --mass-prior tabulated --n 50000 --n-fixed 5000 > logs/test_jax_prior_tab_blanca_${SLURM_JOB_ID}.log 2>&1
tail -1 logs/test_jax_prior_tab_blanca_${SLURM_JOB_ID}.log
if ! grep -q "RESULT: PASS" logs/test_jax_prior_blanca_${SLURM_JOB_ID}.log || \
   ! grep -q "RESULT: PASS" logs/test_jax_loglike_blanca_${SLURM_JOB_ID}.log || \
   ! grep -q "RESULT: PASS" logs/test_tabulated_mass_prior_blanca_${SLURM_JOB_ID}.log || \
   ! grep -q "RESULT: PASS" logs/test_jax_prior_tab_blanca_${SLURM_JOB_ID}.log; then
    echo "numerics tests FAILED on this machine; not profiling. See logs/test_jax_*_blanca_${SLURM_JOB_ID}.log"
    exit 1
fi
fi

echo "===== profile ${PROFILE_ARGS:-(default variants)}"
python -u profile_nss.py --n-it 60 ${PROFILE_ARGS:-} \
    --out results/profile_58858_blanca_${SLURM_JOB_ID}.json \
    2>&1 | grep -v Warning | tee logs/profile_58858_blanca_${SLURM_JOB_ID}.log

echo "===== $(date) done"
