#!/bin/bash -l
# Many different NSS fits in the production layout (21 tasks per node, CURC loadbalance, one
# physical core per fit), as a job ARRAY: element i runs lines [20 i, 20 i + 20) of a task list,
# one fit per line:   CATALOG ID DUST SEED   (blanca/make_dust_tasks.py writes one).
# Production settings otherwise (nss_jax.py defaults): n_live 1000, num_delete 100, 32 inner steps,
# tabulated mass prior, mass integrated out, 1e6 prior start, fagn prior flat1_s025, no MIRI.
#
# Settings (via --export=ALL,NAME=value,...):
#   TASKLIST   task list, relative to nss_test/            (default blanca/dust_tasks.txt)
#   RUN        output folder name under results/            (default dust48)
#   CHUNK      fits per array element                       (default 20)
#   EXTRA      extra nss_jax.py options                     (default --no-full-npz: only the json,
#              the log and the _draws.npz are written, not the ~14 MB full sample file)
# Chunk size: lb hands each worker a new fit as soon as it finishes one, so an element of CHUNK fits
# keeps its 20 cores busy until the list runs out; the idle tail is about one fit per worker. Bigger
# chunks waste less (production uses 200); a preempted, requeued element redoes only the fits that
# had not finished. Elements = ceil(lines / CHUNK).
# Submit (from /projects/$USER/nss_blanca/nss_test/blanca, after `module load slurm/blanca`):
#   sbatch --array=0-28 tasklist_layout.sbatch
# Output: results/$RUN/<DUST>/<ID>_<DUST>_seed<S>.{json,log} and <...>_draws.npz (4000 equal-weight
# posterior draws, float32: the file to copy back), plus the full .npz only if EXTRA is set without
# --no-full-npz (never copy those back).
#
#SBATCH --account=blanca-casa
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --constraint="epyc9354|gold5418y|5418y|milan|icelake"
# CPU types within ~10% of the fastest per NSS iteration (STATUS.md section 6 item 13). On Blanca,
# `milan` is only the two EPYC 7413 nodes (the EPYC 7713 nodes carry epyc7713/epyc7713p instead);
# `icelake` is three Gold 6342 nodes plus one untimed 32-core node.
# For large runs also allow EPYC 7452 and Gold 6230 (~1.35x slower, many more nodes): add
# |rome|cascadelake on the command line, --constraint="epyc9354|gold5418y|5418y|milan|icelake|rome|cascadelake".
# Those tags also cover some untimed models (the 128- and 256-core Rome nodes, the non-40-core
# Cascade Lake nodes, which may include the Gold 6254).
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=21
#SBATCH --hint=nomultithread
#SBATCH --mem=32G                  # 20 fits x 1.37 GB peak private memory with malloc_tools.py (the step compile, all fits at once; worst CPU type, EPYC 7413): ~27.4 GB (memprof jobs 29478805, 29479115-20, 2026-11-11)
#SBATCH --job-name=nss_tasklist
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

NSS_ROOT=/projects/$USER/nss_blanca/nss_test
export EFSPS_PARENT=/projects/$USER/nss_blanca/efsps-photfit
NSS_ENV=/projects/$USER/software/anaconda/envs/nss

module purge
module load loadbalance
if [ ! -x "$NSS_ENV/bin/python" ]; then echo "no python in $NSS_ENV on $(hostname)"; exit 1; fi
export PATH=$NSS_ENV/bin:$PATH
if [ -z "$CURC_LB_BIN" ]; then echo "no loadbalance module on $(hostname)"; exit 1; fi
export SLURM_EXPORT_ENV=ALL
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

TASKLIST=${TASKLIST:-blanca/dust_tasks.txt}
RUN=${RUN:-dust48}
CHUNK=${CHUNK:-20}
EXTRA=${EXTRA---no-full-npz}     # set EXTRA= (empty) to also write the full .npz
I=${SLURM_ARRAY_TASK_ID:?submit as an array, e.g. --array=0-28}
cd "$NSS_ROOT" || exit 1
[ -f "$TASKLIST" ] || { echo "no task list $TASKLIST"; exit 1; }
echo "===== $(date) on $(hostname), job ${SLURM_ARRAY_JOB_ID}_${I}, restart ${SLURM_RESTART_COUNT:-0}"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2); node CPUs $(nproc --all); job allowed $(nproc)"
mkdir -p "$NSS_ROOT/results/$RUN"
TASKS=$NSS_ROOT/results/$RUN/taskfile_${SLURM_ARRAY_JOB_ID}_${I}.txt
FIRST=$((I * CHUNK + 1)); LAST=$((I * CHUNK + CHUNK))
sed -n "${FIRST},${LAST}p" "$TASKLIST" | while read -r CAT ID DUST SEED; do
    OUT=$NSS_ROOT/results/$RUN/$DUST
    mkdir -p "$OUT"
    STEM=${ID}_${DUST}_seed${SEED}
    # a preempted, requeued element skips fits that already finished
    if [ -f "$OUT/${STEM}.json" ]; then continue; fi
    echo "cd $NSS_ROOT && { python -u -c 'import os; a = sorted(os.sched_getaffinity(0)); c = {open(f\"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list\").read().strip() for i in a}; print(\"affinity\", a, \"=\", len(c), \"physical core(s)\", flush=True)'; /usr/bin/time -v python -u nss_jax.py --catalog $CAT --id $ID --tag ${DUST}_seed${SEED} --seed $SEED --dust-prior $DUST --save-draws 4000 $EXTRA --out $OUT; } > $OUT/${STEM}.log 2>&1"
done > "$TASKS"
echo "$(wc -l < "$TASKS") fits to run (lines $FIRST-$LAST of $TASKLIST)"
if [ -s "$TASKS" ]; then $CURC_LB_BIN/mpirun lb "$TASKS"; fi
echo "===== $(date) done; finished fits in results/$RUN: $(ls $NSS_ROOT/results/$RUN/*/*_seed*.json 2>/dev/null | wc -l)"
