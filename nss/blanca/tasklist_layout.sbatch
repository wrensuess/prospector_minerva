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
#SBATCH --constraint="epyc9354|gold5418y|5418y|milan|icelake|rome|cascadelake"
# DEFAULT (Wren, 2026-11-13): every CPU type within ~1.35x of the fastest per NSS iteration
# (STATUS.md section 6 item 13): EPYC 9354, Gold 5418Y, EPYC 7413 (`milan` is only the two 7413
# nodes; the EPYC 7713 nodes carry epyc7713/epyc7713p), Gold 6342 (`icelake`, three nodes plus one
# untimed 32-core node), and EPYC 7452 (`rome`) and Gold 6230 (`cascadelake`), ~1.35x slower but
# many more nodes. `rome`/`cascadelake` also cover some untimed models (the 128- and 256-core Rome
# nodes, the non-40-core Cascade Lake nodes, which may include the Gold 6254). Gold 6130 (skylake)
# and E5-2680 v4 (broadwell), 1.7-1.85x, stay out. Only the fast types, e.g. for timing:
#   --constraint="epyc9354|gold5418y|5418y|milan|icelake"
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
echo "===== $(date) fits done; finished fits in results/$RUN: $(ls $NSS_ROOT/results/$RUN/*/*_seed*.json 2>/dev/null | wc -l)"

# Spectra and rest-frame colours (spectra_nss.py) for this element's galaxies: ONE process
# after the fits, so E-FSPS SpecModel (~1.5 GB) is loaded once per node, not once per fit.
# Restartable: galaxies that already have spectra are skipped, so a preempted and requeued
# element redoes only what is missing; each file is replaced atomically. Skipped with a
# message (the fits are unaffected) if the efsps env, E-FSPS or the library is missing.
# Catch-up for anything left without spectra: make_catalog_nss.py writes missing_spectra.txt;
# run spectra_nss.py --list on it. SPECTRA=0 turns the step off.
SPECTRA=${SPECTRA:-1}
EFSPS_ENV=${EFSPS_ENV:-/projects/$USER/software/anaconda/envs/efsps}
export EFSPS_SRC=${EFSPS_SRC:-/projects/$USER/nss_blanca/E-FSPS}
export EFSPS_LIB=${EFSPS_LIB:-/projects/$USER/nss_blanca/efsps_lib/c3k_hr-mist-afe_cc1b132_slim}
if [ "$SPECTRA" = 1 ]; then
    if [ -x "$EFSPS_ENV/bin/python" ] && [ -d "$EFSPS_SRC/efsps" ] && [ -f "$EFSPS_LIB/meta_data.pkl" ]; then
        POSTLIST=$NSS_ROOT/results/$RUN/postlist_${SLURM_ARRAY_JOB_ID}_${I}.txt
        sed -n "${FIRST},${LAST}p" "$TASKLIST" | while read -r CAT ID DUST SEED; do
            echo "$NSS_ROOT/results/$RUN/$DUST/${ID}_${DUST}_seed${SEED}_post.h5"
        done > "$POSTLIST"
        EFSPS_SIDECAR=0 /usr/bin/time -v "$EFSPS_ENV/bin/python" -u spectra_nss.py --libdir "$EFSPS_LIB" --list "$POSTLIST"
    else
        echo "spectra step SKIPPED: need $EFSPS_ENV/bin/python, $EFSPS_SRC/efsps and $EFSPS_LIB/meta_data.pkl"
    fi
fi
echo "===== $(date) done"
