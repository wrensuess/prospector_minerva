#!/bin/bash -l
# Many different NSS fits in the production layout (21 tasks per node, CURC loadbalance, one
# physical core per fit), as a job ARRAY: element i runs lines [CHUNK i + 1, CHUNK (i + 1)] of a
# task list, one fit per line:   CATALOG ID DUST SEED   (blanca/make_dust_tasks.py writes one).
# Production settings otherwise (nss_jax.py defaults): n_live 1000, num_delete 100, 32 inner steps,
# tabulated mass prior, mass integrated out, 1e6 prior start, fagn prior flat1_s025, no MIRI.
#
# Each element works until every one of its fits is fitted, postprocessed and has spectra:
#   - passes: pipeline_status.py says what is missing (from the files: a valid json, a valid
#     _post.h5 of the current format, spectra); the fits (or postprocessing reruns from a kept
#     full .npz) run under `mpirun lb`, 20 at a time; then spectra_nss.py runs as NSPEC parallel
#     processes on the element's galaxies (SpecModel is built once per process). Then again,
#     until nothing is left (at most MAX_PASSES passes per run of the job).
#   - preemption: Slurm requeues the element (--requeue); the next run starts with the status.
#   - time limit: Slurm signals USR1 10 min before it and the element requeues itself (at most
#     MAX_REQUEUE restarts in all), instead of ending as TIMEOUT.
#   - a stage that runs to its end without producing its output (a crash, not a kill) counts as a
#     failure of that fit; after MAX_FAILS the fit is given up and listed, and the element exits 1.
#   - every write is atomic (json, _post.h5, spectra), so a kill at any moment leaves either the old
#     file or the new one; stale temporary files are removed at the next status.
# Exit status: 0 all done; 1 some fits given up (see the .out); 2 spectra wanted but their setup is
# missing (fits done; see blanca/README.md, "Spectra step").
#
# Settings (via --export=ALL,NAME=value,...):
#   TASKLIST   task list, relative to nss_test/            (default blanca/dust_tasks.txt)
#   RUN        output folder name under results/            (default dust48)
#   CHUNK      fits per array element                       (default 20)
#   EXTRA      extra nss_jax.py options                     (default --no-full-npz: only the json,
#              the log, _post.h5 and _draws.npz are written, not the ~14 MB full sample file)
#   SPECTRA    1 (default) spectra are part of "done"; 0 skips them
#   NSPEC      parallel spectra processes                   (default 10; ~2.5 GB each, fits the 32 GB)
#   MAX_FAILS  failures of one fit's stage before giving up (default 3)
#   MAX_PASSES passes per run of the job                    (default MAX_FAILS + 1)
#   MAX_REQUEUE self-requeues at the time limit, counted with preemptions (default 20)
#   CHAOS      1: inject kills, corruption and requeues (blanca/chaos_monkey.sh), for the restart
#              test only (blanca/README.md, "Restart test"). Never in production.
# Chunk size: lb hands each worker a new fit as soon as it finishes one, so an element of CHUNK fits
# keeps its 20 cores busy until the list runs out; the idle tail is about one fit per worker. Bigger
# chunks waste less (production uses 200).
# Submit (from /projects/$USER/nss_blanca/nss_test/blanca, after `module load slurm/blanca`):
#   sbatch --array=0-28 tasklist_layout.sbatch
# Output: results/$RUN/<DUST>/<ID>_<DUST>_seed<S>.{json,log}, _post.h5 (with spectra) and _draws.npz
# (4000 equal-weight posterior draws, float32), plus the full .npz only if EXTRA is set without
# --no-full-npz or the fit's own postprocessing failed (never copy those back).
# Check a run by hand (login node): see blanca/README.md, "Checking a run".
#
#SBATCH --account=blanca-casa
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
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
#SBATCH --open-mode=append         # keep every run of a requeued element in one .out (default truncates)

NSS_ROOT=${NSS_ROOT:-/projects/$USER/nss_blanca/nss_test}
export EFSPS_PARENT=${EFSPS_PARENT:-/projects/$USER/nss_blanca/efsps-photfit}
NSS_ENV=${NSS_ENV:-/projects/$USER/software/anaconda/envs/nss}
TIMEV=${TIMEV:-/usr/bin/time -v}
SCONTROL=$(command -v scontrol)          # before `module purge`, which can drop it from PATH

if command -v module > /dev/null 2>&1; then module purge; module load loadbalance; fi
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
SPECTRA=${SPECTRA:-1}
NSPEC=${NSPEC:-10}
MAX_FAILS=${MAX_FAILS:-3}
MAX_PASSES=${MAX_PASSES:-$((MAX_FAILS + 1))}
MAX_REQUEUE=${MAX_REQUEUE:-20}
I=${SLURM_ARRAY_TASK_ID:?submit as an array, e.g. --array=0-28}
JOBREF=${SLURM_ARRAY_JOB_ID}_${I}
RESTART=${SLURM_RESTART_COUNT:-0}
cd "$NSS_ROOT" || exit 1
[ -f "$TASKLIST" ] || { echo "no task list $TASKLIST"; exit 1; }
echo "===== $(date) on $(hostname), job $JOBREF, restart $RESTART"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2); node CPUs $(nproc --all 2>/dev/null); job allowed $(nproc 2>/dev/null)"
RES=$NSS_ROOT/results/$RUN
WORK=$RES/work
mkdir -p "$WORK" "$RES/spectra_logs"
FIRST=$((I * CHUNK + 1)); LAST=$((I * CHUNK + CHUNK))
P=$WORK/${SLURM_ARRAY_JOB_ID}_${I}_r${RESTART}      # prefix of this run's work lists

EFSPS_ENV=${EFSPS_ENV:-/projects/$USER/software/anaconda/envs/efsps}
export EFSPS_SRC=${EFSPS_SRC:-/projects/$USER/nss_blanca/E-FSPS}
export EFSPS_LIB=${EFSPS_LIB:-/projects/$USER/nss_blanca/efsps_lib/c3k_hr-mist-afe_cc1b132_slim}
SPEC_SETUP_MISSING=0
if [ "$SPECTRA" = 1 ] && ! { [ -x "$EFSPS_ENV/bin/python" ] && [ -d "$EFSPS_SRC/efsps" ] && [ -f "$EFSPS_LIB/meta_data.pkl" ]; }; then
    echo "!! spectra wanted but setup missing: need $EFSPS_ENV/bin/python, $EFSPS_SRC/efsps and $EFSPS_LIB/meta_data.pkl; fits only"
    SPECTRA=0; SPEC_SETUP_MISSING=1
fi

status() {
    python -u pipeline_status.py --root "$NSS_ROOT" --run "$RUN" --tasklist "$TASKLIST" \
        --first $FIRST --last $LAST --spectra $SPECTRA --max-fails $MAX_FAILS "$@"
}

# time limit: requeue this element (Slurm signals the batch shell 10 min before; lb runs in the
# background so the trap fires at once)
on_time_limit() {
    echo "===== $(date) time limit near (restart $RESTART of at most $MAX_REQUEUE)"
    if [ "$RESTART" -ge "$MAX_REQUEUE" ]; then
        echo "!! MAX_REQUEUE reached; not requeueing. Resubmit this element to finish it."
    elif [ -z "$SCONTROL" ]; then
        echo "!! no scontrol found; cannot requeue. Resubmit this element to finish it."
    else
        echo "requeueing $JOBREF"; $SCONTROL requeue "$JOBREF"
    fi
    exit 0
}
trap on_time_limit USR1

run_lb() {          # $1: task file (one shell command per line), 20 at a time, one core each
    $CURC_LB_BIN/mpirun lb "$1" &
    wait $!
}

chaos() { if [ "${CHAOS:-0}" = 1 ]; then BATCH_PID=$$ SCONTROL=$SCONTROL JOBREF=$JOBREF RES=$RES RUN=$RUN \
          RESTART=$RESTART bash "$NSS_ROOT/blanca/chaos_monkey.sh" "$@"; fi; }

RC=3
for PASS in $(seq 1 $MAX_PASSES); do
    echo "===== $(date) pass $PASS"
    chaos before_pass $PASS
    # ---- fits, and postprocessing reruns from a kept full .npz
    status --write "$P" --stage fits; RC=$?
    if [ $RC -ne 3 ]; then break; fi
    TASKS=$P.fits_tasks
    while read -r CAT ID DUST SEED; do
        OUT=$RES/$DUST; mkdir -p "$OUT"; STEM=${ID}_${DUST}_seed${SEED}
        echo "cd $NSS_ROOT && { python -u -c 'import os; a = sorted(os.sched_getaffinity(0)); c = {open(f\"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list\").read().strip() for i in a}; print(\"affinity\", a, \"=\", len(c), \"physical core(s)\", flush=True)'; $TIMEV python -u nss_jax.py --catalog $CAT --id $ID --tag ${DUST}_seed${SEED} --seed $SEED --dust-prior $DUST --save-draws 4000 $EXTRA --out $OUT; } > $OUT/${STEM}.log 2>&1"
    done < "$P.fit" > "$TASKS"
    while read -r CAT ID DUST SEED; do
        OUT=$RES/$DUST; STEM=${ID}_${DUST}_seed${SEED}
        echo "cd $NSS_ROOT && $TIMEV python -u postprocess_nss.py --catalog $CAT --id $ID --npz $OUT/$STEM.npz --json $OUT/$STEM.json --out $OUT/${STEM}_post.h5 > $OUT/${STEM}_postprocess.log 2>&1"
    done < "$P.post" >> "$TASKS"
    if [ -s "$TASKS" ]; then
        echo "$(wc -l < "$P.fit") fits and $(wc -l < "$P.post") postprocessing reruns to run (lines $FIRST-$LAST of $TASKLIST)"
        run_lb "$TASKS"
        echo "===== $(date) fits done"
        status --record-failures fits "$P.fit" "$P.post"
    fi
    chaos after_fits $PASS
    # ---- spectra: NSPEC processes at once, each over its share of the galaxies
    if [ "$SPECTRA" = 1 ]; then
        status --write "$P" --stage spectra --nspec $NSPEC > /dev/null
        STASKS=$P.spec_tasks
        for L in "$P".spec.*; do
            [ -f "$L" ] || continue
            echo "cd $NSS_ROOT && EFSPS_SIDECAR=0 EFSPS_SRC=$EFSPS_SRC EFSPS_LIB=$EFSPS_LIB EFSPS_PARENT=$EFSPS_PARENT $TIMEV $EFSPS_ENV/bin/python -u spectra_nss.py --libdir $EFSPS_LIB --list $L > $RES/spectra_logs/$(basename "$L")_p${PASS}.log 2>&1"
        done > "$STASKS"
        if [ -s "$STASKS" ]; then
            echo "$(date) spectra: $(cat "$P".spec.* | wc -l) galaxies in $(wc -l < "$STASKS") processes"
            run_lb "$STASKS"
            echo "===== $(date) spectra done"
            status --record-failures spectra "$P".spec.*
        fi
    fi
done
echo "===== $(date) final state:"
status; RC=$?
if [ $RC -eq 0 ] && [ $SPEC_SETUP_MISSING = 1 ]; then RC=2; fi
case $RC in
    0) echo "===== done: every fit fitted, postprocessed$( [ "$SPECTRA" = 1 ] && echo ' and with spectra')" ;;
    1) echo "!! some fits were given up after $MAX_FAILS failures (listed above)" ;;
    2) echo "!! fits done but no spectra: spectra setup missing" ;;
    3) echo "!! work left after $MAX_PASSES passes (failure counts keep this bounded): requeueing"
       if [ "$RESTART" -lt "$MAX_REQUEUE" ] && [ -n "$SCONTROL" ]; then $SCONTROL requeue "$JOBREF"; exit 0; fi
       echo "!! cannot requeue (MAX_REQUEUE reached or no scontrol): resubmit this element" ;;
esac
exit $RC
