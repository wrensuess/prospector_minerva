#!/bin/bash -l
# Many different NSS fits in the production layout (21 tasks per node, CURC loadbalance, one
# physical core per fit), as a job ARRAY: element i runs lines [CHUNK i + 1, CHUNK (i + 1)] of a
# task list, one fit per line:   CATALOG ID DUST SEED   (blanca/make_dust_tasks.py writes one).
# Production settings otherwise (nss_jax.py defaults): n_live 1000, num_delete 100, 32 inner steps,
# tabulated mass prior, mass integrated out, 1e6 prior start, fagn prior flat1_s025, no MIRI.
#
# Each element works until every one of its fits is fitted, postprocessed and has spectra:
#   - lock: the element takes RESULTS_ROOT/RUN/.state/lock_<first>-<last> (its Slurm job id); a
#     second job for the same lines while the holder is still in squeue exits 7 at once.
#   - passes: pipeline_status.py says what is missing (from the files: a fitted json, a valid
#     _post.h5 of the current format matching it, spectra) and removes the stale outputs of what
#     it schedules; the fits (or postprocessing reruns from a kept full .npz) run under
#     `mpirun lb`, 20 at a time; then spectra_nss.py runs as NSPEC parallel processes on the
#     element's galaxies (SpecModel is built once per process). Then again, until nothing is left
#     (at most MAX_PASSES passes per run of the job).
#   - each task line writes its own exit code (<pass dir>/<stem>.rc) after its process ends. A
#     fit is charged a failure only if it ended (exit code there, or a json saying timed_out /
#     capped / nonfinite) without its output; a fit with no exit code was killed (preemption,
#     requeue, mpirun's death), never a failure. lb's own exit status is printed, and leftover
#     processes of a pass are killed before the next one.
#   - circuit breaker: if most fits of a pass fail (>= BREAKER_FRAC and >= BREAKER_MIN of the fits
#     that ended and had not failed before), the node or setup is at fault: no failures are
#     recorded, a diagnosis is printed, the element requeues itself once (it may land on another
#     node); a second trip in a row stops it (exit 6).
#   - preemption: Slurm requeues the element (--requeue); the next run starts with the status.
#   - time limit: Slurm signals USR1 10 min before it and the element requeues itself, instead of
#     ending as TIMEOUT. Self-requeues (time limit, work left after MAX_PASSES, breaker) are
#     counted in .state/job_<arrayjob>_<i>.self_requeues and bounded by MAX_REQUEUE; preemptions
#     are not counted (SLURM_RESTART_COUNT counts both, so it is only printed).
#   - after MAX_FAILS failures of a stage the fit is given up and listed (exit 3).
#   - every write is atomic (json, _post.h5, spectra, exit codes), so a kill at any moment leaves
#     either the old file or the new one; stale temporary files are removed by the next pass.
# After a successful `scontrol requeue` of itself the script waits for Slurm to stop it (requeue of
# a running job terminates it and puts it back to PENDING, same job id), so a requeued run has no
# exit status of its own; it only exits if the requeue failed or did not take effect.
#
# Exit status (Slurm's ExitCode of the element's last run; also in blanca/README.md):
#   0  done: every fit fitted, postprocessed and with spectra (or BAD_INPUT: final, listed)
#   1  setup missing before any work (no python env, no loadbalance module, no task list,
#      cannot create the results folder); nothing ran
#   3  done except fits given up after MAX_FAILS failures (listed at the end of the .out)
#   4  fits done but no spectra: spectra wanted but their setup is missing (README, "Spectra step")
#   5  work left but not requeued: MAX_REQUEUE self-requeues used up, no scontrol, or
#      `scontrol requeue` failed or did not stop the job; resubmit the element
#   6  circuit breaker tripped twice in a row (or its requeue failed): node or setup problem,
#      no failures were recorded; see the diagnosis in the .out
#   7  another live job holds this element's lock (duplicate submission); nothing ran
#   8  pipeline_status.py itself failed (a bug or a broken environment); see the .out
#   9  malformed task-list line (its line number is in the .out)
#   (a bash error before the first status could also give 1, 2 or 127)
#
# Settings (via --export=ALL,NAME=value,...):
#   TASKLIST   task list, relative to nss_test/            (default blanca/dust_tasks.txt);
#              its catalogue paths are relative to nss_test/ too
#   RUN        output folder name under RESULTS_ROOT        (default dust48)
#   RESULTS_ROOT where run folders go                       (default $NSS_ROOT/results; production:
#              /scratch/alpine/$USER/nss_results, since /projects is nearly full)
#   CHUNK      fits per array element                       (default 20)
#   EXTRA      extra nss_jax.py options                     (default --no-full-npz: only the json,
#              the log, _post.h5 and _draws.npz are written, not the ~14 MB full sample file)
#   SPECTRA    1 (default) spectra are part of "done"; 0 skips them
#   NSPEC      parallel spectra processes                   (default 10; ~2.5 GB each, fits the 32 GB)
#   MAX_FAILS  failures of one fit's stage before giving up (default 3)
#   MAX_PASSES passes per run of the job                    (default MAX_FAILS + 1)
#   MAX_REQUEUE self-requeues per job (preemptions not counted) (default 20)
#   BREAKER_MIN, BREAKER_FRAC circuit breaker thresholds    (default 5, 0.5)
#   CHAOS      1: inject kills, corruption and requeues (blanca/chaos_monkey.sh), for the restart
#              test only (blanca/README.md, "Restart test"). Never in production.
# Chunk size: lb hands each worker a new fit as soon as it finishes one, so an element of CHUNK fits
# keeps its 20 cores busy until the list runs out; the idle tail is about one fit per worker. Bigger
# chunks waste less (production uses 200).
# Submit (from /projects/$USER/nss_blanca/nss_test/blanca, after `module load slurm/blanca`):
#   sbatch --array=0-28 tasklist_layout.sbatch
# Output: RESULTS_ROOT/$RUN/<DUST>/<ID>_<DUST>_seed<S>.{json,log}, _post.h5 (with spectra) and
# _draws.npz (4000 equal-weight posterior draws, float32), plus the full .npz only if EXTRA is set
# without --no-full-npz or the fit's own postprocessing failed (never copy those back).
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
SCONTROL=$(command -v scontrol)          # before `module purge`, which can drop them from PATH
SQUEUE=$(command -v squeue)

if command -v module > /dev/null 2>&1; then module purge; module load loadbalance; fi
if [ ! -x "$NSS_ENV/bin/python" ]; then echo "no python in $NSS_ENV on $(hostname)"; exit 1; fi
export PATH=$NSS_ENV/bin:$PATH
if [ -z "$CURC_LB_BIN" ]; then echo "no loadbalance module on $(hostname)"; exit 1; fi
export SLURM_EXPORT_ENV=ALL
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

TASKLIST=${TASKLIST:-blanca/dust_tasks.txt}
RUN=${RUN:-dust48}
RESULTS_ROOT=${RESULTS_ROOT:-$NSS_ROOT/results}
CHUNK=${CHUNK:-20}
EXTRA=${EXTRA---no-full-npz}     # set EXTRA= (empty) to also write the full .npz
SPECTRA=${SPECTRA:-1}
NSPEC=${NSPEC:-10}
MAX_FAILS=${MAX_FAILS:-3}
MAX_PASSES=${MAX_PASSES:-$((MAX_FAILS + 1))}
MAX_REQUEUE=${MAX_REQUEUE:-20}
BREAKER_MIN=${BREAKER_MIN:-5}
BREAKER_FRAC=${BREAKER_FRAC:-0.5}
I=${SLURM_ARRAY_TASK_ID:?submit as an array, e.g. --array=0-28}
JOBREF=${SLURM_ARRAY_JOB_ID}_${I}
JOBID=${SLURM_JOB_ID:-$JOBREF}           # the element's own job id: kept across requeues
RESTART=${SLURM_RESTART_COUNT:-0}
cd "$NSS_ROOT" || exit 1
[ -f "$TASKLIST" ] || { echo "no task list $TASKLIST"; exit 1; }
echo "===== $(date) on $(hostname), job $JOBREF (id $JOBID), restart $RESTART"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2); node CPUs $(nproc --all 2>/dev/null); job allowed $(nproc 2>/dev/null)"
RES=$RESULTS_ROOT/$RUN
WORK=$RES/work
STATE=$RES/.state
mkdir -p "$WORK" "$STATE" "$RES/spectra_logs" || { echo "cannot create $RES"; exit 1; }
FIRST=$((I * CHUNK + 1)); LAST=$((I * CHUNK + CHUNK))
echo "results in $RES; lines $FIRST-$LAST of $TASKLIST"

# ---- one job per element: the lock ------------------------------------------------------------
LOCK=$STATE/lock_${FIRST}-${LAST}
HAVE_LOCK=0
lock_holder() { awk 'NR == 1 {print $1}' "$LOCK" 2>/dev/null; }
job_alive() {       # $1 job id: 0 if squeue lists it pending/running/requeued, 1 if gone or
    local out k     # finished, 2 if squeue keeps failing
    [ -n "$SQUEUE" ] || return 2
    for k in 1 2 3; do
        if out=$($SQUEUE -h -j "$1" -o %T 2>&1); then
            case "$out" in
                ''|COMPLETED|CANCELLED*|FAILED|TIMEOUT|NODE_FAIL|BOOT_FAIL|DEADLINE|OUT_OF_MEMORY|SPECIAL_EXIT) return 1 ;;
                *) return 0 ;;
            esac
        fi
        case "$out" in *"nvalid job id"*) return 1 ;; esac     # left the queue long ago
        sleep 10
    done
    return 2
}
take_lock() {
    local h
    h=$(lock_holder)
    if [ -n "$h" ] && [ "$h" != "$JOBID" ]; then
        job_alive "$h"
        case $? in
            0) echo "!! lines $FIRST-$LAST of $RUN are being run by job $h (still in squeue; lock $LOCK)."
               echo "!! This job ($JOBID) is a duplicate and stops without touching anything. Cancel one of them."
               exit 7 ;;
            2) echo "!! cannot ask squeue whether job $h (holder of $LOCK) is alive; stopping to be safe. Resubmit later."
               exit 7 ;;
            *) echo "lock $LOCK was left by job $h, which is gone: taking it over" ;;
        esac
    fi
    echo "$JOBID $JOBREF $(hostname) $(date +%s)" > "$LOCK.tmp.$$" && mv "$LOCK.tmp.$$" "$LOCK"
    sleep 2                                         # two jobs racing for a free lock: last writer wins
    h=$(lock_holder)
    if [ "$h" != "$JOBID" ]; then
        echo "!! job $h took the lock $LOCK at the same moment; this job ($JOBID) stops."
        exit 7
    fi
    HAVE_LOCK=1
}
finish() {          # $1: exit status; the element ends for good (no requeue): give the lock back
    if [ "$HAVE_LOCK" = 1 ] && [ "$(lock_holder)" = "$JOBID" ]; then rm -f "$LOCK"; fi
    echo "===== $(date) exit $1"
    exit "$1"
}
take_lock

# ---- counters kept across requeues (same job id), reset by a new submission --------------------
SELFQ=$STATE/job_${JOBREF}.self_requeues
BREAKS=$STATE/job_${JOBREF}.breaker_trips
count() { local n; n=$(cat "$1" 2>/dev/null); case "$n" in ''|*[!0-9]*) echo 0 ;; *) echo "$n" ;; esac; }
set_count() { echo "$2" > "$1.tmp.$$" && mv "$1.tmp.$$" "$1"; }
echo "self-requeues so far: $(count "$SELFQ") of at most $MAX_REQUEUE (restart count $RESTART includes preemptions)"

# leftovers of earlier runs of this element (work lists, exit codes of killed passes)
rm -rf "$WORK/${JOBREF}_r"*

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
        --results-root "$RESULTS_ROOT" --first $FIRST --last $LAST --spectra $SPECTRA \
        --max-fails $MAX_FAILS --breaker-min $BREAKER_MIN --breaker-frac $BREAKER_FRAC "$@"
}
status_failed() {   # $1: exit status of pipeline_status.py outside its documented set
    if [ "$1" = 13 ]; then echo "!! malformed task list (see above): fix it and resubmit"; finish 9; fi
    echo "!! pipeline_status.py failed with exit $1 (a Python error or bad arguments, see above)"; finish 8
}

# ---- self-requeue: on success Slurm stops this job; we wait here for that ---------------------
requeue_self() {    # $1: why. Returns only if the element was NOT requeued.
    local n
    n=$(count "$SELFQ")
    if [ "$n" -ge "$MAX_REQUEUE" ]; then
        echo "!! $n self-requeues used (MAX_REQUEUE $MAX_REQUEUE); not requeueing ($1). Resubmit this element to finish it."
        return 1
    fi
    if [ -z "$SCONTROL" ]; then echo "!! no scontrol found; cannot requeue ($1). Resubmit this element."; return 1; fi
    set_count "$SELFQ" $((n + 1))
    echo "requeueing $JOBREF ($1; self-requeue $((n + 1)) of at most $MAX_REQUEUE)"
    if ! $SCONTROL requeue "$JOBREF"; then
        echo "!! scontrol requeue $JOBREF failed; resubmit this element."
        return 1
    fi
    sleep 300 &                                     # Slurm terminates the job now
    wait $!
    echo "!! requeue of $JOBREF accepted but the job was not stopped within 5 min"
    return 1
}

# time limit: requeue this element (Slurm signals the batch shell 10 min before; lb runs in the
# background so the trap fires at once). No failures are recorded for the interrupted pass.
on_time_limit() {
    trap '' USR1
    echo "===== $(date) time limit near"
    requeue_self "time limit" || finish 5
}
trap on_time_limit USR1

breaker() {         # the circuit breaker tripped in this pass
    local n
    n=$(count "$BREAKS"); set_count "$BREAKS" $((n + 1))
    if [ "$n" -ge 1 ]; then
        echo "!! circuit breaker tripped twice in a row (here on $(hostname)): stopping. Fix the setup or node, then resubmit."
        finish 6
    fi
    echo "!! circuit breaker: requeueing once (the next run may get another node)"
    requeue_self "circuit breaker" || finish 6
}

descendants() { local c; for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done; }
kill_leftovers() {  # $1: pass dir. Anything of this pass still alive after lb returned
    local p all=""
    for p in $(pgrep -f -- "$1/"); do all="$all $p $(descendants "$p")"; done
    [ -n "${all// /}" ] || return 0
    echo "!! $(echo $all | wc -w | tr -d ' ') leftover process(es) of this pass still running after lb ended: killing them"
    kill -KILL $all 2> /dev/null
    sleep 1
}

run_lb() {          # $1: task file (one shell command per line), 20 at a time, one core each
    local rc
    $CURC_LB_BIN/mpirun lb "$1" &
    wait $!
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "!! mpirun lb exited $rc: only tasks that wrote their own exit code are judged; the rest were killed, not failed"
    fi
    kill_leftovers "$(dirname "$1")"
    return $rc
}

after_record() {    # $1: exit status of --record-failures
    case $1 in
        0) set_count "$BREAKS" 0 ;;
        12) breaker ;;
        *) status_failed "$1" ;;
    esac
}

# rc file of one task line, written after its process ends (atomically; after a non-zero exit
# wait 2 s first, so a task killed together with the whole job never leaves one)
rc_tail() { echo "rc=\$?; [ \$rc -eq 0 ] || sleep 2; echo \$rc > $1.tmp && mv $1.tmp $1"; }

chaos() { if [ "${CHAOS:-0}" = 1 ]; then BATCH_PID=$$ SCONTROL=$SCONTROL JOBREF=$JOBREF RES=$RES RUN=$RUN \
          WORK=$WORK RESTART=$RESTART bash "$NSS_ROOT/blanca/chaos_monkey.sh" "$@"; fi; }

RC=10
for PASS in $(seq 1 $MAX_PASSES); do
    echo "===== $(date) pass $PASS"
    PD=$WORK/${JOBREF}_r${RESTART}_p${PASS}       # this pass's work lists and exit codes
    rm -rf "$PD"; mkdir -p "$PD"
    chaos before_pass $PASS
    # ---- fits, and postprocessing reruns from a kept full .npz
    status --write "$PD/w" --stage fits --clean; RC=$?
    case $RC in 0|11) rm -rf "$PD"; break ;; 10) ;; *) status_failed $RC ;; esac
    TASKS=$PD/fits_tasks
    BREAK_ALL=0
    if [ "${CHAOS:-0}" = 1 ] && [ -f "$RES/.chaos_break_all" ]; then rm -f "$RES/.chaos_break_all"; BREAK_ALL=1; fi
    while read -r CAT ID DUST SEED; do
        OUT=$RES/$DUST; mkdir -p "$OUT"; STEM=${ID}_${DUST}_seed${SEED}
        FITCMD="python -u nss_jax.py --catalog $CAT --id $ID --tag ${DUST}_seed${SEED} --seed $SEED --dust-prior $DUST --save-draws 4000 $EXTRA --out $OUT"
        if [ $BREAK_ALL = 1 ]; then FITCMD="python -u -c \"raise SystemExit('CHAOS: simulated broken setup, every fit fails at once')\""; fi
        echo "cd $NSS_ROOT && { python -u -c 'import os; a = sorted(os.sched_getaffinity(0)); c = {open(f\"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list\").read().strip() for i in a}; print(\"affinity\", a, \"=\", len(c), \"physical core(s)\", flush=True)'; $TIMEV $FITCMD; } > $OUT/${STEM}.log 2>&1; $(rc_tail "$PD/$STEM.rc")"
    done < "$PD/w.fit" > "$TASKS"
    while read -r CAT ID DUST SEED; do
        OUT=$RES/$DUST; STEM=${ID}_${DUST}_seed${SEED}
        echo "cd $NSS_ROOT && $TIMEV python -u postprocess_nss.py --catalog $CAT --id $ID --npz $OUT/$STEM.npz --json $OUT/$STEM.json --out $OUT/${STEM}_post.h5 > $OUT/${STEM}_postprocess.log 2>&1; $(rc_tail "$PD/$STEM.rc")"
    done < "$PD/w.post" >> "$TASKS"
    if [ -s "$TASKS" ]; then
        echo "$(wc -l < "$PD/w.fit" | tr -d ' ') fits and $(wc -l < "$PD/w.post" | tr -d ' ') postprocessing reruns to run"
        run_lb "$TASKS"
        echo "===== $(date) fits done"
        status --record-failures fits "$PD/w.fit" "$PD/w.post" --rc-dir "$PD"; after_record $?
    fi
    chaos after_fits $PASS
    # ---- spectra: NSPEC processes at once, each over its share of the galaxies
    if [ "$SPECTRA" = 1 ]; then
        status --write "$PD/w" --stage spectra --nspec $NSPEC > /dev/null; R=$?
        case $R in 0|10|11) ;; *) status_failed $R ;; esac
        STASKS=$PD/spec_tasks
        for L in "$PD"/w.spec.*; do
            [ -f "$L" ] || continue
            echo "cd $NSS_ROOT && EFSPS_SIDECAR=0 EFSPS_SRC=$EFSPS_SRC EFSPS_LIB=$EFSPS_LIB EFSPS_PARENT=$EFSPS_PARENT $TIMEV $EFSPS_ENV/bin/python -u spectra_nss.py --libdir $EFSPS_LIB --list $L --progress $L.started > $RES/spectra_logs/$(basename "$PD").$(basename "$L").log 2>&1; $(rc_tail "$L.rc")"
        done > "$STASKS"
        if [ -s "$STASKS" ]; then
            echo "$(date) spectra: $(cat "$PD"/w.spec.* | wc -l | tr -d ' ') galaxies in $(wc -l < "$STASKS" | tr -d ' ') processes"
            run_lb "$STASKS"
            echo "===== $(date) spectra done"
            SPECLISTS=$(ls "$PD"/w.spec.* | grep -E '\.spec\.[0-9]+$')
            status --record-failures spectra $SPECLISTS; after_record $?
        fi
    fi
    rm -rf "$PD"
done
echo "===== $(date) final state:"
status; RC=$?
case $RC in
    0) if [ $SPEC_SETUP_MISSING = 1 ]; then echo "!! fits done but no spectra: spectra setup missing"; finish 4; fi
       echo "===== done: every fit fitted, postprocessed$( [ "$SPECTRA" = 1 ] && echo ' and with spectra') (BAD_INPUT fits, if any, are listed above)"
       finish 0 ;;
    11) echo "!! some fits were given up after $MAX_FAILS failures (listed above)"; finish 3 ;;
    10) echo "!! work left after $MAX_PASSES passes"
        requeue_self "work left" || finish 5 ;;
    *) status_failed $RC ;;
esac
finish 8
