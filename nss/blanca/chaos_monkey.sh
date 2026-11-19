#!/bin/bash
# Fault injection for the restart test ONLY (blanca/README.md, "Restart test"): called by
# tasklist_layout.sbatch when CHAOS=1, with BATCH_PID, SCONTROL, JOBREF, RES, WORK, RUN, RESTART set.
#   chaos_monkey.sh before_pass PASS | after_fits PASS
# What it does, by restart (Slurm's SLURM_RESTART_COUNT):
#   restart 0, pass 1   make every fit of the pass fail at once (the job script swaps each fit for
#                       a command that exits 1: a broken setup) -> expect the circuit breaker: no
#                       failure recorded, a diagnosis, one self-requeue
#   restart 1, pass 1   once CHAOS_N_DONE (default 5) fits have finished: `scontrol requeue` while
#                       the others still run (a preemption in the middle of the fits) -> the fits
#                       it kills must have zero failures; the damage below needs 5 finished fits
#   restart 2, pass 1   before the status: damage finished fits as a killed or failed process
#                       could: truncate one json (-> refit), delete one _post.h5 with the full .npz
#                       kept (-> postprocessing rerun), delete one _post.h5 and its .npz (-> refit),
#                       leave stale temporary files (-> removed), cut one _post.h5 short and pad it
#                       with zeros so it still opens (-> redone); then SIGKILL one running fit 60 s
#                       into it, preferring one the preemption did not kill (-> one counted
#                       failure, refitted later), and 30 s later SIGKILL
#                       `mpirun lb` itself with the other fits in flight (-> those fits killed, not
#                       failed: zero failures, refitted)
#   restart 2, after the fits: SIGKILL one spectra process mid-run, then signal USR1 to the job
#                       (what Slurm sends 10 min before the time limit -> the job requeues itself)
#   restart >= 3        nothing: the job must finish everything cleanly
# Everything it does is printed with "CHAOS" (in the job's .out) and the expected outcome of
# each damaged fit goes to $RES/chaos_expect.txt for blanca/check_restart_test.py.
EVENT=$1; PASS=$2
N_DONE=${CHAOS_N_DONE:-5}
say() { echo "CHAOS [restart $RESTART pass $PASS $EVENT] $*"; }
FIT_PAT="^python -u nss_jax.py .*--out $RES/"
SPEC_PAT="^/[^ ]*bin/python -u spectra_nss.py .*--list $WORK/"
LB_PAT="mpirun lb $WORK/"
EXP=$RES/chaos_expect.txt
expect() { echo "$1 $2 $3 $(date +%s)" >> "$EXP"; }     # stem what outcome time

done_stems() {      # finished fits (json and _post.h5), as stems
    for j in "$RES"/*/*_seed*.json; do
        [ -f "$j" ] || continue
        s=${j%.json}
        [ -f "${s}_post.h5" ] && echo "$s"
    done
}
running_fits() {    # id_tag of every fit running now
    for p in $(pgrep -f "$FIT_PAT"); do
        ps -o command= -p "$p" | grep -o -- '--id [^ ]* --tag [^ ]*' | awk '{print $2"_"$4}'
    done | sort -u
}

pick_fit() {        # pid of a running fit, preferring one the simulated preemption did not kill
    local p k first=""      # (so each fit's failure count has one cause)
    for p in $(pgrep -f "$FIT_PAT"); do
        k=$(ps -o command= -p "$p" | grep -o -- '--id [^ ]* --tag [^ ]*' | awk '{print $2"_"$4}')
        [ -n "$k" ] || continue
        [ -n "$first" ] || first=$p
        if ! grep -q "^$k preempted " "$EXP" 2>/dev/null; then echo "$p"; return; fi
    done
    echo "$first"
}

if [ "$RESTART" = 0 ] && [ "$EVENT" = before_pass ] && [ "$PASS" = 1 ]; then
    : > "$EXP"
    touch "$RES/.chaos_break_all"
    say "every fit of this pass will fail at once (simulated broken setup) -> expect the circuit breaker, no failures recorded, one requeue"
    expect - all_fits_fail breaker
fi

if [ "$RESTART" = 1 ] && [ "$EVENT" = before_pass ] && [ "$PASS" = 1 ]; then
    (
        for _ in $(seq 1 720); do                                   # at most 1 h
            n=$(ls "$RES"/*/*_seed*_post.h5 2>/dev/null | wc -l)
            if [ "$n" -ge "$N_DONE" ]; then
                sleep 5                                             # let their jsons land
                R=$(running_fits)
                say "$n fits finished, $(echo $R | wc -w | tr -d ' ') still running: simulated preemption (scontrol requeue $JOBREF)"
                for x in $R; do expect "$x" preempted zero_fails; done
                $SCONTROL requeue "$JOBREF"
                exit 0
            fi
            sleep 5
        done
        say "gave up waiting for $N_DONE finished fits"
    ) &
fi

if [ "$RESTART" = 2 ] && [ "$EVENT" = before_pass ] && [ "$PASS" = 1 ]; then
    STEMS=($(done_stems))
    say "${#STEMS[@]} finished fits found"
    if [ ${#STEMS[@]} -ge 1 ]; then
        s=${STEMS[0]}; head -c 200 "$s.json" > "$s.json.cut" && mv "$s.json.cut" "$s.json"
        say "truncated $(basename "$s").json -> expect a refit"
        expect "$(basename "$s")" truncated_json refit
    fi
    if [ ${#STEMS[@]} -ge 2 ]; then
        s=${STEMS[1]}; rm -f "${s}_post.h5"
        if [ -f "$s.npz" ]; then
            say "deleted $(basename "$s")_post.h5, kept its .npz -> expect a postprocessing rerun"
            expect "$(basename "$s")" deleted_post rerun_postprocess
        else
            say "deleted $(basename "$s")_post.h5 (no .npz: run with EXTRA= to test the rerun) -> expect a refit"
            expect "$(basename "$s")" deleted_post refit
        fi
    fi
    if [ ${#STEMS[@]} -ge 3 ]; then
        s=${STEMS[2]}; rm -f "${s}_post.h5" "$s.npz"
        say "deleted $(basename "$s")_post.h5 and .npz -> expect a refit"
        expect "$(basename "$s")" deleted_post_and_npz refit
    fi
    if [ ${#STEMS[@]} -ge 4 ]; then
        s=${STEMS[3]}
        echo junk > "${s}_post.h5.tmp.99999"; echo junk > "${s}_post.h5.spectra_tmp.99999"
        echo junk > "$s.json.tmp.99999"
        say "left 3 stale temporary files for $(basename "$s") -> expect them removed"
        expect "$(basename "$s")" stale_tmp_files removed
    fi
    if [ ${#STEMS[@]} -ge 5 ]; then
        s=${STEMS[4]}
        python -c "import sys; p = sys.argv[1]; b = open(p, 'rb').read(); k = int(len(b) * 0.7); open(p + '.cut', 'wb').write(b[:k] + b'\0' * (len(b) - k))" "${s}_post.h5" \
            && mv "${s}_post.h5.cut" "${s}_post.h5"
        say "cut $(basename "$s")_post.h5 to 70% and padded it with zeros (it still opens) -> expect it redone"
        expect "$(basename "$s")" truncated_openable_post redone_post
    fi
    (
        for _ in $(seq 1 360); do                                   # at most 30 min
            pid=$(pgrep -n -f "$FIT_PAT")
            if [ -n "$pid" ]; then
                sleep 60
                pid=$(pick_fit)
                [ -n "$pid" ] || { say "no fit running any more to kill"; exit 0; }
                cmd=$(ps -o command= -p "$pid" | grep -o -- '--id [^ ]* --tag [^ ]*')
                k=$(echo "$cmd" | awk '{print $2"_"$4}')
                say "SIGKILL fit pid $pid ($cmd) mid-run -> expect one counted failure and a refit"
                expect "$k" killed_fit refit
                kill -KILL "$pid"
                sleep 30
                R=$(running_fits | grep -vx "$k")
                lb=$(pgrep -f "$LB_PAT")
                if [ -z "$lb" ]; then say "mpirun lb no longer running; cannot kill it"; exit 0; fi
                say "SIGKILL mpirun lb (pid $lb) with $(echo $R | wc -w | tr -d ' ') fit(s) in flight -> expect them killed, not failed"
                for x in $R; do expect "$x" mpirun_killed zero_fails; done
                expect - mpirun_killed lb_exit_reported
                kill -KILL $lb
                exit 0
            fi
            sleep 5
        done
    ) &
fi

if [ "$RESTART" = 2 ] && [ "$EVENT" = after_fits ] && [ "$PASS" = 1 ]; then
    (
        for _ in $(seq 1 360); do
            pid=$(pgrep -n -f "$SPEC_PAT")
            if [ -n "$pid" ]; then
                sleep 35
                pid=$(pgrep -n -f "$SPEC_PAT")
                if [ -n "$pid" ]; then
                    say "SIGKILL spectra process pid $pid mid-run -> expect its galaxies redone"
                    kill -KILL "$pid"
                else
                    say "spectra processes already finished; nothing to kill"
                fi
                sleep 2
                say "USR1 to the job (as Slurm before the time limit) -> expect a self-requeue"
                kill -USR1 "$BATCH_PID"
                exit 0
            fi
            sleep 5
        done
    ) &
fi
exit 0
