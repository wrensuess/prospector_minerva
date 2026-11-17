#!/bin/bash
# Fault injection for the restart test ONLY (blanca/README.md, "Restart test"): called by
# tasklist_layout.sbatch when CHAOS=1, with BATCH_PID, SCONTROL, JOBREF, RES, RUN, RESTART set.
#   chaos_monkey.sh before_pass PASS | after_fits PASS
# What it does, by restart (Slurm's SLURM_RESTART_COUNT):
#   restart 0, pass 1   once CHAOS_N_DONE (default 4) fits have finished: `scontrol requeue` while
#                       the others still run (a preemption in the middle of the fits); the damage
#                       below needs 4 finished fits
#   restart 1, pass 1   before the status: damage finished fits as a killed or failed process
#                       could: truncate one json (-> refit), delete one _post.h5 with the full .npz
#                       kept (-> postprocessing rerun), delete one _post.h5 and its .npz (-> refit),
#                       leave stale temporary files (-> removed); then SIGKILL one running fit
#                       60 s into it (-> counted as a failure, refitted later)
#   restart 1, after the fits: SIGKILL one spectra process mid-run, then signal USR1 to the job
#                       (what Slurm sends 10 min before the time limit -> the job requeues itself)
#   restart >= 2        nothing: the job must finish everything cleanly
# Everything it does is printed with "CHAOS:" (in the job's .out) and the expected outcome of
# each damaged fit goes to $RES/chaos_expect.txt for blanca/check_restart_test.py.
EVENT=$1; PASS=$2
N_DONE=${CHAOS_N_DONE:-4}
say() { echo "CHAOS [restart $RESTART pass $PASS $EVENT] $*"; }
FIT_PAT="^python -u nss_jax.py .*results/$RUN/"
SPEC_PAT="^/[^ ]*bin/python -u spectra_nss.py .*results/$RUN/"

done_stems() {      # finished fits (json and _post.h5), as stems
    for j in "$RES"/*/*_seed*.json; do
        [ -f "$j" ] || continue
        s=${j%.json}
        [ -f "${s}_post.h5" ] && echo "$s"
    done
}

if [ "$RESTART" = 0 ] && [ "$EVENT" = before_pass ] && [ "$PASS" = 1 ]; then
    (
        for _ in $(seq 1 720); do                                   # at most 1 h
            n=$(ls "$RES"/*/*_seed*.json 2>/dev/null | wc -l)
            if [ "$n" -ge "$N_DONE" ]; then
                running=$(pgrep -f "$FIT_PAT" | wc -l | tr -d ' ')
                say "$n fits finished, $running still running: simulated preemption (scontrol requeue $JOBREF)"
                $SCONTROL requeue "$JOBREF"
                exit 0
            fi
            sleep 5
        done
        say "gave up waiting for $N_DONE finished fits"
    ) &
fi

if [ "$RESTART" = 1 ] && [ "$EVENT" = before_pass ] && [ "$PASS" = 1 ]; then
    STEMS=($(done_stems))
    say "${#STEMS[@]} finished fits found"
    : > "$RES/chaos_expect.txt"
    if [ ${#STEMS[@]} -ge 1 ]; then
        s=${STEMS[0]}; head -c 200 "$s.json" > "$s.json.cut" && mv "$s.json.cut" "$s.json"
        say "truncated $(basename "$s").json -> expect a refit"
        echo "$(basename "$s") truncated_json refit" | sed "s|$| $(date +%s)|" >> "$RES/chaos_expect.txt"
    fi
    if [ ${#STEMS[@]} -ge 2 ]; then
        s=${STEMS[1]}; rm -f "${s}_post.h5"
        if [ -f "$s.npz" ]; then
            say "deleted $(basename "$s")_post.h5, kept its .npz -> expect a postprocessing rerun"
            echo "$(basename "$s") deleted_post rerun_postprocess" | sed "s|$| $(date +%s)|" >> "$RES/chaos_expect.txt"
        else
            say "deleted $(basename "$s")_post.h5 (no .npz: run with EXTRA= to test the rerun) -> expect a refit"
            echo "$(basename "$s") deleted_post refit" | sed "s|$| $(date +%s)|" >> "$RES/chaos_expect.txt"
        fi
    fi
    if [ ${#STEMS[@]} -ge 3 ]; then
        s=${STEMS[2]}; rm -f "${s}_post.h5" "$s.npz"
        say "deleted $(basename "$s")_post.h5 and .npz -> expect a refit"
        echo "$(basename "$s") deleted_post_and_npz refit" | sed "s|$| $(date +%s)|" >> "$RES/chaos_expect.txt"
    fi
    if [ ${#STEMS[@]} -ge 4 ]; then
        s=${STEMS[3]}
        echo junk > "${s}_post.h5.tmp.99999"; echo junk > "${s}_post.h5.spectra_tmp.99999"
        echo junk > "$s.json.tmp.99999"
        say "left 3 stale temporary files for $(basename "$s") -> expect them removed"
        echo "$(basename "$s") stale_tmp_files removed" | sed "s|$| $(date +%s)|" >> "$RES/chaos_expect.txt"
    fi
    (
        for _ in $(seq 1 360); do                                   # at most 30 min
            pid=$(pgrep -n -f "$FIT_PAT")
            if [ -n "$pid" ]; then
                sleep 60
                pid=$(pgrep -n -f "$FIT_PAT")
                [ -n "$pid" ] || { say "no fit running any more to kill"; exit 0; }
                cmd=$(ps -o command= -p "$pid" | grep -o -- '--id [^ ]* --tag [^ ]*')
                say "SIGKILL fit pid $pid ($cmd) mid-run -> expect a counted failure and a refit"
                echo "$(echo "$cmd" | awk '{print $2"_"$4}') killed_fit refit" | sed "s|$| $(date +%s)|" >> "$RES/chaos_expect.txt"
                kill -KILL "$pid"
                exit 0
            fi
            sleep 5
        done
    ) &
fi

if [ "$RESTART" = 1 ] && [ "$EVENT" = after_fits ] && [ "$PASS" = 1 ]; then
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
