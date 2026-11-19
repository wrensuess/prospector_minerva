"""Checks the outcome of the restart test (blanca/README.md, "Restart test"): a task-list job run
with CHAOS=1 (blanca/chaos_monkey.sh), i.e. a pass where every fit fails at once, preempted
mid-fit, outputs damaged, a fit, `mpirun lb` and a spectra process killed, and a time-limit
requeue. Passes if the job nevertheless ended with every fit cleanly fitted, postprocessed and
with spectra, and no failure charged that was not a real one:

 1. every fit of the task list is DONE (pipeline_status.py: fitted json, valid _post.h5 of the
    current format matching it, spectra) or BAD_INPUT (at least --expect-bad-input of them: the
    task with an id that is not in its catalogue), nothing given up;
 2. no temporary files left anywhere in the run;
 3. every json is complete and records no unresolved postprocessing error; every _post.h5 opens
    and its draws are finite;
 4. each damage in chaos_expect.txt had its expected outcome: the circuit breaker tripped (and
    recorded nothing), the truncated json and the deleted files were redone (newer than the
    damage), the zero-padded _post.h5 was redone, the postprocessing rerun is recorded in the json
    (rerun_standalone), stale files gone, the killed fit has exactly one counted failure; the fits
    killed by the simulated preemption and by mpirun's death have none, and mpirun's exit was
    reported;
 5. failure counts overall: no fit has a fits failure except the SIGKILLed one; at most one
    spectra failure in all (the galaxy the killed spectra process was working on); the BAD_INPUT
    task has none and was not retried;
 6. the job's .out (all runs of the element are appended to it): restarts 0-3 happened, every
    CHAOS action is there, and the last run ends with "===== done" and exit 0.

    python blanca/check_restart_test.py --root /projects/$USER/nss_blanca/nss_test --run restart_test \
        --tasklist blanca/restart_test_tasks.txt --out blanca/nss_tasklist_<jobid>_0.out
Ends RESULT: PASS or RESULT: FAIL.
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pipeline_status as PS                                    # noqa: E402

ok_all = True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--tasklist", required=True)
    p.add_argument("--results-root", default=None, help="as the job's RESULTS_ROOT (default <root>/results)")
    p.add_argument("--out", required=True, help="the job's .out file (Slurm appends every restart)")
    p.add_argument("--expect-bad-input", type=int, default=1,
                   help="BAD_INPUT tasks the task list has (restart_test_tasks.txt: 1)")
    a = p.parse_args()
    import h5py
    import numpy as np
    r = PS.Run(a.root, a.run, a.tasklist, results_root=a.results_root)
    res = r.res

    # 1. states
    fits = [r.classify(line, spectra=True, max_fails=10 ** 9) for line in r.lines]
    bad = [(f.state, os.path.basename(f.stem)) for f in fits if f.state not in ("DONE", "BAD_INPUT")]
    badin = [f for f in fits if f.state == "BAD_INPUT"]
    check("1 every fit DONE (fitted, postprocessed, spectra) or BAD_INPUT", not bad,
          f"{len(r.lines) - len(bad)}/{len(r.lines)}" + (f"; not done: {bad}" if bad else ""))
    check(f"1b {a.expect_bad_input} BAD_INPUT task(s) (an id not in its catalogue)",
          len(badin) == a.expect_bad_input, [f"{os.path.basename(f.stem)}: {f.reason}" for f in badin])
    fails = {os.path.basename(f.stem): r.fails(f.stem) for f in fits}
    gave_up = [b for b, s in fails.items() if max(s.get(k, 0) for k in ("fits", "spectra")) >= 3]
    check("1c nothing reached the give-up count", not gave_up, str(gave_up) if gave_up else "")

    # 2. temporary files
    tmps = [f for pat in ("*.tmp.*", "*.spectra_tmp.*", "*.cut")
            for f in glob.glob(os.path.join(res, "**", pat), recursive=True)]
    check("2 no temporary files left", not tmps, ", ".join(os.path.basename(t) for t in tmps[:5]))

    # 3. json and h5 contents
    probs = []
    for f in fits:
        if f.state == "BAD_INPUT":
            continue
        stem = f.stem
        try:
            s = json.load(open(stem + ".json"))
            pp = s.get("postprocess", {})
            if "error" in pp and not pp.get("rerun_standalone"):
                probs.append(f"{os.path.basename(stem)}: postprocess error {pp['error']}")
        except Exception as e:
            probs.append(f"{os.path.basename(stem)}.json: {e!r}")
            continue
        try:
            with h5py.File(stem + "_post.h5", "r") as h:
                lm = h["draws/logmass"][()]
                sb = h["spectra/bestfit"][()]
                if not (np.isfinite(lm).all() and np.isfinite(sb).all()):
                    probs.append(f"{os.path.basename(stem)}_post.h5: non-finite draws or spectrum")
        except Exception as e:
            probs.append(f"{os.path.basename(stem)}_post.h5: {e!r}")
    check("3 every json complete and every _post.h5 readable and finite", not probs, "; ".join(probs[:5]))

    # 4. each damage had its expected outcome
    out = open(a.out).read() if os.path.exists(a.out) else ""
    exp_file = os.path.join(res, "chaos_expect.txt")
    expect = [l.split() for l in open(exp_file) if l.strip()] if os.path.exists(exp_file) else []
    whats = {e[1] for e in expect}
    need = {"all_fits_fail", "preempted", "truncated_json", "deleted_post", "deleted_post_and_npz",
            "stale_tmp_files", "truncated_openable_post", "killed_fit", "mpirun_killed"}
    check("4 chaos_expect.txt lists every kind of damage", need <= whats, f"missing {sorted(need - whats)}")
    # failures each fit is expected to have: one per deliberate SIGKILL (killed_fit), none for
    # anything else. A fit can be preempted at one restart and SIGKILLed at another; its single
    # failure then belongs to the SIGKILL, so a "zero failures" check allows for it.
    killed = {}
    for e in expect:
        if e[1] == "killed_fit":
            killed[e[0]] = killed.get(e[0], 0) + 1
    by_base = {os.path.basename(f.stem): f.stem for f in fits}
    for base, what, outcome, t_damage in expect:
        t_damage = float(t_damage)
        if outcome == "breaker":
            check("4 every fit failing at once -> CIRCUIT BREAKER and a requeue",
                  "CIRCUIT BREAKER" in out and "circuit breaker; self-requeue" in out)
            continue
        if outcome == "lb_exit_reported":
            check("4 mpirun's death reported", "!! mpirun lb exited" in out)
            continue
        stem = by_base.get(base)
        if stem is None:
            check(f"4 {base} {what}", False, "not in the task list")
            continue
        if outcome == "refit":
            check(f"4 {base} {what} -> refit", os.path.getmtime(stem + ".json") > t_damage
                  and PS.json_ok(stem + ".json"), "json rewritten after the damage")
        elif outcome == "rerun_postprocess":
            pp = json.load(open(stem + ".json")).get("postprocess", {})
            check(f"4 {base} {what} -> postprocessing rerun", pp.get("rerun_standalone") is True
                  and os.path.exists(stem + "_postprocess.log"), "json records rerun_standalone")
        elif outcome == "redone_post":
            check(f"4 {base} {what} -> redone", os.path.getmtime(stem + "_post.h5") > t_damage
                  and PS.post_state(stem + "_post.h5") == "spectra", "_post.h5 rewritten and valid")
        elif outcome == "removed":
            left = glob.glob(glob.escape(stem) + "*99999")
            check(f"4 {base} {what} -> removed", not left, str(left) if left else "")
        elif outcome == "zero_fails":
            n = fails[base].get("fits", 0) - killed.get(base, 0)
            note = f" (its {killed[base]} SIGKILL failure set aside)" if base in killed else ""
            check(f"4 {base} {what} -> killed, not failed (zero failures{note})", n == 0, f"fails {fails[base]}")
        if what == "killed_fit":
            n = fails[base].get("fits", 0)
            check(f"4 {base} killed fit counted as exactly one failure", n == killed[base], f"fails {fails[base]}")

    # 5. failure counts overall
    extra = {b: s.get("fits", 0) for b, s in fails.items() if s.get("fits", 0) != killed.get(b, 0)}
    check("5a no fits failure except the SIGKILLed fit's (breaker, preemption, mpirun death charge none)",
          not extra, str(extra) if extra else "")
    nspec = sum(s.get("spectra", 0) for s in fails.values())
    check("5b at most one spectra failure in all (galaxies the killed process never reached: none)",
          nspec <= 1, f"{nspec}")
    for f in badin:
        b = os.path.basename(f.stem)
        check(f"5c BAD_INPUT {b}: no failure, not retried after it said bad_input",
              fails[b].get("fits", 0) == 0 and not os.path.exists(f.stem + ".log.prev2"),
              f"fails {fails[b]}")

    # 6. the job log
    restarts = sorted({int(x) for x in re.findall(r"^===== .* restart (\d+)$", out, re.M)})
    check("6a restarts 0, 1, 2, 3 happened", {0, 1, 2, 3} <= set(restarts), f"restarts seen {restarts}")
    chaos = re.findall(r"^CHAOS .*$", out, re.M)
    for key in ("every fit of this pass will fail", "simulated preemption", "truncated", "cut ",
                "SIGKILL fit", "SIGKILL mpirun", "USR1"):
        check(f"6b CHAOS action '{key}' happened", any(key in c for c in chaos))
    check("6c requeue on USR1", "(time limit; self-requeue" in out)
    last = out.split("===== ")[-2:] if out else ["", ""]
    check("6d the last run ends with '===== done' and exit 0", last[0].startswith("done")
          and last[1].strip().endswith("exit 0"), " | ".join(x.strip()[:60] for x in last))
    print("\nCHAOS log:\n" + "\n".join("  " + c for c in chaos))
    print(f"\nRESULT: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
