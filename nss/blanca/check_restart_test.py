"""Checks the outcome of the restart test (blanca/README.md, "Restart test"): a task-list job run
with CHAOS=1 (blanca/chaos_monkey.sh), i.e. preempted mid-fit, outputs damaged, a fit and a spectra
process killed, and a time-limit requeue. Passes if the job nevertheless ended with every fit
cleanly fitted, postprocessed and with spectra:

 1. every fit of the task list is DONE (pipeline_status.py: valid json, _post.h5 of the current
    format, spectra), nothing given up;
 2. no temporary files left anywhere in the run;
 3. every json is complete and records no unresolved postprocessing error; every _post.h5 opens
    and its draws are finite;
 4. each damage in chaos_expect.txt had its expected outcome: the truncated json and the deleted
    files were redone (newer than the damage), the postprocessing rerun is recorded in the json
    (rerun_standalone), the killed fit has a counted failure and was refitted, stale files gone;
 5. the job's .out (all runs of the element are appended to it): restarts 0, 1 and 2 happened,
    every CHAOS action is there, and the last run ends with "===== done".

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
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--tasklist", required=True)
    p.add_argument("--out", required=True, help="the job's .out file (Slurm appends every restart)")
    a = p.parse_args()
    import h5py
    import numpy as np
    r = PS.Run(a.root, a.run, a.tasklist)
    res = r.res

    # 1. states
    states = [r.classify(line, spectra=True, max_fails=10 ** 9, clean=False) for line in r.lines]
    bad = [(st, os.path.basename(s)) for st, s, _ in states if st != "DONE"]
    check("1 every fit DONE (fitted, postprocessed, spectra)", not bad,
          f"{len(r.lines) - len(bad)}/{len(r.lines)}" + (f"; not done: {bad}" if bad else ""))
    gave_up = [os.path.basename(s) for s in glob.glob(os.path.join(res, ".state", "*.json"))
               if max(json.load(open(s)).get(k, 0) for k in ("fits", "spectra")) >= 3]
    check("1b nothing reached the give-up count", not gave_up, str(gave_up) if gave_up else "")

    # 2. temporary files
    tmps = [f for pat in ("*.tmp.*", "*.spectra_tmp.*", "*.json.cut")
            for f in glob.glob(os.path.join(res, "**", pat), recursive=True)]
    check("2 no temporary files left", not tmps, ", ".join(os.path.basename(t) for t in tmps[:5]))

    # 3. json and h5 contents
    probs = []
    for _, stem, _ in states:
        try:
            s = json.load(open(stem + ".json"))
            pp = s.get("postprocess", {})
            if "error" in pp and not pp.get("rerun_standalone"):
                probs.append(f"{os.path.basename(stem)}: postprocess error {pp['error']}")
        except Exception as e:
            probs.append(f"{os.path.basename(stem)}.json: {e!r}")
            continue
        try:
            with h5py.File(stem + "_post.h5", "r") as f:
                lm = f["draws/logmass"][()]
                sb = f["spectra/bestfit"][()]
                if not (np.isfinite(lm).all() and np.isfinite(sb).all()):
                    probs.append(f"{os.path.basename(stem)}_post.h5: non-finite draws or spectrum")
        except Exception as e:
            probs.append(f"{os.path.basename(stem)}_post.h5: {e!r}")
    check("3 every json complete and every _post.h5 readable and finite", not probs, "; ".join(probs[:5]))

    # 4. each damage had its expected outcome
    exp_file = os.path.join(res, "chaos_expect.txt")
    expect = [l.split() for l in open(exp_file) if l.strip()] if os.path.exists(exp_file) else []
    check("4 chaos_expect.txt lists the damage done", len(expect) >= 4, f"{len(expect)} entries")
    for base, what, outcome, t_damage in expect:
        t_damage = float(t_damage)
        stem = next((s for _, s, _ in states if os.path.basename(s) == base), None)
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
        elif outcome == "removed":
            left = glob.glob(glob.escape(stem) + "*99999")
            check(f"4 {base} {what} -> removed", not left, str(left) if left else "")
        if what == "killed_fit":
            f = r.fails(stem)
            check(f"4 {base} killed fit counted as a failure", f.get("fits", 0) >= 1, f"fails {f}")

    # 5. the job log
    out = open(a.out).read() if os.path.exists(a.out) else ""
    restarts = sorted({int(x) for x in re.findall(r"^===== .* restart (\d+)$", out, re.M)})
    check("5a restarts 0, 1, 2 happened", {0, 1, 2} <= set(restarts), f"restarts seen {restarts}")
    chaos = re.findall(r"^CHAOS .*$", out, re.M)
    for key in ("simulated preemption", "truncated", "SIGKILL fit", "USR1"):
        check(f"5b CHAOS action '{key}' happened", any(key in c for c in chaos))
    check("5c requeue on USR1", "requeueing" in out)
    last = out.split("===== ")[-1] if out else ""
    check("5d the last run ends with '===== done'", last.startswith("done"), last.strip()[:80])
    print("\nCHAOS log:\n" + "\n".join("  " + c for c in chaos))
    print(f"\nRESULT: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
