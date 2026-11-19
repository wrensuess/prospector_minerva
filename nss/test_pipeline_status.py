"""Unit test of pipeline_status.py on synthetic run folders (no fitting): every state a killed,
crashed or half-written process can leave, and the bookkeeping the job script relies on.

 1. classification: missing / truncated / unreadable json; json "status" ok, timed_out, capped,
    nonfinite, bad_input, and files from before the status key (logZ, timed_out true); json
    without post (with / without a full .npz, and with a corrupt .npz); corrupt, truncated,
    truncated-but-openable (zero-filled tail), old-format, unreadable _post.h5; a _post.h5 whose
    logZ is not the json's (old json + new h5); spectra marked done but missing; post without
    spectra (wanted or not); complete;
 2. read-only by default: a status run without --clean changes nothing (temporary files, logs,
    outputs); --clean removes stale temporary files; --record-failures never cleans;
 3. --write --stage fits / spectra: the work lists, NSPEC chunks covering every SPEC galaxy once;
    with --clean the stale json/_post.h5/.npz/_draws.npz of a fit to be redone are removed, its
    log kept as .log.prev<k>, and a postprocessing rerun's invalid _post.h5 removed;
 4. --record-failures fits from evidence only: exit code file non-zero or zero without output ->
    failure; no exit code file (killed: preemption, mpirun's death) -> no failure; a json with
    status timed_out -> failure noted "timed_out"; BAD_INPUT never; MAX_FAILS gives up, give-ups
    survive across runs;
 5. circuit breaker: most fits of a pass fail -> exit 12, NOTHING recorded, diagnosis with log
    lines; a few failures -> recorded; fits that had already failed don't trip it;
 6. spectra attribution: only galaxies a process that ended had started are charged; a killed
    process (no exit code) charges nothing; every process failing before its first galaxy trips
    the breaker;
 7. exit codes 0 / 10 / 11 / 12 / 13 (none is Python's own 1 or 2); BAD_INPUT alone gives 0;
    a malformed task-list line is reported with its line number;
 8. --results-root: a run outside <root>/results;
 9. one directory listing per results directory per invocation (the scan is linear).

    python test_pipeline_status.py
Ends RESULT: PASS or RESULT: FAIL.
"""
import glob
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline_status as PS                                    # noqa: E402

ok_all = True
NAMES = ["zred", "logmass", "dust2"]
NB = 5


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail else ''}")


def make_json(stem, valid=True, logz=1.0, **extra):
    d = {"id": "x", "logZ": logz, "postprocess": {}}
    d.update(extra)
    if "status" in extra and extra["status"] == "bad_input":
        d.pop("logZ")
    s = json.dumps(d, indent=1)
    open(stem + ".json", "w").write(s if valid else s[:15])


def make_post(stem, version=PS.FORMAT_VERSION, spectra=False, corrupt=False, logz=1.0,
              spectra_broken=False):
    path = stem + "_post.h5"
    if corrupt:
        open(path, "wb").write(b"\x89HDF\r\n\x1a\n" + b"\0" * 100)
        return
    pc = np.array([16., 50., 84.])
    with h5py.File(path, "w") as f:
        f.attrs.update({"format_version": version, "logZ": logz, "percents": pc,
                        "spectra": "not computed"})
        d = f.create_group("draws")
        for n in NAMES + ["total_mass", "loglike"]:
            d.create_dataset(n, data=np.arange(PS.N_DRAWS, dtype=np.float32) + 1)
        d.create_dataset("sfr", data=np.ones((PS.N_DRAWS, 3), np.float32), compression="gzip", shuffle=True)
        d.create_dataset("index", data=np.arange(PS.N_DRAWS, dtype=np.int64) + 7)
        p = f.create_group("percentiles")
        for n in NAMES:
            p.create_dataset(n, data=np.ones(3))
        p.create_dataset("sfh", data=np.ones((10, 3)))
        b = f.create_group("bestfit")
        b.create_dataset("theta", data=np.ones(len(NAMES)))
        b.attrs["names"] = json.dumps(NAMES)
        b.create_dataset("model_flux", data=np.ones(NB))
        f.create_dataset("model_flux_percentiles", data=np.ones((NB, 3)))
        o = f.create_group("obs")
        o.create_dataset("bands", data=np.array([f"b{i}" for i in range(NB)], dtype=h5py.string_dtype()))
        for k in ("flux", "err"):
            o.create_dataset(k, data=np.ones(NB))
        o.create_dataset("valid", data=np.array([1, 1, 1, 0, 1], np.int8))
        f.create_group("fit").attrs["summary_json"] = json.dumps({"logZ": logz, "n_bands_used": 4})
        if spectra:
            g = f.create_group("spectra")
            g.create_dataset("wavelength_rest", data=np.arange(20.))
            g.create_dataset("bestfit", data=np.ones(20, np.float32))
            g.create_dataset("percentiles", data=np.ones((20, 3), np.float32))
            p.create_dataset("rest_UVJugi_colors", data=np.ones((4, 3)))
            p.create_dataset("rest_UVJugi_colors_map", data=np.ones(4))   # best fit, no percentile axis
            if not spectra_broken:
                d.create_dataset("rest_mags", data=np.ones((PS.N_DRAWS, 10), np.float32))
            f.attrs["spectra"] = PS.SPECTRA_DONE + " 2026-11-16"


def truncate_openable(path, frac=0.7):
    """Cut the file and pad it back with zeros: still opens, but its tail is gone."""
    b = open(path, "rb").read()
    open(path, "wb").write(b[:int(len(b) * frac)] + b"\0" * (len(b) - int(len(b) * frac)))


def make_npz(stem, corrupt=False):
    np.savez_compressed(stem + ".npz", theta=np.zeros((3, 2)), w=np.ones(3), names=np.array(["a", "b"]))
    if corrupt:
        b = open(stem + ".npz", "rb").read()
        open(stem + ".npz", "wb").write(b[: len(b) // 2])


def run(root, *args, run="t", tasks="tasks.txt"):
    p = subprocess.run([sys.executable, os.path.join(HERE, "pipeline_status.py"), "--root", root,
                        "--run", run, "--tasklist", tasks, *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def snapshot(d):
    return {os.path.relpath(f, d): os.path.getmtime(f) for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)}


root = tempfile.mkdtemp()
res = os.path.join(root, "results", "t", "mass")
os.makedirs(res)
cases = {   # id: (setup, expected state with spectra wanted, without)
    "1": ("nothing", "FIT", "FIT"),
    "2": ("truncated json", "FIT", "FIT"),
    "3": ("json, no post, no npz", "FIT", "FIT"),
    "4": ("json, no post, npz", "POST", "POST"),
    "5": ("json, no post, corrupt npz", "FIT", "FIT"),
    "6": ("json, corrupt post, npz", "POST", "POST"),
    "7": ("json, old-format post", "FIT", "FIT"),
    "8": ("json, post without spectra", "SPEC", "DONE"),
    "9": ("json, post without spectra", "SPEC", "DONE"),
    "10": ("json, post with spectra", "DONE", "DONE"),
    "11": ("post with spectra, truncated json", "FIT", "FIT"),
    "12": ("status ok json, post with spectra", "DONE", "DONE"),
    "13": ("status timed_out json, post", "FIT", "FIT"),
    "14": ("status capped json, post", "FIT", "FIT"),
    "15": ("status nonfinite json, post", "FIT", "FIT"),
    "16": ("legacy json timed_out true, post", "FIT", "FIT"),
    "17": ("status bad_input json", "BAD_INPUT", "BAD_INPUT"),
    "18": ("status ok json without logZ", "FIT", "FIT"),
    "19": ("truncated-but-openable post (zero-filled tail), no npz", "FIT", "FIT"),
    "20": ("truncated-but-openable post, npz", "POST", "POST"),
    "21": ("post logZ differs from the json (old json + new post)", "FIT", "FIT"),
    "22": ("spectra marked done, rest_mags missing", "FIT", "FIT"),
    "23": ("post truncated (shorter file)", "FIT", "FIT"),
}
with open(os.path.join(root, "tasks.txt"), "w") as fh:
    for i in cases:
        fh.write(f"cat.fits {i} mass 0\n")
S = {i: os.path.join(res, f"{i}_mass_seed0") for i in cases}
make_json(S["2"], valid=False)
for i in ("3", "4", "5", "6", "7", "8", "9", "10", "19", "20", "21", "22", "23"):
    make_json(S[i])
make_npz(S["4"]); make_npz(S["5"], corrupt=True); make_npz(S["6"]); make_npz(S["20"])
make_post(S["6"], corrupt=True); make_post(S["7"], version=PS.FORMAT_VERSION - 1)
make_post(S["8"]); make_post(S["9"]); make_post(S["10"], spectra=True)
make_post(S["11"], spectra=True); make_json(S["11"], valid=False)
make_json(S["12"], status="ok"); make_post(S["12"], spectra=True)
for i, st in (("13", "timed_out"), ("14", "capped"), ("15", "nonfinite")):
    make_json(S[i], status=st); make_post(S[i], spectra=True)
make_json(S["16"], timed_out=True); make_post(S["16"], spectra=True)
make_json(S["17"], status="bad_input", reason="id 17 not in cat.fits")
make_json(S["18"], status="ok"); d18 = json.load(open(S["18"] + ".json")); d18.pop("logZ")
json.dump(d18, open(S["18"] + ".json", "w"))
for i in ("19", "20"):
    make_post(S[i], spectra=True); truncate_openable(S[i] + "_post.h5")
make_post(S["21"], spectra=True, logz=2.5)
make_post(S["22"], spectra=True, spectra_broken=True)
make_post(S["23"], spectra=True)
b = open(S["23"] + "_post.h5", "rb").read(); open(S["23"] + "_post.h5", "wb").write(b[:len(b) // 2])
open(S["3"] + ".log", "w").write("old log\n")
stale = [S["10"] + "_post.h5.tmp.123", S["10"] + "_post.h5.spectra_tmp.4", S["1"] + ".json.tmp.9"]
for f in stale:
    open(f, "w").write("junk")

# 1. classification
r = PS.Run(root, "t", "tasks.txt", waits=())
for want_spec, col in ((True, 1), (False, 2)):
    got = {line[1]: r.classify(line, spectra=want_spec).state for line in r.lines}
    wrong = {i: (cases[i][0], got[i], cases[i][col]) for i in cases if got[i] != cases[i][col]}
    check(f"1 classification (spectra {'wanted' if want_spec else 'not wanted'})", not wrong, str(wrong))
f19 = r.classify(r.lines[18])
check("1b truncated-but-openable _post.h5 is reported as damaged", any("damaged" in n for n in f19.notes), f19.notes)
f21 = r.classify(r.lines[20])
check("1c old json + new _post.h5 is not a valid pair", any("differs from the fit's json" in n for n in f21.notes), f21.notes)
check("1d legacy files: plain json with logZ and post is DONE (case 10), timed_out true is not (16)", True)
# unreadable file (present, permission denied): retried, then damaged, and said so
up = os.path.join(root, "unreadable")
os.makedirs(up)
make_json(os.path.join(up, "u")); make_post(os.path.join(up, "u"), spectra=True)
os.chmod(os.path.join(up, "u_post.h5"), 0)
if os.access(os.path.join(up, "u_post.h5"), os.R_OK):
    check("1e unreadable _post.h5 (skipped: running as root)", True)
else:
    import time as _t
    t0 = _t.time()
    st, det = PS.post_info(os.path.join(up, "u_post.h5"), waits=(0.05, 0.05))
    check("1e unreadable _post.h5: retried, then 'damaged' with 'unreadable' in the reason (not 'absent')",
          st == "damaged" and "unreadable" in det and _t.time() - t0 >= 0.1, f"{st}: {det[:60]}")
    os.chmod(os.path.join(up, "u_post.h5"), stat.S_IRUSR | stat.S_IWUSR)
    os.chmod(os.path.join(up, "u.json"), 0)
    k, det, _ = PS.json_info(os.path.join(up, "u.json"), waits=(0.01,))
    check("1f unreadable json: 'unreadable', not 'absent'", k == "unreadable", det[:60])
    os.chmod(os.path.join(up, "u.json"), stat.S_IRUSR | stat.S_IWUSR)
orig_checks, orig_to = PS._h5_checks, PS.H5_TIMEOUT
PS._h5_checks, PS.H5_TIMEOUT = (lambda f, js: __import__("time").sleep(30)), 1.0
st, det = PS.post_info(S["10"] + "_post.h5", waits=())
PS._h5_checks, PS.H5_TIMEOUT = orig_checks, orig_to
check("1h a _post.h5 that hangs the HDF5 reader: killed after H5_TIMEOUT, 'damaged'", st == "damaged"
      and "took more than" in det, f"{st}: {det}")
PS._h5_checks = lambda f, js: os._exit(0)
st, det = PS.post_info(S["10"] + "_post.h5", waits=())
PS._h5_checks = orig_checks
check("1i a _post.h5 that crashes the HDF5 reader: 'damaged'", st == "damaged" and "crashed" in det, det)
check("1g absent file is 'absent' / None", PS.json_info(os.path.join(up, "nope.json"))[0] == "absent"
      and PS.post_info(os.path.join(up, "nope_post.h5"))[0] is None)

# 2. read-only by default
before = snapshot(root)
rc, out = run(root, "--list")
rc, out = run(root, "--write", os.path.join(root, "ro"), "--stage", "fits")
after = snapshot(root)
new = set(after) - set(before)
check("2a status without --clean changes no file (only the work lists it was asked to write)",
      all(after.get(k) == v for k, v in before.items()) and new <= {"ro.fit", "ro.post"}, sorted(new))
open(os.path.join(root, "none.txt"), "w").close()
rc, out = run(root, "--record-failures", "fits", os.path.join(root, "none.txt"))
check("2b --record-failures leaves temporary files alone", all(os.path.exists(f) for f in stale))
rc, out = run(root, "--write", os.path.join(root, "w"), "--stage", "fits", "--clean")
check("2c a status run with --clean removes the stale temporary files", not any(os.path.exists(f) for f in stale),
      [l for l in out.splitlines() if "stale" in l])

# 3. work lists and refit cleanup
fit = open(os.path.join(root, "w.fit")).read().split("\n")
post = open(os.path.join(root, "w.post")).read().split("\n")
fit_ids = sorted(l.split()[1] for l in fit if l)
post_ids = sorted(l.split()[1] for l in post if l)
want_fit = sorted(i for i in cases if cases[i][1] == "FIT")
check("3a fit list", fit_ids == want_fit, str(fit_ids))
check("3b postprocessing-rerun list", post_ids == ["20", "4", "6"], str(post_ids))
check("3c exit 10 while work is left", rc == 10, f"exit {rc}")
check("3d log of a fit to be redone kept as .log.prev1",
      os.path.exists(S["3"] + ".log.prev1") and not os.path.exists(S["3"] + ".log"))
gone = [i for i in ("11", "13", "21", "23") if not any(os.path.exists(S[i] + s) for s in (".json", "_post.h5"))]
check("3e stale json and _post.h5 of fits to be redone removed (no old json can pair with a new h5)",
      gone == ["11", "13", "21", "23"], gone)
check("3f stale .npz of a refit removed, kept for a postprocessing rerun; rerun's bad _post.h5 removed",
      not os.path.exists(S["5"] + ".npz") and os.path.exists(S["20"] + ".npz")
      and not os.path.exists(S["20"] + "_post.h5") and os.path.exists(S["20"] + ".json"))
check("3g BAD_INPUT json kept, not scheduled", os.path.exists(S["17"] + ".json") and "17" not in fit_ids)
rc, out = run(root, "--write", os.path.join(root, "w"), "--stage", "spectra", "--nspec", "10")
chunks = sorted(glob.glob(os.path.join(root, "w.spec.*")))
paths = [l.strip() for c in chunks for l in open(c) if l.strip()]
check("3h spectra lists: one chunk per galaxy when NSPEC > galaxies, each once",
      len(chunks) == 2 and sorted(paths) == sorted([S["8"] + "_post.h5", S["9"] + "_post.h5"]), str(chunks))
rc, out = run(root, "--write", os.path.join(root, "w"), "--stage", "spectra", "--nspec", "1")
chunks = sorted(glob.glob(os.path.join(root, "w.spec.*")))
check("3i NSPEC 1: one chunk with both, old chunks removed", len(chunks) == 1
      and len(open(chunks[0]).read().split()) == 2)

# 4. failures from evidence
rcd = os.path.join(root, "rc")
os.makedirs(rcd)
def rcf(i, code):
    open(os.path.join(rcd, f"{i}_mass_seed0.rc"), "w").write(f"{code}\n")
make_json(S["1"]); make_post(S["1"], spectra=True); rcf("1", 0)   # "1" got fitted in the pass
rcf("2", 1)                                    # crashed
rcf("3", 0)                                    # exit 0 without output
# "5": no rc file: killed (preemption / mpirun death)
make_json(S["7"], status="timed_out")          # "7": wrote a timed_out json (no rc: still counts)
rcf("4", 137)                                  # postprocessing rerun OOM-killed, no output
fl = [os.path.join(root, "w.fit"), os.path.join(root, "w.post")]
rc, out = run(root, "--record-failures", "fits", *fl, "--rc-dir", rcd)
r = PS.Run(root, "t", "tasks.txt", waits=())
F = {i: r.fails(S[i]) for i in cases}
check("4a a fit that succeeded is not counted", F["1"].get("fits", 0) == 0, F["1"])
check("4b crashed (exit 1) and exit-0-without-output counted", F["2"].get("fits") == 1 and F["3"].get("fits") == 1)
check("4c no exit code file (killed) is NOT counted", F["5"].get("fits", 0) == 0 and F["11"].get("fits", 0) == 0,
      f"{F['5']} {F['11']}")
check("4d a json with status timed_out counts, noted timed_out", F["7"].get("fits") == 1
      and "timed_out" in F["7"]["history"][-1], F["7"].get("history"))
check("4e failed postprocessing rerun counted", F["4"].get("fits") == 1, F["4"])
check("4f BAD_INPUT never counted", F["17"].get("fits", 0) == 0)
check("4g exit 0 and the killed fits reported as not counted", rc == 0 and "killed before ending" in out,
      [l for l in out.splitlines() if "pass:" in l])
for _ in range(2):
    run(root, "--record-failures", "fits", *fl, "--rc-dir", rcd)
rc, out = run(root, "--max-fails", "3", "--list")
check("4h after 3 failures: GAVE_UP", "GAVE_UP   2_mass_seed0" in out and "GAVE UP" in out,
      [l for l in out.splitlines() if "2_mass" in l][:1])

# 5. circuit breaker (fits)
broot = tempfile.mkdtemp()
bres = os.path.join(broot, "results", "t", "mass")
os.makedirs(bres)
brc = os.path.join(broot, "rc")
os.makedirs(brc)
with open(os.path.join(broot, "tasks.txt"), "w") as fh:
    for i in range(10):
        fh.write(f"cat.fits b{i} mass 0\n")
for i in range(10):
    open(os.path.join(bres, f"b{i}_mass_seed0.log"), "w").write(
        "starting\nFileNotFoundError: [Errno 2] No such file or directory: 'cat.fits'\n")
    open(os.path.join(brc, f"b{i}_mass_seed0.rc"), "w").write("1\n")
shutil.copy(os.path.join(broot, "tasks.txt"), os.path.join(broot, "w.fit"))
open(os.path.join(broot, "w.post"), "w").close()
bl = [os.path.join(broot, "w.fit"), os.path.join(broot, "w.post")]
rc, out = run(broot, "--record-failures", "fits", *bl, "--rc-dir", brc)
check("5a every fit of the pass failed: exit 12 (breaker)", rc == 12, f"exit {rc}")
check("5b breaker: no failure recorded", not os.path.exists(os.path.join(broot, "results", "t", ".state"))
      or not os.listdir(os.path.join(broot, "results", "t", ".state")))
check("5c breaker diagnosis shows the logs' error lines", "CIRCUIT BREAKER" in out and "No such file" in out,
      out.splitlines()[1][:100] if len(out.splitlines()) > 1 else out)
for i in range(6):                             # 4 of 10 fail: below 50% -> recorded
    make_json(os.path.join(bres, f"b{i}_mass_seed0"))
    make_post(os.path.join(bres, f"b{i}_mass_seed0"), spectra=True)
    open(os.path.join(brc, f"b{i}_mass_seed0.rc"), "w").write("0\n")
rc, out = run(broot, "--record-failures", "fits", *bl, "--rc-dir", brc)
check("5d 4 of 10 failed: normal, 4 recorded", rc == 0 and "4 fits failure(s) recorded" in out, out.splitlines()[-1])
rc, out = run(broot, "--record-failures", "fits", *bl, "--rc-dir", brc, "--breaker-min", "2")
check("5e the same 4 failing again (already failed before) don't trip the breaker", rc == 0, f"exit {rc}")
rc, out = run(broot, "--record-failures", "fits", *bl, "--rc-dir", brc, "--breaker-min", "2",
              "--breaker-frac", "0.3")
check("5f still exit 0 with lower thresholds: only fresh fits are judged", rc == 0, f"exit {rc}")

# 6. spectra attribution
sroot = tempfile.mkdtemp()
sres = os.path.join(sroot, "results", "t", "mass")
os.makedirs(sres)
with open(os.path.join(sroot, "tasks.txt"), "w") as fh:
    for i in range(8):
        fh.write(f"cat.fits s{i} mass 0\n")
SS = [os.path.join(sres, f"s{i}_mass_seed0") for i in range(8)]
for s in SS:
    make_json(s); make_post(s)
wd = os.path.join(sroot, "work")
os.makedirs(wd)
L0, L1 = os.path.join(wd, "w.spec.0"), os.path.join(wd, "w.spec.1")
open(L0, "w").writelines(s + "_post.h5\n" for s in SS[:4])
open(L1, "w").writelines(s + "_post.h5\n" for s in SS[4:])
make_post(SS[0], spectra=True)                     # list 0: s0 done, s1 started and not, s2 working
open(L0 + ".started", "w").writelines(s + "_post.h5\n" for s in SS[:3])
open(L0 + ".rc", "w").write("137\n")               # OOM-killed while on s2; s3 never started
open(L1 + ".started", "w").writelines(s + "_post.h5\n" for s in SS[4:6])   # list 1: no rc: killed
rc, out = run(sroot, "--record-failures", "spectra", L0, L1)
rs = PS.Run(sroot, "t", "tasks.txt")
sf = [rs.fails(s).get("spectra", 0) for s in SS]
check("6a only galaxies an ended process had started are charged (s1 finished w/o output, s2 being worked on)",
      sf == [0, 1, 1, 0, 0, 0, 0, 0] and rc == 0, sf)
check("6b the note says which", "was working on it" in json.dumps(rs.fails(SS[2])) and
      "finished it" in json.dumps(rs.fails(SS[1])))
sroot2 = tempfile.mkdtemp()
shutil.copytree(os.path.join(sroot, "results"), os.path.join(sroot2, "results"))
shutil.rmtree(os.path.join(sroot2, "results", "t", ".state"))
shutil.copy(os.path.join(sroot, "tasks.txt"), sroot2)
wd2 = os.path.join(sroot2, "work"); os.makedirs(wd2)
for k, part in ((0, SS[:4]), (1, SS[4:])):
    L = os.path.join(wd2, f"w.spec.{k}")
    open(L, "w").writelines(s.replace(sroot, sroot2) + "_post.h5\n" for s in part)
    open(L + ".rc", "w").write("1\n")              # failed at SpecModel setup, started nothing
rc, out = run(sroot2, "--record-failures", "spectra", *sorted(glob.glob(os.path.join(wd2, "w.spec.?"))))
check("6c every spectra process failed before its first galaxy: breaker (12), nothing charged",
      rc == 12 and not os.path.exists(os.path.join(sroot2, "results", "t", ".state")), f"exit {rc}")

# 7. exit codes
for i in cases:
    if i in ("2", "17"):
        continue
    make_json(S[i]); make_post(S[i], spectra=True)
rc, out = run(root)
st = {l.split()[1].split("_")[0]: l.split()[0] for l in run(root, "--list")[1].splitlines() if "_mass_seed0" in l}
gave = sorted(i for i, s in st.items() if s == "GAVE_UP")
check("7a exit 11 when only given-up (and BAD_INPUT) fits are left", rc == 11 and gave == ["2"], f"exit {rc}, gave up {gave}")
check("7b BAD_INPUT listed with its reason", "BAD_INPUT cat.fits 17 mass 0: id 17 not in cat.fits" in out)
make_json(S["2"]); make_post(S["2"], spectra=True)
rc, out = run(root)
check("7c exit 0 when everything is DONE or BAD_INPUT (a given-up fit that got done counts as done)",
      rc == 0, out.splitlines()[0])
open(os.path.join(root, "bad.txt"), "w").write("cat.fits 1 mass 0\n\ncat.fits 2 mass\n")
rc, out = run(root, tasks="bad.txt")
check("7d malformed task list: exit 13, names line 3, no traceback", rc == 13 and "line 3" in out
      and "Traceback" not in out, out.strip()[:120])
rc, out = run(root, "--first", "1", "--last", "1", tasks="bad.txt")
check("7e lines outside the element's range are not judged", rc in (0, 10), f"exit {rc}")
check("7f no documented exit code is Python's own 1 or 2",
      not {PS.EXIT_DONE, PS.EXIT_WORK_LEFT, PS.EXIT_GAVE_UP, PS.EXIT_BREAKER, PS.EXIT_BAD_TASKLIST} & {1, 2})

# 8. --results-root
rr = tempfile.mkdtemp()
os.makedirs(os.path.join(rr, "t2", "mass"))
shutil.copy(S["10"] + ".json", os.path.join(rr, "t2", "mass", "10_mass_seed0.json"))
shutil.copy(S["10"] + "_post.h5", os.path.join(rr, "t2", "mass", "10_mass_seed0_post.h5"))
open(os.path.join(root, "one.txt"), "w").write("cat.fits 10 mass 0\n")
rc, out = run(root, "--results-root", rr, run="t2", tasks="one.txt")
check("8 --results-root: run found outside <root>/results", rc == 0 and "DONE 1" in out, out.splitlines()[0])

# 9. one listing per directory
calls = []
orig = os.scandir
os.scandir = lambda p: calls.append(p) or orig(p)
try:
    r9 = PS.Run(root, "t", "tasks.txt", waits=())
    for line in r9.lines:
        r9.classify(line, clean=True)
finally:
    os.scandir = orig
check("9 one listing of the results directory for all fits", calls.count(res) == 1, f"{len(calls)} scandir calls")

for d in (root, broot, sroot, sroot2, rr):
    shutil.rmtree(d, ignore_errors=True)
print(f"\nRESULT: {'PASS' if ok_all else 'FAIL'}")
sys.exit(0 if ok_all else 1)
