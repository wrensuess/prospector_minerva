"""Unit test of pipeline_status.py on synthetic run folders (no fitting): every state a killed,
crashed or half-written process can leave, and the bookkeeping the job script relies on.

 1. classification: missing json, truncated json, json without post (with / without a full .npz,
    and with a corrupt .npz), corrupt _post.h5, old-format _post.h5, post without spectra (with
    spectra wanted or not), complete;
 2. stale temporary files are removed by a status run, not by --record-failures;
 3. --write --stage fits / spectra: the work lists, NSPEC chunks covering every SPEC galaxy once,
    the log of a fit about to be redone kept as .log.prev<k>;
 4. --record-failures counts only what is still undone; MAX_FAILS gives up (exit 1 when nothing
    else is left), and give-ups survive across runs (state in .state/);
 5. exit codes 0 / 3 / 1.

    python test_pipeline_status.py
Ends RESULT: PASS or RESULT: FAIL.
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline_status as PS                                    # noqa: E402

ok_all = True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def make_json(stem, valid=True):
    s = json.dumps({"id": "x", "logZ": 1.0, "postprocess": {}}, indent=1)
    open(stem + ".json", "w").write(s if valid else s[:15])


def make_post(stem, version=PS.FORMAT_VERSION, spectra=False, corrupt=False):
    path = stem + "_post.h5"
    if corrupt:
        open(path, "wb").write(b"\x89HDF\r\n\x1a\n" + b"\0" * 100)
        return
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = version
        for g in ("draws", "percentiles", "bestfit"):
            f.create_group(g)
        f.attrs["spectra"] = "not computed"
        if spectra:
            f.create_dataset("spectra/bestfit", data=np.ones(3))
            f.attrs["spectra"] = PS.SPECTRA_DONE + " 2026-11-16"


def make_npz(stem, corrupt=False):
    np.savez_compressed(stem + ".npz", theta=np.zeros((3, 2)), w=np.ones(3), names=np.array(["a", "b"]))
    if corrupt:
        b = open(stem + ".npz", "rb").read()
        open(stem + ".npz", "wb").write(b[: len(b) // 2])


def run(root, *args):
    p = subprocess.run([sys.executable, os.path.join(HERE, "pipeline_status.py"), "--root", root,
                        "--run", "t", "--tasklist", "tasks.txt", *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


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
}
with open(os.path.join(root, "tasks.txt"), "w") as fh:
    for i in cases:
        fh.write(f"cat.fits {i} mass 0\n")
S = {i: os.path.join(res, f"{i}_mass_seed0") for i in cases}
make_json(S["2"], valid=False)
for i in ("3", "4", "5", "6", "7", "8", "9", "10"):
    make_json(S[i])
make_npz(S["4"]); make_npz(S["5"], corrupt=True); make_npz(S["6"])
make_post(S["6"], corrupt=True); make_post(S["7"], version=PS.FORMAT_VERSION - 1)
make_post(S["8"]); make_post(S["9"]); make_post(S["10"], spectra=True)
make_post(S["11"], spectra=True); make_json(S["11"], valid=False)
open(S["3"] + ".log", "w").write("old log\n")
stale = [S["10"] + "_post.h5.tmp.123", S["10"] + "_post.h5.spectra_tmp.4", S["1"] + ".json.tmp.9"]
for f in stale:
    open(f, "w").write("junk")

# 1. classification
r = PS.Run(root, "t", "tasks.txt")
for want_spec, col in ((True, 1), (False, 2)):
    got = {line[1]: r.classify(line, spectra=want_spec, clean=False)[0] for line in r.lines}
    wrong = {i: (cases[i][0], got[i], cases[i][col]) for i in cases if got[i] != cases[i][col]}
    check(f"1 classification (spectra {'wanted' if want_spec else 'not wanted'})", not wrong, str(wrong))

# 2. record-failures does not clean; a status run does
open(os.path.join(root, "none.txt"), "w").close()
rc, out = run(root, "--record-failures", "fits", os.path.join(root, "none.txt"))
check("2a --record-failures leaves temporary files alone", all(os.path.exists(f) for f in stale))
rc, out = run(root, "--write", os.path.join(root, "w"), "--stage", "fits")
check("2b a status run removes the stale temporary files", not any(os.path.exists(f) for f in stale), out.splitlines()[0])

# 3. work lists
fit = open(os.path.join(root, "w.fit")).read().split("\n")
post = open(os.path.join(root, "w.post")).read().split("\n")
fit_ids = sorted(l.split()[1] for l in fit if l)
post_ids = sorted(l.split()[1] for l in post if l)
check("3a fit list", fit_ids == sorted(["1", "2", "3", "5", "7", "11"]), str(fit_ids))
check("3b postprocessing-rerun list", post_ids == ["4", "6"], str(post_ids))
check("3c exit 3 while work is left", rc == 3, f"exit {rc}")
check("3d log of a fit to be redone kept as .log.prev1",
      os.path.exists(S["3"] + ".log.prev1") and not os.path.exists(S["3"] + ".log"))
rc, out = run(root, "--write", os.path.join(root, "w"), "--stage", "spectra", "--nspec", "10")
chunks = sorted(glob.glob(os.path.join(root, "w.spec.*")))
paths = [l.strip() for c in chunks for l in open(c) if l.strip()]
check("3e spectra lists: one chunk per galaxy when NSPEC > galaxies, each once",
      len(chunks) == 2 and sorted(paths) == sorted([S["8"] + "_post.h5", S["9"] + "_post.h5"]), str(chunks))
rc, out = run(root, "--write", os.path.join(root, "w"), "--stage", "spectra", "--nspec", "1")
chunks = sorted(glob.glob(os.path.join(root, "w.spec.*")))
check("3f NSPEC 1: one chunk with both, old chunks removed", len(chunks) == 1
      and len(open(chunks[0]).read().split()) == 2)

# 4. failures: fix some, record, give up after max-fails
make_json(S["1"]); make_post(S["1"], spectra=True)          # "1" got fitted in the pass
rc, out = run(root, "--record-failures", "fits", os.path.join(root, "w.fit"), os.path.join(root, "w.post"))
f1, f2 = r.fails(S["1"]), r.fails(S["2"])
check("4a a fit that succeeded is not counted", f1.get("fits", 0) == 0, str(f1))
check("4b a fit still undone is counted", f2.get("fits") == 1, str(f2))
for _ in range(2):
    run(root, "--record-failures", "fits", os.path.join(root, "w.fit"), os.path.join(root, "w.post"))
rc, out = run(root, "--max-fails", "3", "--list")
check("4c after 3 failures: GAVE_UP", "GAVE_UP  2_mass_seed0" in out and "GAVE UP" in out,
      [l for l in out.splitlines() if "2_mass" in l][0])
rc, out = run(root, "--record-failures", "spectra", *glob.glob(os.path.join(root, "w.spec.*")))
check("4d spectra failures counted separately", r.fails(S["8"]).get("spectra") == 1 and
      r.fails(S["8"]).get("fits", 0) == 0, str(r.fails(S["8"])))

# 5. exit codes: finish everything except the given-up ones -> 1; give-ups gone -> 0
for i in ("3", "4", "5", "6", "7", "8", "9", "11"):
    make_json(S[i]); make_post(S[i], spectra=True)
rc, out = run(root)
st = {l.split()[1].split("_")[0]: l.split()[0] for l in run(root, "--list")[1].splitlines() if "_mass_seed0" in l}
gave = sorted(i for i, s in st.items() if s == "GAVE_UP")
check("5a exit 1 when only given-up fits are left", rc == 1 and gave == ["2"], f"exit {rc}, gave up {gave}")
make_json(S["2"]); make_post(S["2"], spectra=True)
rc, out = run(root)
check("5b exit 0 when everything is DONE (a given-up fit that got done counts as done)", rc == 0,
      out.splitlines()[-1])

print(f"\nRESULT: {'PASS' if ok_all else 'FAIL'}")
sys.exit(0 if ok_all else 1)
