"""Where each fit of a task list stands, and what still has to run: the restart logic of the
Blanca task-list jobs (blanca/tasklist_layout.sbatch), also usable by hand to check a run.

Each task-list line (CATALOG ID DUST SEED) is one fit, stem results/<RUN>/<DUST>/<ID>_<DUST>_seed<SEED>.
Its state, from the files alone (nothing is trusted that a killed process could have left half
written):

  FIT    no valid json (missing, truncated, unreadable), or the json is valid but there is no
         valid _post.h5 and no full .npz to postprocess from: (re)run the fit (nss_jax.py, which
         postprocesses at its end and writes the json LAST).
  POST   valid json, no valid _post.h5, but a readable full .npz (nss_jax.py keeps it when its
         own postprocessing fails): rerun postprocess_nss.py on it.
  SPEC   valid json and _post.h5, no spectra yet (spectra_nss.py), when spectra are wanted.
  DONE   everything there.
  GAVE_UP  a stage that already failed --max-fails times (a process that ran to its end in a
         pass and did not produce its output; being killed by preemption or a requeue does not
         count, because the job script records failures only after a pass ends normally).

A valid _post.h5 opens, has draws/percentiles/bestfit and the current FORMAT_VERSION of
postprocess_nss.py (an older one is redone). Spectra are done when the file's "spectra" attribute
starts with spectra_nss.DONE. Stale temporary files of killed processes are removed.

Failure counts live in results/<RUN>/.state/<stem>.json, so they survive requeues.

    # what the job script runs before each stage; writes PREFIX.fit / PREFIX.post (task-list
    # lines) or PREFIX.spec.<k> (lists of _post.h5 paths, NSPEC chunks)
    python pipeline_status.py --root R --run RUN --tasklist T --first 1 --last 200 --write PREFIX --stage fits
    python pipeline_status.py ... --write PREFIX --stage spectra --nspec 10
    # after a pass that ended normally: count failures of what was scheduled
    python pipeline_status.py ... --record-failures fits PREFIX.fit PREFIX.post
    python pipeline_status.py ... --record-failures spectra PREFIX.spec.0 PREFIX.spec.1 ...
    # by hand: the state of a whole run (no --first/--last: every line)
    python pipeline_status.py --root R --run RUN --tasklist T [--spectra 0] [--list]

Exit status: 0 all DONE; 3 work left that will be retried; 1 nothing left to retry but some
GAVE_UP (listed). Imports only h5py (no numpy/jax/efsps), so it is fast.
"""
import argparse
import ast
import glob
import json
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def _const(path, name):
    """A module-level constant read from the source, so this script needs no efsps/jax imports."""
    m = re.search(rf"^{name}\s*=\s*(.+?)\s*(#.*)?$", open(path).read(), re.M)
    return ast.literal_eval(m.group(1))                         # a literal: int or str


FORMAT_VERSION = _const(os.path.join(HERE, "postprocess_nss.py"), "FORMAT_VERSION")
SPECTRA_DONE = _const(os.path.join(HERE, "spectra_nss.py"), "DONE")


def json_ok(path):
    try:
        with open(path) as fh:
            s = json.load(fh)
        return isinstance(s, dict) and "logZ" in s
    except (OSError, ValueError):
        return False


def post_state(path):
    """None (missing/unreadable/old format), 'post' (valid, no spectra) or 'spectra'."""
    import h5py
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, "r") as f:
            if int(f.attrs.get("format_version", -1)) != FORMAT_VERSION:
                return None
            if not all(k in f for k in ("draws", "percentiles", "bestfit")):
                return None
            if str(f.attrs.get("spectra", "")).startswith(SPECTRA_DONE) and "spectra/bestfit" in f:
                return "spectra"
            return "post"
    except (OSError, KeyError, ValueError):
        return None


def npz_ok(path):
    """A readable full sample file (zip intact, the arrays postprocess_nss.py needs)."""
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            return {"theta.npy", "w.npy", "names.npy"} <= names and z.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


class Run:
    def __init__(self, root, run, tasklist, first=None, last=None):
        self.root, self.run = root, run
        self.res = os.path.join(root, "results", run)
        self.state_dir = os.path.join(self.res, ".state")
        lines = [l.split() for l in open(os.path.join(root, tasklist) if not os.path.isabs(tasklist)
                                          else tasklist) if l.strip()]
        a = (first or 1) - 1
        b = last if last is not None else len(lines)
        self.lines = lines[a:b]

    def stem(self, line):
        cat, i, dust, seed = line
        return os.path.join(self.res, dust, f"{i}_{dust}_seed{seed}")

    # failure counts -------------------------------------------------------------------------
    def _state_file(self, stem):
        return os.path.join(self.state_dir, os.path.basename(stem) + ".json")

    def fails(self, stem):
        try:
            return json.load(open(self._state_file(stem)))
        except (OSError, ValueError):
            return {"fits": 0, "spectra": 0}

    def add_fail(self, stem, stage, note):
        os.makedirs(self.state_dir, exist_ok=True)
        s = self.fails(stem)
        s[stage] = s.get(stage, 0) + 1
        s.setdefault("history", []).append(f"{stage} failed: {note}")
        tmp = self._state_file(stem) + f".tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(s, fh, indent=1)
        os.replace(tmp, self._state_file(stem))

    # state ----------------------------------------------------------------------------------
    @staticmethod
    def clean_stale(stem):
        """Temporary files of killed processes (only between passes: nothing of this element runs)."""
        gone = []
        for pat in (".json.tmp.*", "_post.h5.tmp.*", "_post.h5.spectra_tmp.*"):
            for f in glob.glob(glob.escape(stem) + pat):
                os.remove(f)
                gone.append(os.path.basename(f))
        return gone

    def classify(self, line, spectra=True, max_fails=3, clean=True):
        stem = self.stem(line)
        stale = self.clean_stale(stem) if clean else []
        ps = post_state(stem + "_post.h5")
        if not json_ok(stem + ".json"):
            st, stage = "FIT", "fits"
        elif ps is None:
            st, stage = ("POST", "fits") if npz_ok(stem + ".npz") else ("FIT", "fits")
        elif spectra and ps != "spectra":
            st, stage = "SPEC", "spectra"
        else:
            return "DONE", stem, stale
        if self.fails(stem).get(stage, 0) >= max_fails:
            return "GAVE_UP", stem, stale
        return st, stem, stale


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="nss_test/ (task list and results/ are under it)")
    p.add_argument("--run", required=True)
    p.add_argument("--tasklist", required=True)
    p.add_argument("--first", type=int, default=None, help="first task-list line (1-based)")
    p.add_argument("--last", type=int, default=None, help="last task-list line (inclusive)")
    p.add_argument("--spectra", type=int, default=1, help="1: a fit needs spectra to be DONE")
    p.add_argument("--max-fails", type=int, default=3)
    p.add_argument("--write", default=None, help="prefix of the work lists to write")
    p.add_argument("--stage", choices=("fits", "spectra"), default="fits")
    p.add_argument("--nspec", type=int, default=10, help="spectra work lists (parallel processes)")
    p.add_argument("--record-failures", nargs="+", metavar=("STAGE", "LIST"), default=None)
    p.add_argument("--list", action="store_true", help="print every fit's state")
    a = p.parse_args()
    r = Run(a.root, a.run, a.tasklist, a.first, a.last)

    if a.record_failures:
        stage, lists = a.record_failures[0], a.record_failures[1:]
        n = 0
        for lf in lists:
            if not os.path.exists(lf):
                continue
            for l in open(lf):
                l = l.strip()
                if not l:
                    continue
                if stage == "fits":
                    line = l.split()
                    st, stem, _ = r.classify(line, spectra=False, max_fails=10 ** 9, clean=False)
                    if st in ("FIT", "POST"):
                        r.add_fail(stem, "fits", f"still {st} after a pass that ended normally")
                        print(f"  failed: {os.path.basename(stem)} ({st})")
                        n += 1
                else:
                    stem = l[:-len("_post.h5")]
                    if post_state(l) != "spectra":
                        r.add_fail(stem, "spectra", "no spectra after a pass that ended normally")
                        print(f"  failed: {os.path.basename(stem)} (spectra)")
                        n += 1
        print(f"pipeline_status: {n} {stage} failure(s) recorded")
        return 0

    counts = {k: [] for k in ("DONE", "FIT", "POST", "SPEC", "GAVE_UP")}
    stale_all = []
    for line in r.lines:
        st, stem, stale = r.classify(line, spectra=bool(a.spectra), max_fails=a.max_fails)
        counts[st].append((line, stem))
        stale_all += stale
        if a.list:
            print(f"  {st:8s} {os.path.basename(stem)}")
    if stale_all:
        print(f"pipeline_status: removed {len(stale_all)} stale temporary file(s): "
              f"{', '.join(stale_all[:5])}{' ...' if len(stale_all) > 5 else ''}")
    print("pipeline_status: " + ", ".join(f"{k} {len(v)}" for k, v in counts.items())
          + f"  ({len(r.lines)} fits; {r.run})")
    for line, stem in counts["GAVE_UP"]:
        hist = r.fails(stem).get("history", [])
        print(f"  GAVE UP {' '.join(line)}: {hist[-1] if hist else ''} "
              f"(see {os.path.basename(stem)}.log*)")

    if a.write:
        if a.stage == "fits":
            for k, key in (("fit", "FIT"), ("post", "POST")):
                with open(f"{a.write}.{k}", "w") as fh:
                    fh.writelines(" ".join(line) + "\n" for line, _ in counts[key])
            # keep the log of a fit that is about to be redone (the rerun overwrites it)
            for line, stem in counts["FIT"]:
                if os.path.exists(stem + ".log"):
                    k = 1
                    while os.path.exists(f"{stem}.log.prev{k}"):
                        k += 1
                    os.replace(stem + ".log", f"{stem}.log.prev{k}")
        else:
            for f in glob.glob(glob.escape(a.write) + ".spec.*"):
                os.remove(f)
            todo = [stem + "_post.h5" for _, stem in counts["SPEC"]]
            k = min(a.nspec, len(todo))
            for j in range(k):
                with open(f"{a.write}.spec.{j}", "w") as fh:
                    fh.writelines(t + "\n" for t in todo[j::k])
    if counts["FIT"] or counts["POST"] or counts["SPEC"]:
        return 3
    return 1 if counts["GAVE_UP"] else 0


if __name__ == "__main__":
    sys.exit(main())
