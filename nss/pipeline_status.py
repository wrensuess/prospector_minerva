"""Where each fit of a task list stands, and what still has to run: the restart logic of the
Blanca task-list jobs (blanca/tasklist_layout.sbatch), also usable by hand to check a run.

Each task-list line (CATALOG ID DUST SEED) is one fit, stem <RESULTS_ROOT>/<RUN>/<DUST>/<ID>_<DUST>_seed<SEED>
(RESULTS_ROOT: --results-root, default <root>/results). Its state, from the files alone (nothing
is trusted that a killed process could have left half written):

  FIT       no fitted json: missing, truncated, unreadable, or a json whose "status" says the fit
            did not succeed ("timed_out", "capped", "nonfinite"); or a fitted json but no valid
            _post.h5 and no full .npz to postprocess from: (re)run the fit (nss_jax.py, which
            postprocesses at its end and writes the json LAST).
  POST      fitted json, no valid _post.h5, but a readable full .npz (nss_jax.py keeps it when its
            own postprocessing fails): rerun postprocess_nss.py on it.
  SPEC      fitted json and valid _post.h5, no spectra yet (spectra_nss.py), when spectra are wanted.
  DONE      everything there.
  BAD_INPUT the fit's json has "status": "bad_input" (nss_jax.py found nothing to fit: the id is
            not in the catalogue, no usable band). Final: never retried, never a failure; listed
            with its reason. Delete the json to retry it.
  GAVE_UP   a stage that already failed --max-fails times (see "Failures" below).

A json counts as fitted if it is valid and has "status": "ok", or (files written before the
status key existed) no "status" but a "logZ" and no "timed_out": true.

A valid _post.h5: opens; has the current FORMAT_VERSION of postprocess_nss.py (an older one is
redone); every object in it can be visited and the last element of every dataset read (catches a
truncated or zero-filled file that still opens); dataset shapes agree with each other and with
the fit summary stored in it; its logZ is the one in the fit's json (an old json next to a new
_post.h5, or the reverse, is not a valid pair). Spectra are done when the file's "spectra"
attribute starts with spectra_nss.DONE and the spectra datasets pass the same checks.
A file that exists but cannot be read (an I/O error, e.g. a network filesystem hiccup) is read
again a few times; if it stays unreadable it is treated as damaged and reported as such. The
look inside a _post.h5 runs in a forked child that is killed after H5_TIMEOUT (60 s): a damaged
file can make the HDF5 library hang or crash, which then counts as damaged instead of stalling.
Each results directory is listed once per invocation (the scan is linear in the number of files).

Failures (--record-failures, run by the job script after each pass): a fit or postprocessing
rerun counts as failed only on evidence that its own process ended without its output: its exit
code file (<rc-dir>/<stem>.rc, written by the task line after the process ends) is there and the
fit is still undone, or its json reports "timed_out" / "capped" / "nonfinite". A fit without an
exit code file was killed (preemption, requeue, mpirun's death), which is never a failure. A
spectra galaxy is charged only if its process ended (list.rc) after STARTING it (list.started,
written by spectra_nss.py --progress) and it has no spectra; galaxies the process never reached
are not charged. Circuit breaker: if, among fits (galaxies) that had not failed before, at least
--breaker-min that ended in this pass failed, and they are at least --breaker-frac of those that
ended, or every spectra process that ended failed before starting any galaxy, the node or the
setup is at fault, not the galaxies: nothing is recorded, a diagnosis with the first error lines
of the logs is printed, and the exit status is 12. Failure counts live in
<RESULTS_ROOT>/<RUN>/.state/<stem>.json, so they survive requeues.

READ-ONLY by default: a status run by hand changes nothing. --clean (the job script, between
passes, while it holds the element's lock) also removes stale temporary files of killed
processes, keeps the log of a fit about to be redone as .log.prev<k>, and removes the stale json,
_post.h5, .npz and _draws.npz of every fit scheduled for a refit (and the invalid _post.h5 of a
postprocessing rerun), so that a fit killed later can never leave an old json next to a new
_post.h5.

    # what the job script runs before each stage; writes PREFIX.fit / PREFIX.post (task-list
    # lines) or PREFIX.spec.<k> (lists of _post.h5 paths, NSPEC chunks)
    python pipeline_status.py --root R --run RUN --tasklist T --first 1 --last 200 --write PREFIX --stage fits --clean
    python pipeline_status.py ... --write PREFIX --stage spectra --nspec 10 --clean
    # after a pass: judge what was scheduled (exit codes in --rc-dir, default: the lists' folder)
    python pipeline_status.py ... --record-failures fits PREFIX.fit PREFIX.post
    python pipeline_status.py ... --record-failures spectra PREFIX.spec.0 PREFIX.spec.1 ...
    # by hand: the state of a whole run (no --first/--last: every line); read-only
    python pipeline_status.py --root R --run RUN --tasklist T [--results-root D] [--spectra 0] [--list]

Exit status (chosen not to collide with Python's own 1 = uncaught exception, 2 = usage error):
   0  every fit DONE or BAD_INPUT (--record-failures: failures recorded normally)
  10  work left that will be retried (FIT / POST / SPEC)
  11  nothing left to retry, but some fits GAVE_UP (listed)
  12  --record-failures only: circuit breaker tripped, nothing recorded
  13  the task list has a malformed line (line number printed)
Imports only h5py (no numpy/jax/efsps), so it is fast.
"""
import argparse
import ast
import json
import os
import re
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

EXIT_DONE, EXIT_WORK_LEFT, EXIT_GAVE_UP, EXIT_BREAKER, EXIT_BAD_TASKLIST = 0, 10, 11, 12, 13
FAILED_STATUSES = ("timed_out", "capped", "nonfinite")
READ_WAITS = (0.5, 1.0, 2.0)          # seconds between read attempts of a present file
H5_TIMEOUT = 60.0                     # s for one look inside a _post.h5 (a damaged file can hang HDF5)
TMP_RE = re.compile(r"^(?P<base>.+?)(?:\.json\.tmp\.|_post\.h5\.tmp\.|_post\.h5\.spectra_tmp\.)\d+$")
ERR_RE = re.compile(r"error|exception|traceback|killed|no such file|denied|no space|"
                    r"cannot|failed|abort|segmentation|memory|terminated", re.I)


def _const(path, name):
    """A module-level constant read from the source, so this script needs no efsps/jax imports."""
    m = re.search(rf"^{name}\s*=\s*(.+?)\s*(#.*)?$", open(path).read(), re.M)
    return ast.literal_eval(m.group(1))                         # a literal: int or str


FORMAT_VERSION = _const(os.path.join(HERE, "postprocess_nss.py"), "FORMAT_VERSION")
N_DRAWS = _const(os.path.join(HERE, "postprocess_nss.py"), "N_DRAWS")
SPECTRA_DONE = _const(os.path.join(HERE, "spectra_nss.py"), "DONE")


class TaskListError(Exception):
    pass


# ------------------------------------------------------------------------- reading with retries
def _read_raw(path, waits=READ_WAITS):
    """The file's bytes; None if it does not exist; raises OSError if it exists but stays
    unreadable after len(waits) retries (a transient error on a network filesystem gets time)."""
    err = None
    for w in (0.0,) + tuple(waits):
        if w:
            time.sleep(w)
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            err = e
    raise err


def json_info(path, waits=READ_WAITS):
    """(kind, detail, data): kind 'absent', 'unreadable', 'invalid', 'ok' (fitted), 'bad_input'
    (detail = reason) or 'failed' (detail = the status: timed_out / capped / nonfinite / ...)."""
    try:
        raw = _read_raw(path, waits)
    except OSError as e:
        return "unreadable", f"json unreadable: {e!r}", None
    if raw is None:
        return "absent", "", None
    try:
        s = json.loads(raw)
    except ValueError as e:
        return "invalid", f"json invalid ({e.__class__.__name__}: truncated or corrupt)", None
    if not isinstance(s, dict):
        return "invalid", "json is not an object", None
    st = s.get("status")
    if st is None:                                     # files from before the status key
        if "logZ" not in s:
            return "invalid", "json has neither status nor logZ", s
        if s.get("timed_out") is True:
            return "failed", "timed_out", s
        return "ok", "", s
    if st == "ok":
        return ("ok", "", s) if "logZ" in s else ("invalid", "status ok but no logZ", s)
    if st == "bad_input":
        return "bad_input", str(s.get("reason", "no reason given")), s
    return "failed", str(st), s


def json_ok(path):
    return json_info(path)[0] == "ok"


def _h5_checks(f, js):
    """Raises ValueError (with the reason) if the open _post.h5 is not a complete, consistent file.
    Returns 'post' or 'spectra'."""
    import h5py
    if int(f.attrs.get("format_version", -1)) != FORMAT_VERSION:
        raise _OldFormat(f"format_version {f.attrs.get('format_version')} != {FORMAT_VERSION}")
    for g in ("draws", "percentiles", "bestfit", "obs", "fit"):
        if g not in f:
            raise ValueError(f"no {g} group")
    dsets = {}

    def visit(name, obj):                              # touches every object header
        if isinstance(obj, h5py.Dataset):
            dsets[name] = obj.shape
            obj[tuple(s - 1 for s in obj.shape)] if obj.size else None   # last element: data there
    f.visititems(visit)
    # data that cannot be exactly zero: a zero-filled region (a file cut short and padded, or
    # blocks that never reached the disk) reads as 0 there without any HDF5 error. These
    # datasets lie throughout the file (draws/loglike, draws/index and obs/err are written last).
    # In 1052 production files none of them ever ended in 0 (2026-11-16).
    for k, ok in (("draws/loglike", lambda v: v == v and v != 0), ("draws/total_mass", lambda v: v > 0),
                  ("draws/index", lambda v: v > 0), ("obs/err", lambda v: v != v or v > 0),
                  ("bestfit/model_flux", lambda v: v != 0), ("model_flux_percentiles", lambda v: v != 0),
                  ("spectra/wavelength_rest", lambda v: v > 0), ("spectra/bestfit", lambda v: v != 0),
                  ("draws/rest_mags", lambda v: v != 0)):
        if k in dsets and dsets[k] and not ok(float(f[k][tuple(s - 1 for s in dsets[k])])):
            raise ValueError(f"{k} ends in an impossible value (zero-filled or cut short)")
    names = json.loads(f["bestfit"].attrs["names"])
    if f["bestfit/theta"].shape != (len(names),):
        raise ValueError("bestfit/theta does not match its names")
    nd = {s[0] for n, s in dsets.items() if n.startswith("draws/") and s and n != "draws/rest_mags"}
    if nd != {N_DRAWS}:
        raise ValueError(f"draws lengths {sorted(nd)} != {N_DRAWS}")
    missing = [n for n in names if f"draws/{n}" not in dsets]
    if missing:
        raise ValueError(f"draws lack {missing[:3]}")
    nb = dsets["obs/bands"][0]
    for k in ("obs/valid", "obs/flux", "obs/err", "bestfit/model_flux", "model_flux_percentiles"):
        if dsets.get(k, (None,))[0] != nb:
            raise ValueError(f"{k} shape {dsets.get(k)} does not match {nb} bands")
    npc = len(f.attrs["percents"])
    bad = [n for n, s in dsets.items() if n.startswith("percentiles/") and not n.endswith("_map")
           and (not s or s[-1] != npc)]                  # *_map: the best fit's values (spectra_nss)
    if bad:
        raise ValueError(f"percentiles without {npc} columns: {bad[:3]}")
    s = json.loads(f["fit"].attrs["summary_json"])
    if int(s.get("n_bands_used", -1)) != int(f["obs/valid"][()].sum()):
        raise ValueError("number of valid bands differs from the fit summary")
    lz = float(f.attrs["logZ"])
    if not _same(lz, s.get("logZ")):
        raise ValueError("logZ differs from the fit summary inside the file")
    if js is not None and "logZ" in js and not _same(lz, js.get("logZ")):
        raise ValueError("logZ differs from the fit's json (json and _post.h5 from different fits)")
    if not str(f.attrs.get("spectra", "")).startswith(SPECTRA_DONE):
        return "post"
    nw = dsets.get("spectra/wavelength_rest", (None,))[0]
    if nw is None or dsets.get("spectra/bestfit") != (nw,) or dsets.get("spectra/percentiles") != (nw, npc) \
            or dsets.get("draws/rest_mags", (None,))[0] != N_DRAWS:
        raise ValueError("spectra marked done but their datasets are missing or the wrong shape")
    return "spectra"


class _OldFormat(Exception):
    pass


def _same(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return a == b or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _h5_check_once(path, js):
    """(state, detail) from one look inside the file: run in a child process (os.fork) that is
    killed after H5_TIMEOUT s, because the HDF5 library can loop forever (or crash) on a damaged
    file: a _post.h5 whose last 1% was zero-filled hung h5py reading an attribute (2026-11-17)."""
    import select
    import signal
    import h5py                                        # imported once here, inherited by each child
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                       # child: never returns
        try:
            os.close(r)
            try:
                with h5py.File(path, "r") as f:
                    res = [_h5_checks(f, js), ""]
            except _OldFormat as e:
                res = ["old_format", str(e)]
            except FileNotFoundError:
                res = [None, ""]
            except Exception as e:                     # bad content, or a transient read error
                res = ["error", f"{e.__class__.__name__}: {e}"]
            os.write(w, json.dumps(res).encode())
        finally:
            os._exit(0)
    os.close(w)
    data, t_end = b"", time.time() + H5_TIMEOUT
    try:
        while True:
            left = t_end - time.time()
            if left <= 0 or not select.select([r], [], [], left)[0]:
                os.kill(pid, signal.SIGKILL)
                return "hung", f"reading it took more than {H5_TIMEOUT:g} s (HDF5 stuck on a damaged file)"
            chunk = os.read(r, 65536)
            if not chunk:
                break
            data += chunk
    finally:
        os.close(r)
        os.waitpid(pid, 0)
    if not data:
        return "error", "the HDF5 reader crashed on it"
    st, det = json.loads(data)
    return st, det


def post_info(path, js=None, waits=READ_WAITS):
    """(state, detail): state None (absent), 'old_format', 'damaged' (detail: why; includes a file
    that stays unreadable, or that hangs or crashes the HDF5 library), 'post' (valid, no spectra)
    or 'spectra'. js: the fit's json data."""
    err = None
    for attempt, w in enumerate((0.0,) + tuple(waits[-1:])):   # the content: two attempts
        if w:
            time.sleep(w)
        try:
            raw = _read_raw(path, waits)
        except OSError as e:
            return "damaged", f"_post.h5 unreadable ({e!r}) after {len(waits)} retries"
        if raw is None:
            return None, ""
        st, det = _h5_check_once(path, js)
        if st == "hung":
            return "damaged", f"_post.h5 damaged: {det}"
        if st != "error":
            return st, det
        err = det                                      # bad content, or a transient error: again
    return "damaged", f"_post.h5 damaged: {err}"


def post_state(path, js=None):
    """Backwards-compatible: None (not usable), 'post' or 'spectra'."""
    st, _ = post_info(path, js)
    return st if st in ("post", "spectra") else None


def npz_ok(path):
    """A readable full sample file (zip intact, the arrays postprocess_nss.py needs)."""
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            return {"theta.npy", "w.npy", "names.npy"} <= names and z.testzip() is None
    except (OSError, zipfile.BadZipFile, ValueError):
        return False


# ------------------------------------------------------------------------- the run
def read_tasklist(path, first=None, last=None):
    """Non-blank lines first..last (1-based, inclusive, counted over non-blank lines), each
    checked: CATALOG ID DUST SEED. Raises TaskListError naming the file line."""
    rows = []
    with open(path) as fh:
        for lineno, l in enumerate(fh, 1):
            if l.strip():
                rows.append((lineno, l.split()))
    a = (first or 1) - 1
    b = last if last is not None else len(rows)
    rows = rows[a:b]
    for lineno, f in rows:
        if len(f) != 4:
            raise TaskListError(f"{path} line {lineno}: expected 4 fields (CATALOG ID DUST SEED), "
                                f"got {len(f)}: {' '.join(f)!r}")
        if not re.fullmatch(r"-?\d+", f[3]) or "/" in f[1] or "/" in f[2]:
            raise TaskListError(f"{path} line {lineno}: SEED must be an integer and ID, DUST "
                                f"plain names: {' '.join(f)!r}")
    return [f for _, f in rows]


class Fit:
    __slots__ = ("line", "stem", "state", "stage", "notes", "reason")

    def __init__(self, line, stem):
        self.line, self.stem, self.state, self.stage, self.notes, self.reason = line, stem, None, None, [], ""


class Run:
    def __init__(self, root, run, tasklist, first=None, last=None, results_root=None, waits=READ_WAITS):
        self.root, self.run, self.waits = root, run, waits
        self.res = os.path.join(results_root or os.path.join(root, "results"), run)
        self.state_dir = os.path.join(self.res, ".state")
        tl = tasklist if os.path.isabs(tasklist) else os.path.join(root, tasklist)
        self.lines = read_tasklist(tl, first, last)
        self._dirs = {}                                # dir -> set of names (one listing each)
        self._fails = None

    def stem(self, line):
        cat, i, dust, seed = line
        return os.path.join(self.res, dust, f"{i}_{dust}_seed{seed}")

    # one listing per directory ---------------------------------------------------------------
    def listing(self, d):
        if d not in self._dirs:
            names = set()
            for w in (0.0,) + tuple(self.waits):
                if w:
                    time.sleep(w)
                try:
                    with os.scandir(d) as it:
                        names = {e.name for e in it}
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    continue
            else:
                raise OSError(f"cannot list {d}")
            tmps = {}
            prev = {}
            for n in names:
                m = TMP_RE.match(n)
                if m:
                    tmps.setdefault(m.group("base"), []).append(n)
                m = re.match(r"^(.+)\.log\.prev(\d+)$", n)
                if m:
                    prev[m.group(1)] = max(prev.get(m.group(1), 0), int(m.group(2)))
            self._dirs[d] = (names, tmps, prev)
        return self._dirs[d]

    def has(self, path):
        return os.path.basename(path) in self.listing(os.path.dirname(path))[0]

    def _forget(self, path):
        self.listing(os.path.dirname(path))[0].discard(os.path.basename(path))

    def remove(self, path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        self._forget(path)

    # failure counts -------------------------------------------------------------------------
    def _state_file(self, stem):
        return os.path.join(self.state_dir, os.path.basename(stem) + ".json")

    def fails(self, stem):
        if self._fails is None:                        # one listing of .state, not a stat per fit
            try:
                self._fails = {n for n in os.listdir(self.state_dir) if n.endswith(".json")}
            except FileNotFoundError:
                self._fails = set()
        if os.path.basename(self._state_file(stem)) not in self._fails:
            return {"fits": 0, "spectra": 0}
        try:
            return json.load(open(self._state_file(stem)))
        except (OSError, ValueError):
            return {"fits": 0, "spectra": 0}

    def add_fail(self, stem, stage, note):
        os.makedirs(self.state_dir, exist_ok=True)
        s = self.fails(stem)
        s[stage] = s.get(stage, 0) + 1
        s.setdefault("history", []).append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {stage} failed: {note}")
        tmp = self._state_file(stem) + f".tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(s, fh, indent=1)
        os.replace(tmp, self._state_file(stem))
        if self._fails is not None:
            self._fails.add(os.path.basename(self._state_file(stem)))

    # state ----------------------------------------------------------------------------------
    def stale_tmps(self, stem):
        return list(self.listing(os.path.dirname(stem))[1].get(os.path.basename(stem), []))

    def clean_stale(self, stem):
        """Temporary files of killed processes (only with --clean: nothing of this element runs)."""
        gone = self.stale_tmps(stem)
        d = os.path.dirname(stem)
        for n in gone:
            self.remove(os.path.join(d, n))
        self.listing(d)[1].pop(os.path.basename(stem), None)
        return gone

    def classify(self, line, spectra=True, max_fails=3, clean=False):
        """Returns a Fit with state, stage ('fits' / 'spectra' / None) and notes. Read-only unless
        clean (stale temporary files removed)."""
        fit = Fit(line, self.stem(line))
        stem = fit.stem
        if clean:
            gone = self.clean_stale(stem)
            if gone:
                fit.notes.append(f"removed stale {', '.join(gone)}")
        js_path = stem + ".json"
        kind, detail, js = json_info(js_path, self.waits) if self.has(js_path) else ("absent", "", None)
        if kind == "bad_input":
            fit.state, fit.reason = "BAD_INPUT", detail
            return fit
        if kind != "ok":
            fit.state, fit.stage = "FIT", "fits"
            if detail:
                fit.notes.append(detail)
        else:
            h5 = stem + "_post.h5"
            ps, pdet = post_info(h5, js, self.waits) if self.has(h5) else (None, "")
            if pdet:
                fit.notes.append(pdet)
            if ps in ("post", "spectra"):
                if spectra and ps != "spectra":
                    fit.state, fit.stage = "SPEC", "spectra"
                else:
                    fit.state = "DONE"
                    return fit
            else:
                npz = stem + ".npz"
                fit.state = "POST" if (self.has(npz) and npz_ok(npz)) else "FIT"
                fit.stage = "fits"
        if self.fails(stem).get(fit.stage, 0) >= max_fails:
            fit.reason = fit.state
            fit.state = "GAVE_UP"
        return fit

    def prepare_refit(self, fit):
        """--clean: before a fit is rerun, its stale outputs go (and its log is kept as .log.prev<k>)."""
        stem = fit.stem
        removed = []
        if fit.state == "FIT":
            for suf in (".json", "_post.h5", ".npz", "_draws.npz"):
                if self.has(stem + suf):
                    self.remove(stem + suf)
                    removed.append(suf)
            if self.has(stem + ".log"):
                d = os.path.dirname(stem)
                prev = self.listing(d)[2]
                b = os.path.basename(stem)
                k = prev.get(b, 0) + 1
                os.replace(stem + ".log", f"{stem}.log.prev{k}")
                prev[b] = k
                self._forget(stem + ".log")
        elif fit.state == "POST" and self.has(stem + "_post.h5"):
            self.remove(stem + "_post.h5")
            removed.append("_post.h5")
        return removed


# ------------------------------------------------------------------------- after a pass
def _rc(path):
    """Exit code from an exit-code file; None if absent (the process was killed, not ended)."""
    try:
        return int(open(path).read().strip())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return -1                                      # there but unreadable: it did end


def _error_lines(log, n=6):
    try:
        lines = [l.rstrip() for l in open(log, errors="replace")]
    except OSError:
        return [f"(no log {os.path.basename(log)})"]
    hits = [l for l in lines if ERR_RE.search(l)]
    return (hits or lines)[-n:] or ["(empty log)"]


def record_failures(r, stage, lists, rc_dir=None, breaker_min=5, breaker_frac=0.5, out=print):
    """Judge one pass; returns EXIT_DONE or EXIT_BREAKER (then nothing is recorded)."""
    charged = []                                       # (stem, note, log, fresh)
    killed = ended = ended_fresh = bad_input = 0
    setup_fail, proc_ended = [], 0
    for lf in lists:
        if not os.path.exists(lf):
            continue
        entries = [l.strip() for l in open(lf) if l.strip()]
        if stage == "fits":
            d = rc_dir or os.path.dirname(os.path.abspath(lf))
            postrun = lf.endswith(".post")
            for l in entries:
                line = l.split()
                fit = r.classify(line, spectra=False, max_fails=10 ** 9)
                base = os.path.basename(fit.stem)
                rc = _rc(os.path.join(d, base + ".rc"))
                fresh = r.fails(fit.stem).get("fits", 0) == 0
                kind, detail, _ = json_info(fit.stem + ".json", r.waits)
                log = fit.stem + ("_postprocess.log" if postrun else ".log")
                if fit.state == "BAD_INPUT":
                    bad_input += 1
                    ended += 1
                    ended_fresh += fresh
                elif kind == "failed":                 # the fit ended and said why
                    ended += 1
                    ended_fresh += fresh
                    charged.append((fit.stem, f"status {detail}", log, fresh))
                elif fit.state in ("FIT", "POST"):
                    if rc is None:
                        killed += 1
                    else:
                        ended += 1
                        ended_fresh += fresh
                        what = "postprocessing rerun" if postrun else "fit"
                        charged.append((fit.stem, f"{what} exit {rc} without its output "
                                        f"(still {fit.state})", log, fresh))
                else:
                    ended += 1
                    ended_fresh += fresh
        else:
            rc = _rc(lf + ".rc")
            try:
                started = [l.strip() for l in open(lf + ".started") if l.strip()]
            except FileNotFoundError:
                started = []
            log = os.path.join(r.res, "spectra_logs",
                               f"{os.path.basename(os.path.dirname(os.path.abspath(lf)))}."
                               f"{os.path.basename(lf)}.log")
            if rc is not None:
                proc_ended += 1
                if rc != 0 and not started:
                    setup_fail.append((lf, rc, log))
            for i, path in enumerate(entries):
                stem = path[:-len("_post.h5")]
                if post_state(path) == "spectra":
                    if rc is not None and path in started:
                        ended += 1
                        ended_fresh += r.fails(stem).get("spectra", 0) == 0
                    continue
                if rc is None:
                    killed += 1
                elif path in started:
                    fresh = r.fails(stem).get("spectra", 0) == 0
                    ended += 1
                    ended_fresh += fresh
                    last = path == started[-1]
                    charged.append((stem, f"spectra process exit {rc}; "
                                    f"{'was working on it' if last else 'finished it'} without spectra",
                                    log, fresh))
                # never started: not charged
    n_fresh = sum(1 for c in charged if c[3])
    trip = n_fresh >= breaker_min and n_fresh >= breaker_frac * max(ended_fresh, 1)
    trip_setup = stage == "spectra" and proc_ended >= 1 and len(setup_fail) == proc_ended
    out(f"pipeline_status: {stage} pass: {ended} ended, {killed} killed before ending (not counted),"
        f" {len(charged)} failed" + (f", {bad_input} BAD_INPUT" if bad_input else ""))
    if trip or trip_setup:
        out("!! CIRCUIT BREAKER: " + (
            f"{n_fresh} of the {ended_fresh} {stage} that ended in this pass (and had not failed "
            f"before) failed (threshold: >= {breaker_min} and >= {breaker_frac:.0%})" if trip else
            f"every spectra process that ended ({proc_ended}) failed before starting any galaxy")
            + ". The node or the setup is at fault, not the galaxies: NO failures recorded.")
        shown = [(os.path.basename(c[0]), c[1], c[2]) for c in charged[:3]] if trip else \
                [(os.path.basename(lf), f"exit {rc}", log) for lf, rc, log in setup_fail[:3]]
        for name, note, log in shown:
            out(f"   {name}: {note}; {log}:")
            for l in _error_lines(log):
                out(f"      | {l}")
        return EXIT_BREAKER
    for stem, note, log, _ in charged:
        r.add_fail(stem, stage, note)
        out(f"  failed: {os.path.basename(stem)} ({note})")
    if bad_input and ended and bad_input >= max(breaker_min, breaker_frac * ended):
        out(f"!! {bad_input} of {ended} fits are BAD_INPUT: check that the task list names the right "
            f"catalogue (BAD_INPUT is final; delete those jsons to retry them)")
    out(f"pipeline_status: {len(charged)} {stage} failure(s) recorded")
    return EXIT_DONE


# ------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="nss_test/ (task list and code are under it)")
    p.add_argument("--run", required=True)
    p.add_argument("--tasklist", required=True)
    p.add_argument("--results-root", default=None,
                   help="where run folders are (default <root>/results); results in RESULTS_ROOT/RUN")
    p.add_argument("--first", type=int, default=None, help="first task-list line (1-based)")
    p.add_argument("--last", type=int, default=None, help="last task-list line (inclusive)")
    p.add_argument("--spectra", type=int, default=1, help="1: a fit needs spectra to be DONE")
    p.add_argument("--max-fails", type=int, default=3)
    p.add_argument("--write", default=None, help="prefix of the work lists to write")
    p.add_argument("--stage", choices=("fits", "spectra"), default="fits")
    p.add_argument("--nspec", type=int, default=10, help="spectra work lists (parallel processes)")
    p.add_argument("--clean", action="store_true",
                   help="job script only: remove stale temporary files and the stale outputs of "
                        "fits about to be redone (never while a job of this element runs)")
    p.add_argument("--record-failures", nargs="+", metavar=("STAGE", "LIST"), default=None)
    p.add_argument("--rc-dir", default=None, help="folder of the fits' exit-code files <stem>.rc")
    p.add_argument("--breaker-min", type=int, default=5)
    p.add_argument("--breaker-frac", type=float, default=0.5)
    p.add_argument("--list", action="store_true", help="print every fit's state")
    a = p.parse_args(argv)
    try:
        r = Run(a.root, a.run, a.tasklist, a.first, a.last, a.results_root)
    except TaskListError as e:
        print(f"pipeline_status: MALFORMED TASK LIST: {e}")
        return EXIT_BAD_TASKLIST

    if a.record_failures:
        stage, lists = a.record_failures[0], a.record_failures[1:]
        if stage not in ("fits", "spectra"):
            p.error("--record-failures STAGE must be fits or spectra")
        return record_failures(r, stage, lists, a.rc_dir, a.breaker_min, a.breaker_frac)

    keys = ("DONE", "FIT", "POST", "SPEC", "BAD_INPUT", "GAVE_UP")
    counts = {k: [] for k in keys}
    cleaned, damaged = [], []
    for line in r.lines:
        fit = r.classify(line, spectra=bool(a.spectra), max_fails=a.max_fails, clean=a.clean)
        counts[fit.state].append(fit)
        for n in fit.notes:
            (cleaned if n.startswith("removed") else damaged).append((fit, n))
        if a.list:
            extra = f"  [{fit.reason}]" if fit.reason else ""
            print(f"  {fit.state:9s} {os.path.basename(fit.stem)}{extra}"
                  + (f"  ({'; '.join(fit.notes)})" if fit.notes else ""))
    if cleaned:
        print(f"pipeline_status: removed stale temporary files of {len(cleaned)} fit(s): "
              + ", ".join(n[len('removed stale '):] for _, n in cleaned[:3]) + (" ..." if len(cleaned) > 3 else ""))
    print("pipeline_status: " + ", ".join(f"{k} {len(v)}" for k, v in counts.items())
          + f"  ({len(r.lines)} fits; {r.run}; {r.res})")
    serious = [(f, n) for f, n in damaged if "unreadable" in n or "damaged" in n or "differs" in n]
    for f, n in serious[:20]:
        print(f"  DAMAGED {os.path.basename(f.stem)}: {n}")
    if len(serious) > 20:
        print(f"  ... and {len(serious) - 20} more damaged files")
    for f in counts["BAD_INPUT"]:
        print(f"  BAD_INPUT {' '.join(f.line)}: {f.reason} (final; not retried)")
    for f in counts["GAVE_UP"]:
        hist = r.fails(f.stem).get("history", [])
        print(f"  GAVE UP {' '.join(f.line)}: {hist[-1] if hist else ''} "
              f"(see {os.path.basename(f.stem)}.log*)")

    if a.write:
        if a.stage == "fits":
            for k, key in (("fit", "FIT"), ("post", "POST")):
                with open(f"{a.write}.{k}", "w") as fh:
                    fh.writelines(" ".join(f.line) + "\n" for f in counts[key])
            if a.clean:
                n = sum(bool(r.prepare_refit(f)) for f in counts["FIT"] + counts["POST"])
                if n:
                    print(f"pipeline_status: removed the stale outputs of {n} fit(s) about to be redone")
        else:
            d, b = os.path.split(os.path.abspath(a.write))
            for f in os.listdir(d):
                if f.startswith(b + ".spec."):
                    os.remove(os.path.join(d, f))
            todo = [f.stem + "_post.h5" for f in counts["SPEC"]]
            k = min(a.nspec, len(todo))
            for j in range(k):
                with open(f"{a.write}.spec.{j}", "w") as fh:
                    fh.writelines(t + "\n" for t in todo[j::k])
    if counts["FIT"] or counts["POST"] or counts["SPEC"]:
        return EXIT_WORK_LEFT
    return EXIT_GAVE_UP if counts["GAVE_UP"] else EXIT_DONE


if __name__ == "__main__":
    sys.exit(main())
