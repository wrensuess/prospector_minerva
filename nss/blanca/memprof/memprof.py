"""Memory timeline of one nss_jax.py fit, for STATUS.md section 6 item 7 ("bring memory down").
The fit code is NOT changed: phases are marked by wrapping nss_jax functions at run time, and
stdout lines are time-stamped.

Recorded (every 20 ms, and at every phase mark):
  Linux:  /proc/self/status VmRSS, VmHWM (peak RSS, what /usr/bin/time reports), RssAnon
          (private memory: what the Slurm memory cgroup charges per process), RssFile (shared
          library and file pages, charged once per node, not per fit).
          glibc heap from mallinfo2 (mallinfo on glibc < 2.33), every 0.5 s and at marks:
          in use = uordblks + hblkhd; held free by malloc = fordblks; mmapped = hblkhd.
  macOS:  resident, phys_footprint and its lifetime peak (proc_pid_rusage); for local tests.

Variants (all optional):
  env PRIOR_BATCH=N   prior-start batch size N instead of nss_jax's 20000 (in memory only).
                      NB: this also changes the random prior draws (nss_jax folds the key per
                      batch), so the fit differs from the default by Monte Carlo noise only.
  glibc allocator settings (MALLOC_ARENA_MAX etc.) are set in the environment by the caller.

usage: python memprof.py --label LABEL --memout OUT.json -- <nss_jax.py arguments>
"""
import ctypes
import json
import os
import sys
import threading
import time

argv = sys.argv[1:]
cut = argv.index("--")
opts, nss_args = argv[:cut], argv[cut + 1:]
LABEL = opts[opts.index("--label") + 1]
MEMOUT = opts[opts.index("--memout") + 1]
NSS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, NSS)
LINUX = sys.platform.startswith("linux")
GB = 1e9

# ------------------------------------------------------------------ probes
if LINUX:
    _libc = ctypes.CDLL("libc.so.6")
    _FIELDS = ["arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks",
               "uordblks", "fordblks", "keepcost"]
    if hasattr(_libc, "mallinfo2"):
        class _MI(ctypes.Structure):
            _fields_ = [(f, ctypes.c_size_t) for f in _FIELDS]
        _libc.mallinfo2.restype = _MI
        _mallinfo, MALLINFO = _libc.mallinfo2, "mallinfo2"
    else:                                   # int fields: wrap at 4 GB, read as unsigned
        class _MI(ctypes.Structure):
            _fields_ = [(f, ctypes.c_uint) for f in _FIELDS]
        _libc.mallinfo.restype = _MI
        _mallinfo, MALLINFO = _libc.mallinfo, "mallinfo (32-bit fields)"

    def heap():
        m = _mallinfo()
        return {"heap_in_use": m.uordblks + m.hblkhd, "heap_free_held": m.fordblks,
                "heap_mmapped": m.hblkhd, "heap_arena": m.arena}

    def usage():
        out = {}
        with open("/proc/self/status") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                if k in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"):
                    out[k] = int(v.split()[0]) * 1024
        return out
else:
    MALLINFO = None
    _lp = ctypes.CDLL("/usr/lib/libproc.dylib")

    class _RU(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("v", ctypes.c_uint64 * 40)]

    def heap():
        return {}

    def usage():
        r = _RU()
        _lp.proc_pid_rusage(os.getpid(), 4, ctypes.byref(r))
        return {"VmRSS": r.v[6], "footprint": r.v[7], "footprint_peak": r.v[28]}

T0 = time.time()
samples, marks = [], []
peak_anon = [0, ""]
phase = ["start"]
stop = threading.Event()


def sampler():
    last_heap = 0.0
    while not stop.is_set():
        t = time.time() - T0
        u = usage()
        if t - last_heap > 0.5:
            u.update(heap())
            last_heap = t
        samples.append((round(t, 3), u))
        a = u.get("RssAnon", u.get("footprint", 0))
        if a > peak_anon[0]:
            peak_anon[:] = [a, phase[0]]
        stop.wait(0.02)


def fmt(u):
    if LINUX:
        s = (f"rss {u['VmRSS']/GB:5.2f} hwm {u['VmHWM']/GB:5.2f} anon {u['RssAnon']/GB:5.2f} "
             f"file {u['RssFile']/GB:5.2f}")
        if "heap_in_use" in u:
            s += f" | heap used {u['heap_in_use']/GB:5.2f} free-held {u['heap_free_held']/GB:5.2f}"
        return s
    return (f"rss {u['VmRSS']/GB:5.2f} | footprint {u['footprint']/GB:5.2f} "
            f"peak {u['footprint_peak']/GB:5.2f}")


def mark(name, set_phase=None):
    if set_phase:
        phase[0] = set_phase
    t = time.time() - T0
    u = usage()
    u.update(heap())
    marks.append((round(t, 3), name, u, phase[0]))
    sys.__stdout__.write(f"[mem {t:7.1f}s {fmt(u)} GB] {name}\n")
    sys.__stdout__.flush()


class Tee:
    def write(self, s):
        for line in s.splitlines():
            if line.strip():
                ln = line.strip()
                mark(ln[:120], "NSS loop" if ln.startswith("it ") else None)

    def flush(self):
        pass


threading.Thread(target=sampler, daemon=True).start()
mark(f"start {LABEL}; env " + " ".join(f"{k}={os.environ[k]}" for k in sorted(os.environ)
                                       if k.startswith(("MALLOC_", "NSS_", "PRIOR_BATCH", "XLA_FLAGS",
                                                        "LD_PRELOAD"))) + f"; {MALLINFO}", "imports")
import numpy as np                     # noqa: E402
import jax                             # noqa: E402
import nss_jax                         # noqa: E402
import mass_marginal                   # noqa: E402
mark("imported jax, blackjax, efsps, nss code", "setup")



def set_default(f, name, value):
    names = f.__code__.co_varnames[:f.__code__.co_argcount]
    d = list(f.__defaults__)
    d[names.index(name) - (len(names) - len(d))] = value
    f.__defaults__ = tuple(d)


# one log line (= one memory mark) per NSS iteration; printing only, no effect on the fit
set_default(nss_jax.run_nss_jax, "log_every", 1)
if os.environ.get("PRIOR_BATCH"):
    set_default(nss_jax.run_nss_jax, "prior_batch", int(os.environ["PRIOR_BATCH"]))
    mark(f"prior_batch default set to {int(os.environ['PRIOR_BATCH'])}")


def wrap(mod, name, ph_in, ph_out):
    f = getattr(mod, name)

    def g(*a, **k):
        mark(f"enter {name}", ph_in)
        r = f(*a, **k)
        mark(f"exit  {name}", ph_out)
        return r
    setattr(mod, name, g)


wrap(nss_jax, "build", "setup", "setup")
wrap(nss_jax, "prior_start", "prior start", "compile NSS step, map start points")
wrap(nss_jax, "finalise", "finalise", "finalise")
wrap(nss_jax, "log_weights", "log_weights", "mass draws, theta")
wrap(nss_jax, "summarise", "summarise", "write")
_init = mass_marginal.MassMarginal.__init__


def _mm_init(self, *a, **k):
    mark("enter MassMarginal.__init__", "MassMarginal init")
    _init(self, *a, **k)
    mark("exit  MassMarginal.__init__", "compile prior-start batch")


mass_marginal.MassMarginal.__init__ = _mm_init

status = "ok"
sys.stdout = Tee()
sys.argv = ["nss_jax.py"] + nss_args
try:
    nss_jax.main()
except BaseException as e:                  # record and re-raise after writing the timeline
    status = f"failed: {e!r}"
    raise
finally:
    sys.stdout = sys.__stdout__
    stop.set()
    time.sleep(0.05)
    mark("done", "done")
    # peak private memory and RSS in each phase (a phase runs from its mark to the next)
    phase_peaks, mi, cur = {}, 0, "start"
    for t, u in samples:
        while mi < len(marks) and marks[mi][0] <= t:
            cur = marks[mi][3] if len(marks[mi]) > 3 else cur
            mi += 1
        p = phase_peaks.setdefault(cur, {"private": 0, "rss": 0})
        p["private"] = max(p["private"], u.get("RssAnon", u.get("footprint", 0)))
        p["rss"] = max(p["rss"], u.get("VmRSS", 0))
    final = usage()
    final.update(heap())
    out = {"label": LABEL, "status": status, "platform": sys.platform, "mallinfo": MALLINFO,
           "env": {k: os.environ[k] for k in os.environ
                   if k.startswith(("MALLOC_", "NSS_", "PRIOR_BATCH", "XLA_FLAGS", "LD_PRELOAD"))},
           "nss_args": nss_args, "peak_private": peak_anon[0], "peak_private_phase": peak_anon[1],
           "final": final, "marks": marks,
           "phase_peaks": phase_peaks,
           "samples": samples[::5]}            # every 0.1 s
    with open(MEMOUT, "w") as fh:
        json.dump(out, fh)
    sys.__stdout__.write(f"[mem] peak private {peak_anon[0]/GB:.3f} GB during '{peak_anon[1]}'; "
                         f"wrote {MEMOUT}\n")
