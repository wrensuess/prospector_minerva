"""Provenance of a fit or product: which code, libraries, machine and inputs made it; and
the safe (atomic, fsynced, NaN-free) JSON writing the fit outputs use.

Code version, in this order:
  1. git, when this folder is a checkout of the repo (nss_jax.py tracked there);
  2. BUNDLE_INFO.txt next to this file, when the code runs from the Blanca bundle
     (blanca/make_bundle.sh writes it: the bundle unpacks to nss_test/ + efsps-photfit/,
     with no .git, and records the commits it was built from as key=value lines);
  3. "unknown" only if neither exists.
efsps-photfit's commit the same way: git if its folder ($EFSPS_PARENT) is a checkout,
else the bundle's record.

"dirty" means: a tracked file under nss/ differs from the commit, or an untracked code file
(.py, .sh, .sbatch) sits there (the bundle copies those, so they run). make_bundle.sh uses
the same rule, so a fit from the bundle and one from the checkout it was built from agree.

    from provenance import provenance, code_version, write_json_atomic
    summary["provenance"] = provenance(catalog=path, emulator={"file": ..., "md5": ...})
    write_json_atomic(path, summary)        # tmp + fsync + os.replace, NaN/inf -> null
"""
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_INFO = os.path.join(HERE, "BUNDLE_INFO.txt")
EFSPS_DIR = os.environ.get("EFSPS_PARENT", "/Users/wren/Projects/MINERVA/efsps-photfit")
LIBRARIES = ("jax", "jaxlib", "blackjax", "numpy", "scipy", "h5py", "astropy")
SLURM_VARS = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT")
CODE_SUFFIXES = (".py", ".sh", ".sbatch")

_CACHE = {}


# ------------------------------------------------------------------ code version
def _git(path, *cmd):
    """stdout of `git -C path cmd`, or None if git fails (not a checkout, no git)."""
    try:
        r = subprocess.run(["git", "-C", path, *cmd], capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    return r.stdout if r.returncode == 0 else None


def _modified(path, rel="."):
    """Files that make the checkout 'dirty' (see the module doc), as git prints them
    (relative to the repo root)."""
    out = _git(path, "status", "--porcelain", "--untracked-files=all", "--", rel)
    if out is None:
        return None
    files = []
    for line in out.splitlines():
        code, name = line[:2], line[3:]
        if code == "??" and not name.endswith(CODE_SUFFIXES):
            continue
        files.append(name)
    return files


def _git_version(path, tracked_file):
    """Commit, branch, dirty flag and modified files of the checkout holding `path`; None
    unless `tracked_file` (relative to path) is tracked there, so code unpacked inside
    some unrelated repository is not credited to that repository's commit."""
    if _git(path, "ls-files", "--error-unmatch", tracked_file) is None:
        return None
    head = _git(path, "rev-parse", "HEAD")
    if head is None:
        return None
    branch = (_git(path, "rev-parse", "--abbrev-ref", "HEAD") or "").strip()
    mod = _modified(path)
    return {"commit": head.strip(), "branch": branch, "dirty": bool(mod), "modified": mod or []}


def read_bundle_info(path=BUNDLE_INFO):
    """key=value lines of BUNDLE_INFO.txt (other lines are for people and are skipped);
    {} if there is no such file."""
    if not os.path.exists(path):
        return {}
    info = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def _as_bool(v):
    return {"true": True, "false": False}.get(str(v).strip().lower())


def code_version():
    """Version of the nss code and of efsps-photfit (cached per process). The keys "git" and
    "dirty" are the ones postprocess_nss.code_version() always returned."""
    if "code" in _CACHE:
        return dict(_CACHE["code"])
    bundle = read_bundle_info()
    g = _git_version(HERE, "nss_jax.py")
    if g is not None:
        out = {"git": g["commit"], "dirty": g["dirty"], "branch": g["branch"],
               "modified": g["modified"], "source": "git"}
    elif bundle.get("nss_commit"):
        mod = bundle.get("nss_modified", "")
        out = {"git": bundle["nss_commit"], "dirty": _as_bool(bundle.get("nss_dirty")),
               "branch": bundle.get("nss_branch", ""),
               "modified": [m for m in mod.split(",") if m], "source": "bundle",
               "bundle_created_utc": bundle.get("created_utc"),
               "bundle_builder_host": bundle.get("builder_host")}
    else:
        out = {"git": "unknown (no git checkout and no BUNDLE_INFO.txt)", "dirty": None,
               "branch": None, "modified": None, "source": "unknown"}
    e = _git_version(EFSPS_DIR, os.path.join("efsps_photfit", "__init__.py"))
    if e is not None:
        out.update(efsps_commit=e["commit"], efsps_dirty=e["dirty"], efsps_source="git")
    elif bundle.get("efsps_commit"):
        out.update(efsps_commit=bundle["efsps_commit"],
                   efsps_dirty=_as_bool(bundle.get("efsps_dirty")), efsps_source="bundle")
    else:
        out.update(efsps_commit="unknown", efsps_dirty=None, efsps_source="unknown")
    _CACHE["code"] = out
    return dict(out)


# ------------------------------------------------------------------ machine and libraries
def library_versions():
    if "versions" not in _CACHE:
        from importlib import metadata
        v = {"python": platform.python_version()}
        for lib in LIBRARIES:
            try:
                v[lib] = metadata.version(lib)
            except Exception:
                v[lib] = None
        _CACHE["versions"] = v
    return dict(_CACHE["versions"])


def cpu_model():
    if "cpu" not in _CACHE:
        name = None
        try:
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo") as fh:
                    for line in fh:
                        if line.startswith("model name"):
                            name = line.split(":", 1)[1].strip()
                            break
            else:
                r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                   capture_output=True, text=True, timeout=10)
                name = r.stdout.strip() or None
        except Exception:
            pass
        _CACHE["cpu"] = name or platform.processor() or "unknown"
    return _CACHE["cpu"]


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def provenance(catalog=None, emulator=None):
    """JSON-serialisable record of code, libraries, machine, job and inputs.
    catalog: the photometry file (path, size, mtime; NOT hashed: ~460 MB per field).
    emulator: {"file", "md5"} as the fit already records it (or {"path"} to hash a file)."""
    cv = code_version()
    p = {"nss_commit": cv["git"], "nss_dirty": cv["dirty"], "nss_branch": cv.get("branch"),
         "nss_modified": cv.get("modified"), "nss_source": cv["source"],
         "efsps_commit": cv["efsps_commit"], "efsps_dirty": cv["efsps_dirty"],
         "efsps_source": cv["efsps_source"]}
    if cv["source"] == "bundle":
        p["bundle_created_utc"] = cv.get("bundle_created_utc")
        p["bundle_builder_host"] = cv.get("bundle_builder_host")
    if emulator is not None:
        e = dict(emulator)
        if "md5" not in e and e.get("path") and os.path.exists(e["path"]):
            e["md5"] = file_md5(e["path"])
        if "path" in e:
            e.setdefault("file", os.path.basename(e.pop("path")))
        p["emulator_file"], p["emulator_md5"] = e.get("file"), e.get("md5")
    p["versions"] = library_versions()
    p["hostname"] = socket.gethostname()
    p["cpu_model"] = cpu_model()
    p["slurm"] = {k: os.environ[k] for k in SLURM_VARS if k in os.environ}
    p["created_utc"] = utc_now()
    if catalog is not None:
        c = {"path": os.path.abspath(catalog)}
        try:
            st = os.stat(catalog)
            c["size_bytes"] = int(st.st_size)
            c["mtime_utc"] = datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
        except OSError as e:
            c["error"] = repr(e)
        p["catalog"] = c
    p["env"] = {k: os.environ.get(k) for k in ("XLA_FLAGS", "NSS_EMU_DTYPE")}
    return p


# ------------------------------------------------------------------ safe writing
def finite_or_none(obj):
    """Copy of a JSON-like object with every non-finite float (NaN, +-inf; python or numpy)
    replaced by None, so the file holds no NaN/Infinity tokens (not valid JSON)."""
    if isinstance(obj, dict):
        return {k: finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_or_none(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def fsync_path(path):
    """Flush a written file to disk (before an os.replace makes it the real one)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_atomic(path, obj, indent=1):
    """Write obj as JSON to path atomically: tmp file, fsync, os.replace. Non-finite floats
    become null and allow_nan=False makes sure no NaN/Infinity token gets through."""
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            json.dump(finite_or_none(obj), fh, indent=indent, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
