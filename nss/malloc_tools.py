"""Give freed memory back to the system during an NSS fit (glibc only). These calls change
how memory is allocated, never what is computed: fit outputs are bitwise identical with or
without them (Blanca memprof jobs, STATUS.md section 6 item 7).

Why (Blanca job 29475943): the data a fit holds is only 0.15-0.6 GB, but the XLA compiles
use ~0.85 GB for ~30 s, and glibc keeps that memory after it is freed (~0.8 GB held free
for the rest of the fit). At the end of the run, the large arrays (log_weights, mass draws,
theta) then grow the heap further, to a per-fit peak of ~1.85 GB.
- release_free_memory(): malloc_trim(0), which returns the free pages inside the heap.
- mmap_large_allocations(): from then on, allocations >= 1 MiB get their own mmap, which is
  returned when freed. Used only after the NSS loop: making it permanent (the env variable
  MALLOC_MMAP_THRESHOLD_) slowed fits by ~5%.

Off on non-glibc systems (e.g. macOS) and when the environment sets NSS_MALLOC_TUNING=0.
"""
import ctypes
import os
import sys

M_MMAP_THRESHOLD = -3          # glibc mallopt parameter number


def _load():
    if not sys.platform.startswith("linux") or os.environ.get("NSS_MALLOC_TUNING", "1") == "0":
        return None
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim, libc.mallopt          # noqa: B018 (both must exist)
    except (OSError, AttributeError):
        return None
    return libc


_libc = _load()
ENABLED = _libc is not None


def release_free_memory():
    """Return the heap's free pages to the system (malloc_trim(0)); no-op when disabled."""
    if _libc is not None:
        _libc.malloc_trim(0)


def mmap_large_allocations(threshold=1 << 20):
    """Serve allocations >= threshold bytes by mmap from now on; no-op when disabled."""
    if _libc is not None:
        _libc.mallopt(M_MMAP_THRESHOLD, int(threshold))
