"""One line per process: does this fit really run on one core?

Prints host, MPI rank, the CPUs this process may run on (its affinity, which XLA sizes
its thread pool from), the process's OS threads after a jitted workload, and that
workload's CPU time / wall time. Run under the production layout by check_binding.sbatch.
"""
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def work(x, W):
    for _ in range(5):
        x = jnp.tanh(x @ W)
    return jnp.sum(jnp.exp(-x[:, :, None] * jnp.linspace(0.0, 1.0, 551)), axis=-1)


def threads():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("Threads:"):
                return int(line.split()[1])


x = jnp.ones((100, 256)) * 0.01
W = jnp.eye(256) * 0.9
work(x, W).block_until_ready()
t, c = time.perf_counter(), time.process_time()
n = 0
while time.perf_counter() - t < 10.0:
    work(x, W).block_until_ready()
    n += 1
wall, cpu = time.perf_counter() - t, time.process_time() - c
rank = next((os.environ[k] for k in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID")
             if k in os.environ), "?")
aff = sorted(os.sched_getaffinity(0))
ncore = len({open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
             for c in aff})
print(f"host {os.uname().nodename} rank {rank:>3s} pid {os.getpid()} jax {jax.__version__} "
      f"affinity {len(aff)} CPU(s) on {ncore} physical core(s) {aff if len(aff) <= 8 else str(aff[:8]) + '...'} "
      f"OS threads {threads()} CPU/wall {cpu / wall:.2f} {1e3 * wall / n:.2f} ms/call "
      f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}", flush=True)
