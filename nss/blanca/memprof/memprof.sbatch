#!/bin/bash -l
# Memory diagnostic for NSS fits (STATUS.md section 6 item 7). Fits at the production defaults
# (nss_jax.py: k=32, 1e6 prior start, mass integrated out), one physical core each, all at once
# on ONE node, so speeds compare directly between variants of the same galaxy and seed.
# Each fit runs under memprof.py (memory timeline; fit code unchanged) and /usr/bin/time -v.
#
# Variants (VARIANTS, space-separated):
#   nofix   NSS_MALLOC_TUNING=0: the code before malloc_tools.py (the reference)
#   fix     the code as it is: malloc_trim after the compiles, large allocations mmapped
#           after the NSS loop (malloc_tools.py). Must be bitwise identical to nofix.
#   split1  fix + XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=1   (default 32)
#   avx2    fix + XLA_FLAGS=--xla_cpu_max_isa=AVX2   (no AVX-512 code; may change last bits)
#   vec256  fix + XLA_FLAGS=--xla_cpu_prefer_vector_width=256
#   noexp   fix + XLA_FLAGS=--xla_llvm_disable_expensive_passes=true   (~30% slower on the Mac)
#   nofuse  fix + XLA_FLAGS=--xla_cpu_use_fusion_emitters=false
#   b5000   fix + prior-start batches of 5000 rows (changes the prior draws)
#   glibc   fix + MALLOC_ARENA_MAX=2, MALLOC_MMAP_THRESHOLD_=1 MiB, MALLOC_TRIM_THRESHOLD_=32 MiB
#   (job 29475943 ran the first set, before malloc_tools.py: base / b5000 / glibc / both)
# IDS (default "1212906 1152437", from minerva_uds_test9.fits) and SEEDS (default "0 1").
# Tasks = |VARIANTS| x |IDS| x |SEEDS| + 1 (the loadbalance master): the default is 29. If
# you change the lists, pass a matching --ntasks to sbatch.
#
# Self-contained: NSS_ROOT is a separate copy of the code (memprof bundle), so nothing in
# /projects/$USER/nss_blanca/nss_test is touched.
# Submit from /projects/$USER/nss_blanca/memprof/nss_test/blanca/memprof (after
# `module load slurm/blanca` on the login node):  sbatch memprof.sbatch
#
#SBATCH --account=blanca-casa
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --constraint="epyc9354|gold5418y|5418y"
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=29
#SBATCH --hint=nomultithread
#SBATCH --mem=96G                  # generous on purpose: we want the unconstrained peaks
#SBATCH --job-name=nss_memprof
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

NSS_ROOT=/projects/$USER/nss_blanca/memprof/nss_test
export EFSPS_PARENT=/projects/$USER/nss_blanca/memprof/efsps-photfit
NSS_ENV=/projects/$USER/software/anaconda/envs/nss

module purge
module load loadbalance
if [ ! -x "$NSS_ENV/bin/python" ]; then echo "no python in $NSS_ENV on $(hostname)"; exit 1; fi
export PATH=$NSS_ENV/bin:$PATH
if [ -z "$CURC_LB_BIN" ]; then echo "no loadbalance module on $(hostname)"; exit 1; fi
export SLURM_EXPORT_ENV=ALL
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CATALOG=minerva_uds_test9.fits
VARIANTS=${VARIANTS:-nofix fix split1 avx2 vec256 noexp nofuse}
IDS=${IDS:-1212906 1152437}
SEEDS=${SEEDS:-0 1}
OUT=$NSS_ROOT/results/memprof_${SLURM_JOB_ID}
mkdir -p "$OUT"
TASKS=$OUT/taskfile.txt
echo "===== $(date) on $(hostname), job $SLURM_JOB_ID, restart ${SLURM_RESTART_COUNT:-0}"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2); node CPUs $(nproc --all); job allowed $(nproc); glibc $(ldd --version | head -1)"
echo "AVX-512: $(grep -m1 -o 'avx512f' /proc/cpuinfo || echo no); variants: $VARIANTS; ids: $IDS; seeds: $SEEDS; tasks $SLURM_NTASKS"
md5sum $NSS_ROOT/nss_jax.py $NSS_ROOT/mass_marginal.py $NSS_ROOT/malloc_tools.py $NSS_ROOT/blanca/memprof/memprof.py

cd "$NSS_ROOT" || exit 1
for id in $IDS; do
  for s in $SEEDS; do
    for v in $VARIANTS; do
      case $v in
        nofix)  ENVS="NSS_MALLOC_TUNING=0" ;;
        fix)    ENVS="" ;;
        split1) ENVS="XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=1" ;;
        avx2)   ENVS="XLA_FLAGS=--xla_cpu_max_isa=AVX2" ;;
        vec256) ENVS="XLA_FLAGS=--xla_cpu_prefer_vector_width=256" ;;
        noexp)  ENVS="XLA_FLAGS=--xla_llvm_disable_expensive_passes=true" ;;
        nofuse) ENVS="XLA_FLAGS=--xla_cpu_use_fusion_emitters=false" ;;
        b5000)  ENVS="PRIOR_BATCH=5000" ;;
        glibc)  ENVS="MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=33554432" ;;
        *) echo "unknown variant $v"; exit 1 ;;
      esac
      L=${v}_${id}_seed${s}
      echo "cd $NSS_ROOT && { python -u -c 'import os; a = sorted(os.sched_getaffinity(0)); c = {open(f\"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list\").read().strip() for i in a}; print(\"$L affinity\", a, \"=\", len(c), \"physical core(s)\", flush=True)'; env $ENVS /usr/bin/time -v python -u blanca/memprof/memprof.py --label $L --memout $OUT/$L.mem.json -- --catalog $CATALOG --id $id --tag $L --seed $s --out $OUT; } > $OUT/$L.log 2>&1"
    done
  done
done > "$TASKS"
n=$(wc -l < "$TASKS")
if [ $((n + 1)) -ne "$SLURM_NTASKS" ]; then echo "WARNING: $n fits but $SLURM_NTASKS tasks (want $((n + 1)))"; fi

$CURC_LB_BIN/mpirun lb "$TASKS"

echo "===== job memory cgroup (what Slurm enforces; all fits together)"
for f in $(find /sys/fs/cgroup -maxdepth 8 -path "*job_${SLURM_JOB_ID}*" \
             \( -name memory.max_usage_in_bytes -o -name memory.peak \) 2>/dev/null); do
  echo "$f: $(awk '{printf "%.2f GB", $1/1e9}' "$f")"
done
echo "===== summary"
python $NSS_ROOT/blanca/memprof/memprof_summary.py "$OUT"
echo "===== $(date) done"
