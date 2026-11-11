#!/bin/bash -l
# 20 complete NSS fits of ONE galaxy at once, in the production layout (21 tasks per node,
# CURC loadbalance, preemptable), one fit per lb rank, seeds 0-19 (free redshift). Measures:
#   - the real per-fit wall time with 19 neighbours on the node (memory bandwidth, caches);
#   - NSS run-to-run scatter: logZ and posterior across 20 seeds of the same galaxy.
#
# Settings (all optional, via --export=ALL,NAME=value,...):
#   CATALOG     catalogue path, relative to nss_test/ or absolute   (default uncover_test.csv)
#               UNCOVER or MINERVA FITS, or efsps CSV (photometry.read_photometry)
#   ID          object id                                           (default 58858)
#   INNER       NSS slice steps per new point (num_inner_steps)      (default 32, chosen 2026-11-11)
#   NLIVE       live points                                         (default 1000)
#   MASS_PRIOR  exact or tabulated (tabulated_mass_prior.py)        (default tabulated)
#   NPRIOR      prior start: begin from the best NLIVE of this many prior draws
#               (nss_jax.py --n-prior-start)                        (default 1e6; 0 = standard start)
#   SEEDS       seeds to run, as "first last"                       (default "0 19")
#   MASSCOORD   marginal (mass integrated out of the likelihood, mass_marginal.py; needs
#               MASS_PRIOR=tabulated) or cdf (logmass through its prior inverse CDF, the
#               earlier standard) (nss_jax.py --mass-coord)          (default marginal)
#   e.g. sbatch --export=ALL,CATALOG=minerva_uds_test9.fits,ID=1212906 fullfit_layout.sbatch   (production defaults)
#
# Core binding (measured on Blanca by the prospector-minerva session, 2026-10-08, OpenMPI
# 4.1.1): on nodes without hyperthreading, mpirun binds each lb worker to its own core. On
# hyperthreaded nodes (e.g. Xeon Gold 6130), 21 tasks get 11 physical cores, mpirun does
# not bind, and 20 fits share them (~2x slower); `--bind-to core` fails there.
# --hint=nomultithread (below) gives 21 physical cores, and each worker is bound to one.
# Each fit's log starts with its affinity, so check that every fit got one physical core.
# BIND (default empty) passes extra mpirun options, for experiments only.
#
#SBATCH --account=blanca-casa
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=21
#SBATCH --hint=nomultithread
#SBATCH --mem=48G                  # 20 fits x up to 2.3 GB peak RSS each (measured 2026-11-11); 30G ran out
#SBATCH --job-name=nss_fullfit_layout
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ---- paths (see README.md)
NSS_ROOT=/projects/$USER/nss_blanca/nss_test
export EFSPS_PARENT=/projects/$USER/nss_blanca/efsps-photfit
NSS_ENV=/projects/$USER/software/anaconda/envs/nss

module purge   # (slurm/blanca is loaded on the LOGIN node before sbatch; inside a job it only gives an Lmod error)
module load loadbalance
# the env's own python, not `module load anaconda` + `source activate`: on 2026-11-09 node
# blanca-g6-u6 had no anaconda module and every task of a job failed within a second there
if [ ! -x "$NSS_ENV/bin/python" ]; then echo "no python in $NSS_ENV on $(hostname)"; exit 1; fi
export PATH=$NSS_ENV/bin:$PATH
if [ -z "$CURC_LB_BIN" ]; then echo "no loadbalance module on $(hostname)"; exit 1; fi
export SLURM_EXPORT_ENV=ALL
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CATALOG=${CATALOG:-uncover_test.csv}
ID=${ID:-58858}
INNER=${INNER:-32}
NLIVE=${NLIVE:-1000}
MASS_PRIOR=${MASS_PRIOR:-tabulated}
NPRIOR=${NPRIOR:-1e6}
SEEDS=${SEEDS:-0 19}
MASSCOORD=${MASSCOORD:-marginal}
if [ "$NPRIOR" = "0" ]; then PS_TAG=""; else PS_TAG=_ps${NPRIOR}; fi
if [ "$MASSCOORD" = "cdf" ]; then MC_TAG=""; else MC_TAG=_mm; fi
PS_OPT="--n-prior-start $NPRIOR"
OUT=$NSS_ROOT/results/blanca_layout_${SLURM_JOB_ID}_${ID}_k${INNER}_L${NLIVE}_${MASS_PRIOR}${PS_TAG}${MC_TAG}
mkdir -p "$OUT"
TASKS=$OUT/taskfile.txt
echo "===== $(date) on $(hostname), job $SLURM_JOB_ID, restart ${SLURM_RESTART_COUNT:-0}, BIND='${BIND:-}'"
echo "galaxy $ID from $CATALOG; NLIVE=$NLIVE, INNER=$INNER, MASS_PRIOR=$MASS_PRIOR, NPRIOR=$NPRIOR, MASSCOORD=$MASSCOORD, seeds $SEEDS"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2); node CPUs $(nproc --all); job allowed $(nproc)"

cd "$NSS_ROOT" || exit 1
for s in $(seq $SEEDS); do
    echo "cd $NSS_ROOT && { python -u -c 'import os; a = sorted(os.sched_getaffinity(0)); c = {open(f\"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list\").read().strip() for i in a}; print(\"seed $s affinity\", a, \"=\", len(c), \"physical core(s)\", flush=True)'; /usr/bin/time -v python -u nss_jax.py --catalog $CATALOG --id $ID --tag seed$s --seed $s --n-live $NLIVE --num-delete 100 --num-inner-steps $INNER --mass-prior $MASS_PRIOR $PS_OPT --mass-coord $MASSCOORD --out $OUT; } > $OUT/seed$s.log 2>&1"
done > "$TASKS"

$CURC_LB_BIN/mpirun ${BIND:-} lb "$TASKS"

echo "===== per-fit results"
python - "$OUT" <<'EOF'
import glob, json, os, re, sys
import numpy as np
out = sys.argv[1]
rows = []
for fn in sorted(glob.glob(os.path.join(out, "*_seed*.json"))):
    s = json.load(open(fn))
    log = open(os.path.join(out, s["tag"] + ".log")).read()
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
    cpu = re.search(r"Percent of CPU this job got: (\d+)%", log)
    rows.append((s["tag"], s["logZ"], s["logZ_err"], s["percentiles"]["zred"][1],
                 s["percentiles"]["logmass"][1], s["wall_s"], s.get("compile_s"),
                 int(rss.group(1)) / 1e6 if rss else None, cpu.group(1) if cpu else "?",
                 s.get("z_at_max_logL", float("nan")), s["iterations"],
                 s.get("prior_start", {}).get("eval_s", 0.0)))
print(f"{'fit':8s} {'logZ':>8s} {'err':>5s} {'z50':>7s} {'z@maxL':>7s} {'logM50':>7s} "
      f"{'iters':>5s} {'wall s':>7s} {'prior s':>7s} {'compile':>7s} {'RSS GB':>6s} {'CPU%':>5s}")
for r in rows:
    print(f"{r[0]:8s} {r[1]:8.2f} {r[2]:5.2f} {r[3]:7.3f} {r[9]:7.3f} {r[4]:7.3f} {r[10]:5d} "
          f"{r[5]:7.0f} {r[11]:7.1f} {r[6]:7.1f} {r[7] if r[7] is None else round(r[7], 2)!s:>6s} "
          f"{r[8]:>5s}")
if rows:
    lz = np.array([r[1] for r in rows]); w = np.array([r[5] for r in rows])
    print(f"\n{len(rows)} fits: logZ mean {lz.mean():.2f}, scatter {lz.std(ddof=1):.2f} "
          f"(mean quoted err {np.mean([r[2] for r in rows]):.2f}); wall median {np.median(w):.0f} s, "
          f"range {w.min():.0f}-{w.max():.0f} s")
EOF
echo "===== $(date) done"
