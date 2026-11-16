#!/bin/bash -l
# After task-list runs (tasklist_layout.sbatch) finish: count finished and failed fits, then build
# the collected HDF5 files and the FITS SPS catalogue (make_catalog_nss.py) for every run, field
# and seed. Submit with a dependency on the arrays, e.g.
#   sbatch --dependency=afterany:<job1>:<job2> --export=ALL,RUNS=specz1k_pp:blanca/pp_specz1k_tasks.txt+dust48_pp:blanca/pp_dust48_tasks.txt catalog_runs.sbatch
# RUNS: '+'-separated RUN:TASKLIST pairs (no spaces or commas: --export splits on commas) (RUN = the results/ folder tasklist_layout wrote to;
# TASKLIST relative to nss_test/). Each task list names its photometric catalogues; a catalogue's
# fits are picked by id prefix (UDS ids start with 1, COSMOS with 2), because make_catalog_nss.py
# needs one band set per catalogue. One catalogue per seed (seed s = the fits *_seed<s>_post.h5).
# Output: results/<RUN>/catalog/<catalogue stem>_seed<s>/ (quant/sfh/chains/spec h5 and
# sps_catalog/*.fits), and results/<RUN>/catalog/fit_report.txt (fits expected, finished, missing,
# and logs with errors). Safe to rerun: everything is rebuilt from the per-fit files.
#
#SBATCH --account=blanca-casa
#SBATCH --partition=blanca
#SBATCH --qos=preemptable
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=nss_catalog
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

NSS_ROOT=/projects/$USER/nss_blanca/nss_test
export EFSPS_PARENT=/projects/$USER/nss_blanca/efsps-photfit
NSS_ENV=/projects/$USER/software/anaconda/envs/nss
module purge
if [ ! -x "$NSS_ENV/bin/python" ]; then echo "no python in $NSS_ENV on $(hostname)"; exit 1; fi
export PATH=$NSS_ENV/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$NSS_ROOT" || exit 1
: "${RUNS:?set RUNS, e.g. --export=ALL,RUNS=\"specz1k_pp:blanca/pp_specz1k_tasks.txt\"}"
echo "===== $(date) on $(hostname), job $SLURM_JOB_ID, restart ${SLURM_RESTART_COUNT:-0}"

for PAIR in ${RUNS//+/ }; do
    RUN=${PAIR%%:*}; TASKLIST=${PAIR#*:}
    RES=$NSS_ROOT/results/$RUN
    [ -d "$RES" ] || { echo "!! $RUN: no $RES"; continue; }
    [ -f "$TASKLIST" ] || { echo "!! $RUN: no task list $TASKLIST"; continue; }
    mkdir -p "$RES/catalog"
    REPORT=$RES/catalog/fit_report.txt
    # ---- which fits finished (json written last by nss_jax.py), which have post.h5, which failed
    python - "$TASKLIST" "$RES" > "$REPORT" <<'EOF'
import os, sys, json
tl, res = sys.argv[1:3]
exp, done, nopost, missing, pp_err = 0, 0, [], [], []
for line in open(tl):
    cat, i, dust, seed = line.split()
    exp += 1
    stem = os.path.join(res, dust, f"{i}_{dust}_seed{seed}")
    if not os.path.exists(stem + ".json"):
        missing.append(f"{cat} {i} {dust} {seed}")
        continue
    done += 1
    post = stem + "_post.h5"                       # nss_jax.py: <out>/<id>_<tag>_post.h5
    if not os.path.exists(post):
        nopost.append(f"{cat} {i} {dust} {seed}")
    try:
        s = json.load(open(stem + ".json"))
        if isinstance(s.get("postprocess"), dict) and "error" in s["postprocess"]:
            pp_err.append(f"{i} seed{seed}: {s['postprocess']['error']}")
    except Exception as e:
        pp_err.append(f"{i} seed{seed}: unreadable json ({e})")
print(f"fits expected {exp}, finished {done}, missing {len(missing)}, finished without post.h5 {len(nopost)}, postprocessing errors {len(pp_err)}")
for title, lst in (("MISSING (rerun these task-list lines)", missing), ("NO POST.H5", nopost), ("POSTPROCESS ERRORS", pp_err)):
    if lst:
        print(f"\n{title}:"); print("\n".join(lst))
EOF
    echo "== $RUN: $(head -1 "$REPORT")  (details: $REPORT)"
    # ---- one catalogue per photometric catalogue and seed
    for CAT in $(awk '{print $1}' "$TASKLIST" | sort -u); do
        case "$CAT" in
            *uds*) PFX=1 ;;
            *cosmos*) PFX=2 ;;
            *) echo "!! $CAT: field unknown (no uds/cosmos in the name); skipped"; continue ;;
        esac
        for SEED in $(awk -v c="$CAT" '$1 == c {print $4}' "$TASKLIST" | sort -un); do
            OUT=$RES/catalog/$(basename "$CAT" .fits)_seed$SEED
            N=$(ls "$RES"/*/${PFX}*_seed${SEED}_post.h5 2>/dev/null | wc -l)
            if [ "$N" -eq 0 ]; then echo "   $CAT seed $SEED: no post.h5 files; skipped"; continue; fi
            rm -rf "$OUT"
            python -u make_catalog_nss.py --indir "$RES" --pattern "${PFX}*_seed${SEED}_post.h5" \
                --catalog "$CAT" --out "$OUT" --workers ${SLURM_CPUS_PER_TASK:-1} > "$OUT.log" 2>&1
            RC=$?
            echo "   $CAT seed $SEED: $N files -> $OUT (exit $RC; $(tail -1 "$OUT.log"))"
        done
    done
done
echo "===== $(date) done"
