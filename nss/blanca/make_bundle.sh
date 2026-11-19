#!/bin/bash
# Build the one file to copy to Blanca: nss_blanca_bundle.tar.gz (~30 MB). It unpacks to
# nss_test/ (this nss/ folder, renamed to the layout the Blanca job scripts expect) and
# efsps-photfit/ (a clean clone of jrleja/efsps-photfit at the pinned commit).
#
#   EFSPS_PARENT=~/path/to/efsps-photfit bash nss/blanca/make_bundle.sh [OUT]
#                                                   -> OUT (default ~/nss_blanca_bundle.tar.gz)
# The private MINERVA test catalogues (minerva_*.fits subsets, e.g. minerva_uds_test9.fits,
# minerva_cosmos_bimodal12.fits; NOT in the public repo) are included if they sit in nss/
# (copy them there from the private handoff folder first).
set -euo pipefail
NSS=$(cd "$(dirname "$0")/.." && pwd)
EFSPS=${EFSPS_PARENT:?set EFSPS_PARENT to your efsps-photfit clone}
OUT=${1:-$HOME/nss_blanca_bundle.tar.gz}
PIN=6d5ab6d                                   # efsps-photfit commit the NSS code was validated with
if [ -d "$EFSPS/.git" ] && [ "$(git -C "$EFSPS" rev-parse --short=7 HEAD)" != "$PIN" ]; then
    echo "WARNING: $EFSPS is at $(git -C "$EFSPS" rev-parse --short=7 HEAD), not the validated $PIN" >&2
fi
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/nss_test/results"
( cd "$NSS" && cp *.py uncover_test.csv restframe_filters_*.npz surviving_fraction_*.npz "$STAGE/nss_test/" && cp -R blanca "$STAGE/nss_test/" \
  && cp results/58858_free_L1000.npz results/58858_free_L1000.json \
        results/58858_jax_free_L1000.npz results/58858_jax_free_L1000.json "$STAGE/nss_test/results/" )
shopt -s nullglob
subsets=("$NSS"/minerva_*.fits)
if [ ${#subsets[@]} -gt 0 ]; then
    cp "${subsets[@]}" "$STAGE/nss_test/"
    echo "MINERVA subsets in the bundle: $(cd "$NSS" && ls minerva_*.fits | tr '\n' ' ')" >&2
else
    echo "note: no minerva_*.fits in $NSS; the MINERVA test galaxies will not be in the bundle" >&2
fi
rsync -a --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
    "$EFSPS/" "$STAGE/efsps-photfit/"
# provenance: there is no .git on Blanca, so record the commits the bundle was built from.
# The first lines are for people; provenance.py reads the key=value lines. "dirty" uses
# provenance.py's rule: a tracked file differs from the commit, or an untracked code file
# (.py .sh .sbatch, which the bundle copies) is present. Modified files: comma-separated,
# relative to the repository root, as `git status --porcelain` prints them.
modified() {    # $1 = checkout; prints the files that make it dirty, comma-separated
    { git -C "$1" status --porcelain --untracked-files=all -- . 2>/dev/null || true; } \
        | awk '{ code = substr($0, 1, 2); name = substr($0, 4)
                 if (code != "??" || name ~ /\.(py|sh|sbatch)$/) print name }' | paste -sd, -
}
dirty() {       # $1 = checkout, $2 = its modified list -> true / false / unknown (no checkout)
    if ! git -C "$1" rev-parse HEAD >/dev/null 2>&1; then echo unknown
    elif [ -n "$2" ]; then echo true; else echo false; fi
}
NSS_MOD=$(modified "$NSS")
EFSPS_MOD=$(modified "$EFSPS")
{
    echo "built $(date '+%Y-%m-%d %H:%M %Z')"
    echo "nss: $(git -C "$NSS" rev-parse HEAD 2>/dev/null || echo unknown)" \
         "$( [ -n "$NSS_MOD" ] && echo '(with uncommitted changes)')"
    echo "efsps-photfit: $(git -C "$EFSPS" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "nss_commit=$(git -C "$NSS" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "nss_branch=$(git -C "$NSS" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "nss_dirty=$(dirty "$NSS" "$NSS_MOD")"
    echo "nss_modified=$NSS_MOD"
    echo "efsps_commit=$(git -C "$EFSPS" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "efsps_dirty=$(dirty "$EFSPS" "$EFSPS_MOD")"
    echo "efsps_modified=$EFSPS_MOD"
    echo "efsps_pinned=$PIN"
    echo "created_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "builder_host=$(hostname)"
} > "$STAGE/nss_test/BUNDLE_INFO.txt"
find "$STAGE" \( -name '__pycache__' -o -name '.DS_Store' \) -prune -exec rm -rf {} +
tar -czf "$OUT" -C "$STAGE" nss_test efsps-photfit
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)); md5 $(md5 -q "$OUT" 2>/dev/null || md5sum "$OUT" | cut -d' ' -f1)"
