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
( cd "$NSS" && cp *.py uncover_test.csv "$STAGE/nss_test/" && cp -R blanca "$STAGE/nss_test/" \
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
find "$STAGE" \( -name '__pycache__' -o -name '.DS_Store' \) -prune -exec rm -rf {} +
tar -czf "$OUT" -C "$STAGE" nss_test efsps-photfit
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)); md5 $(md5 -q "$OUT" 2>/dev/null || md5sum "$OUT" | cut -d' ' -f1)"
