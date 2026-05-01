#!/usr/bin/env bash
# Launch the Streamlit reviewer for a given annotation package.
#
# Usage:
#   bash scripts/run_reviewer.sh data/annotation_packages/match_2026_04_15
#   bash scripts/run_reviewer.sh data/annotation_packages/match --port 8502

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <package_dir> [--port PORT]"
    exit 1
fi

PACKAGE_DIR="$1"
shift

PORT=8501
while [ "$#" -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

streamlit run "${ROOT_DIR}/src/track_annotation/reviewer/app.py" \
    --server.port "${PORT}" \
    --server.headless true \
    -- --package "${PACKAGE_DIR}"
