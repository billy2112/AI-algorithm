#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

echo "== Sanity checks =="
python3 scripts/run_toy_sanity.py

echo "== HSAM digits supervised + continual =="
python3 scripts/run_hsam_digits.py

echo "== HSAM ablations =="
python3 scripts/run_hsam_ablations.py

echo "== HSAM stream self-improvement probe =="
python3 scripts/run_hsam_stream_self_improve.py

echo "== Aggregate tables/plots =="
python3 scripts/aggregate_results.py

echo "Done. See research/03_experiments/* and research/04_results/*."
echo "Note: baselines_digits and char_csm have saved artifacts; rerun with:"
echo "  python3 scripts/run_baselines_digits.py"
echo "  python3 scripts/run_char_csm.py"
