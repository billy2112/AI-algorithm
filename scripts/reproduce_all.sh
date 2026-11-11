#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

# May take several minutes (torch baselines/continual training). CPU-only.
python3 scripts/run_baselines_digits.py
python3 scripts/run_char_csm.py
bash scripts/reproduce_headline.sh
