#!/usr/bin/env bash
# Builds the backtracking lookups, enriches every merged/<judge>/<model>.jsonl
# with the 5 requested fields (lang_meta, eval, root_id, language, category),
# and prints a before/after coverage report. Does not modify merged/ --
# enriched files land in output/<judge>/<model>.jsonl.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Step 1: build lookups ==="
python3 build_lookups.py

echo
echo "=== Step 2: enrich merged files ==="
python3 enrich.py

echo
echo "=== Step 3: before/after field-presence report ==="
python3 verify_report.py
