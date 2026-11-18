#!/usr/bin/env bash
# Merge 3K+6K judge outputs per judge/model, reconcile the merged output's
# schema (add_missing_fields), then aggregate the 3 judges (Mistral, Qwen,
# Sonnet) into one consensus verdict per model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR/../09_llm_as_judge"
MERGED_DIR="$SCRIPT_DIR/merged"
RECONCILED_DIR="$SCRIPT_DIR/add_missing_fields/output"
OUT_DIR="$SCRIPT_DIR/aggregated"

MODELS=(
  aya_expanse_32b
  claude_haiku_4_5
  gemini_2_5_flash
  gpt_5_4_mini
  gpt_oss_120b
  mistral_large_123b
)

echo "=== Step 1: merging 3K + 6K per judge/model ==="
python3 "$SCRIPT_DIR/merge_3k_6k.py" \
  --base-dir "$BASE_DIR" \
  --out-dir "$MERGED_DIR"

echo
echo "=== Step 2: reconciling missing/inconsistent fields (root_id, language, category, eval, lang_meta) ==="
python3 "$SCRIPT_DIR/add_missing_fields/build_lookups.py"
python3 "$SCRIPT_DIR/add_missing_fields/enrich.py"

echo
echo "=== Step 3: aggregating Mistral + Qwen + Sonnet per model ==="
mkdir -p "$OUT_DIR"
for model in "${MODELS[@]}"; do
  echo "--- $model ---"
  python3 "$SCRIPT_DIR/aggregate_judge_output.py" \
    --mistral "$RECONCILED_DIR/mistral/${model}.jsonl" \
    --qwen "$RECONCILED_DIR/qwen/${model}.jsonl" \
    --sonnet "$RECONCILED_DIR/sonnet/${model}.jsonl" \
    --output "$OUT_DIR/${model}_aggregated.jsonl"
done

echo
echo "Done. Aggregated files in $OUT_DIR"
