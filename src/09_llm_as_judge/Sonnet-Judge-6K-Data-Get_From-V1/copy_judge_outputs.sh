#!/usr/bin/env bash
# Copies Sonnet-4.6 judge results (6K data) from v1.0 into this folder,
# one file per judged model.
set -euo pipefail

SRC_ROOT="/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/06_llm_as_judge"
DST_DIR="/home/malam/projects/benchmarks/ai-safety-sept26/src/09_llm_as_judge/Sonnet-Judge-6K-Data-Get_From-V1"

# model_key:output_filename
MAPPINGS=(
  "aya:aya_judge_output.jsonl"
  "gemini:gemini_judge_output.jsonl"
  "gpt-5.4-mini:gpt-5.4-mini_judge_output.jsonl"
  "gpt-oss:gpt-oss_judge_output.jsonl"
  "haiku:haiku_judge_output.jsonl"
  "mistral:mistral_judge_output.jsonl"
)

mkdir -p "$DST_DIR"

for mapping in "${MAPPINGS[@]}"; do
  model_key="${mapping%%:*}"
  dst_name="${mapping##*:}"
  src_file="$SRC_ROOT/$model_key/merged/anthropic/claude_sonnet_4_6/output.jsonl"
  dst_file="$DST_DIR/$dst_name"

  if [ ! -f "$src_file" ]; then
    echo "MISSING source for '$model_key': $src_file" >&2
    exit 1
  fi

  echo "Copying $model_key -> $dst_name"
  cp -- "$src_file" "$dst_file"
done

echo "Done. Files in $DST_DIR:"
ls -la "$DST_DIR"
