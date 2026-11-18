#!/bin/bash
DIR=/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/05_responses
OUT=/home/malam/projects/benchmarks/ai-safety-sept26/src/09_llm_as_judge/Mistral-Judge-6K-Data
CUDA_VISIBLE_DEVICES=0,1 PATH="/home/malam/miniconda3/envs/vllm-serve/bin:$PATH" nohup /home/malam/miniconda3/envs/vllm-serve/bin/python vllm_judge.py \
  --input-file "$DIR/open/aya/merged/aya_expanse:32b_merged.jsonl" \
  --output-file "$OUT/mistral_large_2411_awq_judge_aya_expanse_32b.jsonl" \
  --response-type open \
  --input-file "$DIR/open/gpt/merged/gpt_oss:120b_merged.jsonl" \
  --output-file "$OUT/mistral_large_2411_awq_judge_gpt_oss_120b.jsonl" \
  --response-type open \
  --input-file "$DIR/open/mistral/merged/mistral_large:123b_merged.jsonl" \
  --output-file "$OUT/mistral_large_2411_awq_judge_mistral_large_123b.jsonl" \
  --response-type open \
  --input-file "$DIR/closed/closed_responses/anthropic/claude_haiku_4_5.jsonl" \
  --output-file "$OUT/mistral_large_2411_awq_judge_claude_haiku_4_5.jsonl" \
  --response-type closed_flat \
  --input-file "$DIR/closed/closed_responses/gemini/gemini_2_5_flash.jsonl" \
  --output-file "$OUT/mistral_large_2411_awq_judge_gemini_2_5_flash.jsonl" \
  --response-type closed_flat \
  --input-file "$DIR/closed/closed_responses/openai/gpt_5_4_mini.jsonl" \
  --output-file "$OUT/mistral_large_2411_awq_judge_gpt_5_4_mini.jsonl" \
  --response-type closed_flat \
  --judge-model TechxGenus/Mistral-Large-Instruct-2411-AWQ \
  --quantization awq_marlin \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 8192 \
  --max-new-tokens 384 \
  --batch-size 2000 \
  --resume \
  > "$OUT/mistral_large_2411_awq_judge_6k_all.log" 2>&1 &
echo "PID: $!"
