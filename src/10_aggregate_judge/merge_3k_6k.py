"""
Merge the 3K-sample and 6K-sample judge outputs, per judge and per generator
model, into one combined jsonl per (judge, model).

Each judge folder (Mistral / Qwen / Sonnet) was run twice, once over a 3K
sample split and once over a 6K sample split, using different filenames per
judge. This script concatenates the two halves for the SAME judge and the
SAME generator model only -- it never mixes models or judges together.

Output layout:
    <out-dir>/mistral/<model_key>.jsonl
    <out-dir>/qwen/<model_key>.jsonl
    <out-dir>/sonnet/<model_key>.jsonl
"""

import argparse
import json
import os

MODEL_KEYS = [
    "aya_expanse_32b",
    "claude_haiku_4_5",
    "gemini_2_5_flash",
    "gpt_5_4_mini",
    "gpt_oss_120b",
    "mistral_large_123b",
]

# judge_name -> (subdir_3k, filename_pattern_3k, subdir_6k, filename_pattern_6k)
# filename patterns are keyed by model_key since naming isn't systematic
# across judges.
JUDGE_FILES = {
    "mistral": {
        "3k_dir": "Mistral-Judge-3K-Data",
        "6k_dir": "Mistral-Judge-6K-Data",
        "3k_name": lambda m: f"mistral_large_2411_awq_judge_{m}.jsonl",
        "6k_name": lambda m: f"mistral_large_2411_awq_judge_{m}.jsonl",
    },
    "qwen": {
        "3k_dir": "Qwen-Judge-3K-Data",
        "6k_dir": "Qwen-Judge-6K-Data",
        "3k_name": lambda m: f"qwen35_122b_gptq_judge_{m}.jsonl",
        "6k_name": lambda m: f"qwen35_122b_gptq_judge_v1_{m}.jsonl",
    },
    "sonnet": {
        "3k_dir": "Sonnet-Judge-3K-Data",
        "6k_dir": "Sonnet-Judge-6K-Data-Get_From-V1",
        "3k_name": lambda m: {
            "aya_expanse_32b": "aya_expanse_32b_10k_new_judged.jsonl",
            "claude_haiku_4_5": "claude_haiku_4_5_judged.jsonl",
            "gemini_2_5_flash": "gemini_2_5_flash_judged.jsonl",
            "gpt_5_4_mini": "gpt_5_4_mini_judged.jsonl",
            "gpt_oss_120b": "gpt-oss-120b_judged.jsonl",
            "mistral_large_123b": "mistral_large_123b_merged_judged.jsonl",
        }[m],
        "6k_name": lambda m: {
            "aya_expanse_32b": "aya_judge_output.jsonl",
            "claude_haiku_4_5": "haiku_judge_output.jsonl",
            "gemini_2_5_flash": "gemini_judge_output.jsonl",
            "gpt_5_4_mini": "gpt-5.4-mini_judge_output.jsonl",
            "gpt_oss_120b": "gpt-oss_judge_output.jsonl",
            "mistral_large_123b": "mistral_judge_output.jsonl",
        }[m],
    },
}


def load_jsonl(path):
    rows = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def match_key(row, judge_name):
    """(root_id_short, language) join key, robust to the schema differences
    between Mistral/Qwen (explicit fields) and Sonnet (packed custom_id)."""
    if judge_name == "sonnet":
        parts = str(row.get("custom_id", "")).split("__")
        if len(parts) < 3:
            return None
        root_short = parts[1].split("aegis_root_")[-1][:5]
        language = parts[2]
    else:
        root_id = row.get("root_id") or ""
        if not root_id:
            return None
        root_short = root_id.split("aegis_root_")[-1][:5]
        language = row.get("language_name")
    if not root_short or not language:
        return None
    return (root_short, language)


def merge_one(judge_name, model_key, base_dir, out_dir):
    cfg = JUDGE_FILES[judge_name]
    path_3k = os.path.join(base_dir, cfg["3k_dir"], cfg["3k_name"](model_key))
    path_6k = os.path.join(base_dir, cfg["6k_dir"], cfg["6k_name"](model_key))

    rows_3k = load_jsonl(path_3k)
    rows_6k = load_jsonl(path_6k)

    if not rows_3k:
        print(f"[WARN] {judge_name}/{model_key}: no rows from 3K file ({path_3k})")
    if not rows_6k:
        print(f"[WARN] {judge_name}/{model_key}: no rows from 6K file ({path_6k})")

    merged = rows_3k + rows_6k

    seen = {}
    dup_count = 0
    for row in merged:
        key = match_key(row, judge_name)
        if key is None:
            continue
        if key in seen:
            dup_count += 1
        seen[key] = row
    if dup_count:
        print(f"[WARN] {judge_name}/{model_key}: {dup_count} duplicate (root_id, language) "
              f"keys across the 3K+6K files -- keeping the later occurrence")

    out_path = os.path.join(out_dir, judge_name, f"{model_key}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for row in merged:
            f.write(json.dumps(row) + "\n")

    print(f"[OK] {judge_name}/{model_key}: {len(rows_3k)} (3K) + {len(rows_6k)} (6K) "
          f"= {len(merged)} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "09_llm_as_judge"),
        help="Directory containing the judge folders "
             "(Mistral-Judge-3K-Data, Qwen-Judge-6K-Data, etc.)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "merged"),
        help="Where to write merged/<judge>/<model>.jsonl",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=list(JUDGE_FILES.keys()),
        choices=list(JUDGE_FILES.keys()),
        help="Subset of judges to merge (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_KEYS,
        choices=MODEL_KEYS,
        help="Subset of generator models to merge (default: all)",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    out_dir = os.path.abspath(args.out_dir)

    sonnet_6k_dir = os.path.join(base_dir, JUDGE_FILES["sonnet"]["6k_dir"])
    if "sonnet" in args.judges and not any(
        f.endswith(".jsonl") for f in os.listdir(sonnet_6k_dir)
    ):
        raise SystemExit(
            f"[ERROR] {sonnet_6k_dir} has no .jsonl files yet.\n"
            f"Run its copy_judge_outputs.sh first:\n"
            f"  bash {os.path.join(sonnet_6k_dir, 'copy_judge_outputs.sh')}"
        )

    for judge_name in args.judges:
        for model_key in args.models:
            merge_one(judge_name, model_key, base_dir, out_dir)


if __name__ == "__main__":
    main()
