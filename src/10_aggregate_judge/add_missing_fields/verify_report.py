#!/usr/bin/env python3
"""
Before/after field-presence comparison for the 5 target fields across every
merged/<judge>/<model>.jsonl vs. its enriched output/<judge>/<model>.jsonl
counterpart. Prints a table and writes reports/before_after_field_presence.txt.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MERGED_DIR = os.path.join(HERE, "..", "merged")
OUTPUT_DIR = os.path.join(HERE, "output")
REPORTS_DIR = os.path.join(HERE, "reports")

FIELDS = ["lang_meta", "eval", "root_id", "language", "category"]

JUDGES = ["mistral", "qwen", "sonnet"]
MODELS = [
    "aya_expanse_32b", "claude_haiku_4_5", "gemini_2_5_flash",
    "gpt_5_4_mini", "gpt_oss_120b", "mistral_large_123b",
]


def count_present(path):
    total = 0
    present = {f: 0 for f in FIELDS}
    if not os.path.isfile(path):
        return total, present
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total += 1
            for fld in FIELDS:
                if d.get(fld) is not None:
                    present[fld] += 1
    return total, present


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    lines = []
    lines.append("Before (merged/) vs After (output/) -- field presence, per file")
    lines.append("=" * 118)
    header = f"{'file':<24}{'rows':>7}" + "".join(f"{f + '(before->after)':>22}" for f in FIELDS)
    lines.append(header)
    lines.append("-" * 118)

    for judge_name in JUDGES:
        for model in MODELS:
            before_path = os.path.join(MERGED_DIR, judge_name, f"{model}.jsonl")
            after_path = os.path.join(OUTPUT_DIR, judge_name, f"{model}.jsonl")
            if not os.path.isfile(before_path):
                continue
            total_b, present_b = count_present(before_path)
            total_a, present_a = count_present(after_path)
            row = f"{judge_name + '/' + model:<24}{total_a:>7}"
            for fld in FIELDS:
                row += f"{present_b[fld]:>9} -> {present_a[fld]:<9}"
            lines.append(row)

    lines.append("-" * 118)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(REPORTS_DIR, "before_after_field_presence.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
