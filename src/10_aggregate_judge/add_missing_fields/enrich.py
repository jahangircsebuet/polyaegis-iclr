#!/usr/bin/env python3
"""
Adds the 5 fields requested for correct aggregation -- lang_meta, eval,
root_id, language, category -- to every row of every merged/<judge>/<model>.jsonl
file, wherever they're missing. judge_id/custom_id are left alone (not needed
for aggregation).

Existing, already-populated fields are never overwritten -- only genuinely
missing/null fields are filled in, so this is safe to run against the
current merged/ files and re-run idempotently.

Reads:  ../merged/<judge>/<model>.jsonl
        lookups/*.json   (built by build_lookups.py -- run that first)
Writes: output/<judge>/<model>.jsonl   (enriched copy; merged/ is untouched)
        reports/enrichment_report.json + .txt
"""

import json
import os

import common

HERE = os.path.dirname(os.path.abspath(__file__))
LOOKUPS_DIR = os.path.join(HERE, "lookups")
MERGED_DIR = os.path.join(HERE, "..", "merged")
OUTPUT_DIR = os.path.join(HERE, "output")
REPORTS_DIR = os.path.join(HERE, "reports")

SEP = "\x1f"

JUDGES = ["mistral", "qwen", "sonnet"]
MODELS = [
    "aya_expanse_32b",
    "claude_haiku_4_5",
    "gemini_2_5_flash",
    "gpt_5_4_mini",
    "gpt_oss_120b",
    "mistral_large_123b",
]


def load_json(name):
    with open(os.path.join(LOOKUPS_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def unpack_composite(d):
    out = {}
    for k, v in d.items():
        a, b = k.split(SEP, 1)
        out[(a, b)] = v
    return out


def load_lookups():
    lang_meta_by_language = load_json("lang_meta_by_language.json")
    sample_master_exact = unpack_composite(load_json("sample_master_exact.json"))
    sample_master_shortkey = unpack_composite(load_json("sample_master_shortkey.json"))
    root_label_lookup = load_json("root_label_lookup.json")
    source_id_to_root_id = load_json("source_id_to_root_id.json")
    return (
        lang_meta_by_language, sample_master_exact, sample_master_shortkey,
        root_label_lookup, source_id_to_root_id,
    )


def resolve_identity(row, sample_master_exact, sample_master_shortkey, source_id_to_root_id):
    """Returns (root_id, language, rec, root_id_status) where rec is the
    matched sample-master record (or None if unresolved), root_id is always
    normalised to the canonical 'aegis_root_...' form when resolvable, and
    root_id_status is one of 'already_correct', 'normalized',
    'unresolved_legacy' (row had a root_id but it's a legacy id with no
    known mapping), or 'missing' (row had no root_id/custom_id lead at all)."""
    raw_root_id = row.get("root_id")
    language = row.get("language_name") or row.get("language")

    if raw_root_id:
        root_id, status = common.normalize_root_id(raw_root_id, source_id_to_root_id)
        rec = sample_master_exact.get((root_id, language)) if language else None
        if rec is None and language is None:
            for (rid, lang), candidate in sample_master_exact.items():
                if rid == root_id:
                    language, rec = lang, candidate
                    break
        return root_id, language, rec, status

    # No root_id at all -> only known case is a sonnet packed custom_id.
    cid = row.get("custom_id")
    if cid:
        root_short, lang_key = common.parse_sonnet_custom_id(cid)
        if root_short is not None:
            rec = sample_master_shortkey.get((root_short, lang_key))
            if rec is not None:
                return rec.get("root_id"), rec.get("language_name"), rec, "missing"

    return None, None, None, "missing"


def build_eval(category, rec, language, lang_meta, root_label_lookup, root_id):
    f1 = rec.get("bertscore_f1") if rec else None
    comet = rec.get("comet_score") if (rec and rec.get("comet_supported")) else float("nan")
    quality_bucket, combined_score = common.quality_bucket_and_combined(f1, comet)
    tier = (lang_meta or {}).get("resource_level")
    label = root_label_lookup.get(root_id, "unsafe") if root_id else "unsafe"
    return {
        "category": category,
        "tier": tier,
        "label": label,
        "f1": f1,
        "comet": comet,
        "combined_score": combined_score,
        "quality_bucket": quality_bucket,
    }


def enrich_row(row, lang_meta_by_language, sample_master_exact, sample_master_shortkey,
                root_label_lookup, source_id_to_root_id, stats):
    root_id, language, rec, root_id_status = resolve_identity(
        row, sample_master_exact, sample_master_shortkey, source_id_to_root_id
    )

    # ---- root_id --------------------------------------------------------------
    # Every row's root_id must end up either absent or in the canonical
    # 'aegis_root_...' form -- never left as a legacy bare-hash AEGIS id.
    if not row.get("root_id"):
        if root_id:
            row["root_id"] = root_id
            stats["root_id_added"] += 1
        else:
            stats["root_id_unresolved"] += 1
    elif root_id_status == "already_correct":
        stats["root_id_already_present"] += 1
    elif root_id_status == "normalized":
        row["root_id"] = root_id  # replace legacy bare-hash id with aegis_root_ form
        stats["root_id_normalized"] += 1
    else:  # unresolved_legacy: no known mapping for this legacy id
        stats["root_id_unresolved_legacy"] += 1

    # ---- language -----------------------------------------------------------
    if not row.get("language"):
        chosen_language = row.get("language_name") or language
        if chosen_language:
            row["language"] = chosen_language
            stats["language_added"] += 1
        else:
            stats["language_unresolved"] += 1
    else:
        stats["language_already_present"] += 1
    language = row.get("language") or language

    # ---- category -----------------------------------------------------------
    if not row.get("category"):
        existing_eval = row.get("eval") or {}
        category = existing_eval.get("category") or (rec.get("category") if rec else None)
        if category:
            row["category"] = category
            stats["category_added"] += 1
        else:
            stats["category_unresolved"] += 1
    else:
        stats["category_already_present"] += 1
    category = row.get("category")

    # ---- lang_meta ----------------------------------------------------------
    if not row.get("lang_meta"):
        lm = lang_meta_by_language.get(language) if language else None
        if lm:
            row["lang_meta"] = lm
            stats["lang_meta_added"] += 1
        else:
            stats["lang_meta_unresolved"] += 1
    else:
        stats["lang_meta_already_present"] += 1
    lang_meta = row.get("lang_meta")

    # ---- eval -----------------------------------------------------------------
    existing_eval = row.get("eval")
    if not existing_eval or not all(k in existing_eval for k in ("label", "f1", "comet", "quality_bucket")):
        new_eval = build_eval(category, rec, language, lang_meta, root_label_lookup, row.get("root_id"))
        if existing_eval:
            merged_eval = dict(new_eval)
            merged_eval.update({k: v for k, v in existing_eval.items() if v is not None})
            row["eval"] = merged_eval
        else:
            row["eval"] = new_eval
        if rec is None and category is None:
            stats["eval_reconstructed_incomplete"] += 1
        else:
            stats["eval_added"] += 1
    else:
        stats["eval_already_present"] += 1

    if rec is None and root_id_status == "missing":
        stats["fully_unresolved_rows"] += 1

    return row


def new_stats():
    return {
        "total_rows": 0,
        "root_id_already_present": 0, "root_id_normalized": 0,
        "root_id_added": 0, "root_id_unresolved": 0, "root_id_unresolved_legacy": 0,
        "language_already_present": 0, "language_added": 0, "language_unresolved": 0,
        "category_already_present": 0, "category_added": 0, "category_unresolved": 0,
        "lang_meta_already_present": 0, "lang_meta_added": 0, "lang_meta_unresolved": 0,
        "eval_already_present": 0, "eval_added": 0, "eval_reconstructed_incomplete": 0,
        "fully_unresolved_rows": 0,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading lookups...")
    (lang_meta_by_language, sample_master_exact, sample_master_shortkey,
     root_label_lookup, source_id_to_root_id) = load_lookups()
    print(f"  lang_meta_by_language : {len(lang_meta_by_language)} languages")
    print(f"  sample_master_exact   : {len(sample_master_exact)} keys")
    print(f"  sample_master_shortkey: {len(sample_master_shortkey)} keys")
    print(f"  root_label_lookup     : {len(root_label_lookup)} root_ids")
    print(f"  source_id_to_root_id  : {len(source_id_to_root_id)} legacy id mappings")

    report = {}

    for judge_name in JUDGES:
        for model in MODELS:
            in_path = os.path.join(MERGED_DIR, judge_name, f"{model}.jsonl")
            if not os.path.isfile(in_path):
                continue
            out_dir = os.path.join(OUTPUT_DIR, judge_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{model}.jsonl")

            stats = new_stats()
            rows_out = []
            for row in common.load_jsonl(in_path):
                stats["total_rows"] += 1
                rows_out.append(
                    enrich_row(
                        row, lang_meta_by_language, sample_master_exact,
                        sample_master_shortkey, root_label_lookup,
                        source_id_to_root_id, stats
                    )
                )

            with open(out_path, "w", encoding="utf-8") as f:
                for row in rows_out:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")

            report[f"{judge_name}/{model}"] = stats
            print(f"[OK] {judge_name}/{model}: {stats['total_rows']} rows -> {out_path}")

    with open(os.path.join(REPORTS_DIR, "enrichment_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    write_text_report(report)
    print("\nReports written to reports/enrichment_report.json and reports/enrichment_report.txt")


def write_text_report(report):
    lines = []
    lines.append("Enrichment report -- add_missing_fields")
    lines.append("=" * 120)
    header = (
        f"{'file':<22}{'rows':>7}"
        f"{'rid_norm':>9}{'rid+':>6}{'rid?leg':>8}{'rid?':>6}"
        f"{'lang+':>7}{'lang?':>6}"
        f"{'cat+':>6}{'cat?':>6}"
        f"{'lm+':>6}{'lm?':>5}"
        f"{'eval+':>7}{'eval~':>7}"
    )
    lines.append(header)
    lines.append("-" * 120)
    for key, s in report.items():
        lines.append(
            f"{key:<22}{s['total_rows']:>7}"
            f"{s['root_id_normalized']:>9}{s['root_id_added']:>6}"
            f"{s['root_id_unresolved_legacy']:>8}{s['root_id_unresolved']:>6}"
            f"{s['language_added']:>7}{s['language_unresolved']:>6}"
            f"{s['category_added']:>6}{s['category_unresolved']:>6}"
            f"{s['lang_meta_added']:>6}{s['lang_meta_unresolved']:>5}"
            f"{s['eval_added']:>7}{s['eval_reconstructed_incomplete']:>7}"
        )
    lines.append("-" * 120)
    lines.append("legend: rid_norm = legacy bare-hash root_id rewritten to aegis_root_ form")
    lines.append("        rid+ = root_id added where fully missing, rid?leg = legacy id with no known mapping (left as-is)")
    lines.append("        X+ = field added by this script, X? = could not be resolved (left as-is)")
    lines.append("        eval~ = eval added but with some sub-fields still unresolved")
    with open(os.path.join(REPORTS_DIR, "enrichment_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
