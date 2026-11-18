#!/usr/bin/env python3
"""
Builds the three backtracking lookup tables used by enrich.py and dumps them
to lookups/*.json so they can be inspected/audited independently of the
enrichment run itself. See common.py for how each is constructed and why
each source was chosen.

Usage:
    python3 build_lookups.py
"""

import json
import os

import common

HERE = os.path.dirname(os.path.abspath(__file__))
LOOKUPS_DIR = os.path.join(HERE, "lookups")
MERGED_DIR = os.path.join(HERE, "..", "merged")

SEP = "\x1f"  # unlikely-to-collide separator for composite JSON keys


def dump(obj, name):
    path = os.path.join(LOOKUPS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=True)
    print(f"  wrote {path}")


def main():
    os.makedirs(LOOKUPS_DIR, exist_ok=True)

    print("=== Building lang_meta lookup (language -> lang_meta) ===")
    lang_meta_lookup, lang_meta_stats = common.build_lang_meta_lookup(MERGED_DIR)
    print(f"  observed (hand-curated) languages : {len(lang_meta_stats['observed_languages'])}")
    print(f"  fallback-built (from config)       : {len(lang_meta_stats['fallback_built_languages'])}")
    print(f"  unresolvable                       : {len(lang_meta_stats['unresolvable_languages'])}")
    if lang_meta_stats["unresolvable_languages"]:
        print(f"    !! {lang_meta_stats['unresolvable_languages']}")
    dump(lang_meta_lookup, "lang_meta_by_language.json")
    dump(lang_meta_stats, "lang_meta_build_stats.json")

    print("\n=== Building legacy AEGIS id -> aegis_root_ id mapping ===")
    source_id_to_root_id = common.build_source_id_to_root_id()
    print(f"  legacy id mappings: {len(source_id_to_root_id)}")
    dump(source_id_to_root_id, "source_id_to_root_id.json")

    print("\n=== Building per-sample master index (root_id, language) -> record ===")
    exact_index, shortkey_index = common.build_sample_master(source_id_to_root_id)
    print(f"  exact_index entries    : {len(exact_index)}")
    print(f"  shortkey_index entries : {len(shortkey_index)}")

    exact_dump = {
        f"{rid}{SEP}{lang}": rec for (rid, lang), rec in exact_index.items()
    }
    shortkey_dump = {
        f"{rs}{SEP}{lk}": rec for (rs, lk), rec in shortkey_index.items()
    }
    dump(exact_dump, "sample_master_exact.json")
    dump(shortkey_dump, "sample_master_shortkey.json")

    print("\n=== Building root_id -> AEGIS prompt_label lookup ===")
    root_label_lookup = common.build_root_label_lookup()
    print(f"  root_id entries: {len(root_label_lookup)}")
    dump(root_label_lookup, "root_label_lookup.json")

    print("\n=== Coverage check: legacy root_id normalisation against merged/ ===")
    bad_ids = set()
    for jd in ("mistral", "qwen", "sonnet"):
        d = os.path.join(MERGED_DIR, jd)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".jsonl"):
                continue
            for row in common.load_jsonl(os.path.join(d, fn)):
                rid = row.get("root_id")
                if rid and not rid.startswith("aegis_root_"):
                    bad_ids.add(rid)
    resolved = sum(1 for x in bad_ids if x in source_id_to_root_id)
    print(f"  legacy (non aegis_root_) root_ids found in merged/: {len(bad_ids)}")
    print(f"  resolvable via source_id_to_root_id mapping        : {resolved}")
    print(f"  unresolvable (left as legacy id, flagged in report): {len(bad_ids) - resolved}")

    print("\nDone.")


if __name__ == "__main__":
    main()
