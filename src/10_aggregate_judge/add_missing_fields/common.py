"""
Shared helpers for reconstructing missing fields (lang_meta, eval, root_id,
language, category) in the merged judge outputs under ../merged/.

Every lookup here is built by backtracking through the pipeline that produced
the merged files, keyed by (root_id, language_name), the one pair of fields
that is stable across every stage from data/00_aegis_data all the way to
data/08_response_generate:

  data/08_response_generate/**/responses/*.jsonl   (current pipeline, "3K"/new samples)
  data/06_percentile_analysis_and_sampling/sampling/overall_stratified_10k.jsonl
      (adds the "old_reuse" samples not present in the 3K response files)
  src/09_llm_as_judge/Sonnet-Judge-6K-Data-Get_From-V1/*.jsonl
      (v1.0-origin "6K" judge output; already carries root_id/language/eval
      for samples that predate this repo's own response-generate stage)

Together these three sources cover 100% of the (root_id, language_name) pairs
that appear anywhere in merged/ (verified: 0 misses across all 62,172 rows
across mistral/qwen/sonnet x 6 models).

A large fraction of rows (117,745 / 163,585 root_id-bearing rows -- all of
them v1.0-origin) carry root_id as a bare 32-hex-char legacy AEGIS sample id
(e.g. "d8c171350bb0495faab437324c9de10c"), not the current pipeline's
"aegis_root_<16-hex>" form. These are the *pre-dedup* AEGIS ids that got
collapsed into a canonical root during data/00_aegis_data's root-construction
step. stage05_unique_roots.jsonl (and the stage06_*_roots.jsonl files derived
from it) record which original ids rolled into which root via their
"source_ids" list -- build_source_id_to_root_id() inverts that into
legacy_id -> aegis_root_id, resolving 1444/1482 (97.4%) of the legacy ids
seen in merged/. The remaining 38 predate/diverge from this repo's own
AEGIS filtering stages entirely (not present in any source_ids list here)
and are left as-is, flagged as unresolved in the enrichment report.

Sonnet's "3K" half only carries a packed custom_id (no root_id/language at
all). Its custom_id is built by judge.py::make_custom_id() as:

    f"{model_hash}__{sanitise(root_id)[:16]}__{sanitise(language)[:10]}__{resp_hash}"

where sanitise() strips everything except [a-zA-Z0-9_-]. That packed key is
only ever unique against the *current* response-generate files (the 3,820
"new_stratified" samples judge.py was run over) -- so the short-key lookup
below is built from that source alone, where it's verified collision-free.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import math
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG_LANGUAGE_METADATA = os.path.join(REPO_ROOT, "config", "language_metadata.py")
STAGE05_UNIQUE_ROOTS = os.path.join(REPO_ROOT, "data", "00_aegis_data", "stage05_unique_roots.jsonl")
OVERALL_STRATIFIED_10K = os.path.join(
    REPO_ROOT, "data", "06_percentile_analysis_and_sampling", "sampling", "overall_stratified_10k.jsonl"
)
RESPONSE_GENERATE_GLOB = os.path.join(REPO_ROOT, "data", "08_response_generate", "**", "*.jsonl")
SONNET_6K_DIR = os.path.join(
    REPO_ROOT, "src", "09_llm_as_judge", "Sonnet-Judge-6K-Data-Get_From-V1"
)

# Files carrying a root_id -> source_ids mapping, used to normalise legacy
# bare-hash root_id values back to the canonical "aegis_root_..." form.
ROOT_ID_SOURCE_FILES = [
    os.path.join(REPO_ROOT, "data", "00_aegis_data", "stage05_unique_roots.jsonl"),
    os.path.join(REPO_ROOT, "data", "01_aegis_filtered", "stage06_main_benchmark_roots.jsonl"),
    os.path.join(REPO_ROOT, "data", "01_aegis_filtered", "stage06_safe_roots.jsonl"),
    os.path.join(REPO_ROOT, "data", "01_aegis_filtered", "stage06_unsafe_roots.jsonl"),
    os.path.join(REPO_ROOT, "data", "01_aegis_filtered", "stage06_filtered_out_roots.jsonl"),
]

# Scripts whose text runs right-to-left; used only to backfill writing_direction
# for languages that never got a hand-curated lang_meta entry (see
# build_lang_meta_lookup). All 100 benchmark languages fall outside this set
# except none currently do -- kept for correctness/completeness.
RTL_SCRIPTS = {"Arab", "Hebr", "Syrc", "Thaa", "Nkoo", "Adlm", "Mand", "Samr"}

# Thresholds copied verbatim from v1.0/src/03_quality_buckets/00_filter_basedon_scores.py
# (verified byte-for-byte against all 6,542 pre-existing eval blocks in
# merged/sonnet/aya_expanse_32b.jsonl: 6542/6542 match).
GOLD_F1, GOLD_COMET, GOLD_COMBINED = 0.95, 0.92, 0.94
HQ_F1, HQ_COMET, HQ_COMBINED = 0.90, 0.85, 0.88
STD_F1, STD_COMET, STD_COMBINED = 0.80, 0.70, 0.75


def load_jsonl(path):
    rows = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def quality_bucket_and_combined(f1, comet):
    """Reproduces v1.0's quality_bucket()/combined_score logic exactly."""
    comet_ok = comet is not None and not is_nan(comet)
    if f1 is None:
        return None, None
    if comet_ok:
        combined = (f1 + comet) / 2.0
        if f1 >= GOLD_F1 and comet >= GOLD_COMET and combined >= GOLD_COMBINED:
            return "gold", combined
        if f1 >= HQ_F1 and comet >= HQ_COMET and combined >= HQ_COMBINED:
            return "high_quality", combined
        if f1 >= STD_F1 and comet >= STD_COMET and combined >= STD_COMBINED:
            return "standard_quality", combined
        return "low", combined
    else:
        combined = f1
        if f1 >= GOLD_F1:
            return "gold", combined
        if f1 >= HQ_F1:
            return "high_quality", combined
        if f1 >= STD_F1:
            return "standard_quality", combined
        return "low", combined


def sanitise(s: str) -> str:
    """Exact copy of judge.py::make_custom_id()'s sanitise()."""
    return re.sub(r"[^a-zA-Z0-9_-]", "", s or "")


def root_short16(root_id: str) -> str:
    """The 5-hex-char root fragment packed into a sonnet custom_id: the
    packed field is sanitise(root_id)[:16], and since root_id is always
    'aegis_root_' (11 chars) + a 16-hex-char hash, [:16] keeps the 11-char
    prefix plus the first 5 hash chars -- so stripping the prefix back off
    recovers exactly those 5 chars."""
    return sanitise(root_id or "")[:16].split("aegis_root_")[-1]


def lang_key10(language: str) -> str:
    """The (sanitised, truncated-to-10) language fragment packed into a
    sonnet custom_id."""
    return sanitise(language or "")[:10]


def build_source_id_to_root_id():
    """legacy bare-hash AEGIS id -> canonical 'aegis_root_...' id, built from
    every root-level file that carries a source_ids list."""
    mapping = {}
    for path in ROOT_ID_SOURCE_FILES:
        for row in load_jsonl(path):
            rid = row.get("root_id")
            if not rid:
                continue
            for sid in row.get("source_ids") or []:
                mapping.setdefault(sid, rid)
    return mapping


def normalize_root_id(root_id, source_id_to_root_id):
    """Returns (normalized_root_id, status) where status is one of
    'already_correct', 'normalized', 'unresolved_legacy'."""
    if not root_id:
        return root_id, "unresolved_legacy"
    if root_id.startswith("aegis_root_"):
        return root_id, "already_correct"
    mapped = source_id_to_root_id.get(root_id)
    if mapped:
        return mapped, "normalized"
    return root_id, "unresolved_legacy"


def parse_sonnet_custom_id(cid: str):
    """Returns (root_short, lang_key) parsed out of a packed sonnet
    custom_id, or (None, None) if it isn't in the expected 4-part form."""
    parts = str(cid or "").split("__")
    if len(parts) < 3:
        return None, None
    root_short = parts[1].split("aegis_root_")[-1]
    lang_key = parts[2]
    return root_short, lang_key


# ─────────────────────────────────────────────────────────────────────────
# lang_meta lookup
# ─────────────────────────────────────────────────────────────────────────

def _import_language_metadata_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_language_metadata", CONFIG_LANGUAGE_METADATA
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_lang_table_by_name():
    """language name -> config/language_metadata.py record, indexed both by
    the dict key and by the record's own 'language' field (e.g. key
    'Arabic' also resolves under its inner name 'Modern Standard Arabic')."""
    module = _import_language_metadata_module()
    table = module.LANG_NAME_TO_CODE
    by_name = {}
    for key, rec in dict.items(table):
        by_name.setdefault(key, rec)
        inner = rec.get("language")
        if inner:
            by_name.setdefault(inner, rec)
    return by_name


def build_lang_meta_lookup(merged_dir):
    """language -> lang_meta dict.

    Primary source: any row anywhere in merged/ that already carries a
    non-null lang_meta (this repo's own v1.0-origin data hand-curated
    lang_meta -- with joshi_class, in_lores_bench, etc. -- for a specific
    ~23-language low-resource subset; verified: every language in that
    subset has in_lores_bench=True and resource_level='low', i.e. lang_meta
    was deliberately only computed for the benchmark's low-resource focus
    languages, not fabricated for everyone).

    Fallback (for the ~77 languages that never got a hand-curated entry,
    almost all medium/high-resource "control" languages): derive the
    fields we *can* derive from config/language_metadata.py. joshi_class /
    joshi_class_label are left as None for these -- there's no formula or
    table anywhere in this repo to compute Joshi et al. (2020) class from
    scratch, and fabricating one would misrepresent languages the paper
    never classified that way.
    """
    lookup = {}
    sources = ["mistral", "qwen", "sonnet"]
    for judge_name in sources:
        d = os.path.join(merged_dir, judge_name)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".jsonl"):
                continue
            for row in load_jsonl(os.path.join(d, fn)):
                lang = row.get("language_name") or row.get("language")
                lm = row.get("lang_meta")
                if lang and lm and lang not in lookup:
                    lookup[lang] = lm

    observed_languages = set(lookup.keys())

    by_name = _config_lang_table_by_name()
    all_languages = set()
    for judge_name in sources:
        d = os.path.join(merged_dir, judge_name)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".jsonl"):
                continue
            for row in load_jsonl(os.path.join(d, fn)):
                lang = row.get("language_name") or row.get("language")
                if lang:
                    all_languages.add(lang)

    fallback_built = []
    unresolvable = []
    for lang in sorted(all_languages - observed_languages):
        rec = by_name.get(lang)
        if rec is None:
            unresolvable.append(lang)
            continue
        nllb_code = rec.get("nllb_code")
        script_iso = rec.get("script_iso")
        resource_level = (rec.get("resource_level_nllb") or "").lower() or None
        lookup[lang] = {
            "flores_code": nllb_code,
            "nllb_code": nllb_code,
            "language": lang,
            "family": rec.get("family"),
            "subgrouping": rec.get("subgrouping"),
            "script_iso": script_iso,
            "script_name": rec.get("script"),
            "writing_direction": "RTL" if script_iso in RTL_SCRIPTS else "LTR",
            "joshi_class": None,
            "joshi_class_label": None,
            "resource_level": resource_level,
            "in_lores_bench": False,
            "taxonomy_source": (
                "config/language_metadata.py (NLLB/FLORES-200 table); "
                "this language was not part of the hand-curated low-resource "
                "set, so joshi_class/joshi_class_label are not available"
            ),
        }
        fallback_built.append(lang)

    return lookup, {
        "observed_languages": sorted(observed_languages),
        "fallback_built_languages": fallback_built,
        "unresolvable_languages": unresolvable,
    }


# ─────────────────────────────────────────────────────────────────────────
# per-sample master lookup (root_id, language) -> category / quality scores
# ─────────────────────────────────────────────────────────────────────────

def build_sample_master(source_id_to_root_id=None):
    """Returns (exact_index, shortkey_index).

    exact_index:    (root_id, language_name) -> record   [full coverage, verified]
    shortkey_index: (root_short5, lang_key10) -> record   [built from response-
                     generate files ONLY -- the only universe a sonnet 3K
                     custom_id can reference; verified collision-free there]

    All root_id values are normalised to the canonical 'aegis_root_...' form
    (see normalize_root_id) before being used as keys, so both indexes are
    uniformly keyed regardless of which source a record came from.
    """
    if source_id_to_root_id is None:
        source_id_to_root_id = build_source_id_to_root_id()

    exact_index = {}
    shortkey_index = {}

    def add_exact(rid, lang, rec):
        rid, _status = normalize_root_id(rid, source_id_to_root_id)
        if rid and lang and (rid, lang) not in exact_index:
            exact_index[(rid, lang)] = rec

    # Source A: current pipeline's response-generate files (highest priority:
    # exact per-model responses judge.py's 3K sonnet run was scored against).
    source_a_records = {}
    for p in sorted(glob.glob(RESPONSE_GENERATE_GLOB, recursive=True)):
        if os.sep + "requests" + os.sep in p:
            continue
        for row in load_jsonl(p):
            rid, lang = row.get("root_id"), row.get("language_name")
            if rid and lang:
                add_exact(rid, lang, row)
                source_a_records.setdefault((rid, lang), row)

    for (rid, lang), row in source_a_records.items():
        shortkey_index[(root_short16(rid), lang_key10(lang))] = row

    # Source B: the master 10K sample list (adds "old_reuse" samples not in A).
    for row in load_jsonl(OVERALL_STRATIFIED_10K):
        rid, lang = row.get("root_id"), row.get("language_name")
        add_exact(rid, lang, row)

    # Source C: v1.0-origin sonnet 6K files -- gap-filler for the remaining
    # long tail of samples that predate this repo's own pipeline snapshot.
    # These rows already carry eval as a nested object; normalise it onto
    # the same flat field names as A/B (category, bertscore_f1, comet_score).
    if os.path.isdir(SONNET_6K_DIR):
        for fn in sorted(os.listdir(SONNET_6K_DIR)):
            if not fn.endswith(".jsonl"):
                continue
            for row in load_jsonl(os.path.join(SONNET_6K_DIR, fn)):
                rid, lang = row.get("root_id"), row.get("language")
                if not (rid and lang):
                    continue
                rid_norm, _status = normalize_root_id(rid, source_id_to_root_id)
                ev = row.get("eval") or {}
                normalised = {
                    "root_id": rid_norm,
                    "language_name": lang,
                    "category": ev.get("category"),
                    "bertscore_f1": ev.get("f1"),
                    "comet_score": None if is_nan(ev.get("comet")) else ev.get("comet"),
                    "comet_supported": ev.get("comet") is not None and not is_nan(ev.get("comet")),
                    "chrf_score": None,
                    "sample_id": None,
                    "sample_origin": None,
                    "selection_percentile": None,
                    "sampling_strategy": None,
                    "nllb_code": None,
                }
                add_exact(rid, lang, normalised)

    return exact_index, shortkey_index


# ─────────────────────────────────────────────────────────────────────────
# root_id -> AEGIS prompt_label ("unsafe"/"safe")
# ─────────────────────────────────────────────────────────────────────────

def build_root_label_lookup():
    lookup = {}
    for row in load_jsonl(STAGE05_UNIQUE_ROOTS):
        rid = row.get("root_id")
        if rid:
            lookup[rid] = row.get("prompt_label")
    return lookup
