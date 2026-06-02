#!/usr/bin/env python3
"""
Combine v2 (Chinese corpus) and v3 (multilingual variants) across all batch folders
into a single JSONL file, with multilingual variant flags on every record.

JSONL schema per record:
  sample_id                 : str   — shared across all variants of the same case
  variant_type              : str   — "original" | "instruction_lang" | "content_lang"
  language                  : str   — instruction language (zh / en / ja / ...)
  content_lang              : str?  — video content language (content_lang variants only)
  has_instruction_variants  : bool  — whether instr_* variants exist for this sample
  instruction_variant_langs : list  — which instruction languages are available
  has_content_variants      : bool  — whether content_* variants exist for this sample
  content_variant_langs     : list  — which content languages are available
  ...all original metadata fields...

Evaluation workflow:
  Phase 1 – monolingual baseline : filter variant_type == "original"
  Phase 2 – instruction robustness: group by sample_id, compare instr_* vs zh
  Phase 3 – content robustness   : filter variant_type == "content_lang"

File priority: v3 > v2 (v3 files have sample_id/variant_type already set).
v2 files without a v3 counterpart are backfilled and included.

Usage:
  python scripts/combine_dataset.py \\
      --metadata-dir data/metadata \\
      --output       data/metadata/dataset.jsonl \\
      --batches 0318 0319
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── Filename parsing ──────────────────────────────────────────────────────────

def parse_stem(stem: str) -> tuple[str, str]:
    """
    Returns (base_id, variant_suffix).
    narrative_001              → ("narrative_001", "zh")
    narrative_001_zh           → ("narrative_001", "zh")
    narrative_001_instr_en     → ("narrative_001", "instr_en")
    narrative_001_content_ja   → ("narrative_001", "content_ja")
    """
    m = re.match(r"^(.+?)_(zh|(?:instr|content)_\w+)$", stem)
    if m:
        return m.group(1), m.group(2)
    return stem, "zh"   # v2 file with no language suffix


def variant_type_from_suffix(suffix: str) -> str:
    if suffix == "zh":              return "original"
    if suffix.startswith("instr_"): return "instruction_lang"
    if suffix.startswith("content_"): return "content_lang"
    return "unknown"


def instr_lang_from_suffix(suffix: str) -> str:
    if suffix.startswith("instr_"):   return suffix[6:]
    if suffix.startswith("content_"): return "zh"   # instruction stays Chinese
    return "zh"


def content_lang_from_suffix(suffix: str) -> str | None:
    if suffix.startswith("content_"): return suffix[8:]
    return None


def video_type_key(video_type: str) -> str:
    return {
        "叙事/故事类":      "narrative",
        "电影镜头类":       "cinematic",
        "科幻类（反规律）": "sci_fi",
        "动作类":           "action",
        "日常生活（vlog）": "vlog",
        "商业营销":         "commercial",
        "教育新闻":         "educational",
        "音乐类":           "music",
    }.get(video_type, video_type)


# ── Collection ────────────────────────────────────────────────────────────────

def collect_files(meta_root: Path, batches: list[str]) -> dict[tuple, dict]:
    """
    Returns {(base_id, suffix): {path, batch, tier}}.
    v3 takes priority over v2 for the same key.
    """
    collected: dict[tuple, dict] = {}
    for batch in batches:
        for tier in ("v3", "v2"):
            tier_dir = meta_root / batch / tier
            if not tier_dir.is_dir():
                continue
            for path in sorted(tier_dir.glob("*.json")):
                base_id, suffix = parse_stem(path.stem)
                key = (base_id, suffix)
                if key not in collected or tier == "v3":
                    collected[key] = {"path": path, "batch": batch, "tier": tier}
    return collected


# ── Build variant index ───────────────────────────────────────────────────────

def build_variant_index(collected: dict[tuple, dict]) -> dict[str, dict]:
    """
    For each base_id, collect which instruction langs and content langs exist.
    Returns {base_id: {instr_langs: [...], content_langs: [...]}}.
    """
    index: dict[str, dict] = defaultdict(lambda: {"instr_langs": [], "content_langs": []})
    for (base_id, suffix) in collected:
        if suffix.startswith("instr_"):
            index[base_id]["instr_langs"].append(suffix[6:])
        elif suffix.startswith("content_"):
            index[base_id]["content_langs"].append(suffix[8:])
    return index


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combine v2+v3 metadata into a single JSONL dataset file."
    )
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--output", "-o",  default="data/metadata/dataset.jsonl")
    parser.add_argument("--batches", nargs="+", default=["0318", "0319", "0321"])
    args = parser.parse_args()

    meta_root  = Path(args.metadata_dir)
    output_path = Path(args.output)
    batches     = args.batches

    # ── Step 1: collect ───────────────────────────────────────────────────────
    collected = collect_files(meta_root, batches)
    print(f"Collected {len(collected)} unique entries from batches: {batches}")

    # ── Step 2: assign sample_ids ─────────────────────────────────────────────
    # IDs are always re-assigned from scratch based on the full current dataset.
    # Any sample_id stored in source files is ignored and overwritten.
    base_ids_ordered = sorted({base for (base, _) in collected})
    base_to_sid = {base: f"DB_{i+1:03d}" for i, base in enumerate(base_ids_ordered)}

    # ── Step 3: build variant flags index ────────────────────────────────────
    var_index = build_variant_index(collected)

    # ── Step 4: load + annotate + write JSONL ────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for (base_id, suffix), info in sorted(collected.items()):
        path: Path = info["path"]
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  Warning: cannot read {path}: {e}", file=sys.stderr)
            continue

        sample_id    = base_to_sid[base_id]
        vtype        = variant_type_from_suffix(suffix)
        instr_lang   = instr_lang_from_suffix(suffix)
        content_lang = content_lang_from_suffix(suffix)
        vi           = var_index.get(base_id, {"instr_langs": [], "content_langs": []})

        # Core identification fields (overwrite whatever was in the file)
        doc["sample_id"]    = sample_id
        doc["variant_type"] = vtype
        doc["language"]     = instr_lang
        if content_lang:
            doc["content_lang"] = content_lang
        elif "content_lang" in doc:
            del doc["content_lang"]

        # Unique id for this specific variant, for direct lookup
        if vtype == "original":
            doc["variant_id"] = sample_id                          # e.g. "DB_001"
        elif vtype == "instruction_lang":
            doc["variant_id"] = f"{sample_id}_instr_{instr_lang}" # e.g. "DB_001_instr_en"
        else:
            doc["variant_id"] = f"{sample_id}_content_{content_lang}"  # e.g. "DB_001_content_ja"

        # Variant availability flags.
        # All variants share the same sample_id — use it to look them up:
        #   [r for r in records if r["sample_id"] == doc["sample_id"]]
        # The fields below are shortcuts so you don't have to scan the whole file.
        instr_ids   = [f"{sample_id}_instr_{l}"   for l in sorted(vi["instr_langs"])]
        content_ids = [f"{sample_id}_content_{l}" for l in sorted(vi["content_langs"])]

        doc["has_instruction_variants"]  = len(instr_ids) > 0
        doc["instruction_variant_langs"] = sorted(vi["instr_langs"])
        doc["instruction_variant_ids"]   = instr_ids   # e.g. ["DB_001_instr_en", ...]

        doc["has_content_variants"]      = len(content_ids) > 0
        doc["content_variant_langs"]     = sorted(vi["content_langs"])
        doc["content_variant_ids"]       = content_ids  # e.g. ["DB_001_content_ja", ...]

        # Provenance
        doc["_batch"] = info["batch"]
        doc["_tier"]  = info["tier"]

        records.append(doc)

    with output_path.open("w", encoding="utf-8") as f:
        for doc in records:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Written {len(records)} records → {output_path}")

    # ── Step 4b: write original-only JSONL ───────────────────────────────────
    original_records = [doc for doc in records if doc.get("variant_type") == "original"]
    original_path = output_path.with_name(output_path.stem + "_original.jsonl")
    with original_path.open("w", encoding="utf-8") as f:
        for doc in original_records:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Written {len(original_records)} original records → {original_path}")

    # ── Step 5: distribution report ───────────────────────────────────────────
    dist: dict = {
        "total":            len(records),
        "unique_samples":   len(base_ids_ordered),
        "by_batch":         defaultdict(int),
        "by_variant_type":  defaultdict(int),
        "by_video_type":    defaultdict(int),
        "by_instr_lang":    defaultdict(int),
        "by_content_lang":  defaultdict(int),
        "vtype_x_vt":       defaultdict(lambda: defaultdict(int)),
        "instr_lang_x_vt":  defaultdict(lambda: defaultdict(int)),
    }

    for doc in records:
        vt    = video_type_key(doc.get("video_type", "unknown"))
        vtype = doc.get("variant_type", "unknown")
        lang  = doc.get("language", "zh")
        clang = doc.get("content_lang")
        batch = doc.get("_batch", "unknown")

        dist["by_batch"][batch]           += 1
        dist["by_variant_type"][vtype]    += 1
        dist["by_video_type"][vt]         += 1
        dist["by_instr_lang"][lang]       += 1
        dist["vtype_x_vt"][vtype][vt]     += 1
        dist["instr_lang_x_vt"][lang][vt] += 1
        if clang:
            dist["by_content_lang"][clang] += 1

    # ── Print ─────────────────────────────────────────────────────────────────
    sep = "─" * 64
    print(f"\n{sep}")
    print("  DirectorBench Dataset — Distribution Report")
    print(sep)
    print(f"  Total records    : {dist['total']}")
    print(f"  Unique samples   : {dist['unique_samples']}  (grouped by sample_id)")

    print("\n[ Records by Batch ]")
    for k, v in sorted(dist["by_batch"].items()):
        print(f"  {k} : {v:4d}")

    print("\n[ Records by Variant Type ]")
    for k, v in sorted(dist["by_variant_type"].items(), key=lambda x: -x[1]):
        bar = "█" * (v // 2)
        print(f"  {k:22s} : {v:4d}  {bar}")

    print("\n[ Records by Video Type ]")
    for k, v in sorted(dist["by_video_type"].items(), key=lambda x: -x[1]):
        bar = "█" * (v // 2)
        print(f"  {k:15s} : {v:4d}  {bar}")

    print("\n[ Records by Instruction Language ]")
    for k, v in sorted(dist["by_instr_lang"].items(), key=lambda x: -x[1]):
        bar = "█" * (v // 2)
        print(f"  {k:6s} : {v:4d}  {bar}")

    if dist["by_content_lang"]:
        print("\n[ Records by Content Language  (content_lang variants) ]")
        for k, v in sorted(dist["by_content_lang"].items(), key=lambda x: -x[1]):
            print(f"  {k:6s} : {v:4d}")

    # Cross-tab: variant_type × video_type
    print("\n[ Variant Type × Video Type ]")
    all_vts    = sorted(dist["by_video_type"].keys())
    all_vtypes = sorted(dist["by_variant_type"].keys())
    col_w = 13
    header = f"  {'':22s}" + "".join(f"{vt[:col_w]:>{col_w}}" for vt in all_vts)
    print(header)
    for vtype in all_vtypes:
        row = f"  {vtype:22s}"
        for vt in all_vts:
            cnt = dist["vtype_x_vt"][vtype].get(vt, 0)
            row += f"  {cnt:>{col_w-2}d} "
        print(row)

    # Cross-tab: instruction language × video_type
    print("\n[ Instruction Language × Video Type ]")
    all_langs = sorted(dist["by_instr_lang"].keys())
    header = f"  {'':8s}" + "".join(f"{vt[:col_w]:>{col_w}}" for vt in all_vts)
    print(header)
    for lang in all_langs:
        row = f"  {lang:8s}"
        for vt in all_vts:
            cnt = dist["instr_lang_x_vt"][lang].get(vt, 0)
            row += f"  {cnt:>{col_w-2}d} "
        print(row)

    print(f"\n{sep}\n")

    # Save report alongside JSONL
    report = {}
    for k, v in dist.items():
        if isinstance(v, defaultdict):
            report[k] = {kk: (dict(vv) if isinstance(vv, defaultdict) else vv)
                         for kk, vv in v.items()}
        else:
            report[k] = v
    report_path = output_path.with_name("distribution_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Distribution report → {report_path}")


if __name__ == "__main__":
    main()
