#!/usr/bin/env python3
"""
Filter and clean metadata JSONs from v1/ to produce a curated v2/ dataset.

Steps per file (via a single LLM call):
  1. Detect and replace brand / product names with "xxx"
  2. Judge whether the case contains inappropriate content
     (gambling / drugs / explicit content / gender antagonism / bad values)
  3. Assign a sub-category label for distribution-balance selection

Then select up to the quota per video_type, maximising diversity across sub-categories.
Cases with fewer files than the quota keep all of them.

Quotas:
  narrative   15   cinematic  15   sci_fi       5
  action      10   vlog        5   commercial  10
  educational  5   music       5

Usage:
  python scripts/filter_metadata.py \\
      --input  data/metadata/v1 \\
      --output data/metadata/v2 \\
      --api-key <key> \\
      [--model <endpoint>] [--concurrency 4]
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

ARK_BASE_URL    = 
ARK_MODEL_DEFAULT = 

# ── Target quotas ─────────────────────────────────────────────────────────────
QUOTAS: dict[str, int] = {
    "narrative":   15,
    "cinematic":   15,
    "sci_fi":       5,
    "action":      10,
    "vlog":         5,
    "commercial":  10,
    "educational":  5,
    "music":        5,
}

# ── LLM prompt ────────────────────────────────────────────────────────────────
FILTER_PROMPT = """\
你是一个内容审核与数据清洗专家。请对下面的视频生成元数据 JSON 执行三项任务，并按指定格式输出结果。

## 输入 JSON
{json_content}

## 任务说明

### 任务1：品牌名称脱敏
将 JSON 中所有出现的真实品牌名、公司名、产品名、平台名（含中英文）替换为 "xxx"。
- 判断标准：商业实体的专有名词，如"卢浮宫"是景点而非品牌，不需替换；"可口可乐""Nike""抖音""iPhone"等是品牌，需替换
- 替换须保持 JSON 结构不变，仅替换字符串值中的品牌词

### 任务2：内容合规审查
判断该 case 是否含有以下任一不良内容，有则标记为不合规：
- 黄色内容（色情、性暗示、裸露）
- 赌博内容
- 毒品相关内容
- 明显的性别对立或歧视（如仇女/仇男、性别战争）
- 不良价值观（如鼓励犯罪、暴力美化、反社会行为）

### 任务3：子类别标注
为该 case 分配一个简短的子类别标签（10字以内，中文），用于后续均衡采样。
例如：爱情故事、悬疑犯罪、科幻冒险、亲情温情、喜剧搞笑、自然纪录、产品展示、新闻播报……

## 输出格式（严格 JSON，无注释，无 markdown 代码块）
{{
  "compliant": <true 表示内容合规 | false 表示含不良内容>,
  "violation_reason": "<不合规原因，合规时填 null>",
  "sub_category": "<子类别标签>",
  "cleaned_json": <品牌脱敏后的完整 JSON 对象>
}}

只输出上述 JSON 对象，不要任何其他文字。\
"""

# ── Per-file LLM call ─────────────────────────────────────────────────────────

def process_single(client: OpenAI, model: str, path: Path) -> dict[str, Any] | None:
    """
    Run the LLM filter on one JSON file.
    Returns a dict with keys: compliant, sub_category, cleaned_json, path
    Returns None on failure.
    """
    raw_json = path.read_text(encoding="utf-8")

    prompt = FILTER_PROMPT.format(json_content=raw_json)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    )

    text = response.output_text.strip()
    # Strip accidental markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    result = json.loads(text)
    result["_path"] = str(path)
    result["_filename"] = path.name
    return result


# ── Balanced selection ────────────────────────────────────────────────────────

def select_balanced(cases: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    """
    Select up to `quota` cases from a list, maximising sub_category diversity.
    Uses round-robin across sub-categories sorted by frequency (rarest first).
    """
    if len(cases) <= quota:
        return cases

    # Group by sub_category
    groups: dict[str, list[dict]] = {}
    for c in cases:
        key = c.get("sub_category", "未分类")
        groups.setdefault(key, []).append(c)

    # Sort groups: rarest first so we pick at least one from each
    ordered = sorted(groups.values(), key=len)

    selected: list[dict[str, Any]] = []
    # Round-robin: take one from each group in turn until quota filled
    idx = 0
    pointers = [0] * len(ordered)
    while len(selected) < quota:
        exhausted = True
        for g_idx, group in enumerate(ordered):
            if pointers[g_idx] < len(group):
                selected.append(group[pointers[g_idx]])
                pointers[g_idx] += 1
                exhausted = False
                if len(selected) >= quota:
                    break
        if exhausted:
            break

    return selected[:quota]


# ── Async batch ───────────────────────────────────────────────────────────────

async def process_all_async(
    paths: list[Path],
    client: OpenAI,
    model: str,
    concurrency: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_event_loop()
    results: list[dict[str, Any] | None] = [None] * len(paths)

    async def run(i: int, path: Path) -> None:
        async with sem:
            label = f"[{i+1}/{len(paths)}] {path.name}"
            if verbose:
                print(f"  → {label}", flush=True)
            try:
                result = await loop.run_in_executor(
                    None, process_single, client, model, path
                )
                results[i] = result
                if verbose:
                    status = "✓ compliant" if result["compliant"] else f"✗ blocked: {result['violation_reason']}"
                    print(f"  {status}  [{result['sub_category']}]  {label}", flush=True)
            except Exception as e:
                print(f"  ✗ error {label}: {e}", file=sys.stderr)

    await asyncio.gather(*[run(i, p) for i, p in enumerate(paths)])
    return [r for r in results if r is not None]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter, clean and balance DirectorBench metadata from v1 to v2."
    )
    parser.add_argument("--input",  "-i", default="data/metadata/v1",
                        help="Input directory containing v1 JSON files.")
    parser.add_argument("--output", "-o", default="data/metadata/v2",
                        help="Output directory for filtered JSON files.")
    parser.add_argument("--api-key", "-k", default=None,
                        help="ARK API key. Falls back to env var ARK_API_KEY.")
    parser.add_argument("--model", "-m", default=None,
                        help=f"ARK model endpoint ID (default: env ARK_MODEL or {ARK_MODEL_DEFAULT}).")
    parser.add_argument("--concurrency", "-c", type=int, default=4,
                        help="Concurrent API calls (default: 4).")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress per-file progress output.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        sys.exit("Error: provide --api-key or set ARK_API_KEY.")

    model = args.model or os.environ.get("ARK_MODEL", ARK_MODEL_DEFAULT)
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    verbose    = not args.quiet

    if not input_dir.is_dir():
        sys.exit(f"Error: input directory not found: {input_dir}")

    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        sys.exit(f"Error: no JSON files found in {input_dir}")

    print(f"Found {len(paths)} files in {input_dir}")
    print(f"Running LLM filter  [model: {model}, concurrency: {args.concurrency}]\n")

    client = OpenAI(base_url=ARK_BASE_URL, api_key=api_key)

    # ── Step 1: LLM filter all files ──
    all_results = asyncio.run(
        process_all_async(paths, client, model, args.concurrency, verbose)
    )

    # ── Step 2: Split compliant vs blocked ──
    compliant   = [r for r in all_results if r["compliant"]]
    blocked     = [r for r in all_results if not r["compliant"]]

    print(f"\nCompliant: {len(compliant)}  |  Blocked: {len(blocked)}")
    if blocked:
        print("Blocked cases:")
        for r in blocked:
            print(f"  {r['_filename']}: {r['violation_reason']}")

    # ── Step 3: Group by type and select balanced subset ──
    # Derive type from filename prefix (e.g. "narrative_001.json" → "narrative")
    type_groups: dict[str, list[dict]] = {}
    for r in compliant:
        type_key = re.match(r"([a-z_]+)_\d+", r["_filename"])
        type_key = type_key.group(1) if type_key else "unknown"
        type_groups.setdefault(type_key, []).append(r)

    print("\nSelection summary:")
    selected_all: list[dict[str, Any]] = []
    for type_key, quota in QUOTAS.items():
        group = type_groups.get(type_key, [])
        selected = select_balanced(group, quota)
        selected_all.extend(selected)
        print(f"  {type_key:12s}: {len(group):3d} available → {len(selected):3d} selected  (quota {quota})")

    # ── Step 4: Write output ──
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in selected_all:
        out_path = output_dir / r["_filename"]
        cleaned  = r["cleaned_json"]
        # Attach filter metadata
        cleaned["_sub_category"] = r["sub_category"]
        out_path.write_text(
            json.dumps(cleaned, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"\nSaved {len(selected_all)} files to {output_dir}/")


if __name__ == "__main__":
    main()
