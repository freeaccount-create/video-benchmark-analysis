#!/usr/bin/env python3
"""
Extract structured DirectorBench metadata from conversation CSV files via ARK API.

Supports two CSV input formats:
  Format A (批跑结果): one row per case
    Columns: case_id, account_name, title, scene, prompt, total_turns, ...
  Format B (Agent线上数据): multi-row conversation per user
    Columns: 用户序号, 消息序号, 角色, 内容

Environment variables:
  ARK_API_KEY   : (required) ByteDance ARK API key
  ARK_MODEL     : (optional) model endpoint ID, overrides --model flag

Usage:
  python scripts/extract_metadata.py --input data/xxx.csv --output data/metadata/
  python scripts/extract_metadata.py --input data/ --output data/metadata/
  python scripts/extract_metadata.py --input data/ --output data/metadata/ --concurrency 5
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

ARK_BASE_URL = 
ARK_MODEL_DEFAULT = 

# ── Video type canonical names (see data/README.md) ──────────────────────────
VIDEO_TYPES = {
    "narrative":    "叙事/故事类",
    "cinematic":    "电影镜头类",
    "sci_fi":       "科幻类（反规律）",
    "action":       "动作类",
    "vlog":         "日常生活（vlog）",
    "commercial":   "商业营销",
    "educational":  "教育新闻",
    "music":        "音乐类",
}
VIDEO_TYPES_ZH = "、".join(VIDEO_TYPES.values())

# ── Extraction prompt ─────────────────────────────────────────────────────────
EXTRACT_PROMPT = """\
你是一个视频生成任务的元数据提取专家。请从以下用户与AI助手的对话中，提取出一组视频生成的结构化元数据。

## 对话内容
{conversation}

## 提取规则
1. 如果对话中涉及多个视频生成任务，只提取最核心/最完整的那一组
2. video_type 必须从以下选项中选择一个（输出中文名称）：{video_types}
3. duration_sec：从对话中推断视频总时长（秒），无明确信息则填 null
4. 尽量从对话中提取具体的镜头、音频、故事结构等信息；没有相关信息的字段填 null 或空列表
5. 若对话没有明确的故事三幕结构，story_arc 各字段填 null
6. script 与 visual.shots 均为【按需提取】字段——仅当用户明确指定了必须在视频中出现的台词/旁白或具体场景时才填写；
   若用户只给出风格/氛围描述而未指定具体内容，则两者均填空列表 []
7. 当 script 与 visual.shots 均有内容时，二者的 shot_id 须严格一一对应

## 输出字段说明

### 顶层字段
- meta_id        : 自动生成的唯一标识符，格式为 "类型_编号"，保持填充值不变
- duration_sec   : 视频总时长（秒）。从对话中寻找"X秒""X分钟"等明确表述；无则填 null
- video_type     : 视频类型，从上述选项中选一个中文名称
- main_instruction: 用一句话概括用户最核心的视频创作意图（不超过50字）

### modality_details.text — 文本/剧本层
- story_arc      : 故事三幕结构
  - act1_setup      : 第一幕「建立」——交代背景、引入人物与冲突起点
  - act2_conflict   : 第二幕「冲突」——核心矛盾爆发、情节推进
  - act3_resolution : 第三幕「解决」——冲突化解、结局或情绪落点
- script         : 【按需提取】仅当用户明确要求特定台词/旁白必须出现在视频中时才填写，否则填 []
                   每个元素对应一个镜头，shot_id 与 visual.shots 一一对应
  - shot_id         : 镜头编号
  - duration_sec    : 该镜头时长（秒），无明确信息填 null
  - dialogue        : 该镜头内角色说的对白原文；无对白填空字符串 ""
  - narration       : 该镜头的画外音/旁白原文；无旁白填空字符串 ""
- tone_requirements: 整体情感基调与风格要求，如"romantic_bittersweet""悬疑紧张逐步释放"

### modality_details.visual — 视觉层
- shots          : 【按需提取】仅当用户明确要求特定场景/画面必须出现在视频中时才填写，否则填 []
                   shot_id 与 script 一一对应（若 script 有内容）
  - shot_id         : 镜头编号
  - description     : 镜头的画面内容描述（主体、场景、构图）
  - camera_movement : 运镜方式，如 tracking / push_in / pull_out / orbiting / handheld / crane / zoom / static；无明确信息填 null
  - lighting        : 光线与色调描述，如 "natural_rainy_glow" / "soft_dramatic"；无明确信息填 null
- camera_requirements     : 整个视频中要求出现的运镜类型列表；无则填 []
- consistency_requirements: 跨镜头需要保持一致的元素列表，如 "character_identity" / "clothing" / "lighting_shadow"；无则填 []

### modality_details.audio — 音频层
- dialogue       : 视频是否包含人物对白（true/false）
- lip_sync       : 是否要求对白与口型同步（true/false）
- bgm_style      : 背景音乐风格描述，如 "soft_piano_orchestral"；无明确信息填 null
- sound_effects  : 需要的音效列表，如 ["rain_ambient", "footsteps_in_puddle"]；无则填 []
- tone_control   : 音频情感走向控制，如 "emotional_buildup_to_warm_resolution"；无则填 null
- multi_language : 语言要求，如 "zh" / "en" / "zh_en_switch"；无特殊要求填 null

## 输出格式（严格 JSON，无注释、无 markdown 代码块）
{{
  "meta_id": "{meta_id}",
  "duration_sec": <number | null>,
  "video_type": "<从上述选项选一个中文名称>",
  "main_instruction": "<用户的核心视频生成指令，一句话概括>",
  "modality_details": {{
    "text": {{
      "story_arc": {{
        "act1_setup": "<第一幕描述 | null>",
        "act2_conflict": "<第二幕描述 | null>",
        "act3_resolution": "<第三幕描述 | null>"
      }},
      "script": [],
      "tone_requirements": "<情感/风格基调 | null>"
    }},
    "visual": {{
      "shots": [],
      "camera_requirements": [],
      "consistency_requirements": []
    }},
    "audio": {{
      "dialogue": <true | false>,
      "lip_sync": <true | false>,
      "bgm_style": "<背景音乐风格 | null>",
      "sound_effects": [],
      "tone_control": "<音频情感走向 | null>",
      "multi_language": "<语言要求 | null>"
    }}
  }}
}}

只输出 JSON 对象，不要任何前缀、后缀或解释文字。\
"""

# ── CSV format detection ──────────────────────────────────────────────────────

def detect_format(path: Path) -> str:
    """Return 'A' for 批跑结果 format, 'B' for Agent线上数据 format."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
    if "用户序号" in first_line or "消息序号" in first_line:
        return "B"
    return "A"


# ── Loaders ───────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_format_a(path: Path) -> list[dict[str, Any]]:
    """Load Format A: one case per row. Returns list of cases."""
    cases = []
    for row in _read_csv(path):
        case_id = (row.get("case_id") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        scene = (row.get("scene") or "").strip()
        title = (row.get("title") or "").strip()
        if not prompt:
            continue
        # Build a minimal conversation string from the single prompt
        context_lines = []
        if scene:
            context_lines.append(f"[场景分类: {scene}]")
        if title:
            context_lines.append(f"[内容类型: {title}]")
        context_lines.append(f"用户: {prompt}")
        cases.append({
            "id": case_id or str(len(cases) + 1),
            "conversation": "\n".join(context_lines),
            "source": path.name,
        })
    return cases


def load_format_b(path: Path) -> list[dict[str, Any]]:
    """Load Format B: multi-row conversations. Returns one case per user."""
    rows = _read_csv(path)
    # Propagate 用户序号 across blank rows
    current_uid = None
    user_rows: dict[str, list[dict]] = {}
    uid_order: list[str] = []
    for row in rows:
        uid = (row.get("用户序号") or "").strip()
        if uid:
            current_uid = uid
            if uid not in user_rows:
                user_rows[uid] = []
                uid_order.append(uid)
        if current_uid:
            user_rows[current_uid].append(row)

    cases = []
    for uid in uid_order:
        msgs = user_rows[uid]
        lines = []
        for msg in msgs:
            role = (msg.get("角色") or "").strip()
            content = (msg.get("内容") or "").strip()
            if not content or content == "NULL":
                continue
            if role == "user":
                lines.append(f"用户: {content}")
            elif role == "assistant":
                lines.append(f"助手: {content}")
            # skip tool messages (complex JSON, not needed for extraction)
        if not lines:
            continue
        cases.append({
            "id": uid,
            "conversation": "\n\n".join(lines),
            "source": path.name,
        })
    return cases


# ── LLM extraction ────────────────────────────────────────────────────────────

def build_meta_id(video_type_zh: str, case_id: str) -> str:
    """Build a meta_id like 'narrative_001' from zh type name and case id."""
    reverse = {v: k for k, v in VIDEO_TYPES.items()}
    type_en = reverse.get(video_type_zh, "unknown")
    # Pad case_id if numeric
    padded = case_id.zfill(3) if case_id.isdigit() else case_id
    return f"{type_en}_{padded}"


def extract_single(client: OpenAI, model: str, case: dict[str, Any]) -> dict[str, Any]:
    """Call ARK API to extract metadata for one case. Returns parsed dict."""
    # Use a placeholder meta_id; we'll update it after we know the video_type
    placeholder_id = f"case_{case['id'].zfill(3) if case['id'].isdigit() else case['id']}"

    prompt = EXTRACT_PROMPT.format(
        conversation=case["conversation"],
        video_types=VIDEO_TYPES_ZH,
        meta_id=placeholder_id,
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
    )

    raw = response.output_text.strip()
    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)

    # Fix meta_id now that we know video_type
    video_type_zh = result.get("video_type", "")
    result["meta_id"] = build_meta_id(video_type_zh, case["id"])
    result["_source"] = case["source"]
    result["_case_id"] = case["id"]
    return result


# ── Batch processing ──────────────────────────────────────────────────────────

async def extract_all_async(
    cases: list[dict[str, Any]],
    output_dir: Path,
    concurrency: int,
    model: str,
    api_key: str,
    verbose: bool,
) -> None:
    """Extract metadata for all cases with bounded concurrency."""
    client = OpenAI(
        base_url=ARK_BASE_URL,
        api_key=api_key,
    )
    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_event_loop()
    output_dir.mkdir(parents=True, exist_ok=True)

    async def process(case: dict[str, Any], idx: int, total: int) -> None:
        async with sem:
            label = f"[{idx+1}/{total}] case {case['id']}"
            if verbose:
                print(f"  → {label} ...", flush=True)
            try:
                result = await loop.run_in_executor(
                    None, extract_single, client, model, case
                )
                out_path = output_dir / f"{result['meta_id']}.json"
                out_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if verbose:
                    print(f"  ✓ {label} → {out_path.name}", flush=True)
            except Exception as e:
                print(f"  ✗ {label} failed: {e}", file=sys.stderr)

    total = len(cases)
    tasks = [process(case, i, total) for i, case in enumerate(cases)]
    await asyncio.gather(*tasks)


# ── CLI ───────────────────────────────────────────────────────────────────────

def collect_csv_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DirectorBench metadata from conversation CSVs via ARK API."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a CSV file or a directory containing CSV files.",
    )
    parser.add_argument(
        "--output", "-o", default="data/metadata",
        help="Output directory for extracted JSON files (default: data/metadata).",
    )
    parser.add_argument(
        "--api-key", "-k", default="6b177549-b3df-4549-bef5-00cefd6c30f3",
        help="ARK API key. If omitted, falls back to env var ARK_API_KEY.",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help=f"ARK model endpoint ID (default: env ARK_MODEL or {ARK_MODEL_DEFAULT}).",
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=3,
        help="Number of concurrent API calls (default: 3).",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-case progress output.",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        sys.exit("Error: provide --api-key or set the ARK_API_KEY environment variable.")

    input_path = Path(args.input)
    output_dir = Path(args.output)
    model = args.model or os.environ.get("ARK_MODEL", ARK_MODEL_DEFAULT)
    verbose = not args.quiet

    if not input_path.exists():
        sys.exit(f"Error: input path does not exist: {input_path}")

    csv_files = collect_csv_files(input_path)
    if not csv_files:
        sys.exit(f"Error: no CSV files found in {input_path}")

    all_cases: list[dict[str, Any]] = []
    for csv_path in csv_files:
        fmt = detect_format(csv_path)
        if verbose:
            print(f"Loading {csv_path.name} (Format {fmt})")
        if fmt == "A":
            cases = load_format_a(csv_path)
        else:
            cases = load_format_b(csv_path)
        print(f"  {len(cases)} cases found.")
        all_cases.extend(cases)

    if not all_cases:
        sys.exit("No cases to process.")

    print(f"\nExtracting metadata for {len(all_cases)} cases → {output_dir}/  [model: {model}]")
    asyncio.run(extract_all_async(all_cases, output_dir, args.concurrency, model, api_key, verbose))
    print("\nDone.")


if __name__ == "__main__":
    main()
