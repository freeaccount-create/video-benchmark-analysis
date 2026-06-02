#!/usr/bin/env python3
"""
Assign sample IDs and generate multilingual variants of metadata JSONs.

Two independent capability dimensions are covered:

  Dim-1  instruction_lang  (指令理解能力)
         The user's prompt fields are translated to a foreign language.
         Tests whether the model can understand instructions written in that language.
         Translated fields: main_instruction, story_arc, shot descriptions,
                            tone_requirements, bgm_style, sound_effects, tone_control.

  Dim-2  content_lang  (内容创作能力)
         The instruction stays in Chinese but specifies that the video's
         dialogue / narration must be delivered in a target language.
         Changed fields: script[].dialogue/narration translated to target lang,
                         audio.multi_language updated to target lang,
                         main_instruction appended with a language requirement note.
         Only applicable to cases that have actual dialogue/narration content.

LLM judges suitability for each dimension independently, then selects languages.
All variants (zh original + instruction variants + content variants) share sample_id.

Output filename convention:
  <meta_id>_zh.json              original
  <meta_id>_instr_<lang>.json    instruction-language variant
  <meta_id>_content_<lang>.json  content-language variant

Usage:
  python scripts/add_multilingual.py \\
      --input  data/metadata/v2 \\
      --output data/metadata/v3 \\
      --api-key <key> \\
      [--model <endpoint>] [--concurrency 4]
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

ARK_BASE_URL      = 
ARK_MODEL_DEFAULT = 

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "fr": "Français",
    "es": "Español",
    "de": "Deutsch",
    "ar": "العربية",
}

# ── Prompt ────────────────────────────────────────────────────────────────────

MULTILINGUAL_PROMPT = """\
你是一个视频生成数据集的多语言化专家。请对以下视频生成元数据完成两个维度的多语言扩充。

## 输入 JSON
{json_content}

---

## 维度一：多语言指令理解
将用户的创作指令翻译成其他语言，用于测试模型理解多语言指令的能力。

### 1-A 适配性判断
判断该 case 的指令是否适合翻译成其他语言（文化通用性）。
不适合的情况：
- 依赖中文双关/谐音/文字游戏
- 强依赖中国特定文化符号且在其他文化无对应
- 内容本身是某种语言的语言教学

### 1-B 语言选择与翻译（仅适合时执行）
从以下语言中选 2～4 种最适合的目标语言：
{language_options}

选择标准：
- 日系动漫/古风等题材优先日语、韩语
- 通用内容（自然、情感、运动）选英语 + 1～2 种主流语言
- 商业营销选英语 + 目标市场语言

翻译字段（保持 JSON 结构，仅翻译字符串值）：
- main_instruction
- modality_details.text.story_arc 的三个子字段（非 null 时）
- modality_details.text.script[] 中每项的 dialogue 和 narration（非空字符串时）
- modality_details.text.tone_requirements（非 null 时）
- modality_details.visual.shots[] 中每项的 description
- modality_details.audio.bgm_style（非 null 时）
- modality_details.audio.sound_effects[] 中每个音效名称（非空时）
- modality_details.audio.tone_control（非 null 时）
- modality_details.audio.multi_language：若原值为具体语言代码（如 "zh"），改为目标语言代码；若为 null 则保持 null

翻译要求：
- 忠实原意，影视专业术语（tracking / push-in 等）不翻译
- 对白须符合目标语言口语习惯
- video_type 保留中文原值不翻译

---

## 维度二：多语言视频内容创作
指令保持中文，但要求视频中的对白/旁白使用目标语言，用于测试模型创作多语言视频内容的能力。

### 2-A 适配性判断
判断该 case 是否适合生成"内容语言"变体。
适合条件（满足任一即可）：
- script 中有实际对白（dialogue 非空字符串）
- script 中有实际旁白（narration 非空字符串）
- audio.dialogue 为 true

不适合条件：
- 纯视觉/纯音乐类，完全没有人声内容
- 对白内容高度绑定某种语言（如绕口令、古诗吟诵）

### 2-B 语言选择与内容变体（仅适合时执行）
从以下语言中选 1～3 种目标语言：
{language_options}

每种语言生成一个"内容语言"变体，变化如下：
1. main_instruction：在原中文指令末尾追加一句要求（如"视频中的对白和旁白请使用英语"），不替换原指令
2. modality_details.text.script[] 中每项的 dialogue 和 narration：翻译为目标语言
3. modality_details.audio.multi_language：更新为目标语言代码
4. 其余所有字段保持与原版完全一致

---

## 输出格式（严格 JSON，无注释，无 markdown 代码块）
{{
  "dim1": {{
    "suitable": <true | false>,
    "reason": "<不适合原因 | null>",
    "selected_languages": ["<lang_code>"],
    "translations": {{
      "<lang_code>": {{
        "main_instruction": "<翻译值>",
        "modality_details": {{
          "text": {{
            "story_arc": {{
              "act1_setup": "<翻译值 | null>",
              "act2_conflict": "<翻译值 | null>",
              "act3_resolution": "<翻译值 | null>"
            }},
            "script": [
              {{"shot_id": <原值>, "duration_sec": <原值>, "dialogue": "<翻译值>", "narration": "<翻译值>"}}
            ],
            "tone_requirements": "<翻译值 | null>"
          }},
          "visual": {{
            "shots": [
              {{"shot_id": <原值>, "description": "<翻译值>", "camera_movement": <原值>, "lighting": <原值>}}
            ]
          }},
          "audio": {{
            "bgm_style": "<翻译值 | null>",
            "sound_effects": ["<翻译值>"],
            "tone_control": "<翻译值 | null>",
            "multi_language": "<目标语言代码 | null>"
          }}
        }}
      }}
    }}
  }},
  "dim2": {{
    "suitable": <true | false>,
    "reason": "<不适合原因 | null>",
    "selected_languages": ["<lang_code>"],
    "content_variants": {{
      "<lang_code>": {{
        "main_instruction": "<原中文指令 + 末尾追加语言要求>",
        "script": [
          {{"shot_id": <原值>, "duration_sec": <原值>, "dialogue": "<目标语言翻译>", "narration": "<目标语言翻译>"}}
        ],
        "audio_multi_language": "<目标语言代码>"
      }}
    }}
  }}
}}

当某维度 suitable 为 false 时，selected_languages 填 []，对应翻译/变体字段填 {{}}。
只输出上述 JSON 对象，不要任何其他文字。\
"""

# ── Apply helpers ─────────────────────────────────────────────────────────────

def apply_instruction_translation(
    original: dict[str, Any], trans: dict[str, Any]
) -> dict[str, Any]:
    """Merge dim-1 translation into a copy of original."""
    result = copy.deepcopy(original)
    result["main_instruction"] = trans.get("main_instruction", result["main_instruction"])

    trans_text = trans.get("modality_details", {}).get("text", {})
    orig_text  = result["modality_details"]["text"]

    if trans_text.get("story_arc"):
        for key in ("act1_setup", "act2_conflict", "act3_resolution"):
            val = trans_text["story_arc"].get(key)
            if val is not None:
                orig_text["story_arc"][key] = val

    if trans_text.get("script"):
        trans_map = {item["shot_id"]: item for item in trans_text["script"]}
        for item in orig_text.get("script", []):
            sid = item["shot_id"]
            if sid in trans_map:
                item["dialogue"]  = trans_map[sid].get("dialogue",  item.get("dialogue",  ""))
                item["narration"] = trans_map[sid].get("narration", item.get("narration", ""))

    if trans_text.get("tone_requirements") is not None:
        orig_text["tone_requirements"] = trans_text["tone_requirements"]

    trans_shots = trans.get("modality_details", {}).get("visual", {}).get("shots", [])
    if trans_shots:
        shots_map = {s["shot_id"]: s for s in trans_shots}
        for shot in result["modality_details"]["visual"].get("shots", []):
            sid = shot["shot_id"]
            if sid in shots_map:
                shot["description"] = shots_map[sid].get("description", shot["description"])

    trans_audio = trans.get("modality_details", {}).get("audio", {})
    orig_audio  = result["modality_details"]["audio"]
    if trans_audio.get("bgm_style") is not None:
        orig_audio["bgm_style"] = trans_audio["bgm_style"]
    if trans_audio.get("sound_effects"):
        orig_audio["sound_effects"] = trans_audio["sound_effects"]
    if trans_audio.get("tone_control") is not None:
        orig_audio["tone_control"] = trans_audio["tone_control"]
    if "multi_language" in trans_audio:
        orig_audio["multi_language"] = trans_audio["multi_language"]

    return result


def apply_content_variant(
    original: dict[str, Any], variant: dict[str, Any], lang: str
) -> dict[str, Any]:
    """Merge dim-2 content variant into a copy of original."""
    result = copy.deepcopy(original)
    result["main_instruction"] = variant.get("main_instruction", result["main_instruction"])

    if variant.get("script"):
        script_map = {item["shot_id"]: item for item in variant["script"]}
        for item in result["modality_details"]["text"].get("script", []):
            sid = item["shot_id"]
            if sid in script_map:
                item["dialogue"]  = script_map[sid].get("dialogue",  item.get("dialogue",  ""))
                item["narration"] = script_map[sid].get("narration", item.get("narration", ""))

    result["modality_details"]["audio"]["multi_language"] = variant.get(
        "audio_multi_language", lang
    )
    return result


# ── Build all output variants ─────────────────────────────────────────────────

def build_outputs(
    original: dict[str, Any],
    llm_result: dict[str, Any],
    sample_id: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    """
    Returns list of (variant_suffix, variant_type, json_dict).
    variant_suffix: "zh" / "instr_en" / "content_ja" etc.
    variant_type:   "original" / "instruction_lang" / "content_lang"
    """
    outputs: list[tuple[str, str, dict[str, Any]]] = []

    # ── Original ──
    zh_doc = copy.deepcopy(original)
    zh_doc["sample_id"]    = sample_id
    zh_doc["language"]     = "zh"
    zh_doc["variant_type"] = "original"
    outputs.append(("zh", "original", zh_doc))

    # ── Dim-1: instruction language variants ──
    dim1 = llm_result.get("dim1", {})
    if dim1.get("suitable"):
        for lang in dim1.get("selected_languages", []):
            trans = dim1.get("translations", {}).get(lang)
            if not trans:
                continue
            doc = apply_instruction_translation(original, trans)
            doc["sample_id"]    = sample_id
            doc["language"]     = lang
            doc["variant_type"] = "instruction_lang"
            outputs.append((f"instr_{lang}", "instruction_lang", doc))

    # ── Dim-2: content language variants ──
    dim2 = llm_result.get("dim2", {})
    if dim2.get("suitable"):
        for lang in dim2.get("selected_languages", []):
            variant = dim2.get("content_variants", {}).get(lang)
            if not variant:
                continue
            doc = apply_content_variant(original, variant, lang)
            doc["sample_id"]    = sample_id
            doc["language"]     = "zh"           # instruction still in Chinese
            doc["content_lang"] = lang           # target language for video content
            doc["variant_type"] = "content_lang"
            outputs.append((f"content_{lang}", "content_lang", doc))

    return outputs


# ── LLM call ─────────────────────────────────────────────────────────────────

def process_single(
    client: OpenAI, model: str, path: Path, sample_id: str
) -> list[tuple[str, str, dict[str, Any]]]:
    original = json.loads(path.read_text(encoding="utf-8"))
    lang_options = "\n".join(f"  {code}  {name}" for code, name in SUPPORTED_LANGUAGES.items())

    prompt = MULTILINGUAL_PROMPT.format(
        json_content=json.dumps(original, ensure_ascii=False, indent=2),
        language_options=lang_options,
    )

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )

    text = response.output_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    llm_result = json.loads(text)
    return build_outputs(original, llm_result, sample_id)


# ── Async batch ───────────────────────────────────────────────────────────────

async def process_all_async(
    paths: list[Path], client: OpenAI, model: str, concurrency: int, verbose: bool
) -> list[tuple[Path, list[tuple[str, str, dict[str, Any]]]]]:
    sem  = asyncio.Semaphore(concurrency)
    loop = asyncio.get_event_loop()
    results: list[Any] = [None] * len(paths)

    async def run(i: int, path: Path, sample_id: str) -> None:
        async with sem:
            label = f"[{i+1}/{len(paths)}] {path.name}"
            if verbose:
                print(f"  → {label}", flush=True)
            try:
                variants = await loop.run_in_executor(
                    None, process_single, client, model, path, sample_id
                )
                results[i] = (path, variants)
                if verbose:
                    summary = ", ".join(
                        f"{suffix}({vtype})" for suffix, vtype, _ in variants
                    )
                    print(f"  ✓ {sample_id}  {summary}", flush=True)
            except Exception as e:
                print(f"  ✗ error {label}: {e}", file=sys.stderr)
                results[i] = (path, [])

    await asyncio.gather(*[
        run(i, path, f"DB_{i+1:03d}") for i, path in enumerate(paths)
    ])
    return [r for r in results if r is not None]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign sample IDs and generate multilingual variants (instruction + content)."
    )
    parser.add_argument("--input",  "-i", default="data/metadata/v2")
    parser.add_argument("--output", "-o", default="data/metadata/v3")
    parser.add_argument("--api-key", "-k", default="6b177549-b3df-4549-bef5-00cefd6c30f3",
                        help="ARK API key. Falls back to env var ARK_API_KEY.")
    parser.add_argument("--model", "-m", default=None,
                        help=f"ARK model endpoint (default: env ARK_MODEL or {ARK_MODEL_DEFAULT}).")
    parser.add_argument("--concurrency", "-c", type=int, default=4)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        sys.exit("Error: provide --api-key or set ARK_API_KEY.")

    model      = args.model or os.environ.get("ARK_MODEL", ARK_MODEL_DEFAULT)
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    verbose    = not args.quiet

    if not input_dir.is_dir():
        sys.exit(f"Error: input directory not found: {input_dir}")
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        sys.exit(f"Error: no JSON files found in {input_dir}")

    print(f"Found {len(paths)} files in {input_dir}")
    print(f"Model: {model}  |  Concurrency: {args.concurrency}\n")

    client = OpenAI(base_url=ARK_BASE_URL, api_key=api_key)
    all_results = asyncio.run(
        process_all_async(paths, client, model, args.concurrency, verbose)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    type_counter: dict[str, int] = {}
    total = 0

    for source_path, variants in all_results:
        stem = source_path.stem
        for suffix, vtype, doc in variants:
            out_name = f"{stem}_{suffix}.json"
            (output_dir / out_name).write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            type_counter[vtype] = type_counter.get(vtype, 0) + 1
            total += 1

    print(f"\nOutput: {total} files → {output_dir}/")
    print("Breakdown by variant_type:")
    for vtype, count in sorted(type_counter.items()):
        print(f"  {vtype}: {count}")


if __name__ == "__main__":
    main()
