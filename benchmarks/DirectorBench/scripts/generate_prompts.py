#!/usr/bin/env python3
"""
Generate personalized video prompts by crossing metadata with user profiles.

For each (metadata × profile) pair the LLM writes a realistic user-style
video generation prompt that naturally blends:
  - selective creative requirements from the metadata
    (story arc, shot list, camera moves, audio spec)
  - the personal preferences from the profile
    (priority_weights → adjust depth of each dimension;
     hard_constraints → weave into the narration naturally)
  - user characteristics (expertise level, expression style)

Requirements for each generated prompt:
  1. Realistic – reads like a real human typed it in a chat box
  2. Natural   – matches user expertise level, no rigid language
  3. Complete  – all prompts must contain sufficient information for
                 end-to-end video generation evaluation (no follow-up)
  4. Weighted  – higher-weight dimensions get richer description;
                 lower-weight dimensions get briefer coverage
  5. Diverse   – varied opening styles, sentence structures
  6. Language  – matches the instruction language of the source metadata

Output: one JSONL record per (metadata × profile) pair.

Usage:
  python scripts/generate_prompts.py \\
      --dataset   data/metadata/dataset_original.jsonl \\
      --profiles  data/profiles.json \\
      --output    data/metadata/dataset_personalized.jsonl \\
      --api-key   <API_KEY> \\
      [--model     \\
      [--endpoint  \\
      [--office-network]  \\
      [--concurrency 4]
"""

import argparse
import asyncio
import json
import os
import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import openai

DEFAULT_ENDPOINT    = 
OFFICE_ENDPOINT     = 
DEFAULT_MODEL       = "kimi-k2.5"

# ── Prompt template ───────────────────────────────────────────────────────────

_FEW_SHOT_EXAMPLES = """\
以下六个示例基于同一个元数据（65秒浪漫故事：女孩雨中追逐男孩，三幕弧，分镜序列 tracking→push_in→orbiting→handheld，BGM soft_piano_orchestral，需唇同步），但面向不同 profile。
重点关注：①每条都是完整的视频生成指令，包含足够的信息让模型知道要生成什么 ②不同专业度的用户措辞差异很大（novice 口语化、expert 术语多）③不同关注维度的用户展开的重点不同

---【示例 A — Story-First / intermediate / narrative】---
帮我生成一段一分钟出头的雨中追逐爱情短片，情节是雨巷偶遇到误会爆发再到雨中和好，这个情感弧线的因果要清晰不能突然就原谅了。男孩转身走掉女孩愣在雨里那个停顿很重要，配上钢琴转弦乐把揪心的感觉撑起来。镜头跟着人物走就好不用太花哨，人脸前后别对不上，雨天氛围到位就行。

---【示例 B — Visual-Heavy / expert / technical】---
生成一段65秒的雨巷追逐视频，镜头设计是核心：开场 wide tracking 跟跑，15s 左右 push_in 推到女孩泪眼特写注意推速跟情绪节奏匹配，35s orbiting 环绕相拥半径从大到小收拢，收 handheld 轻微晃动。雨光漫反射要柔不能吃脸部细节高光。故事三幕正常走，BGM soft_piano 配弦乐，唇同步要精准。

---【示例 C — Casual Vlogger / novice / casual】---
帮我做一个一分钟左右的短片，雨天两个人的浪漫故事，女孩追男孩中间有误会最后和好那种。画面要自然好看，有个好听的背景音乐就行，说话的时候嘴型对得上，感觉像真的电影一样。

---【示例 D — Audio & Emotion / intermediate / emotional】---
我想生成一段一分钟的雨中追逐短片，音乐是灵魂，钢琴开始弦乐慢慢叠进来跟着情绪走，误会那段要听出紧张最后和好的时候音乐全放开让人鼻子一酸。故事就是偶遇误会和好三段，对白自然点雨声别盖住说话，画面暖色调有雨的氛围，镜头稳一些不用太炫。

---【示例 E — Creative Dreamer / novice / casual】---
做一个一分钟左右雨天追逐的爱情短片吧，画面梦幻一点不要太写实，雨滴最好有慢动作效果。故事就是女孩追男孩中间闹误会最后和好，要有好听的音乐配着，镜头角度最好有点特别，人物前后长得要一样。

---【示例 F — Detail Obsessive / expert / structured】---
生成65秒视频，三幕结构。Act 1 雨巷偶遇 wide tracking，15s push_in 推泪眼特写，Act 3 orbiting 环绕相拥收 handheld。BGM soft_piano 转弦乐，唇同步精准，雨声混在对白底层。人物外貌全程一致，雨伞物理真实，光线保持柔和的雨天漫射。
---
"""

PROMPT_TEMPLATE = """\
你的任务：模拟一位真实用户在视频生成工具（Kling / Sora / Veo 等）的输入框里打出的【视频生成指令】。

关键定位：这是一条有明确目的的指令——用户想让 AI 生成一段视频。
- 不是跟朋友闲聊，不是写创作方案，不是写导演手记
- 即使最随意的用户，打出来的也是"帮我做XX视频"这种带有指令性的内容
- 每句话都应该在传达一个具体的视频需求，不能有纯粹的感叹或闲聊

## 参考示例（注意六种风格的巨大差异）
{few_shot_examples}

---

## 当前任务

### 视频元数据（指令需要覆盖这些信息，高优先级维度展开、低优先级简要带过）
{metadata_block}

### 用户画像
{profile_block}

---

## 写作规则

### 1. 身份规则（最重要）
你不是 AI，不是导演，你就是这个用户本人在打字。
- 第一人称，口语
- 绝对禁止的词和语气："好的""我们将""让我们""以下是""这段视频将""本片""建议""整体上""综上""总结来说"
- 禁止分标题、加粗、破折号列条目
- 禁止每个维度都面面俱到地覆盖——真人不会这样写

### 1.5. 结构规则（极其重要）
视频生成指令的基本结构是：先明确说明要生成什么视频（任务说明），再描述具体需求。
这是自然的指令结构，就像人在工具输入框里打字一样。

常见的开头模式（选一种自然使用，不要每次都一样）：
- "帮我生成一段XX的视频，……"
- "我想生成一个一分钟的XX短片，……"
- "生成一段视频，内容是XX，……"
- "做一个XX的短片，要求是……"
- 或者直接："一分钟的XX视频，……"

关键：开头要让人一眼看出这是一条视频生成指令，并且知道要生成什么内容。
后面的具体需求自然展开就好，不需要刻意按维度分条也不需要刻意打乱。

### 2. 专业度规则（根据 expertise_level 严格执行）
- novice：完全不用专业术语。说"镜头跟着人跑"不说"tracking shot"，说"切换快一点"不说"whip_pan"，说"嘴型对上"不说"lip_sync"。措辞随意，可以有不确定感（"大概""好像""那种感觉你懂吧"）
- intermediate：偶尔用专业词但更多用口语描述混搭。可以说"推镜头"但不说"push_in focal racking"
- expert：自然使用专业术语，但仍是聊天口吻不是论文口吻

### 3. 信息完整性规则（极其重要）
生成的指令将直接送入视频生成模型进行端到端评测，中间不会有任何追问或补充。因此：
- 指令必须包含足够的信息，让视频生成模型能据此产出可评估的视频
- 所有维度（故事、视觉、音频、同步）都需要覆盖到，但高优先级维度展开更多、低优先级的简要带过
- 不能遗漏关键需求（如时长、故事走向、镜头风格、音乐要求等）
- novice 用户用口语表达这些需求，expert 用术语表达，但信息量要同样充分

### 4. 表达风格规则（根据 expression_style 调整语感，但所有风格都是在下指令）
- casual：口语化但目的明确。可以用"那种""吧"等语气词，但每句仍在描述一个具体需求
- emotional：用感受描述想要的效果（"让人鼻子一酸""看了会紧张"），仍是在说"我要什么效果"
- narrative：用画面描述来传达需求，像在说"我要的视频是这样的：……"
- technical：精确简洁，参数化，像在填工单但用自然语言
- precise：每个要求有明确描述，逻辑清晰
- structured：分层次，简洁干练，接近专业 brief 的电报体
注意：所有风格都必须输出信息完整的指令。区别只是措辞方式（口语vs术语、感性vs理性），不是信息量的多少。

### 5. 开头规则
{opening_directive}
开头必须让人一眼看出这是一条视频生成指令。常见模式："帮我生成一段XX视频""我想生成一个XX短片""做一个XX的片子""生成一段XX视频"等。回顾六个示例，每条的开头都明确表达了"我要生成一个什么视频"。不同 expertise_level 和 expression_style 的用户开头措辞不同（novice 更随意，expert 更简洁），但都要传达清晰的生成指令。

### 6. 篇幅规则
核心关注的维度自然占更多篇幅，低优先级的简要带过。
长度参考（这是输入框，不是文档）：
- novice 用户：150-250 字（口语描述自然会长一些）
- intermediate 用户：150-300 字
- expert 用户：100-350 字（术语简洁所以可以更短，或者因为细节多所以更长）
总之不要超过 350 字，超过就不像真人在输入框里打字了。

### 7. 文风规则
- novice/casual/emotional/narrative 用户：口语化，不用"必须""严格按照"，用"最好能""希望""要是能做到就好了"。可以有口语重复、省略主语、语气词
- expert/technical/structured/precise 用户：可以直接、干脆，甚至用短句电报体，但仍然是聊天口吻不是写文档。可以说"必须精准"但不要说"严格按照以下标准执行"
- 所有用户：hard_constraints 自然融入句子，不单独列举

### 8. 语言规则
与元数据主指令语言保持一致（中文元数据 → 中文输出）。

### 9. 质量红线（出现以下任一条即为不合格输出）
- 没有明确的任务说明（看不出是在要求生成视频）→ 不合格
- 信息不完整，缺少关键维度（如完全没提故事走向、或完全没提视觉要求）→ 不合格
- 读起来像散文、诗歌、小说片段、意识流 → 不合格
- 读起来像在跟朋友闲聊而不是在给工具下指令 → 不合格
- 有句子不承载任何视频需求信息（纯感叹、纯联想）→ 不合格
- 超过 350 字 → 不合格
- 每段开头都用"我想要""我希望" → 不合格
- 用了元数据中没有的华丽修辞（"像星辰般闪烁""如梦似幻的光影""像画一样"）→ 不合格
- novice 用户用了专业术语（tracking/push_in/orbiting/match_cut/lip_sync）→ 不合格

只输出指令文本本身，第一个字就是指令内容，不要任何开场白、解释或标题。\
"""

# ── Metadata / profile formatters ────────────────────────────────────────────

WEIGHT_LABELS = {
    "text_story_arc":   "故事弧/叙事",
    "visual_camera":    "视觉/镜头",
    "audio_emotion":    "音频/情感",
    "cross_modal_sync": "多模态同步",
}


def build_metadata_block(doc: dict) -> str:
    """Build complete metadata block with all details."""
    md = doc.get("modality_details", {})
    lines = [
        f"视频类型：{doc.get('video_type', '')}",
        f"时长：{doc.get('duration_sec', '')} 秒",
        f"主创作指令：{doc.get('main_instruction', '')}",
    ]

    text = md.get("text", {})
    arc  = text.get("story_arc", {})
    if arc:
        lines.append("\n故事弧：")
        lines.append(f"  建置（Act 1）：{arc.get('act1_setup', '')}")
        lines.append(f"  冲突（Act 2）：{arc.get('act2_conflict', '')}")
        lines.append(f"  解决（Act 3）：{arc.get('act3_resolution', '')}")

    if text.get("tone_requirements"):
        lines.append(f"整体基调：{text['tone_requirements']}")

    scripts = text.get("script", [])
    if scripts:
        lines.append("\n分镜脚本：")
        for s in scripts:
            parts = [f"镜头 {s.get('shot_id', '')} ({s.get('duration_sec', '')}s)"]
            if s.get("dialogue"):  parts.append(f"台词：「{s['dialogue']}」")
            if s.get("narration"): parts.append(f"旁白：{s['narration']}")
            lines.append("  " + "  |  ".join(parts))

    visual = md.get("visual", {})
    shots  = visual.get("shots", [])
    if shots:
        lines.append("\n视觉分镜：")
        for sh in shots:
            lines.append(
                f"  镜头 {sh.get('shot_id', '')}：{sh.get('description', '')}，"
                f"运镜 {sh.get('camera_movement', '')}，"
                f"光影 {sh.get('lighting', '')}"
            )

    if visual.get("camera_requirements"):
        lines.append(f"镜头技法：{', '.join(visual['camera_requirements'])}")
    if visual.get("consistency_requirements"):
        lines.append(f"长时序一致性：{', '.join(visual['consistency_requirements'])}")

    audio = md.get("audio", {})
    audio_parts = []
    if audio.get("dialogue"):          audio_parts.append("含对白")
    if audio.get("lip_sync"):          audio_parts.append("需精准唇同步")
    if audio.get("bgm_style"):         audio_parts.append(f"BGM：{audio['bgm_style']}")
    if audio.get("sound_effects"):     audio_parts.append(f"音效：{', '.join(audio['sound_effects'])}")
    if audio.get("tone_control"):      audio_parts.append(f"情感基调：{audio['tone_control']}")
    if audio.get("multi_language"):    audio_parts.append(f"语言：{audio['multi_language']}")
    if audio_parts:
        lines.append(f"\n音频设定：{'；'.join(audio_parts)}")

    return "\n".join(lines)


EXPERTISE_DESCRIPTIONS = {
    "novice":       "完全不懂影视制作，日常用户，不知道任何专业术语",
    "intermediate": "有一定审美和基本概念，偶尔能说出几个术语但更习惯口语描述",
    "expert":       "影视/动画从业者，熟练使用专业术语，但打字仍是聊天口吻",
}

EXPRESSION_DESCRIPTIONS = {
    "casual":      "口语化，自然随意，但信息完整",
    "emotional":   "用感受和体验描述想要的效果",
    "narrative":   "用画面和场景描述来传达需求",
    "technical":   "精确简洁，参数化，像在填工单但用自然语言",
    "precise":     "每个要求有明确描述，逻辑清晰",
    "structured":  "分层次简洁干练，接近专业 brief",
}


def build_profile_block(profile: dict) -> str:
    p       = profile.get("personalization", {})
    weights = p.get("priority_weights", {})
    taste   = p.get("user_taste", {})
    hard    = p.get("hard_constraints", [])

    expertise  = p.get("expertise_level", "intermediate")
    expr_style = p.get("expression_style", "narrative")

    # Assign explicit priority tiers based on weight value
    sorted_dims = sorted(weights.items(), key=lambda x: -x[1])
    tiers: dict[str, str] = {}
    for i, (k, v) in enumerate(sorted_dims):
        if i == 0:
            tiers[k] = "【核心关注】最多篇幅展开"
        elif v >= 0.20:
            tiers[k] = "【重要】适当展开"
        else:
            tiers[k] = "【次要】简要提及即可"

    lines = [
        f"用户类型：{profile.get('name', '')}",
        f"核心诉求：{taste.get('focus', '')}",
        "",
        f"expertise_level：{expertise}（{EXPERTISE_DESCRIPTIONS.get(expertise, '')}）",
        f"expression_style：{expr_style}（{EXPRESSION_DESCRIPTIONS.get(expr_style, '')}）",
        "",
        "各维度优先级（所有维度都需覆盖，区别在于展开程度）：",
    ]
    for k, tier in tiers.items():
        label = WEIGHT_LABELS.get(k, k)
        lines.append(f"  {label}（{int(weights[k]*100)}%）→ {tier}")

    taste_extras = {k: v for k, v in taste.items() if k != "focus"}
    if taste_extras:
        extras = "、".join(f"{k}={v}" for k, v in taste_extras.items())
        lines.append(f"\n其他口味偏好：{extras}")

    if hard:
        lines.append(f"hard_constraints（自然融入句子，不要列举）：{', '.join(hard)}")

    return "\n".join(lines)


# ── Opening style directives (randomly injected to break repetition) ─────────

_OPENING_DIRECTIVES = [
    '这次用"帮我生成一段XX的视频"这种请求式开头，然后自然展开具体需求。',
    '这次用"我想生成一段视频，里面是XX"开头，再补充细节。',
    '这次用"做一个XX的短片"这种简洁指令式开头，然后说具体要求。',
    '这次用"生成一段XX视频"直接开头，不加"帮我""请"等客气词。',
    '这次用"想做一个XX的片子"这种口语化请求开头。',
    '这次先说时长和类型（"一分钟的XX视频"），再说具体需求。',
    '这次用"能不能帮我生成XX"这种商量语气开头。',
    '这次用类比开头（"类似XX那种感觉的视频"），再说具体要什么。',
    '这次先说核心效果（"生成一段让人感觉XX的视频"），再展开。',
    '这次直接说视频内容（"一段XX的视频，要求是……"），不用前缀。',
]


# ── LLM call ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LlmCallConfig:
    """AzureOpenAI call extensions for headers/thinking options."""

    # If None/empty: generate a new UUID per request.
    tt_logid: str | None
    thinking_budget: int = 2000
    no_thinking: bool = False


def call_llm(
    client: openai.AzureOpenAI,
    model: str,
    meta_block: str,
    prof_block: str,
    cfg: LlmCallConfig,
    rng: random.Random | None = None,
) -> str:
    # Pick a random opening directive to break sentence-initial repetition
    _rng = rng or random.Random()
    opening = _rng.choice(_OPENING_DIRECTIVES)

    prompt = PROMPT_TEMPLATE.format(
        few_shot_examples=_FEW_SHOT_EXAMPLES,
        metadata_block=meta_block,
        profile_block=prof_block,
        opening_directive=opening,
    )
    log_id = (cfg.tt_logid or "").strip() or str(uuid.uuid4())
    extra_headers = {"X-TT-LOGID": log_id}

    if cfg.no_thinking:
        extra_body: dict = {
            "thinking": {
                "include_thoughts": False,
                "budget_tokens": 0,
            }
        }
    else:
        extra_body = {
            "thinking": {
                "include_thoughts": True,
                "budget_tokens": cfg.thinking_budget,
            }
        }

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=4096,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


async def process_pair(
    loop: asyncio.AbstractEventLoop,
    semaphore: asyncio.Semaphore,
    client: openai.AzureOpenAI,
    model: str,
    llm_cfg: LlmCallConfig,
    doc: dict,
    profile: dict,
    idx: int,
    total: int,
    pairing_seed: int | None = None,
    pairing_mode: str = "random",
) -> dict:
    meta_block = build_metadata_block(doc)
    prof_block = build_profile_block(profile)

    # Per-pair RNG for opening directive diversity (deterministic if seeded)
    pair_rng = random.Random(f"{pairing_seed or 0}_{idx}")

    async with semaphore:
        generated = await loop.run_in_executor(
            None, lambda: call_llm(client, model, meta_block, prof_block, llm_cfg, pair_rng)
        )

    print(
        f"  [{idx}/{total}] meta_id={doc.get('meta_id', '')} "
        f"sample_id={doc.get('sample_id', '')} × profile {profile['profile_id']} done"
    )

    out = dict(doc)
    out["pair_meta_id"]           = doc.get("meta_id")
    out["pair_profile_id"]        = profile["profile_id"]
    out["profile_id"]             = profile["profile_id"]
    out["profile_name"]           = profile.get("name", "")
    out["profile_personalization"] = profile.get("personalization", {})
    out["generated_prompt"]       = generated
    out["variant_type"]           = "personalized"
    out["pairing_mode"]           = pairing_mode
    if pairing_seed is not None:
        out["pairing_random_seed"] = pairing_seed
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    endpoint = OFFICE_ENDPOINT if args.office_network else (args.endpoint or DEFAULT_ENDPOINT)
    client = openai.AzureOpenAI(
        api_key=args.api_key,
        azure_endpoint=endpoint,
        api_version=args.api_version,
    )
    model = args.model
    llm_cfg = LlmCallConfig(
        tt_logid=args.tt_logid or os.environ.get("X_TT_LOGID") or None,
        thinking_budget=args.thinking_budget,
        no_thinking=args.no_thinking,
    )

    # Load metadata
    dataset_path = Path(args.dataset)
    docs: list[dict] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    print(f"Loaded {len(docs)} metadata records from {dataset_path}")

    # Load profiles (handles both JSON array and bare comma-separated objects)
    profiles_path = Path(args.profiles)
    raw = profiles_path.read_text(encoding="utf-8").strip()
    if not raw.startswith("["):
        raw = "[" + raw + "]"
    profiles: list[dict] = json.loads(raw)
    if not profiles:
        print("Error: profiles list is empty", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(profiles)} profiles from {profiles_path}")

    pair_mode = args.pair_mode
    paired: list[tuple[dict, dict]] = []
    pairing_seed: int | None = None

    if pair_mode == "random":
        rng = random.Random(args.seed)
        for doc in docs:
            paired.append((doc, rng.choice(profiles)))
        pairing_seed = args.seed
        seed_note = f" (seed={args.seed})" if args.seed is not None else ""
        print(
            f"Generating {len(paired)} personalized prompts "
            f"({len(docs)} metadata rows × 1 random profile each{seed_note})…\n"
        )
    elif pair_mode == "fixed":
        if not args.fixed_profile_id:
            print(
                "Error: --fixed-profile-id is required when --pair-mode fixed",
                file=sys.stderr,
            )
            sys.exit(1)
        profile_map = {p.get("profile_id"): p for p in profiles}
        fixed_profile = profile_map.get(args.fixed_profile_id)
        if fixed_profile is None:
            print(
                f"Error: fixed profile_id not found: {args.fixed_profile_id}",
                file=sys.stderr,
            )
            sys.exit(1)
        for doc in docs:
            paired.append((doc, fixed_profile))
        print(
            f"Generating {len(paired)} personalized prompts "
            f"({len(docs)} metadata rows × fixed profile {args.fixed_profile_id})…\n"
        )
    elif pair_mode == "all":
        paired = [(doc, profile) for doc in docs for profile in profiles]
        print(
            f"Generating {len(paired)} personalized prompts "
            f"({len(docs)} metadata rows × {len(profiles)} profiles) [all pairs]…\n"
        )
    else:
        print(f"Error: unsupported pair mode: {pair_mode}", file=sys.stderr)
        sys.exit(1)

    total = len(paired)

    loop      = asyncio.get_event_loop()
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        asyncio.create_task(
            process_pair(
                loop,
                semaphore,
                client,
                model,
                llm_cfg,
                doc,
                profile,
                i + 1,
                total,
                pairing_seed=pairing_seed,
                pairing_mode=pair_mode,
            )
        )
        for i, (doc, profile) in enumerate(paired)
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success, failed = 0, 0
    # Write each record as soon as it finishes to survive interruptions.
    with output_path.open("w", encoding="utf-8") as f:
        for fut in asyncio.as_completed(tasks):
            try:
                r = await fut
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                success += 1
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                failed += 1

    print(f"\nDone. Written {success} records → {output_path}")
    if failed:
        print(f"  {failed} pairs failed (see stderr for details)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate personalized video prompts from metadata × user profiles."
    )
    parser.add_argument("--dataset",     default="data/metadata/sample.jsonl",
                        help="Path to dataset_original.jsonl")
    parser.add_argument("--profiles",    default="data/profiles.jsonl",
                        help="Path to profiles.json")
    parser.add_argument("--output", "-o", default="data/instruction_original_personalized.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--api-key",       default=os.environ.get("API_KEY", ""),
                        help="API key")
    parser.add_argument("--endpoint",      default=None,
                        help=f"Azure endpoint URL (default: {DEFAULT_ENDPOINT})")
    parser.add_argument("--office-network", action="store_true",
                        help=f"Use office-network domain ({OFFICE_ENDPOINT})")
    parser.add_argument("--api-version",   default=DEFAULT_API_VERSION,
                        help=f"Azure API version (default: {DEFAULT_API_VERSION})")
    parser.add_argument("--model",         default=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--concurrency",   type=int, default=4,
                        help="Max parallel LLM calls (default: 4)")
    parser.add_argument(
        "--tt-logid",
        default=os.environ.get("X_TT_LOGID", ""),
        help="Value for X-TT-LOGID header (or env X_TT_LOGID). Empty -> UUID per request.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=2000,
        help="thinking.budget_tokens in extra_body (default: 2000). Ignored with --no-thinking.",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking by sending budget_tokens=0.",
    )
    parser.add_argument(
        "--pair-mode",
        choices=["random", "fixed", "all"],
        default="random",
        help="Profile pairing mode.",
    )
    parser.add_argument(
        "--fixed-profile-id",
        default=None,
        help="Profile id used in fixed mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for random mode.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set API_KEY env var", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
