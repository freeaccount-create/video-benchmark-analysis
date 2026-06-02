# ScriptAgent / The Script is All You Need —— 对白→电影级长视频的智能体框架与其评测

> 论文/项目：The Script is All You Need（arXiv 2601.17737）· 源码：[Tencent/digitalhuman · ScriptAgent](https://github.com/Tencent/digitalhuman)（`ScriptAgent/`）· 模型 [🤗 XD-MU/ScriptAgent](https://huggingface.co/XD-MU/ScriptAgent) · 项目页 [xd-mu.github.io/ScriptIsAllYouNeed](https://xd-mu.github.io/ScriptIsAllYouNeed/) · 评测基准 **ScriptBench**。
>
> 本文聚焦其 **CriticAgent 评测流程**（剧本评测 + 视频评测），`file:line` 引用指向本仓库 `benchmarks/ScriptAgent/code/` 下的真实源码。

## 1. 数据集由来

「The Script is All You Need」的核心论点：**把「长视频生成」这个难问题，归约为「先把粗对白写成专业拍摄剧本，再让剧本驱动逐镜头生成」**。三个智能体串成一条流水线：

1. **ScriptAgent**（GRPO/ms-swift 训练，HF `XD-MU/ScriptAgent`）：粗对白 → 结构化四段式拍摄剧本【人物描述】【场景描述】【角色站位】【对白】。
2. **DirectorAgent**（`code/director_agent.py`）：编排 Sora2/VEO3.1/Kling2.5/Wan2.5 等生成模型逐节点生成镜头，靠**末帧抽取→下一镜头参考图**维持跨镜头连贯，moviepy 拼接成片。
3. **CriticAgent**（`code/critic_agent_script.py` + `code/critic_agent_video.py`）：分别评**剧本质量**与**视频生成保真度**，每类都给**主观分（LLM 评判）+ 客观指标（CLIP/VSA/FVD）**。

评测基准 **ScriptBench** 即由这两个 Critic 的输出（剧本分 + 视频分）组成——它要回答的不是「单个 2s 片段好不好」，而是「对白→专业剧本→分钟级多镜头成片」整条链路每一环的质量。被评的「剧本」与「视频」由 DirectorAgent 批量产出（`run.sh` 读 `test_responses.jsonl` 逐条生成，并写出「成片→对白」映射文件 `video_dialogues.jsonl`，`director_agent.py:654-663`）——该映射正是视频评测的输入。

## 2. 原始数据格式

**被评剧本**（README「Script Format」四段式真实样例）：

```
【Character Description】
Alice: A young woman with long brown hair, wearing a blue dress.
Bob: An elderly man with white beard, in formal suit.
【Scene Description】
A sunny afternoon in a beautiful garden with blooming flowers.
【Character Positions】
1. Alice stands on the left, Bob on the right
2. Both move to the center
【Dialogue】
1. Alice: "What a beautiful day!" (smiling and looking around)
2. Bob: "Indeed, reminds me of my youth." (nostalgic expression)
```

`director_agent.py` 用 `SECTION_PATTERN = re.compile(r"【([^】]+)】：?")`（`:456`）切段，`extract_story_components()`（`:522`）解析成 `StoryComponents`：`nodes`（对白节点）、`characters`、`scene`、`station_nodes`（站位）、`time_spans`（从 `[Xseconds-Yseconds]` 抽取）。

**剧本评测输入**（`evaluate_from_files` 读取，`critic_agent_script.py:458`）：生成剧本来自 `scripts_jsonl` 每行 `data["response"]`（`:479-489`），源对白来自 `dialogues_json` 每项 `item["input"]`（`:492-504`）。

**视频评测输入**（`evaluate_from_folder` 读取，`critic_agent_video.py:1556`）：`mapping_jsonl` 每行一个「成片→剧本」映射（`:1576-1587`）：

```json
{"sora2-pro_001.mp4": "【Scene Description】sunny garden … 【Dialogue】1. Alice: 'What a beautiful day!' …"}
```

源码结构（`benchmarks/ScriptAgent/`）：

```
code/
  critic_agent_script.py   # ★ 剧本评测（4 维主观分，Gemini 2.5 Pro）
  critic_agent_video.py    # ★ 视频评测（4 维主观分 + CLIP/VSA/FVD 客观指标）
  director_agent.py        # 生成管线：剧本解析→逐节点生成→末帧续帧→拼接
  run.sh                   # 批量生成示例
first_frame_list/*.png     # 20 张统一首帧（跨模型共享初始帧）
figures/overview.png
```

## 3. 完整打分流程（pipeline）

### 3.1 剧本评测（`critic_agent_script.py`）

四个维度，0.0–5.0 **连续小数**（`SCRIPT_EVALUATION_PROMPT` `:226-337`）：

| 维度 | 含义 |
|---|---|
| **Format Compliance** | 四段【DIALOGUE】【CHARACTER PROFILES】【SCENE DESCRIPTION】【BLOCKING】是否齐、时间码、运镜/景别标注 |
| **Shot Division Rationality** | 分镜是否贴合叙事节拍与情绪转折，不过碎/过长 |
| **Content Completeness** | 是否补足源对白缺失的可拍摄视觉信息（场景、动作、运镜） |
| **Narrative Coherence** | 镜头序列逻辑是否连贯、与对白上下文是否吻合 |

提示词写入校准锚点（3.0=可用有瑕 / 4.0–4.4=良 / 4.5–4.9=优 / 5.0=近乎完美，`:311-320`），并要求**只返回 JSON**（`:322-337`）。`ScriptEvaluator`（`:340`）经 `DistillInterface`（`:44`）调 Gemini 2.5 Pro（HMAC-SHA1 签名 `get_simple_auth` `:65-75`）。`evaluate()`（`:367`）把 `source_dialogue`+`generated_script` 填进提示发出，再做**三级 JSON 解析兜底**：直接 `json.loads` → 正则抽最长 `{...}` → 首尾大括号截取（`:411-448`）。批量入口 `evaluate_from_files()`（`:458`）按 `min(len)` 配对逐条评，每 10 条存中间结果，`_calculate_average_scores()`（`:626`）对四维求均值写汇总 JSON。

### 3.2 视频评测（`critic_agent_video.py`）

**主观分 + 客观指标**双轨，`VideoEvaluator.evaluate()`（`:1497`）合并进一个 dict。

**主观四维**（`VIDEO_EVALUATION_PROMPT` `:95-148`，对视频+音频直接打 0–5）：

| 维度 | 判据 |
|---|---|
| **Audio-Visual Synchronization** | 爆炸/脚步/手势等视觉事件是否对齐音频时间戳 |
| **Emotional Consistency** | 光影/调色/表情是否匹配剧本情绪强度 |
| **Rhythm Coordination** | 视觉运动节奏（剪切快慢）是否与语音/音频律动协调 |
| **Voice-Lip Sync** | 有人说话时口型与音轨是否同步 |

两个后端二选一（`:1466-1484`）：**GeminiVideoEvaluator**（`:979`）把视频 base64 内联 `inline_data`，>10MB 先 ffmpeg 压到 10MB（`_compress_video` `:1006`），请求带 `videoMetadata.fps=30`、`audioTimestamp=True`、`thinkingBudget=-1`（`:322-335`），同样三级 JSON 兜底（`:1190-1228`）并校验四分数齐全（`:1232-1243`）；**QwenVideoEvaluator**（`:1286`）本地 `Qwen3-Omni-30B-A3B-Instruct`，`use_audio_in_video=True` 把视频里的音频一并喂入（`:1369-1383`）。

**客观三指标**（`VideoMetricsCalculator` `:541`）：

- **CLIP**（`calculate_clip_score` `:580`）：ViT-L/14（失败回退 ViT-B/32，`:556-561`）。视频均匀采样 16–32 帧（`:602-609`），剧本按句切分（最多 10 句，`:622-627`），算**逐帧 max 句相似度再求均值**（`:643-647`），非线性映射到 0–100（`:652`）。
- **VSA**（Video Semantic Accuracy，`calculate_vsa_score` `:661`）：`0.7*CLIP + 0.3*motion_quality*100`（`:718`），其中 `motion_quality = 1/(1+std/mean)` 来自相邻帧 **Farnebäck 光流**幅值的均值/方差——理想是「适度运动、低方差」（`:702-715`）。
- **FVD**（Fréchet Video Distance，`calculate_fvd_score` `:845`）：简化 **I3D**（`class I3D` `:490`）抽 16 帧 clip 特征（`_extract_i3d_features` `:734`）。有参考视频算两高斯的 Fréchet 距离（`_calculate_frechet_distance` `:819`）；**无参考**退化为特征统计质量估计（一致性 0.4 + 时序平滑 0.3 + 激活强度 0.3，映射到 FVD∈[0,30]，`:898-943`）——越低越好。

批量入口 `evaluate_from_folder()`（`:1556`）对文件夹每个视频 `evaluate()`，每 5 个存中间结果，`_calculate_average_scores`（四维主观）+ `_calculate_average_metrics`（CLIP/VSA/FVD）写汇总并打印表（`:1717-1741`）。

## 4. 用到的模型

| 角色 | 实现 | 模型/后端 |
|---|---|---|
| ScriptAgent | HF `XD-MU/ScriptAgent`（ms-swift GRPO） | 对白→剧本 |
| DirectorAgent | `director_agent.py` | Sora2-pro/Sora2、VEO3.1/-fast、Kling、Wan2.5、ViduQ2、Jimeng |
| Critic-Script | `critic_agent_script.py` | Gemini 2.5 Pro（可换 gpt-4） |
| Critic-Video 主观 | `critic_agent_video.py` | Gemini 2.5 Pro / Qwen3-Omni-30B |
| Critic-Video 客观 | 同上 | CLIP ViT-L/14、光流(VSA)、I3D(FVD) |

致谢明确借鉴 **VBench**（视频评测指标）、**LLaMA-Factory**（SFT）、**ms-swift**（GRPO）。

## 5. 实际数据案例全过程

**① 剧本评测一条样本**——以 README 的 Garden 剧本为被评对象，源对白即其【Dialogue】三句的粗粒度版本（`evaluate()` `:367-456`）：

```
source_dialogue = "Alice: 今天天气真好！ / Bob: 是啊，让我想起年轻时。 / Alice: 多讲讲！"
generated_script = "【Character Description】Alice: long brown hair, blue dress …
                    【Scene Description】sunny garden … 【Character Positions】… 【Dialogue】…"
```

1. `SCRIPT_EVALUATION_PROMPT.format(...)` 拼提示（`:381-384`），粗估 token，>30k 告警（`:392-394`）。
2. `client.request(model="gemini-2.5-pro", content_payload=prompt, temperature=0.3)`（`:397-401`），string 载荷被包成 `[{"type":"text","value":prompt}]`（`:105-110`）。
3. 四段齐全、时间码/运镜在【Dialogue】、站位明确 → 按锚点落「优」档，真实形态返回：

```json
{ "Format Compliance": 4.6, "Shot Division Rationality": 4.2,
  "Content Completeness": 4.4, "Narrative Coherence": 4.3,
  "Reasoning": { "Format Compliance": "All four required sections present with time codes and shot types.",
    "Narrative Coherence": "Smooth left→center→bench progression matches the conversation." } }
```

4. `json.loads` 成功直接返回（`:413-415`）；批量模式补 `entry_number/dialogue_preview/script_preview`（`:533-535`），逐维打印 `✓ Format Compliance: 4.60/5.0 …`，进 `_calculate_average_scores()` 均值累加（`:637-644`），汇总进 `script_eval.json`。

**② 视频评测一条样本**——输入来自 `video_dialogues.jsonl` 一行 `{"sora2-pro_001.mp4": "…剧本…"}`，`evaluate(script, "output_story/sora2-pro/final_video/sora2-pro_001.mp4")`（`:1497-1554`）：

1. **主观分**：Gemini 后端将 7s 视频内联 base64，提示嵌入剧本，同时看画面+听音轨返回：

```json
{ "Audio-Visual Synchronization": 4.0, "Emotional Consistency": 4.0,
  "Rhythm Coordination": 3.5, "Voice-Lip Sync": 3.0,
  "Reasoning": { "Voice-Lip Sync": "Lips roughly track 'What a beautiful day' with slight lag." } }
```

2. **客观指标**（`:1531-1544`）：CLIP=16 帧 vs 剧本各句 max 相似度均值→映射后约 **74.x**；VSA=`0.7*74+0.3*motion*100`，花园镜头运动平缓、方差低→motion_quality 高→约 **72.x**；FVD=无参考集走质量估计分支→约 **6.x**（越低越好）。合并进 `result["objective_metrics"]={"CLIP":74.x,"VSA":72.x,"FVD":6.x}`。
3. 带 `video_name/backend` 进 `results`（`:1607-1609`），汇总到 `video_eval_gemini.json`，控制台打印 `Audio-Visual Synchronization: 4.00/5.0 … CLIP: 74.xx …`。

> **端到端回放**：粗对白（Alice/Bob 三句）→ **ScriptAgent** 写成四段式剧本 → 作为 `{"index":1,"response":"…"}` 进 `test_responses.jsonl` → **DirectorAgent** 逐节点喂 sora2/veo3.1、末帧续帧保连贯、拼成 `sora2-pro_001.mp4` 并写 `video_dialogues.jsonl` → **Critic-Script** 拿「源对白 vs 生成剧本」给四维主观分；**Critic-Video** 拿「剧本 vs 成片」给四维主观分 + CLIP/VSA/FVD → 两份 JSON 即 ScriptBench 成绩。

## 6. 指标公式速查表（简介·模型·公式·参数）

> 双轨：主观四维由 LLM 评判（0–5），客观三指标可复现计算。

| 指标 | 简介 | 模型 | 计算公式 + 参数说明 |
|---|---|---|---|
| **剧本主观四维** | 剧本作为拍摄蓝图的完备度 | Gemini 2.5 Pro | 0.0–5.0 连续分，按锚点（3=可用/4=良/4.5=优/5=完美）；维度=Format/ShotDivision/Completeness/Coherence ·· `critic_agent_script.py:226-337` |
| **视频主观四维** | 音画同步/情绪/节奏/口型 | Gemini / Qwen3-Omni | 0–5，模型同时看画面+听音轨；维度=AV-Sync/Emotion/Rhythm/Lip-Sync ·· `critic_agent_video.py:95-148` |
| **CLIP** | 帧–剧本语义一致 | CLIP ViT-L/14 | 16–32 帧 × ≤10 句，逐帧取 max 句相似度再均值，非线性映射到 0–100 ·· `:643-652` |
| **VSA** | 语义+运动质量融合 | CLIP + Farnebäck 光流 | $0.7\cdot\text{CLIP}+0.3\cdot mq\cdot100$，$mq=\dfrac{1}{1+\text{std}/\text{mean}}$（相邻帧光流幅值）·· `:702-718` |
| **FVD** | 与参考分布距离（越低越好） | 简化 I3D | 有参考：两高斯 Fréchet 距离；无参考：一致性0.4+时序平滑0.3+激活0.3→FVD∈[0,30] ·· `:819/898-943` |

**参数说明**：①剧本 Critic 不在乎文采、只在乎「四段是否齐、时间码/运镜/站位是否给足、能否直接交给 DirectorAgent」；②视频双轨——主观抓需要理解力的判断（LLM 强项），客观给可复现数值底座；③VSA 把 CLIP 语义与光流运动质量融合；④工程鲁棒性：三级 JSON 兜底、>10MB 自动压缩、内容审核错误时删节点改剧本继续、断点续生成。

---
**一句话定位**：ScriptAgent/「The Script is All You Need」= 把长视频生成归约为「对白→专业剧本→逐镜头生成」的三智能体闭环（写/拍/评），其 **CriticAgent** 用「主观 LLM 四维 + 客观 CLIP/VSA/FVD」双轨同时评剧本与成片，组成 **ScriptBench**。
