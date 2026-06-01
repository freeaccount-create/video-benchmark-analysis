# SFD（Short Film Dataset）—— 故事级长视频理解基准

> 论文：[Long Story Short: Story-level Video Understanding from 20K Short Films](https://arxiv.org/abs/2406.10221)（Ghermi, Wang, Kalogeiton, Laptev；LIX, École Polytechnique / IP Paris + MBZUAI，2024，2025-01 修订）
> 项目页：https://ridouaneg.github.io/sf20k.html · 数据集：[HuggingFace `rghermi/sf20k`](https://huggingface.co/datasets/rghermi/sf20k) · [Papers with Code: SFD](https://paperswithcode.com/dataset/sfd)

## ⚠️ 关于命名（更正旧版）

本仓库早期把 SFD 误指为 `LittlePey/SFD`（CVPR 2022 的 LiDAR 3D 检测方法），那是**完全错误**的链接。真正的 **SFD = Short Film Dataset**：一个聚焦**长篇幅、多类型故事**的**视频理解评测基准**（不是方法）。论文初版以 1,078 部短片、4,885 道题提出 SFD；正式发布版扩展为 **SF20K（Short-Films 20K）**，规模 **20,143 部业余短片、约 3,582 小时**（平均每片 ~11–13 分钟）。下文以发布版 SF20K 为准。

## 1. 数据集由来

现有电影数据集要么任务偏短时、要么视频不公开、要么因字幕/剧情早被 LLM 预训练吸收而存在**数据泄漏**。SFD/SF20K 为此从 **YouTube / Vimeo** 收集**公开的业余短片**：

- **长篇幅**：平均 ~11–13 分钟，远长于多数视频 QA 数据集，需跨越完整叙事做长时推理。
- **多类型**：横跨剧情(narrative fiction)、纪录(documentary)、动画(animation) 等风格，以及动作/剧情/喜剧/恐怖等题材，视觉风格与叙事方式多样。
- **低泄漏**：业余短片极少被 LLM 见过，缓解了商业电影数据集的泄漏问题。
- 每部短片配一句高层概述 **logline**，并据其衍生问答。

目标：评测模型能否在**整段长视频**上理解人物、设定、关键事件与主题，做**故事级**（而非片段级）的长时推理。

## 2. 原始数据格式

数据托管在 HuggingFace `rghermi/sf20k`，**4 个 split**：`train` / `test` / `test_silent` / `test_expert`（silent=无字幕/对白线索，expert=人工精校的高难子集）。每行就是**一道题**，真实列结构（HF schema，逐字）：

| 列 | 类型 | 说明 |
|----|------|------|
| `question_id` | string | 题 ID，如 `z9HVSEot-O8_00`（`{video_id}_{序号}`） |
| `video_id` | string | 短片 ID（即 YouTube id） |
| `question` | string | 问题文本 |
| `answer` | string | **标准答案文本**（开放式 OEQA 用） |
| `option_0` … `option_4` | string | MCQA 的 **5 个选项**（A–E） |
| `correct_answer` | float64 | 正确选项的**索引**（0–4） |
| `correct_letter` | string | 正确选项**字母**（A–E） |
| `video_url` | string | 短片地址（YouTube/Vimeo），视频本体按需自行下载 |

**真实样本（test split，逐字摘录）**
```json
{
  "question_id": "z9HVSEot-O8_00",
  "video_id": "z9HVSEot-O8",
  "question": "What prompts Nina to move to the city?",
  "answer": "The synopsis does not specify why Nina moves to the city.",
  "option_0": "Nina moves to the city to pursue a career in fashion design.",
  "option_1": "The synopsis does not specify why Nina moves to the city.",
  "option_2": "Nina moves to the city to escape a troubled family situation.",
  "option_3": "Nina moves to the city to study architecture at a prestigious university.",
  "option_4": "Nina moves to the city to attend a specialized cooking school.",
  "correct_answer": 1.0,
  "correct_letter": "B",
  "video_url": "https://www.youtube.com/watch?v=z9HVSEot-O8"
}
```
> 注意「正确答案是『概述里没说』」这类**干扰项**：问答由 LLM 从 logline/synopsis 生成、再经人工筛选与改写，专门构造有挑战性的 distractor。

## 3. 任务与打分流程

两类故事级 videoQA 任务：

- **MCQA（Multiple-Choice QA）**：5 选 1，模型选一个字母，与 `correct_letter` 比对。
- **OEQA（Open-Ended QA）**：自由生成答案文本，与标准 `answer` 比较是否语义正确（**LLM-as-judge** 判对错）。

**输入模态**可分三种设定，用来拆解"看懂"还是"读懂"：
1. **Transcript-only**：只给字幕/ASR 文本 → LLM 答题；
2. **Vision-only**（≈ `test_silent`）：只给画面帧 → 视频模型答题；
3. **Multimodal**：画面 + 字幕一起。

**核心发现**：① 字幕里信号很强，Transcript+LLM 可逼近人类；② 仅用视觉时，现有视频模型**远低于人类**；③ 需要**长时间窗**才能解题；④ 在 `SF20K-Train` 上做大规模指令微调能提升表现。

## 4. 用到的模型

- **被测**：各类 LLM（读 transcript）与视频多模态模型（读帧/帧+字幕），如 GPT 系列、开源 Video-LLM 等（论文表格列多个 baseline），并与**人类**对照。
- **判分辅助**：OEQA 用 **LLM-as-judge**（GPT 类）判生成答案与标准答案是否一致。

## 5. 实际数据案例全过程

以上面 `z9HVSEot-O8_00`（问"是什么促使 Nina 搬到城市？"）为例：

1. **取视频**：按 `video_url` 下载该短片（~10 分钟）。
2. **MCQA**：把 5 个 option（A–E）+ 问题连同视频/字幕送模型 → 模型输出字母；与 `correct_letter="B"` 比对得 0/1。
3. **OEQA**：去掉选项，模型自由作答 → LLM-as-judge 对比标准答案 `"The synopsis does not specify..."` 判对错。
4. **聚合**：MCQA 按 split 算准确率；OEQA 按 judge 判定的正确率；分别对比 transcript / vision / multimodal 三设定与人类基线。

## 6. 指标公式速查表（简介·模型·公式·参数）

> 两个任务都是"准确率"型。记号：`N`=题数，`1[·]`=指示函数。

| 指标 | 简介 | 模型 | 计算公式 + 参数说明 |
|---|---|---|---|
| **MCQA Accuracy** | 5 选 1 选择题准确率 | 被测 LLM / 视频模型 | $\text{Acc}=\dfrac{1}{N}\sum_i \mathbb{1}[\hat\ell_i=\text{correct\_letter}_i]$ ·· `ℓ̂`=模型所选字母(A–E)；5 个选项含 LLM 构造的强干扰项 |
| **OEQA Accuracy** | 开放式问答正确率 | 被测模型 + LLM-as-judge | $\text{Acc}=\dfrac{1}{N}\sum_i \mathbb{1}[\text{Judge}(\hat a_i,\,a_i)=\text{correct}]$ ·· `â`=模型自由生成答案，`a`=标准 `answer`，Judge=GPT 类裁判判语义是否等价 |
| **（分设定/与人对照）** | 拆解视觉 vs 文本能力 | — | 同上准确率，分别在 **transcript-only / vision-only(`test_silent`) / multimodal** 三设定下计算，并与 **Human** 基线对比 |

**参数说明**：①MCQA 为确定性字母匹配，无需裁判；②OEQA 依赖 LLM 裁判，故引入裁判模型；③`test_expert` 为人工精校高难子集、`test_silent` 用于纯视觉评测；④强调长时间窗——短片平均 ~11–13 分钟，截断过短会显著掉分。

---
**一句话定位**：SFD（Short Film Dataset，发布版 SF20K）= 2 万部公开业余短片(多类型、平均十余分钟)上的**故事级长视频问答基准**，含 MCQA(准确率)与 OEQA(LLM 裁判)两任务，专测长时叙事理解；现有视频模型纯视觉下远逊于人类。
