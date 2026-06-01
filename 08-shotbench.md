# ShotBench — 电影摄影语言理解评测

> 仓库：`Vchitect/ShotBench` · 任务：评测多模态大模型(VLM)对**电影摄影语言**的理解能力

## 1. 数据集由来

电影的"镜头语言"（景别、机位、焦段、构图、布光、运镜）是专业知识，VLM 是否真的懂？ShotBench 为此收集了 **200+ 部获奖电影**（多为奥斯卡提名）的画面与片段，由专家标注成 **3500+ 道选择题**，覆盖图像与视频两种模态。团队同时基于 ShotQA 数据训练并开源了专用模型 **ShotVL-3B / ShotVL-7B**（基于 Qwen2.5-VL，SFT + GRPO）。

## 2. 原始数据格式

测试集是一个 **TSV 文件**（默认 `evaluation/data/ShotBench/test.tsv`，`evaluate.py:84` 用 `pd.read_csv(sep="\t")` 读取；HuggingFace `Vchitect/ShotBench` 单独下载，仓库内仅含 `assets/` 演示图）。**7 列**（HF 上的真实 schema）：

| 列 | 类型 | 说明 |
|----|------|------|
| `index` | int64 | 题号 |
| `type` | string | 模态列表的**字符串**，如 `["image"]` 或 `["video"]`（`safe_load` 解析） |
| `path` | string | 媒体相对路径列表的**字符串**，如 `['image/YWE9AAUK.jpg']`、`['video/FopBvLLXKZ0.webm_31.mp4']` |
| `question` | string | 问题文本 |
| `options` | string | 选项**JSON 字典串** `{"A": "...", "B": "...", "C": "...", "D": "..."}` |
| `answer` | string | 标准答案字母（A/B/C/D；Z=拒答） |
| `category` | string | 摄影维度标签（**小写**，如 `shot size`、`camera movement`，用于分组统计） |

> 注意：`type`/`path`/`options` 在 TSV 里都以**字符串**存储，代码用 `json.loads`→失败再 `ast.literal_eval`(`safe_load`) 解析。

**真实样本（HF test split，逐字摘录）**

图像题（`index=1`）：
```
index    1
type     ["image"]
path     ['image/YWE9AAUK.jpg']
question What's the shot size of this shot?
options  {"A": "Extreme Close Up", "B": "Medium Close Up", "C": "Close Up", "D": "Medium Wide"}
answer   D
category shot size
```
视频题（运镜维度）：
```
type     ["video"]
path     ['video/FopBvLLXKZ0.webm_31.mp4']
category camera movement
```

**8 个摄影维度**：Shot Size(景别)、Shot Framing(构图边界)、Camera Angle(机位角度)、Lens Size(焦段)、Lighting Type(布光类型)、Lighting Conditions(曝光条件)、Shot Composition(画面构成)、Camera Movement(运镜)。

## 3. 完整打分流程

**两步**：先推理出答案，再判分。

**① 推理**（`evaluation/shotvl/evaluate.py`）：
```bash
accelerate launch --num_processes 4 evaluation/shotvl/evaluate.py \
    --model ShotVL-3B --reasoning --output-dir eval_results
```
- 加载模型(bf16 + flash-attn2)；按 rank 切分数据 `df.iloc[rank::world_size]`(L85)。
- 构造 prompt（L29）：`Question + Options(A/B/C/D) + "Please select the most likely answer"`；reasoning 模式加 `<think></think><answer></answer>`。
- 视觉处理：图像直接读；视频 `process_vision_info`（max_pixels 360×640、fps 12）。
- 贪心解码 `do_sample=False`，输出存 `predictions_{ts}.xlsx`。

**② 判分**（`evaluation/calculate_scores.py`）：
```bash
OPENAI_API_KEY=... python evaluation/calculate_scores.py --prediction_path <xlsx>
```
答案匹配三级（L25-94）：(1) 直接抽 `<answer>` 里的字母；(2) 选项文本匹配；(3) 前两步失败则调 **GPT-4o** 语义匹配（≤3 次），仍失败则随机/Z。

## 4. 评分方法与输出

- 逐行 `hit = int(pred_letter == answer)`（L93）。
- 按 `category` 分组：`accuracy = correct / total`（L126），8 个维度各一准确率 + 整体平均。
- 输出 Excel 两个 sheet：**Results**（原始+prediction+pred_letter+hit）、**Accuracy**（category/total/correct/accuracy）。
- 参考成绩：GPT-4o ≈ 59.3%，开源 **ShotVL-7B ≈ 70.1%**（SOTA），ShotVL-3B ≈ 65.1%。

## 5. 用到的模型

- **被测**：任意 VLM；官方提供 **ShotVL-3B / ShotVL-7B**（Qwen2.5-VL 架构，`Qwen2_5_VLForConditionalGeneration`）。
- **判分辅助**：**GPT-4o**（仅在前两级字符串匹配失败时做语义对齐，`openai==1.93.0`）。

## 6. 实际数据案例全过程

以一道"景别(Shot Size)"图像选择题为例（典型结构）：
```
question: "What is the shot size of this image?"
options:  {"A":"Extreme Wide Shot","B":"Wide Shot","C":"Medium Shot","D":"Close-up"}
type: ["image"]   path: ["images/xxx.jpg"]   answer: "D"   category: "Shot Size"
```
1. **推理**：把图 + 问题 + 4 选项送 ShotVL，reasoning 模式输出 `<think>面部占满画面…</think><answer>D</answer>`。
2. **抽答案**：`calculate_scores.py` 从 `<answer>` 取出 `D`。
3. **判分**：`pred_letter("D") == answer("D")` → `hit=1`。
4. **聚合**：该题计入 `category="Shot Size"`；最终 `Shot Size 准确率 = 该维度 correct/total`，并汇总 8 维 + 整体平均。
5. **输出**：`eval_results/ShotVL-3B/predictions_{ts}.xlsx`（Results + Accuracy 两表）。

## 7. 指标公式速查表（简介·模型·公式·参数）

> ShotBench 是**选择题准确率**型基准：VLM 选字母 → 判对错 → 按 8 维分组算准确率。无数值回归指标。

| 指标 | 简介 | 模型 | 计算公式 + 参数说明 |
|---|---|---|---|
| **Hit（单题对错）** | 单题预测是否正确 | 被测 VLM（如 ShotVL-3B/7B） | $\text{hit}=\mathbb{1}[\text{pred\_letter}=\text{answer}]$ ·· `pred_letter`三级抽取：①取`<answer>`字母→②选项文本匹配→③GPT-4o 语义对齐(≤3 次)（`calculate_scores.py:25-93`） |
| **Category Accuracy（8 维）** | 每个摄影维度的准确率 | 同上 | $\text{Acc}_{cat}=\dfrac{\#\{\text{该维度 hit}=1\}}{\#\{\text{该维度题数}\}}$ ·· 按 `category` 分组(`:126`)；8 维=景别/构图/机位/焦段/布光类型/曝光/画面构成/运镜 |
| **Overall Accuracy** | 整体平均准确率 | 同上 | $\text{Acc}=\dfrac{\text{correct}}{\text{total}}$ ·· 参考：GPT-4o≈59.3%，ShotVL-7B≈70.1%(SOTA) |

**参数说明**：①推理用贪心解码 `do_sample=False`；视频 `process_vision_info`(max_pixels 360×640, fps 12)；②GPT-4o 仅作答案对齐兜底，不参与打分；③拒答/失败映射为随机或 Z；④输出 Excel 双表 Results + Accuracy。

---
**一句话定位**：ShotBench = 200+ 获奖电影的 3500+ 道摄影语言选择题，VLM 答题→（必要时 GPT-4o 帮忙对齐答案）→按 8 个摄影维度算准确率；并配套开源 ShotVL 模型。
