# RISEBench — 推理驱动的视觉编辑评测

> 仓库：`PhoenixZ810/RISEBench` · RISE = **R**easoning-**I**nformed vi**S**ual **E**diting · 首个聚焦"带推理的图像编辑"的基准

## 1. 数据集由来

普通图像编辑只需"照指令改"，RISEBench 要求模型**先推理、再编辑**——理解世界知识/因果/时间/空间/逻辑后，生成合理的编辑结果。例如"画出香蕉在常温放一年后的样子"，需要推理出"腐烂变黑"。它定义四类推理：

- **Temporal** 时间推理（随时间变化）
- **Causal** 因果推理（动作后果）
- **Spatial** 空间推理（视角/空间关系）
- **Logical** 逻辑推理（谜题/模式）

规模：**64 条**，四类各 16 条（小而精）。

## 2. 原始数据格式

`data/data_total.json`（仓库真实第 0 条）：
```json
{
  "index": "temporal_reasoning_1",
  "category": "temporal_reasoning",
  "instruction": "Draw what it will look like after being kept in a daily environment for a year.",
  "image": "temporal_reasoning_images/1.png",
  "reference": "Rotten, blackened banana."
}
```
字段：`index` / `category` / `instruction`(编辑指令) / `image`(输入图) / `reference`(参考答案，文本或 `reference_img`)。配套图片目录 `{category}_images/`。

## 3. 完整打分流程

入口 `gpt_eval.py`：
```bash
python gpt_eval.py --data data/data_total.json --output outputs/MODEL_NAME
```
`main()`(L326)：加载数据 → 多进程 `track_progress_rich` → 每条调 `eval_vanilla()`(L152) 按 category 构造 2–3 个评判 prompt → 调 `gpt_generate()`(L38, 默认 **gpt-4.1-2025-04-14**, temperature=0) → `extract()`(L263) 正则 `Final Score:\s*(\d+)` 抽分。

### 三个评判维度（GPT 当裁判）
1. **Instruction Reasoning**（推理正确性）—— 输出是否符合 reference 描述，1–5。
2. **Appearance Consistency**（外观一致性）—— 除指令要求外是否保持一致，1–5。
3. **Visual Plausibility**（视觉合理性）—— 是否清晰、物理合理，1–5。
（Logical 类用 0/1 二值，再映射到 1/5。）

### 加权聚合（`calculate_score()` L287）
```
Temporal/Causal/Spatial:  score = 0.3·Consistency + 0.5·Reasoning + 0.2·Plausibility
Logical:                  score = 0.3·Consistency + 0.7·Reasoning
若 Reasoning==1:          score = max(score×0.5, 1)        # 推理崩了重罚
```
完成度 `calculate_completion()`：所有维度都满分(5)才算完成(1)。百分比 `trans_to_percent(s)=25·(s−1)`（1→0%，5→100%）。

## 4. 用到的模型

- **GPT-4.1 (gpt-4.1-2025-04-14)** —— 唯一裁判，对每条数据按维度跑多个评判 prompt（含/不含输入图两种模板）。被测的则是各种图像编辑/生成模型（输出放 `outputs/{MODEL}/`）。

## 5. 实际数据案例全过程

取真实条目 `temporal_reasoning_1`（指令"画出常温放一年后的样子"，输入=完好香蕉图，reference="Rotten, blackened banana"）：

1. **被测模型**输出编辑后图 → `outputs/{MODEL}/temporal_reasoning_1.png`。
2. **三次 GPT-4.1 评判**：
   - Consistency：对比[输入图, 输出图]——除"变质"外香蕉形状/背景是否保持 → 例 4 分；
   - Reasoning：输出是否符合 "rotten, blackened banana" → 例 5 分；
   - Plausibility：图是否清晰合理 → 例 4 分。
3. **抽分**：每个回答末尾 `Final Score: 4/5/4`，正则取出。
4. **加权**：`0.3×4 + 0.5×5 + 0.2×4 = 4.5`；百分比 `25×(4.5−1)=87.5%`；完成度=0（非全 5）。
5. **输出**：`{model}.pkl`（缓存）、`{model}_judge.xlsx`（逐条 Reasoning/Consistency/Plausibility/score/complete）、`{model}_judge.csv`（Overall 及各类别 Score-Origin / Score-Percentage / Accuracy）。

## 6. 指标公式速查表（简介·模型·公式·参数）

> RISEBench 用单一裁判 **GPT-4.1**（temperature=0）对每条编辑结果按三维各打 1–5 分，再加权。抽分正则 `Final Score:\s*(\d+)`。

| 指标 | 简介 | 模型 | 计算公式 + 参数说明 |
|---|---|---|---|
| **Instruction Reasoning** | 输出是否符合 reference（推理对不对） | GPT-4.1 | $r\in\{1..5\}$ ·· 对比输出图与参考答案描述；Logical 类先 0/1 再映射 1/5（`gpt_eval.py:152-263`） |
| **Appearance Consistency** | 除指令要求外是否保持一致 | GPT-4.1 | $c\in\{1..5\}$ ·· 对比[输入图,输出图]，非编辑区域是否未被破坏 |
| **Visual Plausibility** | 是否清晰、物理合理 | GPT-4.1 | $p\in\{1..5\}$ ·· 仅看输出图的成像质量与合理性 |
| **加权总分 score** | 三维加权 | — | 一般类：$s=0.3c+0.5r+0.2p$；Logical：$s=0.3c+0.7r$；若 $r=1$ 则 $s=\max(0.5s,1)$（推理崩了重罚，`calculate_score():287`） |
| **百分比 / 完成度** | 报告口径 | — | $\text{percent}=25\,(s-1)$（1→0%，5→100%）；完成度=`所有维度全为5→1 否则 0` |

**参数说明**：①推理权重最高(0.5/0.7)，体现"先想再改"导向；②Reasoning==1 触发对总分的 0.5 倍重罚且不低于 1；③Logical 类不计 Plausibility，权重重排到推理；④裁判含"带/不带输入图"两种模板。

---
**一句话定位**：RISEBench = 64 条"需推理的编辑题"，用 GPT-4.1 从推理/一致性/合理性三维打分并加权（推理权重最高、推理崩了重罚），专测"会不会先想再改"。
