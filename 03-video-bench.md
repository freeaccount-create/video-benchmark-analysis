# Video-Bench — 以 MLLM 为裁判、贴合人类偏好的视频生成评测

> 仓库：`Video-Bench`（项目页 video-bench.github.io）· 核心思想：用多模态大模型（GPT-4o）当评委，让自动打分与人类偏好高度对齐（论文报告 Spearman 相关 0.40–0.75）

## 1. 数据集由来

传统视频生成指标（FID/FVD/CLIP）与人类主观感受相关性弱。Video-Bench 的动机是：**直接让 MLLM（GPT-4o）像人一样看视频、打分**，并通过多轮对话 / 思维链让评分更稳。它沿用了与 VBench 类似的维度划分，但判分器从"专用 CV 模型"换成了"GPT-4o 提示工程"。

提示词总表 `videobench/VideoBench_full.json`，共 **419 条**。

## 2. 原始数据格式

prompt 元信息 JSON，字段与 VBench 类似（仅 prompt + 该 prompt 要测的维度）：
```json
{ "prompt": "A person is marching", "dimension": ["action", "temporal_consistency", "motion_effects"] }
```
另有 `config.json` 存放 API key 与各维度日志路径（GPT4o_API_KEY、GPT4o_mini_API_KEY、log_path_* 等）。

## 3. 完整打分流程

入口 `evaluate.py`：
```bash
python evaluate.py --dimension color --videos_path ./data/ \
    --config_path ./config.json --models kling videocrafter2
```

调用链（`videobench/__init__.py`）：`evaluate()`(L294) → `evaluate_dimension()`(L241) → `build_full_info_json()`(L38) 收集视频 → 按维度类型分派到三种判分模块（L257-286）：

- **文本对齐类** → `VideoTextAlignment.py`
- **静态质量类** → `staticquality.py`
- **动态质量类** → `dynamicquality.py`

### 维度与判分方法

| 类别 | 维度 | 分值 | 判分方式 |
|------|------|------|---------|
| 静态质量 | imaging_quality, aesthetic_quality | 1–5 | 抽帧(2fps)→base64→GPT-4o 打分，正则 `:\s*(\d+)` 抽分 |
| 动态质量 | temporal_consistency, motion_effects | 1–5 | 全帧并行送 GPT-4o，按 "because" 关键字分离分数 |
| 文本对齐 | overall_consistency(1–5), object/color/action/scene(1–3) | 1–5/1–3 | **多轮对话**：Host(GPT-4o) 描述→Agent(GPT-4o-mini) 提问→Host 回答→Host 汇总评分 |

**文本对齐的多轮架构**（`VideoTextAlignment.py:298-397`）：`host.initial_result()` 生成初始描述 → `host.question()` 由 Agent 提问 → `host.answer()` 回答 → `host.summarize_and_get_results()` 给最终分 → `extract_content_from_result()` 在 "Evaluation Result" 后取 "because" 前最近的数字。

所有维度最后 `average_scores = {model: total/count}` 聚合。

## 4. 用到的模型

- **GPT-4o-2024-08-06**（主裁判：质量打分、文本对齐 Host）
- **GPT-4o-mini**（文本对齐里的提问 Agent）
- 含 `@retry` 重试机制保证 API 稳定。

> 与 VBench 的根本区别：VBench 用 DINO/CLIP/GRiT 等 CV 模型；Video-Bench 用 GPT-4o 提示工程当裁判。

## 5. 实际数据案例全过程

取真实条目 `{"prompt":"A person is marching","dimension":["action","temporal_consistency","motion_effects"]}`：

1. **生成**：被测模型（如 kling）按 prompt 生成视频，放 `./data/{model}/...`。
2. **action（文本对齐，1–3 分）**：
   - Host(GPT-4o) 先看帧生成视频描述；
   - Agent(GPT-4o-mini) 针对"是否在行进/marching"提问；
   - Host 回答并最终给出 `Evaluation Result: 3 because ...`；
   - 脚本取 "because" 前最近数字 → 3。
3. **temporal_consistency / motion_effects（动态质量，1–5）**：全帧并行送 GPT-4o，返回带 "because" 的分数文本，`extract_scores_from_result()` 拆出每个模型的分。
4. **聚合**：每个维度对所有 prompt 求平均，写
   `evaluation_results/{dimension}/{name}_score_results.json`（含 `average_scores` 与逐条 `scores`）+ `_history_results.json`（多轮对话全程）。

---
**一句话定位**：Video-Bench = 维度化 prompt + GPT-4o/GPT-4o-mini 多轮对话当裁判，输出 1–5 / 1–3 分并求均值，主打"贴合人类偏好"。
