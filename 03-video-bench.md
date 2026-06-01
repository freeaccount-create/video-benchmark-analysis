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

## 6. 指标公式速查表（简介·模型·公式·参数）

> Video-Bench 全部维度都是 **VLM 当裁判 → 正则抽分 → 算术平均**，无 CV 数值指标。统一聚合：
> $$\text{Dim\_Score}=\frac{1}{K}\sum_{k=1}^{K}s_k$$
> ·· `K`=该维度视频数；`s_k`=第 k 个视频从 VLM 输出正则抽出的整数分；抽分用 `re.search(r':\s*(\d+)')` 或取 "because" 前最近数字；所有 API `temperature=0`，抽帧 2fps。

| 维度 | 简介 | 模型 | 计算公式 + 参数说明 |
|---|---|---|---|
| **Aesthetic Quality** | 美学（色彩/构图/清晰度） | GPT-4o | `s_k∈{1..5}`，GPT-4o 按 5 级标准直接打分；维度分=`mean(s_k)`（`staticquality.py:85-106`） |
| **Imaging Quality** | 成像（模糊/噪声/曝光） | GPT-4o | 同上，`s_k∈{1..5}` |
| **Temporal Consistency** | 帧间一致（防闪烁/突变） | GPT-4o（多帧） | `s_k∈{1..5}`；多帧并行送入，`extract_scores_from_result()` 取每段 "because" 前数字（`dynamicquality.py:20-121`） |
| **Motion Effects** | 运动是否符合物理/动感 | GPT-4o（多帧） | 同上，`s_k∈{1..5}` |
| **Color** | 物体颜色对不对 | GPT-4o + GPT-4o-mini | `s_k∈{1..3}`；多轮对话 Host 描述→Agent 提问→Host 答→总结给 `[Evaluation Result]: x`，抽 x（`VideoTextAlignment.py:227-296`） |
| **Object Class** | 物体类别对不对 | GPT-4o + GPT-4o-mini | 同 Color 多轮框架，`s_k∈{1..3}` |
| **Scene** | 场景对不对 | GPT-4o + GPT-4o-mini | 同上，`s_k∈{1..3}` |
| **Action** | 主体动作对不对 | GPT-4o + GPT-4o-mini | 同上，`s_k∈{1..3}` |
| **Video-Text Consistency** | 总体内容匹配 | GPT-4o + GPT-4o-mini | 同多轮框架，但 `s_k∈{1..5}`（综合物体/动作/场景/颜色/数量/风格） |

**参数说明**：①前 4 维(质量类)是 GPT-4o 单/多帧**直接打分**(1–5)；②后 5 维(对齐类)走 **Host(GPT-4o)+Agent(GPT-4o-mini) 多轮对话**，其中 Color/Object/Scene/Action 为 1–3 分、整体一致性为 1–5 分；③`s_k` 取整数，缺失/解析失败记 Error 不计入分母。

---
**一句话定位**：Video-Bench = 维度化 prompt + GPT-4o/GPT-4o-mini 多轮对话当裁判，输出 1–5 / 1–3 分并求均值，主打"贴合人类偏好"。
