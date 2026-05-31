# VBench — 文本→视频生成综合评测体系

> 仓库：`Vchitect/VBench` · 论文：VBench: Comprehensive Benchmark Suite for Video Generative Models (CVPR 2024)

## 1. 数据集由来

VBench 是业界首个把视频生成质量**拆成 16 个细粒度维度**、并能公平横向对比不同 T2V 模型的基准。它的核心不是"一堆视频"，而是一套**精心设计的提示词（prompt suite）**：每条 prompt 专门用来考察某个维度（比如"In a still frame, a stop sign"专门测时间抖动），并标注了用于自动判分的辅助信息（`auxiliary_info`）。归一化所用的 Min/Max 区间则由**大规模人类偏好标注**校准而来。

VBench 的提示词总表为 `vbench/VBench_full_info.json`，共 **946 条**。

## 2. 原始数据格式

数据是 prompt 元信息 JSON。两种典型条目：

**(a) 仅质量维度的条目**（无需 auxiliary_info）：
```json
{ "prompt_en": "In a still frame, a stop sign", "dimension": ["temporal_flickering"] }
```

**(b) 语义维度的条目**（带 auxiliary_info 供自动判分）：
```json
{
  "prompt_en": "a bird and a cat",
  "dimension": ["multiple_objects"],
  "auxiliary_info": { "multiple_objects": { "object": "bird and cat" } }
}
```

**被测视频的目录约定**：`vbench_videos/{model}/{dimension}/{prompt}-{index}.mp4`，每条 prompt 通常生成 5 个视频（index 0–4）。

## 3. 完整打分流程

入口 `evaluate.py`：
```bash
python evaluate.py --videos_path $VIDEO_PATH --dimension $DIMENSION
# 多 GPU：torchrun --nproc_per_node=8 evaluate.py ...
```

1. `VBench.evaluate()`（`vbench/__init__.py:171-192`）初始化设备/输出目录。
2. `build_full_info_json()` 加载视频路径+prompt，生成 `{name}_full_info.json`，并按 GPU rank 分发视频。
3. 逐维度动态导入：`from vbench.{dimension} import compute_{dimension}`，调用 `compute_{dimension}(...)`。
4. 收集 `results_dict[dimension] = (overall_score, video_results)`，写 `{name}_eval_results.json`。
5. `scripts/cal_final_score.py` 做归一化 + 加权聚合，得 Quality / Semantic / Total 三个总分。

### 16 个维度（两大类）

**质量维度（7 个）**：subject consistency、background consistency、temporal flickering、motion smoothness、aesthetic quality、imaging quality、dynamic degree

**语义维度（9 个）**：object class、multiple objects、human action、color、spatial relationship、scene、appearance style、temporal style、overall consistency

### 代表性维度算法

| 维度 | 模型 | 算法要点 | 文件 |
|------|------|---------|------|
| Subject Consistency | DINO ViT-B/16 | 帧特征与首帧/相邻帧余弦相似度平均 | `subject_consistency.py` |
| Background Consistency | CLIP ViT-B/32 | 同上，背景帧特征 | `background_consistency.py` |
| Motion Smoothness | AMT-S 插帧 | 插帧重建误差 (255-MAE)/255 | `motion_smoothness.py` |
| Aesthetic Quality | CLIP ViT-L/14 + LAION 美学头 | 帧美学分 /10 平均 | `aesthetic_quality.py` |
| Imaging Quality | MUSIQ | 无参考 IQA 分 /100 | `imaging_quality.py` |
| Temporal Flickering | 像素 MAE | 相邻帧差 (255-MAE)/255 | `temporal_flickering.py` |
| Dynamic Degree | RAFT 光流 | 光流幅度阈值 → 动/静二值 | `dynamic_degree.py` |
| Object/MultiObj/Color/Spatial | GRiT 密集描述 | 检测物体/颜色/边界框是否匹配 | `object_class.py` 等 |
| Human Action | UMT ViT-L | Kinetics-400 动作 Top-5 判定 | `human_action.py` |
| Scene | Tag2Text | 描述是否含目标场景词 | `scene.py` |
| Appearance/Temporal/Overall | CLIP / ViCLIP | 视频-文本相似度 | `appearance_style.py` 等 |

### 聚合（`scripts/cal_final_score.py` + `constant.py`）
```
Normalized[dim] = (raw − Min[dim]) / (Max[dim] − Min[dim]) × Weight[dim]
Quality  = Σ质量维度 / Σ权重     Semantic = Σ语义维度 / Σ权重
Total    = (Quality×4 + Semantic×1) / 5     # QUALITY_WEIGHT=4, SEMANTIC_WEIGHT=1
```
Min/Max 来自人类偏好校准；dynamic degree 权重为 0.5，其余为 1。

## 4. 用到的模型

DINO ViT-B/16、CLIP ViT-B/32、CLIP ViT-L/14 + LAION 美学头、MUSIQ、AMT-S（插帧）、RAFT（光流）、GRiT（密集描述/检测）、UMT ViT-L（动作）、Tag2Text（场景字幕）、ViCLIP（视频-文本）。

## 5. 实际数据案例全过程

取仓库真实条目 `{"prompt_en":"a bird and a cat","dimension":["multiple_objects"],"auxiliary_info":{"multiple_objects":{"object":"bird and cat"}}}`：

1. **生成**：模型用 prompt "a bird and a cat" 生成 5 个视频 → `vbench_videos/{model}/multiple_objects/a bird and a cat-{0..4}.mp4`。
2. **判分**（`multiple_objects.py`）：每个视频抽 16 帧，逐帧用 **GRiT** 输出 (描述, 物体类别)；判定该帧是否**同时**含 `bird` 与 `cat`；成功率 = 同时命中帧数 / 总帧数。
3. **维度分**：5 个视频成功率取平均 → multiple_objects 的 overall_score。
4. **归一化**：`(raw − Min)/(Max − Min) × 1`，归入 Semantic 类。
5. **总分**：Quality×4 + Semantic×1 再 /5。
6. **输出**：`{name}_eval_results.json` 里 `"multiple_objects": [overall_score, [每个视频的 {video_path, video_results}]]`。

---
**一句话定位**：VBench = 一套维度化 prompt + 16 个专用判分器（DINO/CLIP/GRiT/RAFT/MUSIQ/ViCLIP…），归一化加权成 Quality/Semantic/Total，可公平对比 T2V 模型。
