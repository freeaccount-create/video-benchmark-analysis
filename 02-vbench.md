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

## 6. 16 维指标公式速查表（简介·模型·公式·参数）

> 记号：`f_t`=第 t 帧特征，`N`=帧数，`cos`=余弦相似度，`mean_t`=对帧平均。每维先算原始分再 `归一化×权重`。

### 质量类（7 维，不看 prompt）

| 维度 | 简介 | 模型 | 计算公式 + 参数说明 |
|---|---|---|---|
| **Subject Consistency** | 主体跨帧是否稳定 | DINO ViT-B/16 | $S=\frac{1}{N-1}\sum_{t=2}^{N}\frac{\max(0,\cos(f_t,f_{t-1}))+\max(0,\cos(f_t,f_1))}{2}$ ·· `f_t`=第t帧DINO特征(L2归一化)；`f_1`=首帧；与前帧+与首帧相似度取均值，`max(0,·)`截负 |
| **Background Consistency** | 背景跨帧是否稳定 | CLIP ViT-B/32 | 同上公式，`f_t` 换为 CLIP 图像特征 |
| **Temporal Flickering** | 静态视频是否闪烁 | 无（像素级） | $S=\frac{255-\mathrm{mean}_t\,\mathrm{MAE}(I_t,I_{t+1})}{255}$ ·· `MAE`=相邻帧逐像素绝对差均值；`255`=8bit 最大像素值 |
| **Motion Smoothness** | 运动是否平滑连贯 | AMT-S 插帧 | $S=\frac{255-\mathrm{mean}_i\,\mathrm{MAE}(I_i^{gt},\hat I_i)}{255}$ ·· 抽掉奇数帧→AMT 插回 `Î_i`→与真实被抽帧比 MAE |
| **Dynamic Degree** | 是否真有运动（防静态刷分） | RAFT 光流 | 单帧对 $r=\mathrm{mean}(\text{top}5\%\sqrt{u^2+v^2})$；判定 $\text{move}=[\#\{r>\tau\}\ge c]$；维度分=会动视频占比 ·· `u,v`=光流分量；阈值 `τ=6.0·(scale/256)`；门限 `c=round(4N/16)` |
| **Aesthetic Quality** | 美学观感 | CLIP ViT-L/14+LAION 美学头 | $S=\mathrm{mean}_t(a_t/10)$ ·· `a_t`=LAION MLP 对 CLIP 特征打的美学分(0–10)，`/10` 归一化 |
| **Imaging Quality** | 成像技术质量 | MUSIQ | $S=\mathrm{mean}_t(q_t)/100$ ·· `q_t`=MUSIQ 无参考质量分(0–100，含模糊/噪声/失真) |

### 语义类（9 维，看是否符合 prompt）

| 维度 | 简介 | 模型 | 计算公式 + 参数说明 |
|---|---|---|---|
| **Object Class** | 指定物体是否出现 | GRiT | $S=\mathrm{mean}_{vid}\frac{\#\{帧:c^*\in D_t\}}{N}$ ·· `c*`=目标物体类；`D_t`=第t帧 GRiT 检出集合 |
| **Multiple Objects** | 多物体是否同时出现 | GRiT | $S=\mathrm{mean}_{vid}\frac{\#\{帧:K\subseteq D_t\}}{N}$ ·· `K`=prompt 全部物体集合，需一帧内全检出 |
| **Color** | 物体颜色是否对 | GRiT | $S=\mathrm{mean}_{vid}\frac{\#\{帧:物体对且颜色对\}}{\#\{帧:检出该物体\}}$ ·· 分母=检出目标物体帧数，分子=颜色也匹配的帧 |
| **Spatial Relationship** | 空间关系是否对 | GRiT+规则 | 左/右(x向)：$score=\begin{cases}1,&\lvert\Delta x\rvert>\lvert\Delta y\rvert,IoU<0.1\\0.1/IoU,&\lvert\Delta x\rvert>\lvert\Delta y\rvert,IoU\ge0.1\\0,&\text{否则}\end{cases}$ ·· `Δx,Δy`=两物体中心差；`IoU`=两框交并比；`0.1`=阈值；上/下把 x↔y 对调 |
| **Scene** | 场景是否匹配 | Tag2Text | $S=\mathrm{mean}_{vid}\frac{\#\{帧:所有场景词命中\}}{N}$ ·· 对每帧打标，需全部关键词命中(`Σq_flag==len`)才算成功帧 |
| **Appearance Style** | 外观风格匹配（帧级） | CLIP | $S=\mathrm{mean}_t(\mathrm{logit}_{CLIP}(text,f_t)/100)$ ·· `text`=风格词(如"oil painting")；logit=`100·cos` 后再 `/100` |
| **Temporal Style** | 时间风格匹配（视频级） | ViCLIP | $S=\mathrm{mean}_{vid}\,\mathrm{logit}_{ViCLIP}(text,V)$ ·· `V`=整段视频(ViCLIP 视频级编码,含时序) |
| **Overall Consistency** | 总体语义对齐 | ViCLIP | 同上公式，`text` 换成完整 prompt |
| **Human Action** | 人体动作是否对 | UMT ViT-L | $S=\frac{\#\{视频:GT动作\in\text{Top5}_{\ge0.85}\}}{\#视频}$ ·· UMT 输出 Top-5 动作，置信度`≥0.85`才保留，命中标注动作记 1 |

### 聚合公式（`cal_final_score.py`+`constant.py`）
$$\text{Norm}[d]=\frac{\text{raw}[d]-\text{Min}[d]}{\text{Max}[d]-\text{Min}[d]}\times W[d],\quad \text{Total}=\frac{\text{Quality}\times4+\text{Semantic}\times1}{5}$$
·· `Min/Max`=人类偏好校准的归一化区间；`W[d]`=维度权重(dynamic degree=0.5，其余=1)；质量权重 4、语义权重 1。

---
**一句话定位**：VBench = 一套维度化 prompt + 16 个专用判分器（DINO/CLIP/GRiT/RAFT/MUSIQ/ViCLIP…），归一化加权成 Quality/Semantic/Total，可公平对比 T2V 模型。
