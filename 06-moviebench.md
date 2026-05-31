# MovieBench — 电影级长视频生成评测

> 仓库：`showlab/MovieBench` · 任务：多场景、长篇幅、保持角色一致性的"电影级"视频生成

## 1. 数据集由来

MovieBench 关注比单镜头更难的问题：**跨多个场景生成连贯的电影内容并保持角色身份一致**。它基于 **LSMDC / MPII Movie Description** 数据集构建，收集了 **160 部电影**，按"电影 → 场景(scene) → 镜头(shot)"三级组织，并为角色建立人脸图库（Character Bank）用于一致性评测。

数据划分：训练 80 部、测试 6 部（Juno、Bad Santa、Les Miserables、The Ugly Truth、This is 40、Harry Potter and the Prisoner of Azkaban）。

## 2. 原始数据格式

**(a) 电影-场景-镜头映射** `data/movies_scenes.json`（仓库真实片段）：
```json
"1037_The_Curious_Case_Of_Benjamin_Button": {
  "Sence 1: A pile of assorted buttons with a logo.": [
    "1037_..._00.00.04.663-00.00.09.179",
    "1037_..._00.00.11.836-00.00.17.623"
  ],
  "Sence 2: Close-up of a person's face.": [ "..._00.00.26.404-00.00.35.955", ... ]
}
```
即每个场景由若干带**时间码**的 shot 片段组成。

**(b) Shot 级标注** `Annotation_Shot_Desc/`：每个 shot 的角色、风格、剧情(Plot)、背景、镜头运动等。
**(c) 角色库** `Character_Bank/{movie}/{character}/best.jpg`：用于人脸匹配。

## 3. 完整打分流程

```
预测结果 Pre_path/{movie}/{shot}.{mp4|jpg} ──► 加载标注/真值 ──► 逐 shot 抽帧+resize+特征 ──► 各指标平均
```
6 个独立指标脚本（`metrics/`，由 `run.sh` 串起），每个一条命令，例如：
```bash
python Metric_1_clip_score.py --Pre_path <gen> --GT_json_path <ann> --Format Video --Resolution 256 --Frame_Number 25
python Metric_5_Character_ID_Consistency.py --Pre_path <gen> --GT_json_path <ann> --Format Video --Resolution 512 --Image_Format mp4
```

| # | 指标 | 模型 | 衡量 |
|---|------|------|------|
| 1 | CLIP Score | clip-vit-base-patch16 | 生成帧 ↔ shot 描述(Plot) 文本对齐 |
| 2 | Inception Score | InceptionV3 | 生成图像质量+多样性 |
| 3 | Aesthetic Score | CLIP ViT-L/14 + LAION 线性头 | 美学分(0–10) |
| 4 | FID | InceptionV3 (192d, 512×512) | 生成 vs 真实分布距离 |
| 5 | **Character ID Consistency** | DeepFace(fastmtcnn+ArcFace 等) | 角色身份一致性 → P/R/F1 |
| 6 | FVD | I3D | 视频时序分布距离 |

**Metric 5 细节**（`Metric_5_Character_ID_Consistency.py`）：抽 5 帧 → DeepFace 检测人脸(置信度阈值 0.8) → 在 Character Bank 中 `find` 最相似角色(euclidean_l2) → 与标注 `Characters` 比对，算 TP/FP/FN → Precision / Recall / F1。

## 4. 用到的模型

CLIP ViT-B/16（CLIP Score）、InceptionV3（IS / FID）、CLIP ViT-L/14 + LAION 美学头（Aesthetic）、DeepFace(fastmtcnn 检测 + ArcFace/Facenet/VGG-Face 等识别)（角色一致性）、I3D（FVD）。

## 5. 实际数据案例全过程

取 `1037_The_Curious_Case_Of_Benjamin_Button` 的 "Sence 1" 第一个 shot（时间码 `00.00.04.663-00.00.09.179`）：

1. **生成**：模型按该 shot 的标注(Plot/角色/背景)生成视频 → `Pre_path/1037_.../{shot}.mp4`。
2. **CLIP Score**：抽 25 帧，每帧与 Plot 文本 "A pile of assorted buttons with a logo" 算 CLIP 相似度并平均。
3. **Character ID**：抽 5 帧，DeepFace 检测人脸 → 在 `Character_Bank/1037_.../` 找最相似角色 → 与标注角色集合比对 → 累计 TP/FP/FN。
4. **FID / FVD**：与对应真值帧/视频比分布；**IS / Aesthetic** 直接评生成图。
5. **聚合**：每个指标对全部测试 shot 求平均；角色指标输出 Precision/Recall/F1。
6. **输出**：各 `Metric_*.py` 打印对应分数（avg_clip_score、IS mean/std、FID、P/R/F1、Average FVD）。

---
**一句话定位**：MovieBench = LSMDC 160 部电影按场景/镜头组织 + 角色人脸库，用 CLIP/IS/FID/FVD 评画质、用 DeepFace 评跨场景角色一致性，专攻"电影级长视频"。
