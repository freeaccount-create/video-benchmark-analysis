# VinaBench — 视觉叙事生成评测

> 仓库：`Silin159/VinaBench` · VINA = **Vi**sual **Na**rrative · 论文：VinaBench: Benchmark for Faithful and Consistent Visual Narratives (CVPR 2025)

## 1. 数据集由来

**视觉叙事生成**：把一段文字故事转成**一系列图像**讲完整故事。VinaBench 的独特贡献是为样本标注了底层的**常识约束(commonsense constraints)** 与**话语约束(discourse constraints)**，以此同时指导生成、并支撑"忠实度 + 一致性"的细粒度评测。数据汇集自三个已有视觉叙事数据集：

- **VWP**（Visual Writing Prompts）
- **Storyboard20K**（来自 showlab Long-form-Video-Prior 的电影分镜帧）
- **StorySalon**（含原版与短版 short）

每个数据集都有 `{name}_train.json` / `{name}_test.json`。

## 2. 原始数据格式

> 仓库内 `data/` 只含脚本，真实图片/标注 JSON 需按 `data/README.md` 的链接另行下载（VWP 走 Google Drive、Storyboard20K 走原仓库、StorySalon 走 Drive）。

标注结构（由 `data/scripts/create_gold_constraints.py:23-104` 定义）每条含：
- `narrative`：分场景的文本叙事序列
- `image_links`/`key_frames`：每个场景对应图像
- `scene_characters`：每场景出场角色（`present`、`num_present`）
- `global_profile`：角色全局描述（角色名→文本）
- `time` / `location`：每场景时间(early morning…night)/地点
- `captions`：图像标题
- `linked_entities`：叙事↔图像的实体链接
- `setups`：约束格式 `[Characters] 角色(描述) [Time] 时间 [Location] 地点`

## 3. 完整打分流程

入口 `evaluation/evaluate.sh`，分四阶段：

**阶段1 — 基础生成质量**（`MM-Interleaved/calculate_fid_clipi.py`）：FID + CLIP-I（生成图 vs 金标准图）。

**阶段2 — 文本-图像排序**（CLIP）：`prepare_mrr_rank_clip.py` 建图像池取 top-100 候选 → `clip_text_rank.py` 算 **CLIP-T**（叙事↔图像余弦）与 **MRR**（生成图在候选中的倒数排名 `1/(rank+1)`）。

**阶段3 — LLaVA VQA 对齐/一致性**：对齐(entity/character/time/location)、一致性(style/character/location)、VQA-MRR，全部用提问 yes/no 的方式判分。

**阶段4 — MiniCPM VQA**：同阶段3的对齐/一致性指标，换 MiniCPM-V 复测。

### 代表性指标算法
| 指标 | 提问/算法 | 文件 |
|------|----------|------|
| Character Num Align | "How many characters…?"→数字是否等于标注 | `*vqa_character_align.py` |
| Character Attr Align | "Do characters fit descriptions? yes/no" | 同上 |
| Entity Align | 对每个实体 "Does this image contain '{e}'?" 命中率 | `*vqa_entity_align.py` |
| Time/Location Align | "Is this image taken at/in {time|loc}? yes/no" | `*vqa_time/location_align.py` |
| Char/Location Consist | 多图 "Do all images contain same character/location? yes/no" | `*vqa_*_consist.py` |
| Style Consist | 多图 "Are all images in the same style? yes/no" | `*vqa_style_consist.py` |
| CLIP-T / MRR | 叙事-图像余弦 / 候选排名倒数 | `clip_text_rank.py` |

yes/no 类从首 token 的 "Yes/yes/YES" 概率和取分，全样本平均。

## 4. 用到的模型

- **评测器**：CLIP ViT-L/14（CLIP-I、CLIP-T、候选池）、**LLaVA-OneVision-Qwen2-72B** 与 **MiniCPM-V-2.6**（VQA 对齐/一致性，双模型交叉验证）。
- **基线生成器**：MM-Interleaved（主基线）。
- **标注生成（造数据用）**：Llama-3.1-70B（实体抽取/链接/角色描述）、Mantis-8B（图像字幕）、MiniCPM-V（全局风格属性）。

## 5. 实际数据案例全过程

以一条 VWP 测试叙事（设 3 个场景，角色 Anna 出现在场景 1、3；场景时间 morning/night）为例：

1. **生成**：模型按 `narrative` 三句生成 3 张图 → 预测序列。
2. **FID/CLIP-I**：与金标准 3 张图比分布与图像相似度。
3. **CLIP-T/MRR**：每句叙事 ↔ 对应生成图算余弦(CLIP-T)；并在 100 候选里排名 → `mrr=1/(rank+1)`。
4. **VQA 对齐**：对场景1图问 "Is this image taken in the morning? yes/no"（time）、"Does this image contain 'pool'? yes/no"（entity）、"How many characters?" 是否=标注数。
5. **一致性**：把 Anna 出现的场景1、3两张图一起问 "Do all these images contain the same character Anna: {描述}? yes/no" → 角色一致性分。
6. **聚合 & 输出**：每个指标全样本取平均，写成多个 JSON（如 `*_clipt_mrr_scores.json`、`*_llava_vqa_char_consist_scores.json`、`*_minicpm_vqa_*_scores.json`），结构 `{dataset_experiment: 分数}`。

---
**一句话定位**：VinaBench = VWP/Storyboard20K/StorySalon 视觉叙事 + 常识/话语约束标注，用 CLIP（FID/CLIP-I/CLIP-T/MRR）测画质对齐、用 LLaVA+MiniCPM 双 VQA 测角色/时间/地点/风格的对齐与一致性。
