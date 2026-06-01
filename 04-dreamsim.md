# DreamSim — 学习人类感知的图像相似度「度量」

> 仓库：`ssundaram21/dreamsim` · 论文：DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data (NeurIPS 2023)

## ⚠️ 它不是"benchmark"，是一个「度量指标（metric）」

DreamSim 本身是一个**可调用的感知相似度函数** `d = model(img1, img2)`，用来衡量两张图"在人眼看来有多不同"。它填补了低层指标（LPIPS/PSNR/SSIM 只比颜色纹理）与高层嵌入（CLIP 偏语义）之间的**中层视觉属性**（布局、姿态、语义内容）空白。它既可当评测指标，也可当检索/感知损失。其"基准"含义体现在：用 NIGHTS 数据集的人类 2AFC 判断来衡量该度量与人的对齐程度。

## 1. 数据集由来（NIGHTS）

**NIGHTS**（Novel Image Generations with Human-Tested Similarities）：用 Stable Diffusion 2.1 生成图像三元组，再让人做 2AFC 选择（"左右哪张更像参考图"）。共 **20,019 个三元组**（保留 ≥6 票一致的；另有 10 万条未过滤版）。类别来自 ImageNet/CIFAR/Flowers/Food-101/SUN397，提示模板 `"An image of a <category>"`，同类别不同随机种子生成三张。

## 2. 原始数据格式

目录结构（`dataset/README.md`）：
```
nights/
├── ref/000/000.png ...        # 参考图
├── distort/000/000_0.png, 000_1.png ...   # 两个扭曲版本
└── data.csv
```
`data.csv` 关键列（`dataset/dataset.py:15-42`）：`id`、`p`（人类判断标签，0/1）、`ref_path`、`left_path`、`right_path`、`votes`（一致票数，过滤 ≥6）、`split`（train/val/test）、`is_imagenet`。

一条样本 = `(img_ref, img_left, img_right, p, id)`，`p` 表示人类觉得 left 还是 right 更接近 ref。

## 3. 完整打分流程（两层含义）

**(A) 作为度量使用**（前向）`model.py:79-105`：
```python
embed_a = embed(img_a); embed_b = embed(img_b)     # 各 backbone 特征拼接→MLP 投影
distance = 1 - F.cosine_similarity(embed_a, embed_b)
```

**(B) 评测它与人类的对齐**（2AFC 准确率）`evaluation/score.py:9-39`：
```python
d0 = model(ref, left);  d1 = model(ref, right)
score = (d0<d1)*(1-target) + (d1<d0)*target + (d0==d1)*0.5
twoafc_acc = mean(score)     # 越高越对齐人类
```
- `target=0` 表示人选 left 更近：若 `d0<d1`（模型也认为 left 更近）得 1 分。
- 入口 `evaluation/eval_percep.py`，分别评 val / test_imagenet / test_no_imagenet。

**训练**（LoRA 微调）`training/train.py`：Hinge Loss，`logit = d0 − d1`，margin=0.05；只训 LoRA(qkv) + MLP，冻结骨干。

## 4. 用到的模型

**集成（默认，最佳）**：DINO ViT-B/16(cls) + CLIP ViT-B/16(embedding) + OpenCLIP ViT-B/16(embedding)，拼接维度 768+512+512=1792 → MLP。单分支可选 DINOv2/SynCLR/CLIP/OpenCLIP（约 3× 加速）。LoRA 配置：r=16、alpha=8、dropout=0.3（集成）。性能：集成模型 NIGHTS 上 ~96.2% 2AFC（test）。

## 5. 实际数据案例全过程

取 NIGHTS 一条三元组（`ref/000/000.png`, `distort/000/000_0.png`, `distort/000/000_1.png`，假设 `p=1` 即人选 right 更近）：

1. **预处理**：三张图各 `preprocess` 到 (1,3,224,224)。
2. **嵌入**：每张图过 DINO+CLIP+OpenCLIP，cls/embedding 特征拼成 1792 维 → MLP 投影、L2 归一化。
3. **距离**：`d0 = 1 − cos(ref, left)`、`d1 = 1 − cos(ref, right)`。
4. **判分**：因 `target=1`，若 `d1 < d0`（模型也认为 right 更近）→ 该样本得 1 分，否则 0。
5. **聚合**：全测试集求平均 → 2AFC 准确率（如 96.2%），即"该度量与人类感知的一致程度"。

> 若把 DreamSim 当作**别的视频/图像基准里的一个指标**用，就只走第 1–3 步：直接返回两图距离。

## 6. 指标公式速查表（简介·模型·公式·参数）

> DreamSim 是**学习型感知距离**：把三个骨干特征拼接过 MLP，输出一对图像的"知觉差异"，并用人类 2AFC（二选一）标注校准/评测。

| 指标 | 简介 | 模型 | 计算公式 + 参数说明 |
|---|---|---|---|
| **DreamSim Distance** | 一对图像的感知差异（越小越像） | DINO ViT-B/16 + CLIP ViT-B/16 + OpenCLIP ViT-B/16（集成） | $d(a,b)=1-\cos(\phi(a),\phi(b))$ ·· `φ(x)`=concat(DINO 768 + CLIP 512 + OpenCLIP 512 = **1792维**)→投影 MLP→L2 归一化嵌入；`cos`=余弦相似度（`model.py:79-105`） |
| **2AFC Agreement（评测分）** | 与人类"哪张更像参考"的一致率 | 同上 | $\text{score}=\mathbb{1}[d_0<d_1](1-y)+\mathbb{1}[d_1<d_0]\,y+\mathbb{1}[d_0=d_1]\cdot0.5$，准确率=`mean(score)` ·· `d_0=d(ref,left)`、`d_1=d(ref,right)`；`y`=人类标签 `target`(选 right 为 1)；平局记 0.5（`evaluation/score.py:9-39`） |
| **训练损失（造度量用）** | 让距离贴合人类偏好 | 同上 | Hinge：$L=\max(0,\;\text{margin}-(d_1-d_0)\cdot\tilde y)$ ·· `logit=d0−d1`，`margin=0.05`；只训 LoRA(qkv)+MLP，冻结骨干，`r=16, alpha=8, dropout=0.3` |

**参数说明**：①三骨干均 ViT-B/16，输入 224×224，各自取 cls/embedding 后拼接；②距离用 `1−cos` 而非欧氏，范围 ~[0,2]；③评测集 NIGHTS（人类 2AFC 三元组 ref/left/right，≥6 票一致）；④集成模型 test 2AFC ≈ 96.2%。

---
**一句话定位**：DreamSim = 用人类 2AFC 数据（NIGHTS）微调 DINO+CLIP 集成得到的"感知距离函数"，既是度量工具，其自身好坏又用 2AFC 准确率来衡量。
