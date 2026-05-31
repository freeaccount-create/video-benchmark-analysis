# SFD —— ⚠️ 这不是视频/故事 benchmark，而是 3D 目标检测方法

> 仓库：`LittlePey/SFD` · 论文：[Sparse Fuse Dense: Towards High Quality 3D Detection with Depth Completion](https://arxiv.org/abs/2203.09780) (CVPR 2022, Oral)

## ⚠️ 重要说明

SFD 被误列入了"video 相关数据集"清单，但它**完全不是视频/视觉叙事/图像编辑 benchmark**。它是一个**自动驾驶场景下的 LiDAR 3D 目标检测方法**（一个模型/方法的官方实现，而非评测基准），基于 Voxel-R-CNN 和 OpenPCDet 框架构建。它处理**单帧点云+图像**，与其余 9 个生成类基准没有任何关系。这里如实记录其本质，避免混淆。

## 1. 它是什么 / 解决什么任务

- **任务**：在 KITTI 数据集上做 **3D 目标检测**（主要检测 Car 类）。
- **核心思想**：稀疏的 LiDAR 点云信息量不足，SFD 用图像做**深度补全(depth completion，TWISE)** 生成稠密"伪点云"，再把稀疏真实点云与稠密伪点云融合(Sparse Fuse Dense)，提升检测质量。

**输入**：
- 稀疏 LiDAR 点云 (x,y,z,intensity)
- RGB 图像 (image_2)
- 稠密深度图 / 伪点云 (x,y,z,r,g,b,seg,u,v，9 维)
- 标定文件 calib

**输出**：3D 边界框 `(x,y,z,l,w,h,θ)` + 类别 + 置信度。

## 2. 原始数据格式（KITTI）

标准 KITTI 3D 检测格式，目录 `data/kitti_sfd_seguv_twise/`：
- `velodyne/`：`.bin` 点云（float32，每点 4 维）
- `image_2/`：`.png` 左目彩图
- `calib/`：相机/雷达标定 `.txt`
- `label_2/`：标注 `.txt`，**每行一个目标**，KITTI 标准 15 字段：
  ```
  Car 0.00 0 -1.58  587.0 156.4 615.2 189.5  1.48 1.60 3.69  1.84 1.47 8.41  -1.56
  类别 截断 遮挡 观测角  2Dbox(x1 y1 x2 y2)  h w l  3D中心(x y z)  旋转角ry
  ```
- 训练 3712 帧、验证 3769 帧。

## 3. 整体 Pipeline / 如何使用

模型 SFD（`pcdet/models/detectors/sfd.py`）：MeanVFE → VoxelBackBone8x(3D) → BaseBEVBackbone(2D/BEV) → AnchorHeadSingle(RPN) → **SFDHead**(ROI 头，含 RoI Grid/Point/Voxel Pool，用 CPConvs 处理伪点云)。

**训练**：
```bash
cd SFD/tools
scripts/dist_train.sh 8 --cfg_file cfgs/kitti_models/sfd.yaml
```
**评估**：
```bash
scripts/dist_test.sh 8 --cfg_file cfgs/kitti_models/sfd.yaml \
    --ckpt ../output/.../checkpoint_epoch_40.pth
```

## 4. 评测方式（与生成类基准无关）

KITTI 官方指标（`pcdet/.../kitti_object_eval_python/eval.py`）：
- **2D BBox AP、BEV AP、3D AP、AOS**，以及 R40 变体（mAP_3d_R40 等）。
- 召回阈值 [0.3, 0.5, 0.7]，Car 类按 Easy/Moderate/Hard 难度。

## 5. 与其余基准的关系

| 维度 | SFD | 其余 9 个 |
|------|-----|----------|
| 领域 | 自动驾驶 LiDAR 感知 | 视频/图像/叙事生成 |
| 输入 | 单帧点云+图像 | 文本/视频/图像 |
| 性质 | 检测**方法**(模型) | 评测**基准** |
| 指标 | KITTI AP/mAP | FID/FVD/CLIP/VLM 打分等 |

**结论**：与视频生成评测毫无关系，建议从"video benchmark"清单中剔除。

---
**一句话定位**：SFD = CVPR 2022 的 LiDAR+图像融合 3D 检测方法（KITTI Car），是模型而非基准，被误归类到视频 benchmark。
