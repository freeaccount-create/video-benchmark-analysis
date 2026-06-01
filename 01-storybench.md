# StoryBench — 连续故事视频生成基准

> 仓库：`google/storybench` · 论文：[StoryBench: A Multifaceted Benchmark for Continuous Story Visualization](https://arxiv.org/abs/2308.11606) (NeurIPS 2023)

## 1. 数据集由来

StoryBench 由 Google 提出，目标是评测**连续故事可视化（continuous story visualization）**——即模型能否把一段多句的故事文本，连贯地生成为视频。它没有从零造视频，而是**复用三个已有的真实视频数据集**，再用一套自动转换 pipeline 把它们改造成「故事任务」格式：

- **OOPS**：来自 VidLN 标注的真实"翻车/意外"视频，含原始字幕 + 算法生成的故事（可选鼠标轨迹 traces）。
- **UVO**（dense / sparse 两种密度）：开放世界视频，同样基于 VidLN 字幕 + 自动生成故事。
- **DiDeMo**：经典的时序定位（temporal localization）视频数据集，提供原始字幕。

三个源经统一转换后，得到约 **3.6 万条**任务样本，分三个子任务：

| 子任务 | 含义 | 是否给条件帧 |
|--------|------|-------------|
| `action_exe` | 给一句动作描述，生成该动作 | 可有少量条件帧 |
| `story_cont` | 给故事上文 + 条件帧，续生成后续 | 有 |
| `story_gen` | 给背景 + 多句文本，从零生成完整故事 | 无 |

## 2. 原始数据格式

原始数据是**两部分**：

**(a) 视频本体** —— 三个源各自的真实视频（原生 mp4/webm，分辨率/帧率各异）。StoryBench 把它们统一**重采样为 96×160 像素、8 fps**，存成 NumPy 数组 `.npy/.npz`（形状 `[T, H, W, 3]`），存放于 `npy_96x160pix_8fps/...`。

**(b) 文字标注 JSON** —— VidLN/DiDeMo 风格，**不含像素**，关键字段（被 `scripts/create_task_data.py` 读取，行 93-100）：
```json
{
  "background_description": "...",
  "sentence_parts": ["句子1", "句子2", "句子3"],
  "start_times": [0.0, 2.25, 23.1],   // 每句的时间区间（秒）
  "end_times":   [2.25, 23.1, 29.6]
}
```

## 3. 完整打分流程

```
task JSON ──► 模型生成视频(NPZ, [T,H,W*S,3]) ──► 按指标提取特征(缓存 features/) ──► 各指标脚本独立算分
```

1. **准备任务**：`data/tasks/{源}-{split}/{task}.json` 已含 `texts`、`exact_frames_per_prompt`、视频指针与帧范围。
2. **模型生成**：模型读 `texts`(+`background`)，输出 NPZ，数组 `[T, H, W*S, 3]`（S=并排生成的视频数）。
3. **目录组织**：`{model}/{task}/{dataset}/raw/fn0.npz`；真值放 `ground_truth/...`。
4. **特征提取**：每个指标各跑一遍，特征缓存进 `features/{metric}/embeddings_*.npz`。
5. **算分**：每个指标一条命令，例如
   ```bash
   python -m metrics.fid_inception --model=phenaki --task=action_exe \
       --dataset=oops_test --data_dir=/tmp/datadir/ --output_dir=/tmp/out/ --num_videos=4
   ```
   对 `ground_truth` 和模型输出各跑一次再比较。

### 9 个指标

| 指标 | 模型 | 衡量 | 算法 | 文件 |
|------|------|------|------|------|
| FID-Inception | InceptionV3 (2048d) | 帧级真实度 | Fréchet 距离 | `metrics/fid_inception.py` |
| SIM-Inception | InceptionV3 | 帧级相似度 | 余弦×100 | 同上 |
| FID-CLIP | OpenCLIP ViT-L/14-336 | 语义真实度 | Fréchet | `metrics/fid_clip.py` |
| SIM-CLIP | OpenCLIP | 语义相似度 | 余弦×100 | 同上 |
| FVD-I3D | I3D (Kinetics-400) | 视频级时序分布 | 视频级 Fréchet | `metrics/fvd_i3d.py` |
| FVD-InternVideo | InternVideo-MM-L/14 | 视频级时序分布 | 同上 | `metrics/fvd_internvideo.py` |
| VTM-CLIP | OpenCLIP | 视频↔文本对齐 | 点积×100，按句段平均 | `metrics/vtm_clip.py` |
| VTM-InternVideo | InternVideo | 视频↔文本对齐 | 同上 | `metrics/vtm_internvideo.py` |
| PQA-DOVER | DOVER | 感知质量 | 美学+技术分 sigmoid 融合 0–100 | `metrics/pqa_dover.py` |

**Fréchet 距离公式**（`metrics/utils.py:66-118`）：
```
d² = ‖μ₁ − μ₂‖² + Tr(Σ₁ + Σ₂ − 2·√(Σ₁Σ₂))
```

## 4. 用到的模型

- **InceptionV3**（ImageNet 预训练，FID/SIM 帧级特征）
- **OpenCLIP ViT-L/14-336**（CLIP 帧特征 + 文本编码）
- **I3D**（Kinetics-400，视频级时序特征 → FVD）
- **InternVideo-MM-L/14**（多模态视频编码 → FVD / VTM）
- **DOVER**（视频质量评估，输出美学/技术双分）

> 注意：StoryBench **完全不使用 GPT/VLM 当裁判**，是纯「经典自动指标」型基准。

## 5. 实际数据案例全过程

取 `data/tasks/didemo-test/story_cont.json` 的**真实第 0 条**：

**① 原始标注（秒级）**
```json
{
  "sentence_parts": [
    "A group of people and swimmers are standing near the swimming pool ...",
    "The swimmers dive into the water and starts swimming from one end to the another.",
    "The swimmers after touching the grey surface again starts swimming towards the starting point."
  ],
  "start_times": [0.0, 2.25, 23.1],
  "end_times":   [2.25, 23.1, 29.6]
}
```

**② 转换后任务（帧级，仓库实际内容）**
```json
{
  "texts": ["A group of people ...", "The swimmers dive ...", "The swimmers after ..."],
  "exact_frames_per_prompt": [18, 167, 52],
  "npz_video": "storybench/npy_96x160pix_8fps/didemo-test/videos/10052357@N03_7585070382_5762149bfc..npy",
  "npz_video_start_frame": 0, "npz_video_end_frame": 4,
  "npz_gt_video_start_frame": 0, "npz_gt_video_end_frame": null,
  "skip_frames_after_generation": 4,
  "storybench_mode": "story_cont",
  "comment": "10052357@N03_7585070382_5762149bfc."
}
```
转换关键：`(23.1 − 2.25) × 8 fps ≈ 167 帧` → 第二句的 `exact_frames_per_prompt`。

**③ 模型生成**：读 3 句 `texts` + 第 0–4 帧条件，生成约 18+167+52 帧的视频，存 `story_cont/didemo_test/raw/fn0.npz`。

**④ 打分**（举 VTM-CLIP）：
- 逐帧 CLIP 编码生成视频 → 按 `exact_frames_per_prompt` 切成 3 段，每段帧特征取平均；
- 3 句文本 CLIP 编码；
- 每段 `score = (视频emb · 文本emb) × 100`，再对 3 段平均；
- 同时跑 FID/FVD（与真值帧/视频比分布）、DOVER（质量）。

**⑤ 输出**：`{model}/story_cont/didemo_test/{metric}/result.txt`；DOVER 另出逐视频 TSV + aesthetic/technical/overall 三个聚合分，并写 TensorBoard 标量。

## 6. 指标公式速查表（简介·模型·公式·参数）

> 统一记号：`μ,Σ`=特征集合的均值/协方差；`v,t`=视频/文本特征向量；`cos(a,b)=a·b/(‖a‖‖b‖)`；`mean_x`=对 x 求平均。所有 FID/FVD 共用 `metrics/utils.py:66-118` 的 Fréchet 实现。

| 指标 | 简介 | 模型 | 计算公式 + 参数说明 |
|------|------|------|----------------------|
| **FID-Inception** | 帧级真实度（生成帧分布 vs 真实帧分布） | InceptionV3(2048d) | $d^2=\lVert\mu_g-\mu_r\rVert^2+\mathrm{Tr}(\Sigma_g+\Sigma_r-2\sqrt{\Sigma_g\Sigma_r})$ ·· `μ_g,Σ_g`=生成帧 Inception 特征(2048维)的均值/协方差；`μ_r,Σ_r`=真实帧的；越小越真 |
| **FID-CLIP** | 语义层面真实度 | OpenCLIP ViT-L/14-336 | 同上 Fréchet 公式，特征换成 CLIP 图像特征(768维)；测"语义分布"而非像素分布 |
| **SIM-Inception / SIM-CLIP** | 帧级相似度（与真值帧对齐） | InceptionV3 / OpenCLIP | $S=\mathrm{mean}(\,\hat v_g\cdot\hat v_r\,)\times100$ ·· `v̂`=L2 归一化特征；逐帧与真值帧点积(=余弦)后取均值×100 |
| **FVD-I3D** | 视频级时序分布真实度 | I3D(Kinetics-400, 400d) | 同 FID 的 Fréchet 公式，但特征是**整段视频**的 I3D 时空特征(400维)；含运动/时序 |
| **FVD-InternVideo** | 视频级时序分布真实度 | InternVideo-MM-L/14(768d) | 同上，特征换 InternVideo 视频编码(8帧倍数窗口,768维)；`mean(0)`池化后求分布距离 |
| **VTM-CLIP** | 视频↔文本对齐（**帧级**，无时序基线） | OpenCLIP | $S=\mathrm{mean}_{seg}(\hat v_{seg}\cdot\hat t_{seg})\times100$ ·· `v_seg`=该故事段内逐帧 CLIP 特征**平均**后 L2 归一化；`t_seg`=该句 CLIP 文本特征；按 `exact_frames_per_prompt` 切段，段分再平均（`vtm_clip.py:104,127`） |
| **VTM-InternVideo** | 视频↔文本对齐（**视频级**，含时序） | InternVideo | 同上公式，但 `v_seg`=InternVideo 对整段(补齐/采样到 8 帧倍数)的**视频级编码** `mean(0)`+L2；能区分动作顺序（`vtm_internvideo.py:135`） |
| **PQA-DOVER** | 无参考感知质量（美学+技术） | DOVER | $PQA=\tfrac{1}{2}\big(\sigma(\tfrac{a-\mu_a}{\sigma_a})+\sigma(\tfrac{q-\mu_q}{\sigma_q})\big)\times100$ ·· `a,q`=DOVER 美学/技术分支原始分；`σ`=sigmoid；`μ,σ`=各分支预设均值/方差做标准化；两分支融合×100（`pqa_dover.py:57-64`） |

**关键参数**：StoryBench 把所有视频统一到 **96×160 像素、8 fps**；VTM 按故事句段(`exact_frames_per_prompt`)切分逐段算再平均；InternVideo 时序窗口=**8 帧**(224×224)；FID/FVD 的 `√(Σ_gΣ_r)` 用 `scipy.linalg.sqrtm`，奇异时加 `eps=1e-6` 对角扰动。

---
**一句话定位**：StoryBench = 复用 OOPS/UVO/DiDeMo 真实视频 + 自动改造成故事任务，用 FID/FVD/CLIP/VTM/DOVER 等**经典指标**评连续故事生成，不依赖大模型裁判。
