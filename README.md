# Video / Story / Editing Benchmarks — 源码级深度分析

本仓库对 12 个与「视频 / 视觉叙事 / 图像编辑」相关的开源 benchmark 进行了**源码级**分析。每个 benchmark 一篇文档，统一覆盖：

1. **数据集由来** —— 原始数据从哪来、为什么造、对应论文
2. **原始数据格式** —— 原始数据集长什么样（字段、文件类型）
3. **完整打分流程（pipeline）** —— 从模型输出到最终分数的每一步
4. **用到的模型** —— 评测中调用的所有模型 / 网络
5. **一条实际数据案例的全过程** —— 用仓库里真实的一条数据，走通「输入 → 处理 → 打分 → 输出」

> 所有 file:line 引用均来自各 benchmark 的官方仓库源码。

## 目录

| # | Benchmark | 任务 | 打分范式 | 文档 |
|---|-----------|------|---------|------|
| 1 | **StoryBench** (google) | 连续故事视频生成 | 经典自动指标 | [01-storybench.md](./01-storybench.md) |
| 2 | **VBench** (Vchitect) | 文本→视频质量 | 16 维度模型打分 | [02-vbench.md](./02-vbench.md) |
| 3 | **Video-Bench** | 视频生成（贴合人类偏好） | MLLM 当裁判 | [03-video-bench.md](./03-video-bench.md) |
| 4 | **DreamSim** | 感知图像相似度**度量** | 2AFC 人类对齐 | [04-dreamsim.md](./04-dreamsim.md) |
| 5 | **StoryEval** | 故事级事件完成度 | VLM 逐事件判定 | [05-storyeval.md](./05-storyeval.md) |
| 6 | **MovieBench** (showlab) | 电影级长视频生成 | 多指标（含角色一致性） | [06-moviebench.md](./06-moviebench.md) |
| 7 | **RISEBench** | 推理驱动的图像编辑 | GPT-4.1 多维打分 | [07-risebench.md](./07-risebench.md) |
| 8 | **ShotBench** (Vchitect) | 电影摄影语言理解 | 选择题准确率 | [08-shotbench.md](./08-shotbench.md) |
| 9 | **VinaBench** | 视觉叙事生成 | CLIP + VQA 约束对齐 | [09-vinabench.md](./09-vinabench.md) |
| 10 | **SFD** (Short Film Dataset / SF20K) | 故事级长视频理解 QA | MCQA 准确率 + OEQA LLM 裁判 | [10-sfd.md](./10-sfd.md) |
| 11 | **DirectorBench** | 分钟级多镜头视频生成诊断式评测 | 多智能体 DAG + checkpoint 加权 | [11-directorbench.md](./11-directorbench.md) |
| 12 | **ScriptAgent** (The Script is All You Need / ScriptBench) | 对白→剧本→长视频生成 | 主观 LLM 四维 + CLIP/VSA/FVD | [12-scriptagent.md](./12-scriptagent.md) |

## 重要说明

- **第 10 个 SFD = Short Film Dataset（发布版 SF20K）**，是**故事级长视频理解问答基准**（2 万部公开业余短片、多类型、平均十余分钟），含 MCQA + OEQA 两任务。
  - ⚠️ 早期版本曾把 SFD 误指为 `LittlePey/SFD`（CVPR 2022 LiDAR 3D 检测方法），链接错误，现已更正为论文 [Long Story Short (arXiv 2406.10221)](https://arxiv.org/abs/2406.10221) / [项目页](https://ridouaneg.github.io/sf20k.html) / [HF 数据集](https://huggingface.co/datasets/rghermi/sf20k)。
- 打分范式可粗分三类：
  - **经典自动指标**（FID/FVD/CLIP/IS/光流…）：StoryBench、MovieBench、VBench（部分维度）
  - **模型 / VLM 当裁判**（GPT-4o、LLaVA、MiniCPM、Gemini…）：Video-Bench、StoryEval、RISEBench、VinaBench、ShotBench、SFD（OEQA）、DirectorBench、ScriptAgent
  - **度量本身**（学习人类感知）：DreamSim
- 第 11–12 个为**视频生成评测**新增：**DirectorBench**（LangGraph 多智能体 DAG，对分钟级多镜头成片做诊断式评测、输出瓶颈/建议而非单一分）与 **ScriptAgent / The Script is All You Need**（对白→剧本→长视频生成的三智能体闭环，其 CriticAgent 用主观 LLM 四维 + 客观 CLIP/VSA/FVD 双轨组成 ScriptBench）。

---
*本分析基于各仓库截至 2026-05 的 `HEAD`。*
