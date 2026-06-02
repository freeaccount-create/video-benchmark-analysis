# DirectorBench —— 分钟级视频生成的多智能体诊断式评测

> 论文/项目：DirectorBench（arXiv 2605.30090）· 源码：[github.com/jiaminchen-1031/DirectorBench](https://github.com/jiaminchen-1031/DirectorBench) · 类型：**多模态视频生成评测框架**（LangGraph 多智能体 DAG，输出诊断报告而非单一标量分）。
>
> 本文所有 `file:line` 引用均指向本仓库 `benchmarks/DirectorBench/` 下的真实源码。

## 1. 数据集由来

主流视频生成评测（VBench、StoryBench 等）多为「单镜头 / 短视频 / 标量分」范式：给 2–4s 片段跑若干全局指标（运动平滑度、美学、文图一致）取平均。DirectorBench 面向一个不同量级的对象——**分钟级、多镜头、带剧本/对白/BGM 的「成片」**，并把评测组织成**像剧组复盘一样的诊断流程**：

- 不只给总分，而是产出 **bottleneck（瓶颈短板）+ 可执行修改建议 + 叙事化诊断报告**；
- 不是一把大模型梭哈，而是拆成 **5 个专科智能体**（剧本/视频/音频/稳定性/跨模态），每个内部再拆成一组带 Likert 锚点的 **checkpoint**；
- 不用固定权重，而用 **用户画像（profile）** 把「这个用户在乎什么」注入加权——故事派、视觉派、音画同步强迫症得到不同总分。

数据集 `data/metadata/dataset_original.jsonl` 收录 **75 条真实评测样本**（动作/剧情/纪录等多类型），每条是一张「出题卡」：声明这段视频**本应做到什么**（三幕结构、运镜、BGM、音效、对白/唇形同步等），被评对象则是某生成模型针对 `main_instruction` 产出的成片。

## 2. 原始数据格式

样本元数据（`data/metadata/dataset_original.jsonl`，**第一条** `DB_001`，节选真实字段）：

```jsonc
{
  "meta_id": "action_006", "sample_id": "DB_001",
  "duration_sec": 67.0, "video_type": "动作类",
  "main_instruction": "屋顶追逐战（多角度快速切换+跳跃动作）",
  "modality_details": {
    "text": { "story_arc": {
        "act1_setup":      "英雄发现目标",
        "act2_conflict":   "屋顶追逐",
        "act3_resolution": "成功制服" },
      "script": [ { "shot_id": 1, "duration": 18,
                    "dialogue": "别跑！", "narration": "追逐开始" } ],
      "tone_requirements": "intense_exciting" },
    "visual": {
      "shots": [ { "shot_id": 1, "action": "屋顶跳跃",
                   "camera_movement": "tracking", "lighting": "high_contrast" } ],
      "camera_requirements": ["tracking", "whip_pan"],
      "consistency_requirements": ["spatial_layout", "momentum"] },
    "audio": { "dialogue": true, "lip_sync": true,
               "bgm_style": "strong_pulse", "sound_effects": ["footsteps","impact"],
               "tone_control": "fast_rhythm", "multi_language": "zh" }
  },
  "language": "zh", "variant_type": "original"
}
```

另有 `data/profiles.jsonl` 5 个**用户画像**（如 Profile 1「Story-First」：`priority_weights.text_story_arc=0.55`，`hard_constraints` 含 `strong_three_act_arc`、`causal_logic`），决定各维度在总分里的权重。

源码结构（`benchmarks/DirectorBench/directorbench/`）：

```
main.py            # 入口 evaluate_video() / CLI / 批量
graph.py           # LangGraph DAG（5 agent + 编排 + 诊断）
schemas.py         # GraphState / EvalResult / UserProfile / ContentProfile
checkpoints.py     # CHECKPOINTS 注册表：每个子指标 = 一组 CheckpointDef
preprocessing.py   # Phase 0：分镜/音频分离/ASR/转场度量
config.py          # 阈值、grade 边界、LLM 配置
report.py          # JSONL 追加 + 控制台摘要
agents/
  base.py          # ★ checkpoint 评测引擎（所有专科 agent 父类）
  script_agent.py  video_agent.py  audio_agent.py  stability_agent.py
  crossmodal_agent.py  diagnosis.py   # ★ 诊断综合器
```

## 3. 完整打分流程（pipeline）

`build_eval_graph()` 用 LangGraph 的 `StateGraph` 拼成四阶段 DAG（`directorbench/graph.py:34-146`）：

```
                 ┌── script_eval ──┐
 orchestrator ──┼── video_eval ───┼── crossmodal_eval ── diagnosis ── END
   (Phase 0)     ├── audio_eval ───┤      (Phase 2)        (Phase 3)
                 └── stability_eval┘
                      (Phase 1, 并行)
```

- **Phase 0 编排**：`orchestrator_node` 调 `Preprocessor.run()`（`graph.py:60-65`）——分镜检测(PySceneDetect)、音频抽取+分离(ffmpeg+AudioShake)、ASR(Whisper)、转场度量(OpenCV SSIM/直方图/光流)，产物写进共享 `GraphState.preprocessing`，工具成功/失败记录入 `tool_context`（`graph.py:67-76`），供下游复用、避免重复解码。
- **Phase 1 扇出**：编排器对四个专科 agent 各连一条边（`graph.py:125-128`），LangGraph 对无依赖节点**自动并行**。
- **Phase 2 barrier**：四条边汇入 `crossmodal_eval`（`graph.py:131-134`）——跨模态对齐必须等所有单模态就绪。
- **Phase 3**：`crossmodal → diagnosis → END`（`graph.py:137-140`）。`evaluate_video()` 建图→`graph.invoke()`→取 `final_state["diagnosis"]`→`ReportWriter.append()` 落 JSONL（`main.py:110-157`），CLI 按 grade 决定退出码（`D/F` 退 1，`main.py:368-370`）。

**checkpoint 评测引擎（`agents/base.py`，所有专科 agent 共用）**：

1. **ContentProfile 门控**：`_build_content_profile`（`base.py:229`）用 VLM+ASR 判定视频是否「有角色/有手持物/有场景切换/有对白」等布尔属性；每个 `CheckpointDef` 带 `applicable_when`，`_filter_applicable`（`base.py:383-391`）据此过滤——`object_permanence` 门控 `{"has_held_objects":True}`（`checkpoints.py:60`）、`temporal_logic` 门控 `{"has_scene_changes":True}`（`checkpoints.py:102`）。被跳过的 checkpoint **不计入分母**。
2. **防御式单 checkpoint 评测**：`_evaluate_single_checkpoint`（`base.py:769`）用「缺陷优先/怀疑论」提示让 VLM 先找毛病再打分；三重防护——id 不匹配重试、`factual_override`（如 `duration_completeness` 用真实时长直接覆盖 VLM，`base.py:735`）、`_check_reasoning_score_consistency`（`base.py:476`，检测「嘴上说问题、分却给高」的矛盾并重评）。
3. **归一化**：BINARY→`float(raw_val)`，LIKERT→`(raw_val-1)/4.0`（`base.py:1057-1062` / `1270-1275`）。Likert 1–5 配 `RubricAnchor` 锚点，`_build_rubric_text`（`base.py:394`）拼进提示。
4. **聚合**：`_aggregate_checkpoint_score`（`base.py:1304`）对**激活的** checkpoint 按 `weight` 加权平均，并**在激活集合上重新归一化**（被跳过项权重自动补足）。

**视频 agent 的「算法证据层」**（`agents/video_agent.py`，5 个 agent 中唯一带纯算法证据）：`_compute_visual_evidence`（`:157`）算相邻帧 pairwise SSIM、直方图卡方距离、像素差、Haar 人脸计数，翻译成自然语言塞进 VLM 提示。关键的**算法–VLM 分歧复评**（`_eval_temporal_coherence`，`:346`）：

```python
if avg_ssim < 0.3 and r.raw_value >= 3:   # video_agent.py:414
    # VLM 给了 3+，但 avg_ssim<0.3 说明相邻帧极度不同 → 带证据逼 VLM 重评
```

转场质量两层检测（`_eval_transition_quality`，`:559`）：Layer-1 算 `composite=0.5*ssim+0.25*hist+0.25*flow`（`:600`），`<0.6` 标记可疑（`:604`）；Layer-2 只让 VLM 对被标记边界分类「硬切穿帮 vs 合理切换」，惩罚 `penalty=verified_bad*0.08`（`:680`）——避免高动态运镜（whip_pan）被冤枉。

**诊断综合（`agents/diagnosis.py`）**：`_compute_dimension_scores`（`:114`）按 `Σ(score·conf)/Σ(conf)` 做置信度加权（某维度无结果则整体省略、不拉低总分，`:139`）；`_compute_overall_score`（`:164`）用 profile 的 `priority_weights` 加权（映射 `_DIM_TO_WEIGHT_FIELD` `:156-162`），只有有分的维度参与、被跳过维度权重从分母剔除再归一化（`:175-189`）；`_assign_grade`（`:192`）：A≥0.85/B≥0.70/C≥0.55/D≥0.40/否则 F；`_identify_bottlenecks`（`:207`）挑 `score<阈值` 的子指标按分升序、连同 suggestions 进报告；最后 `_generate_narrative` 生成叙事诊断。

## 4. 用到的模型

- **VLM 评测主力**：多模态大模型（GPT-4o / Gemini 类）在各 checkpoint 上看图+读 ASR 打分（`agents/base.py` 评测引擎、`llm_utils.py`）。
- **预处理工具链**：PySceneDetect（分镜）、ffmpeg + AudioShake（音轨分离）、Whisper/Azure（ASR）、OpenCV（SSIM/直方图/光流/Haar 人脸）。
- **算法证据**：纯 OpenCV 数值，用于**反驳/校正** VLM 的乐观判断（视频 agent 专属）。

## 5. 实际数据案例全过程

以 `DB_001`「屋顶追逐战」（67s、动作类）+ **Profile 1「Story-First」**（`--profile-id 1`，`main.py:334-338`）为例：

1. **Phase 0**：编排器把 67s 成片切镜头、抽音轨分离人声/BGM/音效、跑 ASR 拿对白与说话人、对每个镜头边界算 SSIM/直方图/光流，全部写入 `GraphState.preprocessing`。出题卡只给 shot_id 1（18s），分镜检测给出成片**真实**镜头边界。
2. **Phase 1 并行**：
   - **视频 agent**：先 ContentProfile 判定「有英雄+目标两角色→`has_characters=True`」「追逐戏无手持道具→`has_held_objects=False`」「全程屋顶无昼夜变化→`has_scene_changes=False`」。于是 `char_face_consistency / char_clothing` **激活**，`object_permanence / temporal_logic` **被跳过、不计入分母**——`temporal_coherence` 维度实际只在 `char_face / char_clothing / background / scale_proportion / motion_continuity` 上打分。
   - 多角度快切使 OpenCV 算出相邻帧低 `avg_ssim`。若 VLM 在 `char_face_consistency` 给 4 分但 `avg_ssim<0.3`，触发 `video_agent.py:414` 的**分歧复评**：带「SSIM<0.3，3+ 分需证据」提示让 VLM 复看人脸区域——把「运镜快」与「换人穿帮」解耦。
   - 出题卡要 `whip_pan/tracking`，这类运镜在转场 Layer-1 因高光流被标记，但 Layer-2 判为「合理」→**不扣分**。
3. **Phase 2 跨模态**：核对「别跑！」对白是否对上口型、`footsteps/impact` 音效是否踩在跳跃/落地画面上。
4. **Phase 3 诊断**：各维度按置信度加权得分；按 **Profile 1 的 `text_story_arc=0.55`** 加权出总分——剧本维度主导，故只要三幕结构（发现目标→追逐→制服）完整，视觉一致性因快切的小瑕疵对 Story-First 用户总分影响有限（若换 Profile 2「Visual-Heavy」`visual_camera=0.50`，同一份子指标会算出完全不同的总分）。最后定 grade、列瓶颈、生成叙事报告，`ReportWriter.append` 落 JSONL，工具调用轨迹（含耗时）写 `tool_traces.jsonl`（`main.py:31-60`）。

> **一句话回放**：出题卡 `DB_001` → Phase 0 切镜头/抽音轨/ASR/算转场度量 → Phase 1 四 agent 并行（ContentProfile 门控跳过 N/A 项 + 算法–VLM 分歧复评区分快切与穿帮 + 转场两层检测放过合理甩镜）→ Phase 2 跨模态核对口型与音效 → Phase 3 置信度加权 + Profile 加权出总分/grade/瓶颈/叙事 → 落 JSONL。

## 6. 指标公式速查表（简介·模型·公式·参数）

> DirectorBench 是**诊断式**框架：底层是 checkpoint 归一化分，逐级加权聚合为维度分、总分、grade。

| 指标 | 简介 | 模型 | 计算公式 + 参数说明 |
|---|---|---|---|
| **Checkpoint 归一化** | 单个子指标打分 | VLM + 算法证据 | LIKERT：$(\text{raw}-1)/4$（1..5→0..1）；BINARY：$\text{float}(\text{raw})$（0/1）·· `base.py:1057-1062` |
| **子指标聚合** | 一组 checkpoint→一个子指标 | — | $S=\dfrac{\sum_{a}w_a x_a}{\sum_{a}w_a}$ ·· 仅对**激活集** $a$ 求和（门控跳过项权重不入分母）·· `base.py:1304` |
| **转场惩罚** | 坏硬切扣分 | OpenCV+VLM | $\text{penalty}=n_{bad}\times0.08$，其中 `composite=0.5·ssim+0.25·hist+0.25·flow`，`composite<0.6` 标记、VLM 确认才计 ·· `video_agent.py:600/680` |
| **维度分** | agent 多 checkpoint→维度 | — | $S_{dim}=\dfrac{\sum_i s_i c_i}{\sum_i c_i}$ ·· `c`=置信度；无结果维度整体省略 ·· `diagnosis.py:144-146` |
| **总分** | 维度→个性化总分 | — | $\text{Overall}=\dfrac{\sum_{d}S_d w_d}{\sum_{d}w_d}$ ·· `w`=profile `priority_weights`，仅活跃维度参与、权重重归一化 ·· `diagnosis.py:175-189` |
| **Grade** | 字母评级 | — | A≥0.85 / B≥0.70 / C≥0.55 / D≥0.40 / 否则 F ·· `diagnosis.py:192` |

**参数说明**：①checkpoint 用 `applicable_when` 门控「看人下菜」，没角色就不评人脸一致、N/A 项不入分母；②算法证据层（SSIM/直方图/光流/人脸计数）用于**反驳** VLM 乐观偏差，并专门处理「低 SSIM 是快切还是穿帮」的分歧复评；③同一份客观子指标经 5 种 profile 加权得到不同总分与 grade——「评测标准本身因人而异」被显式建模。

---
**一句话定位**：DirectorBench = 用 **LangGraph 多智能体 DAG** 对**分钟级多镜头成片**做**诊断式评测**的框架——5 专科 agent 并行 + checkpoint 门控 + 算法证据层校正 VLM + 用户画像加权，输出瓶颈/建议/叙事报告而非单一跑分。
