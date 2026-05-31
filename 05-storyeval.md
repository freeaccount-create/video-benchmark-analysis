# StoryEval — 故事级"事件完成度"评测

> 仓库：`ypwang61/StoryEval` · 核心思想：用 VLM 逐个判定"故事里的每个事件是否在生成视频中真的发生了"

## 1. 数据集由来

现有 T2V 指标多关注画质/时空一致性，却不回答一个关键问题：**生成的视频有没有把故事里多个连续事件都演出来？** StoryEval 专为此设计——把每个故事分解成 2–4 个原子事件，再让 VLM 逐事件检查完成情况。例如"把大象放进冰箱"= 开门→放象→关门三个事件，好的模型应在一个视频里演全。

数据规模：**423 条 prompt**（2 事件 133 条、3 事件 262 条、4 事件 28 条）。

## 2. 原始数据格式

`prompts/all_prompts.json`，键是视频文件名，值含 prompt / 事件列表 / 分类标签。**仓库真实第 0 条**：
```json
"A_CD_is_inserted_into_a_player_and_then_spins_up.mp4": {
  "prompt": "A CD is inserted into a player, and then spins up.",
  "event_list": ["A CD is inserted into a player", "And then the CD spins up"],
  "class": ["object", "medium", "2_events"]
}
```
`class` 三维：内容类型(human/animal/object) + 难度(easy/medium/hard) + 事件数(2/3/4_events)。

## 3. 完整打分流程

入口 `./evaluate.sh` →
```bash
python evaluate.py --eval_model_type llava \
    --generative_model_list model_x::model_y --prompts_file_name all_prompts.json \
    --repeat_time 3 --max_frames_num 16
```

单个视频流程（`evaluate.py:251-596` `process_videos_in_directory`）：
1. **抽帧**：`load_video()`(utils.py:160) Decord 均匀采样 16–32 帧，统一 512×320。
2. **生成描述**（Step1, L358-425）：VLM 按时间顺序详细描述关键帧。
3. **组装评分问题**（Step2, L427）：把描述 + 原 prompt + 事件列表拼进 `general_template`。
4. **逐事件判定**（Step3, L436-528）：VLM 严格判断每个事件是否完成（模糊/动作不清→判 0），还要检查跨事件主体一致性；末尾输出 `[COMPLETE_LIST]: 1, 0`。
5. **抽取**（L235-248）：正则 `\[COMPLETE_LIST\]:\s*(.*)` 取最后一个匹配里的数字；若长度与事件数不符则重试（≤3 次）。
6. **单次分**：`completion_score = sum(list)/len(list)`。
7. **投票**：每视频跑 `repeat_time=3` 次（seed 0/1/2），逐事件平均 + 总体平均。

### 评分定义
- `completion_list`：长度=事件数的 0/1 列表，如 `[1,0]`。
- `completion_score`：完成事件数 / 总事件数 ∈ [0,1]。
- **严格投票**（vote_type=1）：3 次全判完成才算完成。
- 最终按 class 分组求平均（`summarization.py`）。

## 4. 用到的模型（VLM 裁判，可选其一）

- **LLaVA-OneVision-Qwen2-72B**（默认 `lmms-lab/llava-onevision-qwen2-72b-ov-chat`，需 ≥4×49GB GPU）
- **GPT-4o / GPT-4-Vision**（Azure API，抽 8 帧）
- **Qwen2-VL-72B-Instruct**

## 5. 实际数据案例全过程

取真实条目 `A_CD_is_inserted_into_a_player_and_then_spins_up.mp4`（2 事件）：

1. **输入**：被测模型生成的同名视频 + `event_list = ["A CD is inserted into a player", "And then the CD spins up"]`。
2. **Step1**：VLM 看 16 帧 → 描述"a CD tray opens, a disc is placed, the disc starts spinning..."。
3. **Step2**：把描述 + prompt + 2 个事件拼成评分问题。
4. **Step3**：VLM 判定 → 输出 `... Finally we have [COMPLETE_LIST]: 1, 1`。
5. **单次分**：`(1+1)/2 = 1.0`。
6. **投票**：跑 3 次，假设结果 `[1,1] / [1,0] / [1,1]` → 逐事件平均 `[1.0, 0.667]`，总体 `completion_score_avg = (1.0+0.5+1.0)/3 ≈ 0.83`；严格投票下事件2因非全 1 记 0。
7. **输出**：`results/{model}_llava_final.json`，每条含 `completion_list_avg`、`completion_score_avg` 及 3 次 `outputX`(description/scoring_output/seed)；`summarization.py` 按 class 汇总各类平均完成率。

---
**一句话定位**：StoryEval = 把故事拆成事件，用大 VLM"看视频→逐事件判完成→3 次投票"，最终输出故事完成率，专测叙事完整性而非画质。
