# DirectorBench

**DirectorBench** is a multi-agent evaluation framework for AI-generated **minute-long videos**. Unlike existing benchmarks that focus on short clips (3–20s), DirectorBench targets the unique challenges of long-form video generation: multi-shot narrative coherence, cross-modal alignment, temporal consistency, generation stability, and audio-visual synchronization.

---

## Why Minute-Long Videos Need a Different Benchmark

| Capability | Short Clip (3–20s) | Minute-Long Video |
|---|---|---|
| Shot Count / Transitions | Single-shot or pseudo-continuous | Multi-shot, controllable transitions (cuts / dissolves), varied camera movements |
| Perspective / Camera Control | Simple pan, slight zoom | Director-level: push-in/pull-out, tracking, orbiting, crane/boom, zoom, Dutch angle, handheld |
| Character / Scene Consistency | Locally maintainable | Strong long-term consistency: identity, clothing, lighting, spatial layout |
| Spatiotemporal Realism | Locally reasonable | Cross-shot physical continuity: gravity, momentum, causality |
| Narrative Coherence | Single action or scene | Complete mini-story arc (three-act), causal logic, emotional progression |
| Storyboard / Shot Design | None | Storyboard-level shot intentions (wide → close-up → over-the-shoulder) |
| Dialogue & Narration | None, or simple post-added | Multi-character dialogue, lip-sync, voiceover, ambient audio |

---

## Architecture

```
                             ┌──────────────┐
                             │  Input Data  │
                             │  video_path  │
                             │  script_text │  (optional)
                             │  user_prompt │
                             │  storyboard  │  (optional)
                             └──────┬───────┘
                                    │
                        ┌───────────▼──────────────┐
                        │   Phase 0: Orchestrator   │
                        │  • Shot detection (PySD)  │
                        │  • Boundary frame extract  │
                        │  • Transition metrics      │
                        │  • Audio extraction (ffmpeg)│
                        │  • Audio separation (API)  │
                        │  • ASR transcription (API) │
                        │  • Frame sampling          │
                        └───────────┬──────────────┘
                                    │ dispatch
              ┌─────────┬───────────┼───────────┬──────────┐
              ▼         ▼           ▼           ▼          │
        ┌──────────┐┌──────────┐┌──────────┐┌──────────┐  │
        │  Script  ││  Video   ││  Audio   ││ Stability│  │
        │  Agent   ││  Agent   ││  Agent   ││  Agent   │  │
        └────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘  │
             │           │           │           │         │
             └─────┬─────┴─────┬─────┘           │         │
                   │  barrier  │                 │         │
              ┌────▼───────────▼─────────────────▼─┐       │
              │  Phase 2: Cross-Modal Agent        │       │
              │  • Text ↔ Video alignment          │       │
              │  • Video ↔ Audio alignment         │       │
              │  • Text ↔ Audio alignment          │       │
              │  • Overall multimodal harmony       │       │
              └──────────────┬─────────────────────┘       │
                             │                             │
                    ┌────────▼─────────┐                   │
                    │  Phase 3:        │◄──────────────────┘
                    │  Diagnosis       │
                    │  Synthesizer     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ DiagnosisReport  │
                    │ • Overall score  │
                    │ • Dimension map  │
                    │ • Bottlenecks    │
                    │ • Suggestions    │
                    │ • Radar chart    │
                    └──────────────────┘
```

The framework uses **LangGraph** to define the execution DAG. Phase 1 agents run in parallel; Phase 2 waits for all Phase 1 results via a barrier sync; Phase 3 aggregates everything into the final report.

---

## Evaluation Dimensions

DirectorBench evaluates across **5 specialist agents** producing **14 sub-metrics**, plus a cross-modal alignment phase:

| Agent | Sub-Metric | What It Measures |
|---|---|---|
| **Script Agent** | `script_reasonableness` | Logical flow, cause-effect, plot coherence |
| | `script_novelty` | Originality, avoidance of clichés |
| | `user_requirement_consistency` | Alignment with user prompt |
| | `script_video_fidelity` | Script ↔ video faithfulness (reference mode only) |
| **Video Agent** | `user_demand_fulfillment` | Visual match to user-specified demands |
| | `temporal_coherence` | Smooth transitions, no unjustified discontinuities |
| | `lighting_consistency` | Consistent light sources and shadows across shots |
| | `transition_quality` | Splice artifact detection at clip boundaries (frame-level) |
| **Audio Agent** | `narration_reasonableness` | Dialogue timing, naturalness, emotional fit (skipped if no speech) |
| | `bgm_consistency` | Background music matches mood and pacing (skipped if no BGM) |
| **Stability Agent** | `generation_stability` | Quality maintenance over full minute, no degradation |
| **Cross-Modal Agent** | `text_video_consistency` | Script ↔ video alignment |
| | `video_audio_consistency` | Lip-sync, event-audio matching |
| | `text_audio_consistency` | Script ↔ spoken dialogue alignment |

The **Script Agent** operates in two modes: if a reference script is provided, it evaluates script quality and video-to-script fidelity; if no script is provided, it extracts the narrative from the video via VLM + ASR and evaluates the extracted narrative's quality.

Each sub-metric returns a normalized **EvalResult** with: score (0–1), confidence, evidence chain, and actionable suggestions.

### Grading Scale

| Grade | Score Range | Description |
|---|---|---|
| **A** | ≥ 0.85 | Excellent, broadcast-quality |
| **B** | ≥ 0.70 | Strong, minor imperfections |
| **C** | ≥ 0.55 | Competent, some noticeable issues |
| **D** | ≥ 0.40 | Marginal, significant issues |
| **F** | < 0.40 | Poor, fundamental problems |

---

## Project Structure

```
DirectorBench/
├── config/
│   ├── default_config.yaml          # Default LLM model, thresholds, frame extraction
│   └── user_config_template.yaml    # Template for per-user overrides
│
├── directorbench/
│   ├── __init__.py                  # Package entry
│   ├── schemas.py                   # Pydantic v2 data models (GraphState, EvalResult,
│   │                                #   UserProfile, UserTaste, PriorityWeights, ...)
│   ├── config.py                    # EvalConfig, LLMConfig, VLMConfig, PreprocessConfig
│   ├── llm_utils.py                # Unified LLM client (Azure OpenAI / standard OpenAI)
│   ├── preprocessing.py            # Phase 0: shot detection, audio separation (AudioShake),
│   │                                #   ASR (Azure Whisper API), boundary frame analysis
│   ├── graph.py                    # LangGraph DAG definition (Phase 0 → 1 → 2 → 3)
│   ├── report.py                   # JSONL append-only result persistence + MD renderer
│   ├── main.py                     # CLI entry point + evaluate_video() / evaluate_batch() API
│   ├── example_usage.py            # Quick-start usage examples
│   │
│   └── agents/
│       ├── __init__.py
│       ├── base.py                 # BaseEvalAgent abstract class
│       ├── script_agent.py         # Script evaluation (4 sub-metrics, reference/reference-free)
│       ├── video_agent.py          # Video evaluation (4 sub-metrics, incl. frame-level splice detection)
│       ├── audio_agent.py          # Audio evaluation (2 sub-metrics, absent data → skip)
│       ├── stability_agent.py      # Generation stability (quality degradation detection)
│       ├── crossmodal_agent.py     # Cross-modal alignment (4 sub-metrics, Phase 2)
│       └── diagnosis.py            # Diagnosis Synthesizer (confidence-weighted + profile-weighted)
│
├── baselines/                       # Reference video generation systems
│   └── MovieAgent/                 # Multi-agent movie generation pipeline
│
├── data/
│   ├── profiles.jsonl              # User evaluation profiles (5 archetypes)
│   └── samples/
│       └── sample_metadata.json     # Example metadata for a 90s chase sequence
│
├── render_report.py                # CLI tool: JSONL → Markdown report by report_id
├── setup.py
├── requirements.txt
└── README.md
```

---

### Required Dependencies

```bash
pip install pydantic openai langgraph langchain-core requests
pip install scenedetect[opencv]    # shot detection (pure algorithm)
pip install opencv-python numpy    # transition metrics (SSIM, flow)
```

---

## Quick Start

### Python API

```python
from directorbench.main import evaluate_video, load_all_profiles

# --- Load a profile from profiles.jsonl ---
profiles = load_all_profiles()
visual_user = profiles[1]  # profile_id=2, Visual-Heavy User

# --- With reference script (reference mode) ---
report = evaluate_video(
    video_path="path/to/video.mp4",
    user_prompt="A dramatic chase scene through a neon-lit city at night",
    script_text="Scene 1: Detective spots suspect on rooftop...",
    user_profile=visual_user,      # profiles.jsonl dict, auto-parsed
    output_dir="./eval_outputs",
)

# --- Without script (reference-free mode) ---
# Narrative is extracted from video via VLM + ASR automatically
report = evaluate_video(
    video_path="path/to/video.mp4",
    user_prompt="A dramatic chase scene through a neon-lit city at night",
    output_dir="./eval_outputs",
)

print(f"Grade: {report.overall_grade} | Score: {report.overall_score:.2f}")
for b in report.bottlenecks:
    print(f"  Bottleneck: {b.metric_name} = {b.score:.2f}")
```

### CLI

```bash
# Single video evaluation with profile
python -m directorbench.main \
  --video path/to/video.mp4 \
  --prompt "A dramatic chase scene..." \
  --script "Scene 1: ..." \
  --profile-id 2 \
  --output-dir ./eval_outputs

# Reference-free mode (no script)
python -m directorbench.main \
  --video path/to/video.mp4 \
  --prompt "A dramatic chase scene..." \
  --profile-id 1 \
  --output-dir ./eval_outputs
```

### Viewing Results

All evaluation runs are appended to `eval_outputs/results.jsonl` (never overwrites). Use `render_report.py` to view or export:

```bash
# List all evaluation records
python render_report.py --list
#   #  Report ID       Grade   Score  Profile                 Video Path
#   1  a1b2c3d4e5f6        B    0.72  Visual-Heavy User (#2)  path/to/video.mp4

# Render a specific report as Markdown
python render_report.py --id a1b2c3d4e5f6

# Render the most recent report to a file
python render_report.py --latest --out reports/latest.md
```

---

## Preprocessing Pipeline (Phase 0)

The Orchestrator runs a preprocessing pipeline before dispatching to agents:

1. **Video probing** — ffprobe for duration, FPS, resolution
2. **Shot detection** — PySceneDetect ContentDetector (fallback: uniform segmentation)
3. **Frame extraction** — Representative thumbnail per shot via ffmpeg
4. **Transition analysis** — Boundary frame extraction at each cut point, plus algorithmic metrics (SSIM, colour histogram chi-square distance, Farneback optical flow) computed via OpenCV
5. **Audio extraction** — ffmpeg to isolate audio track
6. **Audio separation** — AudioShake API for dialogue / BGM splitting (fallback: mixed track)
7. **ASR transcription** — Azure OpenAI Whisper API for speech-to-text with segment timestamps (fallback: local Whisper)

All tools have graceful fallbacks — the framework runs even without optional API keys, producing degraded but valid evaluations.

---

## Script Agent: Reference vs Reference-Free Mode

| Mode | When | What Happens |
|---|---|---|
| **Reference** | `script_text` or `storyboard` provided | Evaluates script quality + video-to-script fidelity |
| **Reference-free** | No script provided | Extracts narrative from video (VLM on frame sequences + ASR transcript fusion), then evaluates extracted narrative quality |

In reference-free mode, the agent reconstructs the story arc by analyzing sequential video frames with GPT-4o Vision and aligning ASR dialogue to shots. This extracted narrative is then evaluated for logical coherence and creativity, just like a provided script would be.

---

## Transition Quality Detection

The `transition_quality` sub-metric (Video Agent) detects splice artifacts at clip concatenation points using a two-layer approach:

**Layer 1 — Algorithmic detection** (pure OpenCV, no models): extracts boundary frames at each detected shot cut, computes SSIM, colour histogram distance, and optical flow magnitude. Transitions with low composite quality scores are flagged as suspicious.

**Layer 2 — VLM semantic verification**: flagged boundaries are sent to GPT-4o Vision along with their algorithmic metrics. The VLM classifies each as either an intentional scene change (acceptable) or a same-scene splice artifact (defect), identifying the specific discontinuity type (frame jump, colour shift, character teleport, flicker, or motion break).

---

## Inter-Agent Communication Protocol

Every agent returns standardized `EvalResult` objects — the fundamental unit of inter-agent communication:

```python
EvalResult(
    agent_id    = "video_eval_agent",
    metric_name = "transition_quality",
    score       = 0.68,
    confidence  = 0.85,
    granularity = "frame-level",
    evidence    = [
        Evidence(
            type="splice_artifact",
            timestamp="12.4s",
            issue="Clip splice artifact (colour_shift) at shot 3→4",
            severity="medium",
        )
    ],
    intermediate_repr = {
        "total_transitions": 8,
        "algorithmically_flagged": 2,
        "vlm_verified_bad": 1,
    },
    suggestions = [
        "Add cross-fade or smooth interpolation at 12.4s"
    ],
)
```

Results with `confidence` below the threshold (default 0.6) are flagged for human review in the diagnosis report.

---

## User Profile & Personalization

Each evaluation run is associated with a **user profile** that controls dimension weighting, taste preferences, and hard constraints. Profiles are defined in `data/profiles.jsonl`:

```json
{
  "profile_id": 2,
  "name": "Visual-Heavy User",
  "personalization": {
    "user_taste": {
      "focus": "visual_heavy",
      "wants_bgm": false,
      "camera_movement": "important",
      "lighting": "very_important"
    },
    "priority_weights": {
      "text_story_arc": 0.15,
      "visual_camera": 0.50,
      "audio_emotion": 0.10,
      "cross_modal_sync": 0.25
    },
    "hard_constraints": ["complex_camera_movements", "beautiful_lighting"]
  }
}
```

### Built-in Profile Archetypes

| ID | Name | Primary Weight | Key Preferences |
|----|------|----------------|-----------------|
| 1 | Story-First User | text=0.55 | Three-act arc, causal logic |
| 2 | Visual-Heavy User | visual=0.50 | Camera, lighting; no BGM needed |
| 3 | Audio & Emotion User | audio=0.45 | Emotional BGM, tone control |
| 4 | Sync Perfectionist | sync=0.40 | Lip-sync, text-visual match |
| 5 | Creative Dreamer | visual=0.30, text=0.25 | Fantasy effects, dreamy transitions |

### How Profiles Affect Scoring

**Dimension weighting** — `priority_weights` directly controls the final overall score aggregation. A Visual-Heavy user's video dimension contributes 50% to the total.

**User taste fulfillment** — The `user_requirement_consistency` sub-metric (Script Agent) evaluates whether the video respects taste preferences. If `wants_bgm=false` but the video has BGM, it is penalised as VIOLATED.

**Hard constraints** — Non-negotiable requirements. Violations receive CRITICAL severity and heavily penalise the user consistency score.

### Scoring Mechanism

**Layer 1 — Dimension score** (within each agent): Multiple sub-metrics are combined via **confidence-weighted average**: `Σ(score × confidence) / Σ(confidence)`. High-confidence metrics dominate; low-confidence ones contribute proportionally less but are not crushed.

**Layer 2 — Overall score** (across dimensions): Dimension scores are combined via **profile-weighted average** using `priority_weights`. Dimensions with no data (e.g. audio in a silent video) are **excluded** from the calculation entirely — they don't drag down the score.

```
overall = Σ(dim_score[i] × weight[i]) / Σ(weight[i])
          only for dimensions where results exist
```

---

## Metadata Format

Each video can be described with a JSON metadata file for batch evaluation:

```json
{
  "video_id": "sample_001",
  "video_path": "data/samples/sample_video.mp4",
  "prompt": "A tense chase sequence through a neon-lit city...",
  "duration_seconds": 90.0,
  "model_name": "VideoGen-v2",
  "has_audio": true,
  "shot_descriptions": [
    {
      "shot_index": 0,
      "start_time": 0.0,
      "end_time": 8.0,
      "camera_type": "crane",
      "description": "Aerial crane shot establishing the city skyline.",
      "characters": ["detective", "suspect"]
    }
  ]
}
```

---

## Output

Every evaluation run is **appended** as a single JSON line to `eval_outputs/results.jsonl`. Each record contains the complete inputs (video path, prompt, script, profile) and the full `DiagnosisReport` output — nothing is ever overwritten.

Use `render_report.py --list` to browse records and `render_report.py --id <ID>` to export any record as Markdown.

Console output shows a coloured summary with progress bars; the `DiagnosisReport` includes: overall score/grade (profile-weighted), per-dimension breakdown (confidence-weighted), bottleneck identification, low-confidence items for human review, radar chart data, and an LLM-generated narrative summary.

---

## Running Tests

```bash
pytest tests/ -v
```

