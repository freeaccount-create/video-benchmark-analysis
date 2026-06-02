"""
schemas.py — Core data models for the evaluation framework.

Defines the communication protocol between agents: every Specialist Agent
produces a list of `EvalResult` objects that flow into the Cross-Modal Agent
and finally into the Diagnosis Synthesizer.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Annotated
import operator

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK = "fallback"   # tool failed but a degraded fallback was used
    SKIPPED = "skipped"     # tool not invoked (e.g. missing API key)


class AgentID(str, Enum):
    ORCHESTRATOR = "orchestrator"
    SCRIPT_EVAL = "script_eval_agent"
    VIDEO_EVAL = "video_eval_agent"
    AUDIO_EVAL = "audio_eval_agent"
    STABILITY_EVAL = "stability_eval_agent"
    CROSSMODAL_EVAL = "crossmodal_eval_agent"
    DIAGNOSIS = "diagnosis_synthesizer"


# ---------------------------------------------------------------------------
# Evidence & EvalResult — the inter-agent communication protocol
# ---------------------------------------------------------------------------

class ToolCallRecord(BaseModel):
    """Record of a tool / API call during preprocessing or evaluation.

    These records are collected in GraphState.tool_context so that downstream
    agents can see which tools succeeded, failed, or fell back to degraded
    mode.  The LLM is shown this context and can lower its confidence when
    critical information is missing.
    """
    tool_name: str              # e.g. "PySceneDetect", "AudioShake", "ASR/OpenAI", "Librosa"
    status: ToolStatus
    detail: str = ""            # human-readable explanation of what happened
    elapsed_ms: Optional[float] = None
    affects: list[str] = Field(
        default_factory=list,
        description="Metrics / dimensions this tool's output feeds into",
    )


# ---------------------------------------------------------------------------
# Checkpoint-based evaluation (dynamic rubric system)
# ---------------------------------------------------------------------------

class CheckpointType(str, Enum):
    """How the checkpoint is scored."""
    BINARY = "binary"       # pass/fail → 0 or 1
    LIKERT = "likert"       # 1-5 scale with anchored rubric descriptions
    ORDINAL = "ordinal"     # custom ordered categories (e.g. none/minor/major/critical)


class RubricAnchor(BaseModel):
    """One level in an anchored scoring rubric (e.g. 'Score 3: ...')."""
    value: int              # 1-5 for likert, 0-1 for binary
    label: str              # short label, e.g. "Good", "Poor"
    description: str        # concrete description of what this score means


class CheckpointDef(BaseModel):
    """Definition of a single evaluation checkpoint within a sub-metric.

    Each checkpoint is a specific, narrow question that the VLM/LLM answers.
    The ``applicable_when`` field enables *dynamic rubrics*: checkpoints are
    only activated when the video content matches the condition.

    Example:
        CheckpointDef(
            id="char_clothing_consistency",
            question="Do characters' clothing/accessories remain consistent …",
            checkpoint_type=CheckpointType.LIKERT,
            weight=0.15,
            applicable_when={"has_characters": True},
            rubric=[
                RubricAnchor(value=5, label="Perfect", description="All characters …"),
                ...
            ],
        )
    """
    id: str                                     # unique key, e.g. "char_face_consistency"
    question: str                               # the narrow question posed to VLM/LLM
    checkpoint_type: CheckpointType = CheckpointType.LIKERT
    weight: float = 1.0                         # relative weight (renormalised at runtime)
    applicable_when: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Content-profile conditions that must ALL be true for this "
            "checkpoint to be active.  Empty dict → always active."
        ),
    )
    rubric: list[RubricAnchor] = Field(
        default_factory=list,
        description="Anchored descriptions for each score level (required for likert).",
    )
    na_score: Optional[float] = None            # if set, use this when checkpoint is N/A
                                                 # instead of dropping it from aggregation


class CheckpointResult(BaseModel):
    """The VLM/LLM's answer to a single checkpoint."""
    checkpoint_id: str
    raw_value: int                               # 0-1 for binary, 1-5 for likert
    normalised: float = 0.0                      # mapped to [0, 1]
    reasoning: str = ""
    applicable: bool = True                      # False → was skipped


class ContentProfile(BaseModel):
    """Structured description of what a video contains.

    Generated once per video by a fast VLM pass at the start of evaluation.
    Agents use this to decide which checkpoints are applicable.
    """
    # Entities
    has_characters: bool = False
    character_count: int = 0
    has_dialogue: bool = False
    has_held_objects: bool = False
    has_animals: bool = False

    # Scene
    scene_count: int = 1
    has_scene_changes: bool = False
    is_single_shot: bool = False

    # Style / content
    has_text_overlay: bool = False
    has_special_effects: bool = False
    is_live_action_style: bool = False
    is_animation_style: bool = False
    has_background_music: bool = False

    # Motion
    has_camera_movement: bool = False
    has_fast_motion: bool = False
    has_slow_motion: bool = False

    # Custom / extensible
    extra: dict[str, Any] = Field(default_factory=dict)

    def matches(self, conditions: dict[str, Any]) -> bool:
        """Return True if ALL conditions match this profile.

        Supports dot-notation for ``extra`` keys:
            {"extra.weather": "rain"} checks self.extra.get("weather") == "rain"
        """
        for key, expected in conditions.items():
            if key.startswith("extra."):
                actual = self.extra.get(key[6:])
            else:
                actual = getattr(self, key, None)
            if actual != expected:
                return False
        return True


class Evidence(BaseModel):
    """A single piece of evidence supporting a score."""
    type: str                           # e.g. "shot_transition", "lip_sync", "script_gap"
    timestamp: Optional[str] = None     # e.g. "00:32-00:34"
    shot_index: Optional[int] = None    # which shot (0-indexed)
    issue: str                          # human-readable description
    severity: Severity = Severity.MEDIUM


class EvalResult(BaseModel):
    """
    Standardized output from every Specialist Agent for each sub-metric.
    This is the fundamental unit of inter-agent communication.
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent_id: AgentID
    metric_name: str                    # e.g. "temporal_coherence"
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    granularity: str = "shot-level"     # "frame-level" | "shot-level" | "video-level"
    evidence: list[Evidence] = Field(default_factory=list)
    intermediate_repr: dict[str, Any] = Field(default_factory=dict)
    checkpoint_results: list[CheckpointResult] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Preprocessing outputs
# ---------------------------------------------------------------------------

class ShotSegment(BaseModel):
    """A single shot (cut) detected in the video."""
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    thumbnail_path: Optional[str] = None  # path to representative frame
    # Boundary frames for transition quality analysis
    last_frame_path: Optional[str] = None   # last frame of this shot
    first_frame_path: Optional[str] = None  # first frame of this shot


class AudioSegment(BaseModel):
    """Separated audio track info."""
    track_type: str     # "dialogue" | "bgm" | "sfx" | "mixed"
    path: str
    duration_sec: float


class ASRSegment(BaseModel):
    """A single ASR (speech-to-text) segment."""
    start_sec: float
    end_sec: float
    text: str
    speaker: Optional[str] = None
    confidence: float = 1.0


class TransitionBoundary(BaseModel):
    """Detected transition boundary between two shots, with pre-extracted
    boundary frames and algorithmic metrics for splice quality analysis."""
    from_shot_index: int
    to_shot_index: int
    timestamp_sec: float                       # exact time of the cut
    # Boundary frame paths (extracted during preprocessing)
    frame_before_path: Optional[str] = None    # last frame of preceding shot
    frame_after_path: Optional[str] = None     # first frame of following shot
    # Algorithmic metrics (computed during preprocessing, no local model needed)
    ssim: Optional[float] = None               # structural similarity [0-1]
    histogram_diff: Optional[float] = None     # color histogram chi-square distance
    optical_flow_magnitude: Optional[float] = None  # Farneback flow magnitude at boundary
    # Classification (filled by evaluation agent)
    is_scene_change: Optional[bool] = None     # True = intentional cut; False = mid-scene splice


class PreprocessingOutput(BaseModel):
    """Aggregated output from the Orchestrator's preprocessing phase."""
    video_path: str
    audio_path: Optional[str] = None
    script_text: Optional[str] = None
    storyboard: Optional[list[dict]] = None  # list of shot intentions

    shots: list[ShotSegment] = Field(default_factory=list)
    transitions: list[TransitionBoundary] = Field(default_factory=list)
    audio_segments: list[AudioSegment] = Field(default_factory=list)
    asr_segments: list[ASRSegment] = Field(default_factory=list)

    total_duration_sec: float = 0.0
    fps: float = 24.0
    resolution: tuple[int, int] = (1920, 1080)

    # Tool call records from the preprocessing phase
    tool_records: list[ToolCallRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# User Profile
# ---------------------------------------------------------------------------

class UserTaste(BaseModel):
    """User's aesthetic / functional preferences — loaded from profiles.jsonl."""
    focus: str = "balanced"                     # story_focused | visual_heavy | audio_emotion | sync_important | creative_fantasy | quick_and_natural | comprehensive_control
    wants_bgm: Optional[bool] = None            # None = don't care
    wants_lip_sync: Optional[bool] = None
    wants_complex_camera: Optional[bool] = None
    emotion_depth: str = "medium"               # low | medium | high
    camera_movement: str = "normal"             # normal | important
    lighting: str = "normal"                    # normal | important | very_important
    bgm_important: Optional[bool] = None
    tone_control: str = "normal"                # normal | important | very_important
    wants_dreamy_effects: Optional[bool] = None
    wants_unusual_camera: Optional[bool] = None
    text_visual_alignment: str = "normal"       # normal | important | critical
    prefers_reference: Optional[bool] = None    # prefers reference-based descriptions


class PriorityWeights(BaseModel):
    """Dimension weights matching profiles.jsonl format.
    Keys correspond to the four evaluation dimensions.
    All weights should sum to ~1.0."""
    text_story_arc: float = 0.25
    visual_camera: float = 0.25
    audio_emotion: float = 0.25
    cross_modal_sync: float = 0.25


class UserProfile(BaseModel):
    """
    User-specific evaluation preferences.

    Compatible with profiles.jsonl format:
        {
            "profile_id": 1,
            "name": "Story-First User",
            "personalization": {
                "user_taste": {...},
                "priority_weights": {...},
                "hard_constraints": [...]
            }
        }
    """
    profile_id: int = 0
    name: str = "default"

    # Nested personalization (matches profiles.jsonl structure)
    user_taste: UserTaste = Field(default_factory=UserTaste)
    priority_weights: PriorityWeights = Field(default_factory=PriorityWeights)
    hard_constraints: list[str] = Field(default_factory=list)

    # Prompt-generation metadata (used by generate_prompts.py;
    # stored in profiles.jsonl alongside the evaluation fields)
    expertise_level: str = "intermediate"       # novice | intermediate | expert
    expression_style: str = "narrative"         # casual | emotional | narrative | technical | precise | structured

    # Evaluation granularity control
    enable_frame_level: bool = False   # expensive; default to shot-level
    max_shots_to_sample: Optional[int] = None  # None = evaluate all shots

    # Additional free-form user requirements (from CLI --prompt or custom input)
    custom_requirements: list[str] = Field(default_factory=list)

    @classmethod
    def from_profile_dict(cls, data: dict) -> "UserProfile":
        """Construct from a single line of profiles.jsonl."""
        personalization = data.get("personalization", {})
        return cls(
            profile_id=data.get("profile_id", 0),
            name=data.get("name", "default"),
            user_taste=UserTaste(**personalization.get("user_taste", {})),
            priority_weights=PriorityWeights(**personalization.get("priority_weights", {})),
            hard_constraints=personalization.get("hard_constraints", []),
            expertise_level=personalization.get("expertise_level", "intermediate"),
            expression_style=personalization.get("expression_style", "narrative"),
        )


# ---------------------------------------------------------------------------
# Diagnosis Report
# ---------------------------------------------------------------------------

class BottleneckItem(BaseModel):
    """Identified bottleneck in the generated video."""
    metric_name: str
    score: float
    agent_id: AgentID
    description: str
    suggestions: list[str] = Field(default_factory=list)


class DiagnosisReport(BaseModel):
    """Final structured output from the Diagnosis Synthesizer."""
    report_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Overall
    overall_score: float = 0.0
    overall_grade: str = "N/A"   # A / B / C / D / F

    # Per-dimension scores (weighted)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    # e.g. {"script": 0.78, "video": 0.65, "audio": 0.80, ...}

    # All individual EvalResults
    all_results: list[EvalResult] = Field(default_factory=list)

    # Bottleneck analysis
    bottlenecks: list[BottleneckItem] = Field(default_factory=list)

    # Low-confidence items needing human review
    needs_human_review: list[EvalResult] = Field(default_factory=list)

    # Radar chart data (for visualization)
    radar_data: dict[str, float] = Field(default_factory=dict)

    # Narrative summary (generated by LLM)
    summary: str = ""
    detailed_analysis: str = ""

    # Content profile used for dynamic checkpoint filtering (persisted for reproducibility)
    content_profile: Optional[ContentProfile] = None

    # Snapshot of checkpoint definitions that were actually evaluated.
    # Keyed by metric_name → list of serialised CheckpointDef dicts.
    # This captures the exact rubric anchors used so results are reproducible.
    checkpoint_registry_snapshot: dict[str, list[dict]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# LangGraph State — the shared state flowing through the graph
# ---------------------------------------------------------------------------

class GraphState(BaseModel):
    """
    The mutable state object passed through the LangGraph workflow.
    Each agent reads from and writes to this shared state.
    """
    # Inputs
    video_path: str = ""
    script_text: str = ""
    storyboard: Optional[list[dict]] = None
    audio_path: Optional[str] = None
    user_prompt: str = ""
    user_profile: UserProfile = Field(default_factory=UserProfile)

    # Preprocessing output
    preprocessing: Optional[PreprocessingOutput] = None
    content_profile: Optional[ContentProfile] = None

    # Reference-free narrative extracted by ScriptEvalAgent from the video
    # itself (VLM thumbnails + ASR). Populated only when `script_text` and
    # `storyboard` are both empty. Downstream agents (CrossModalEvalAgent)
    # should read `script_text or extracted_script_text` instead of
    # `script_text` alone, so that text-video / text-audio metrics keep
    # working in reference-free mode.
    extracted_script_text: str = ""

    # Agent results (populated by each agent)
    script_results: list[EvalResult] = Field(default_factory=list)
    video_results: list[EvalResult] = Field(default_factory=list)
    audio_results: list[EvalResult] = Field(default_factory=list)
    stability_results: list[EvalResult] = Field(default_factory=list)
    crossmodal_results: list[EvalResult] = Field(default_factory=list)

    # Final report
    diagnosis: Optional[DiagnosisReport] = None

    # Execution metadata
    errors: list[str] = Field(default_factory=list)
    # Multiple nodes may append logs concurrently in the same LangGraph step.
    # Use an additive channel reducer to avoid InvalidUpdateError.
    execution_log: Annotated[list[str], operator.add] = Field(default_factory=list)

    # Tool-call context: accumulated records from preprocessing + agents.
    # Agents read this to inform the LLM about degraded / missing data so
    # the LLM can adjust its confidence accordingly.
    tool_context: Annotated[list[ToolCallRecord], operator.add] = Field(default_factory=list)
