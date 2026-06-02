"""
config.py — Global configuration for the evaluation framework.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for the LLM backend."""
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.1        # low temp for evaluation consistency
    max_tokens: int = 4096
    api_key: str = ""               # set via env var OPENAI_API_KEY
    base_url: str | None = None     # for local models via vLLM/Ollama


class VLMConfig(BaseModel):
    """Configuration for the Vision-Language Model backend."""
    provider: str = "openai"
    model: str = "gpt-4o"           # GPT-4o supports vision
    max_frames_per_shot: int = 8    # frames to sample per shot for VLM
    api_key: str = ""


class PreprocessConfig(BaseModel):
    """Configuration for the preprocessing pipeline."""
    # Shot detection
    shot_detection_threshold: float = 27.0  # PySceneDetect threshold
    min_shot_duration_sec: float = 0.5

    # ASR
    asr_model: str = "whisper-large-v3"
    asr_language: str = "auto"

    # Audio separation
    separate_audio_tracks: bool = True

    # Frame extraction
    frames_per_shot: int = 8        # representative frames per shot
    frame_output_format: str = "jpg"


class EvalConfig(BaseModel):
    """Master configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)

    # Scoring
    confidence_threshold: float = 0.6   # below this → flag for human review
    bottleneck_threshold: float = 0.5   # score below this → bottleneck

    # Grade boundaries
    grade_boundaries: dict[str, float] = Field(default_factory=lambda: {
        "A": 0.85, "B": 0.70, "C": 0.55, "D": 0.40, "F": 0.0
    })

    # Output
    output_dir: str = "./eval_outputs"
    save_intermediate: bool = True
