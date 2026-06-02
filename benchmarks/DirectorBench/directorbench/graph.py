"""
graph.py — LangGraph DAG definition for the evaluation workflow.

Execution order:
  Phase 0: orchestrator_node (preprocessing + task dispatch)
  Phase 1: script_node, video_node, audio_node, stability_node (PARALLEL)
  Phase 2: crossmodal_node (depends on Phase 1)
  Phase 3: diagnosis_node (depends on Phase 2)

LangGraph handles the DAG scheduling; we use Send() for fan-out
parallelism in Phase 1 and a barrier before Phase 2.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from .agents.audio_agent import AudioEvalAgent
from .agents.crossmodal_agent import CrossModalEvalAgent
from .agents.diagnosis import DiagnosisSynthesizer
from .agents.script_agent import ScriptEvalAgent
from .agents.stability_agent import StabilityEvalAgent
from .agents.video_agent import VideoEvalAgent
from .config import EvalConfig
from .preprocessing import Preprocessor
from .schemas import GraphState

logger = logging.getLogger(__name__)


def build_eval_graph(config: EvalConfig | None = None) -> StateGraph:
    """
    Build and compile the LangGraph evaluation workflow.

    Returns a compiled StateGraph that can be invoked with:
        result = graph.invoke(initial_state)
    """
    config = config or EvalConfig()

    # --- Instantiate agents ---
    preprocessor = Preprocessor(config.preprocess)
    script_agent = ScriptEvalAgent(config)
    video_agent = VideoEvalAgent(config)
    audio_agent = AudioEvalAgent(config)
    stability_agent = StabilityEvalAgent(config)
    crossmodal_agent = CrossModalEvalAgent(config)
    diagnosis = DiagnosisSynthesizer(config)

    # ------------------------------------------------------------------
    # Node functions
    # ------------------------------------------------------------------

    def orchestrator_node(state: GraphState) -> dict[str, Any]:
        """Phase 0: Preprocessing and task setup."""
        logger.info("[Orchestrator] Running preprocessing pipeline...")

        prep_output = preprocessor.run(
            video_path=state.video_path,
            audio_path=state.audio_path,
            script_text=state.script_text,
            storyboard=state.storyboard,
        )

        return {
            "preprocessing": prep_output,
            "execution_log": state.execution_log + [
                f"Orchestrator: preprocessed {prep_output.total_duration_sec:.1f}s video, "
                f"{len(prep_output.shots)} shots, {len(prep_output.asr_segments)} ASR segments"
            ],
            # Propagate preprocessing tool records into the shared tool_context
            # so downstream agents can see which tools succeeded / failed.
            "tool_context": prep_output.tool_records,
        }

    def script_node(state: GraphState) -> dict[str, Any]:
        """Phase 1: Script evaluation."""
        return script_agent(state)

    def video_node(state: GraphState) -> dict[str, Any]:
        """Phase 1: Video evaluation."""
        return video_agent(state)

    def audio_node(state: GraphState) -> dict[str, Any]:
        """Phase 1: Audio evaluation."""
        return audio_agent(state)

    def stability_node(state: GraphState) -> dict[str, Any]:
        """Phase 1: Stability evaluation."""
        return stability_agent(state)

    def crossmodal_node(state: GraphState) -> dict[str, Any]:
        """Phase 2: Cross-modal alignment (depends on Phase 1)."""
        return crossmodal_agent(state)

    def diagnosis_node(state: GraphState) -> dict[str, Any]:
        """Phase 3: Diagnosis synthesis."""
        return diagnosis(state)

    # ------------------------------------------------------------------
    # Build the graph
    # ------------------------------------------------------------------

    # Define the state graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("script_eval", script_node)
    workflow.add_node("video_eval", video_node)
    workflow.add_node("audio_eval", audio_node)
    workflow.add_node("stability_eval", stability_node)
    workflow.add_node("crossmodal_eval", crossmodal_node)
    workflow.add_node("diagnosis", diagnosis_node)

    # --- Edges ---

    # Entry → Orchestrator
    workflow.set_entry_point("orchestrator")

    # Phase 0 → Phase 1 (fan-out to parallel agents)
    # LangGraph executes nodes with no dependency between them in parallel
    workflow.add_edge("orchestrator", "script_eval")
    workflow.add_edge("orchestrator", "video_eval")
    workflow.add_edge("orchestrator", "audio_eval")
    workflow.add_edge("orchestrator", "stability_eval")

    # Phase 1 → Phase 2 (barrier: all Phase 1 must complete)
    workflow.add_edge("script_eval", "crossmodal_eval")
    workflow.add_edge("video_eval", "crossmodal_eval")
    workflow.add_edge("audio_eval", "crossmodal_eval")
    workflow.add_edge("stability_eval", "crossmodal_eval")

    # Phase 2 → Phase 3
    workflow.add_edge("crossmodal_eval", "diagnosis")

    # Phase 3 → END
    workflow.add_edge("diagnosis", END)

    # Compile
    compiled = workflow.compile()

    logger.info("[Graph] Evaluation workflow compiled successfully")
    return compiled


def create_initial_state(
    video_path: str,
    user_prompt: str = "",
    script_text: str = "",
    audio_path: str | None = None,
    storyboard: list[dict] | None = None,
    user_profile: "UserProfile | dict | None" = None,
) -> GraphState:
    """
    Convenience function to create the initial GraphState.

    Args:
        video_path: Path to the generated video file.
        user_prompt: The original user prompt/instruction.
        script_text: The generated script/storyboard text.
        audio_path: Optional separate audio file path.
        storyboard: Optional list of shot intention dicts.
        user_profile: A ``UserProfile`` instance, a raw dict from
                      profiles.jsonl (will be parsed via
                      ``UserProfile.from_profile_dict``), or ``None``
                      for default weights.

    Returns:
        GraphState ready to be passed to graph.invoke().
    """
    from .schemas import UserProfile as UP

    if user_profile is None:
        profile = UP()
    elif isinstance(user_profile, UP):
        profile = user_profile
    elif isinstance(user_profile, dict):
        # Auto-detect profiles.jsonl format vs flat dict
        if "personalization" in user_profile:
            profile = UP.from_profile_dict(user_profile)
        else:
            profile = UP(**user_profile)
    else:
        profile = UP()

    return GraphState(
        video_path=video_path,
        user_prompt=user_prompt,
        script_text=script_text,
        audio_path=audio_path,
        storyboard=storyboard,
        user_profile=profile,
    )
