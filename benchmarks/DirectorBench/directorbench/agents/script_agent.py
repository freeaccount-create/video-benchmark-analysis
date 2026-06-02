"""
script_agent.py — Script/Storyboard Evaluation Agent (Text-Modal)

Two operating modes:
  A. Reference mode   (script_text / storyboard provided by user)
     → evaluate script quality + video-to-script fidelity
  B. Reference-free   (no script provided)
     → extract narrative from video (VLM + ASR) → evaluate extracted narrative

Sub-metrics:
  1. Script Reasonableness  — logical flow, cause-effect, no plot holes
  2. Script Novelty          — originality relative to common tropes
  3. User Requirement Consistency — alignment with user prompt
  4. Script-Video Fidelity   — (reference mode only) does the video follow the script
"""

from __future__ import annotations

import logging
from typing import Any

from ..schemas import (
    AgentID, ContentProfile, EvalResult, Evidence, GraphState,
    Severity, ShotSegment,
)
from ..checkpoints import CHECKPOINTS
from .base import BaseEvalAgent, CONFIDENCE_TOOL_GUIDANCE

logger = logging.getLogger(__name__)


class ScriptEvalAgent(BaseEvalAgent):
    agent_id = AgentID.SCRIPT_EVAL

    # ==================================================================
    # Narrative extraction (reference-free mode)
    # ==================================================================

    def _extract_narrative_from_video(self, state: GraphState) -> str:
        """When no script is provided, extract the narrative flow from the
        video itself using VLM (sequential thumbnails) + ASR transcript.

        Returns a structured narrative string that can be used as a
        stand-in for script_text in downstream evaluations.
        """
        prep = state.preprocessing
        shots = prep.shots if prep else []
        asr_segments = prep.asr_segments if prep else []

        # Collect thumbnails (temporal order)
        thumbnails = [s.thumbnail_path for s in shots if s.thumbnail_path]

        # Build ASR transcript grouped by shot
        asr_text_by_shot = self._align_asr_to_shots(asr_segments, shots)

        # ---- Step 1: VLM extracts visual narrative ----
        visual_narrative = ""
        if thumbnails:
            system_prompt = """You are a professional film analyst. Given sequential frames
from a minute-long video (in temporal order), reconstruct the narrative:

1. Identify the main characters and describe them visually.
2. For each shot/scene, describe what is happening (action, emotion, setting).
3. Identify the narrative arc: setup → conflict/development → resolution (if any).
4. Note any scene transitions and what they signify narratively.

Return JSON:
{
  "characters": [{"name_or_label": "<str>", "visual_description": "<str>"}],
  "scenes": [
    {
      "shot_indices": [<int>, ...],
      "setting": "<str>",
      "action": "<str>",
      "emotion": "<str>"
    }
  ],
  "narrative_arc": {
    "setup": "<str>",
    "development": "<str>",
    "resolution": "<str>"
  },
  "full_narrative": "<str: cohesive paragraph summarising the entire story>"
}"""

            result = self.llm.vision_evaluate(
                system_prompt=system_prompt,
                text_prompt="These are sequential frames from the video, in temporal order. Reconstruct the narrative:",
                image_paths=thumbnails[:16],
            )
            visual_narrative = result.get("full_narrative", "")

            # Store extracted structured data for downstream use
            self._extracted_scenes = result.get("scenes", [])
            self._extracted_characters = result.get("characters", [])
            self._extracted_arc = result.get("narrative_arc", {})

        # ---- Step 2: Integrate ASR dialogue ----
        dialogue_summary = ""
        all_dialogue = []
        for shot_idx, lines in sorted(asr_text_by_shot.items()):
            if lines:
                all_dialogue.append(f"[Shot {shot_idx}] {' '.join(lines)}")
        if all_dialogue:
            dialogue_summary = "\n".join(all_dialogue)

        # ---- Step 3: Fuse visual + dialogue into unified narrative ----
        if visual_narrative and dialogue_summary:
            fuse_prompt = f"""Combine the following visual narrative and dialogue transcript
into a single cohesive script-like narrative.

Visual narrative:
{visual_narrative}

Dialogue transcript:
{dialogue_summary}

Write a unified narrative that integrates both visual actions and dialogue,
in chronological order. Output as plain text (2-4 paragraphs)."""

            fused = self.llm.evaluate(
                "You are a screenwriter reconstructing a script from visual and audio information.",
                fuse_prompt,
            )
            # The LLM may return a dict or a string depending on implementation
            if isinstance(fused, dict):
                narrative = fused.get("narrative", fused.get("text", str(fused)))
            else:
                narrative = str(fused)
        elif visual_narrative:
            narrative = visual_narrative
        elif dialogue_summary:
            narrative = dialogue_summary
        else:
            narrative = ""

        logger.info(f"[ScriptEvalAgent] Extracted narrative from video ({len(narrative)} chars)")
        return narrative

    @staticmethod
    def _align_asr_to_shots(
        asr_segments: list, shots: list[ShotSegment]
    ) -> dict[int, list[str]]:
        """Map each ASR segment to the shot it temporally belongs to."""
        result: dict[int, list[str]] = {}
        for asr in asr_segments:
            mid = (asr.start_sec + asr.end_sec) / 2
            best_shot = 0
            for s in shots:
                if s.start_sec <= mid <= s.end_sec:
                    best_shot = s.index
                    break
            result.setdefault(best_shot, []).append(asr.text)
        return result

    # ==================================================================
    # Sub-metric 1: Script Reasonableness
    # ==================================================================

    def _eval_reasonableness(self, state: GraphState, narrative: str) -> EvalResult:
        """Checkpoint-based script reasonableness evaluation."""
        storyboard = state.storyboard or []
        profile = self._build_content_profile(state)

        extra_ctx = (
            f"Script:\n{narrative}\n\n"
            f"Storyboard shots:\n{self._format_storyboard(storyboard)}\n\n"
            f"Content: characters={profile.has_characters}, "
            f"dialogue={profile.has_dialogue}, "
            f"scene_changes={profile.has_scene_changes}\n\n"
            f"Evaluate the script's logical coherence:"
        )

        score, confidence, cp_results, active_cps, profile = self._checkpoint_evaluate(
            metric_name="script_reasonableness",
            state=state,
            extra_context=extra_ctx,
            profile=profile,
        )

        evidence = []
        for r in cp_results:
            if r.normalised < 0.5:
                evidence.append(Evidence(
                    type="script_logic",
                    issue=f"[{r.checkpoint_id}] {r.reasoning}" if r.reasoning else r.checkpoint_id,
                    severity=Severity.HIGH if r.normalised < 0.25 else Severity.MEDIUM,
                ))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="script_reasonableness",
            score=score,
            confidence=confidence,
            granularity="video-level",
            evidence=evidence,
            checkpoint_results=cp_results,
            intermediate_repr={
                "active_checkpoints": [c.id for c in active_cps],
                "content_profile": profile.model_dump(),
            },
            suggestions=[f"Fix: {e.issue}" for e in evidence if e.severity == Severity.HIGH],
        )

    # ==================================================================
    # Sub-metric 2: Script Novelty / Creativity
    # ==================================================================

    def _eval_novelty(self, state: GraphState, narrative: str) -> EvalResult:
        """Checkpoint-based script novelty evaluation."""
        profile = self._build_content_profile(state)

        extra_ctx = f"Script:\n{narrative}\n\nEvaluate the script's creativity and originality:"

        score, confidence, cp_results, active_cps, profile = self._checkpoint_evaluate(
            metric_name="script_novelty",
            state=state,
            extra_context=extra_ctx,
            profile=profile,
        )

        evidence = []
        for r in cp_results:
            if r.normalised < 0.5:
                evidence.append(Evidence(
                    type="script_creativity",
                    issue=f"[{r.checkpoint_id}] {r.reasoning}" if r.reasoning else r.checkpoint_id,
                    severity=Severity.MEDIUM,
                ))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="script_novelty",
            score=score,
            confidence=confidence,
            granularity="video-level",
            evidence=evidence,
            checkpoint_results=cp_results,
            intermediate_repr={"active_checkpoints": [c.id for c in active_cps]},
        )

    # ==================================================================
    # Sub-metric 3: User Requirement Consistency
    # ==================================================================

    def _eval_user_consistency(self, state: GraphState, narrative: str) -> EvalResult:
        """Check alignment between narrative and user's original prompt,
        including the user profile's taste preferences and hard constraints.

        This is the proper place to penalise things like:
        - User wants BGM but video has none  (or vice-versa)
        - User wants complex camera but video is static
        - hard_constraints not met
        """
        user_prompt_text = state.user_prompt or ""
        profile = state.user_profile
        custom_reqs = profile.custom_requirements
        taste = profile.user_taste
        hard_constraints = profile.hard_constraints

        # Build a structured description of what the user cares about
        taste_description = self._format_user_taste(taste)
        constraints_str = ", ".join(hard_constraints) if hard_constraints else "none"

        system_prompt_raw = """You are evaluating whether a generated video script fulfills the user's requirements AND personal preferences.

You are given:
1. The user's original prompt
2. Custom free-form requirements (if any)
3. User taste profile — a structured description of what the user values
4. Hard constraints — non-negotiable requirements that MUST be met

For each requirement / preference / constraint, determine if it is:
- FULFILLED: clearly present in the script / video plan
- PARTIALLY: somewhat addressed but incomplete
- MISSING: not addressed at all
- VIOLATED: the video does the opposite of what the user wanted (e.g. user wants no BGM but video has BGM)

Scoring guide:
- Hard constraint violated → severe penalty
- Taste preference violated → moderate penalty
- Taste preference "don't care" (null) → no penalty either way

Return JSON:
{
  "score": <float 0-1>,
  "reasoning": "<detailed reasoning>",
  "requirement_status": [
    {"requirement": "<req>", "source": "prompt|custom|taste|hard_constraint", "status": "fulfilled|partially|missing|violated", "detail": "<explanation>"}
  ],
  "confidence": <float 0-1>
}"""

        user_content = f"""User prompt: {user_prompt_text}

Custom requirements: {custom_reqs if custom_reqs else "none"}

User taste profile:
{taste_description}

Hard constraints: {constraints_str}

Generated script / narrative:
{narrative}"""

        system_prompt = self.maybe_add_confidence_guidance(system_prompt_raw, state)
        tool_ctx = self.format_tool_context(state, relevant_metrics=["user_requirement_consistency"])
        user_content += tool_ctx

        result = self.llm.evaluate(system_prompt, user_content)

        evidence = []
        for req in result.get("requirement_status", []):
            status = req.get("status", "")
            if status in ("partially", "missing", "violated"):
                sev = Severity.CRITICAL if status == "violated" and req.get("source") == "hard_constraint" else \
                      Severity.HIGH if status in ("missing", "violated") else Severity.MEDIUM
                evidence.append(Evidence(
                    type="requirement_gap",
                    issue=f"[{req.get('source', '?')}] {req.get('requirement')}: {req.get('detail', '')}",
                    severity=sev,
                ))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="user_requirement_consistency",
            score=float(result.get("score", 0.5)),
            confidence=float(result.get("confidence", 0.8)),
            granularity="video-level",
            evidence=evidence,
            suggestions=[f"Address: {e.issue}" for e in evidence],
            metadata={"requirement_status": result.get("requirement_status", [])},
        )

    @staticmethod
    def _format_user_taste(taste) -> str:
        """Render UserTaste into a human-readable description for the LLM."""
        lines = [f"- Focus area: {taste.focus}"]
        if taste.wants_bgm is not None:
            lines.append(f"- Wants BGM: {'yes' if taste.wants_bgm else 'no'}")
        if taste.wants_lip_sync is not None:
            lines.append(f"- Wants lip-sync: {'yes' if taste.wants_lip_sync else 'no'}")
        if taste.wants_complex_camera is not None:
            lines.append(f"- Wants complex camera movements: {'yes' if taste.wants_complex_camera else 'no'}")
        if taste.emotion_depth != "medium":
            lines.append(f"- Emotion depth: {taste.emotion_depth}")
        if taste.camera_movement != "normal":
            lines.append(f"- Camera movement importance: {taste.camera_movement}")
        if taste.lighting != "normal":
            lines.append(f"- Lighting importance: {taste.lighting}")
        if taste.bgm_important is not None:
            lines.append(f"- BGM importance: {'high' if taste.bgm_important else 'low'}")
        if taste.tone_control != "normal":
            lines.append(f"- Tone control importance: {taste.tone_control}")
        if taste.wants_dreamy_effects is not None:
            lines.append(f"- Wants dreamy/fantasy effects: {'yes' if taste.wants_dreamy_effects else 'no'}")
        if taste.wants_unusual_camera is not None:
            lines.append(f"- Wants unusual camera angles: {'yes' if taste.wants_unusual_camera else 'no'}")
        if taste.text_visual_alignment != "normal":
            lines.append(f"- Text-visual alignment importance: {taste.text_visual_alignment}")
        return "\n".join(lines)

    # ==================================================================
    # Sub-metric 4: Script–Video Fidelity (reference mode only)
    # ==================================================================

    def _eval_script_video_fidelity(self, state: GraphState) -> EvalResult:
        """Compare the given script/storyboard against what actually appears
        in the video.  Only runs when a reference script is provided.

        Uses VLM to check each storyboard shot intention against the
        corresponding video thumbnail.
        """
        prep = state.preprocessing
        shots = prep.shots if prep else []
        script = state.script_text or ""
        storyboard = state.storyboard or []
        thumbnails = [s.thumbnail_path for s in shots if s.thumbnail_path]

        if not thumbnails:
            return EvalResult(
                agent_id=self.agent_id,
                metric_name="script_video_fidelity",
                score=0.5, confidence=0.3,
                metadata={"note": "No video frames available for fidelity check"},
            )

        system_prompt_raw = """You are evaluating how faithfully an AI-generated video follows its script.

Given:
- The original script / storyboard (the intended plan)
- Sequential frames from the generated video (the actual output)

For each scripted scene or shot, determine:
1. Was the scene visually realised in the video?
2. Are the characters present and correctly depicted?
3. Does the action/event match the script description?
4. Are there any scenes in the video NOT in the script (hallucinated content)?
5. Are there any scripted scenes MISSING from the video?

Return JSON:
{
  "score": <float 0-1>,
  "scene_fidelity": [
    {
      "script_scene": "<scene description from script>",
      "status": "faithful|partial|missing|hallucinated",
      "detail": "<str>"
    }
  ],
  "hallucinated_content": ["<description of content not in script>"],
  "missing_from_video": ["<scripted scene not found in video>"],
  "reasoning": "<str>",
  "confidence": <float 0-1>
}"""

        system_prompt = self.maybe_add_confidence_guidance(system_prompt_raw, state)
        tool_ctx = self.format_tool_context(state, relevant_metrics=["script_video_fidelity"])

        text_prompt = f"""Script / Storyboard:
{script}

{self._format_storyboard(storyboard)}

Below are sequential frames from the generated video (in temporal order).
Evaluate how faithfully the video follows the script:{tool_ctx}"""

        result = self.llm.vision_evaluate(
            system_prompt=system_prompt,
            text_prompt=text_prompt,
            image_paths=thumbnails[:16],
        )

        evidence = []
        for item in result.get("missing_from_video", []):
            evidence.append(Evidence(
                type="script_not_in_video",
                issue=f"Scripted scene missing from video: {item}",
                severity=Severity.HIGH,
            ))
        for item in result.get("hallucinated_content", []):
            evidence.append(Evidence(
                type="video_not_in_script",
                issue=f"Video contains unscripted content: {item}",
                severity=Severity.MEDIUM,
            ))
        for item in result.get("scene_fidelity", []):
            if item.get("status") == "partial":
                evidence.append(Evidence(
                    type="partial_fidelity",
                    issue=f"Partially realised: {item.get('script_scene', '')} — {item.get('detail', '')}",
                    severity=Severity.MEDIUM,
                ))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="script_video_fidelity",
            score=float(result.get("score", 0.5)),
            confidence=float(result.get("confidence", 0.7)),
            granularity="shot-level",
            evidence=evidence,
            intermediate_repr={
                "scene_fidelity": result.get("scene_fidelity", []),
            },
            suggestions=[
                f"Regenerate: {e.issue}" for e in evidence
                if e.severity == Severity.HIGH
            ],
        )

    # ==================================================================
    # Main evaluate — dispatches based on whether script is provided
    # ==================================================================

    def evaluate(self, state: GraphState) -> list[EvalResult]:
        results = []

        has_script = bool(state.script_text or state.storyboard)
        # Reset per-run extracted narrative (consumed by __call__ to
        # propagate into GraphState.extracted_script_text).
        self._last_extracted_narrative: str = ""

        if has_script:
            # ---- Reference mode ----
            # Use the provided script directly
            narrative = state.script_text or self._format_storyboard(state.storyboard or [])
            logger.info("[ScriptEvalAgent] Reference mode: evaluating provided script")

            results.append(self._eval_reasonableness(state, narrative))
            results.append(self._eval_novelty(state, narrative))

            # Fidelity: does the video match the script?
            results.append(self._eval_script_video_fidelity(state))

        else:
            # ---- Reference-free mode ----
            # Extract narrative from video (VLM + ASR)
            logger.info("[ScriptEvalAgent] Reference-free mode: extracting narrative from video")
            narrative = self._extract_narrative_from_video(state)
            self._last_extracted_narrative = narrative or ""

            if narrative:
                results.append(self._eval_reasonableness(state, narrative))
                results.append(self._eval_novelty(state, narrative))
            else:
                logger.warning("[ScriptEvalAgent] Could not extract narrative, skipping evaluation")

        # User requirement consistency works in both modes
        if state.user_prompt:
            effective_narrative = narrative if 'narrative' in locals() else ""
            results.append(self._eval_user_consistency(state, effective_narrative))

        return results

    def __call__(self, state: GraphState) -> dict:
        """Override base __call__ so the narrative extracted in
        reference-free mode is written back into GraphState as
        `extracted_script_text`. Downstream agents (CrossModalEvalAgent)
        depend on this for text-video / text-audio consistency metrics.
        """
        update = super().__call__(state)
        narrative = getattr(self, "_last_extracted_narrative", "") or ""
        if narrative and not (state.script_text or state.storyboard):
            update["extracted_script_text"] = narrative
            logger.info(
                f"[ScriptEvalAgent] Propagating extracted narrative "
                f"({len(narrative)} chars) → state.extracted_script_text"
            )
        return update

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _format_storyboard(storyboard: list[dict]) -> str:
        if not storyboard:
            return "(no storyboard provided)"
        lines = []
        for i, shot in enumerate(storyboard):
            lines.append(f"Shot {i+1}: {shot}")
        return "\n".join(lines)
