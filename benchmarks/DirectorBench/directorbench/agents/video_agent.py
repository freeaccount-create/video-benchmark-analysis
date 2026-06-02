"""
video_agent.py — Video Evaluation Agent (Visual-Modal)

Sub-metrics:
  1. User Demand Fulfillment   — do visuals match user-specified demands
  2. Temporal Coherence        — smooth transitions, no flickering/jumps
  3. Lighting/Shadow Consistency — consistent light sources across shots

Strategy: shot-level evaluation → aggregate to video-level.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..schemas import (
    AgentID, ContentProfile, EvalResult, Evidence, GraphState,
    Severity, ShotSegment, ToolStatus,
)
from ..checkpoints import CHECKPOINTS
from .base import BaseEvalAgent, CONFIDENCE_TOOL_GUIDANCE

logger = logging.getLogger(__name__)


class VideoEvalAgent(BaseEvalAgent):
    agent_id = AgentID.VIDEO_EVAL

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _extract_score_confidence(
        self,
        result: dict[str, Any],
        metric_name: str,
        missing_confidence_value: float = 0.25,
    ) -> tuple[float, float, dict[str, Any]] | None:
        """
        Parse score/confidence from LLM JSON strictly.
        - Missing/invalid score: return None (caller should skip this metric)
        - Missing/invalid confidence: degrade confidence and record metadata
        """
        metadata: dict[str, Any] = {}
        if "score" not in result:
            logger.warning(f"[VideoAgent] {metric_name}: missing 'score' field, skipping metric")
            return None

        try:
            score = self._clamp01(float(result["score"]))
        except (TypeError, ValueError):
            logger.warning(f"[VideoAgent] {metric_name}: invalid 'score' value={result.get('score')!r}, skipping metric")
            return None

        if "confidence" in result:
            try:
                confidence = self._clamp01(float(result["confidence"]))
            except (TypeError, ValueError):
                logger.warning(
                    f"[VideoAgent] {metric_name}: invalid 'confidence' value={result.get('confidence')!r}, "
                    f"degrading confidence to {missing_confidence_value:.2f}"
                )
                confidence = missing_confidence_value
                metadata["confidence_degraded"] = "invalid_confidence"
        else:
            logger.warning(
                f"[VideoAgent] {metric_name}: missing 'confidence' field, "
                f"degrading confidence to {missing_confidence_value:.2f}"
            )
            confidence = missing_confidence_value
            metadata["confidence_degraded"] = "missing_confidence"

        return score, confidence, metadata

    # ------------------------------------------------------------------
    # Sub-metric 1: User Demand Fulfillment
    # ------------------------------------------------------------------

    def _eval_user_demand(self, state: GraphState) -> EvalResult | None:
        """
        VLM-based multi-question answering: does the video show
        what the user requested?

        For each shot, ask targeted questions derived from the user prompt.
        """
        prep = state.preprocessing
        shots = prep.shots if prep else []
        user_prompt_text = state.user_prompt or ""

        # Collect thumbnail paths for VLM input
        thumbnails = [s.thumbnail_path for s in shots if s.thumbnail_path]

        if thumbnails and user_prompt_text:
            system_prompt_raw = """You are a video evaluation expert.
Given representative frames from a minute-long generated video and the user's
original prompt, evaluate how well the video fulfills the user's visual demands.

Check for:
1. Are the requested scenes/events visually present?
2. Is the requested style/mood reflected?
3. Are specified characters/objects visible?
4. Are any key elements from the prompt missing?

Return JSON:
{
  "score": <float 0-1>,
  "reasoning": "<detailed analysis>",
  "fulfilled": ["<demand1>", ...],
  "missing": ["<demand1>", ...],
  "shot_scores": [{"shot_index": <int>, "score": <float>, "note": "<str>"}],
  "confidence": <float 0-1>
}"""
            system_prompt = self.maybe_add_confidence_guidance(system_prompt_raw, state)
            tool_ctx = self.format_tool_context(state, relevant_metrics=["user_demand_fulfillment"])

            result = self.llm.vision_evaluate(
                system_prompt=system_prompt,
                text_prompt=f"User prompt: {user_prompt_text}\n\nThese frames are representative shots from the video (in temporal order):{tool_ctx}",
                image_paths=thumbnails[:16],  # limit to avoid token overflow
            )
        else:
            logger.info("[VideoAgent] user_demand_fulfillment skipped: insufficient frames or empty user prompt")
            return None

        parsed = self._extract_score_confidence(result, "user_demand_fulfillment")
        if parsed is None:
            return None
        score, confidence, parse_meta = parsed

        evidence = []
        for item in result.get("missing", []):
            evidence.append(Evidence(
                type="missing_demand", issue=f"Missing visual element: {item}",
                severity=Severity.HIGH,
            ))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="user_demand_fulfillment",
            score=score,
            confidence=confidence,
            granularity="shot-level",
            evidence=evidence,
            intermediate_repr={
                "shot_scores": result.get("shot_scores", []),
            },
            metadata=parse_meta,
            suggestions=[f"Add visual element: {e.issue}" for e in evidence],
        )

    # ------------------------------------------------------------------
    # Algorithmic visual-evidence helpers
    # ------------------------------------------------------------------

    def _compute_visual_evidence(
        self,
        thumbnails: list[str],
    ) -> dict[str, Any]:
        """Compute lightweight algorithmic signals across consecutive frames.

        Returns a dict with:
          - ``pairwise_ssim``: list of SSIM values between consecutive frames
          - ``avg_ssim``: average pairwise SSIM
          - ``pairwise_hist_diff``: list of histogram chi-square distances
          - ``pairwise_pixel_diff``: list of mean absolute pixel differences
          - ``face_cluster_count``: estimated number of distinct faces (None
            if face detection unavailable)
          - ``summary``: human-readable summary for injection into VLM prompt

        All computations use OpenCV only (no GPU, no heavy model).
        Typically < 500 ms for 16 frames.
        """
        result: dict[str, Any] = {
            "pairwise_ssim": [],
            "avg_ssim": None,
            "pairwise_hist_diff": [],
            "pairwise_pixel_diff": [],
            "face_cluster_count": None,
            "face_detection_counts": [],
            "summary": "",
        }

        if len(thumbnails) < 2:
            return result

        t0 = time.perf_counter()

        try:
            import cv2
            import numpy as np
        except ImportError as e:
            logger.warning(f"[VideoAgent] visual evidence skipped — OpenCV unavailable: {e}")
            self._record_tool("VisualEvidence-OpenCV", ToolStatus.FALLBACK,
                              f"OpenCV unavailable: {e}",
                              affects=["temporal_coherence"])
            return result

        frames = []
        for p in thumbnails:
            img = cv2.imread(p)
            if img is not None:
                frames.append(img)

        if len(frames) < 2:
            return result

        # --- Pairwise SSIM + histogram diff + pixel diff ---
        ssim_vals = []
        hist_diffs = []
        pixel_diffs = []

        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            # Resize to common size for consistent comparison
            h = min(f1.shape[0], f2.shape[0], 256)
            w = min(f1.shape[1], f2.shape[1], 256)
            r1 = cv2.resize(f1, (w, h))
            r2 = cv2.resize(f2, (w, h))

            # SSIM (grayscale)
            g1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY)
            try:
                from skimage.metrics import structural_similarity
                ssim_val = structural_similarity(g1, g2)
            except ImportError:
                # Fallback: simplified SSIM-like measure using correlation
                mean1, mean2 = g1.mean(), g2.mean()
                std1, std2 = g1.std(), g2.std()
                cov = ((g1.astype(float) - mean1) * (g2.astype(float) - mean2)).mean()
                C1, C2 = 6.5025, 58.5225
                ssim_val = float(
                    ((2 * mean1 * mean2 + C1) * (2 * cov + C2))
                    / ((mean1**2 + mean2**2 + C1) * (std1**2 + std2**2 + C2))
                )
            ssim_vals.append(round(ssim_val, 3))

            # Histogram chi-square distance (colour)
            h1 = cv2.calcHist([r1], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
            h2 = cv2.calcHist([r2], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            chi_sq = cv2.compareHist(h1.flatten(), h2.flatten(), cv2.HISTCMP_CHISQR)
            hist_diffs.append(round(float(chi_sq), 3))

            # Mean absolute pixel difference (normalized to [0, 1])
            diff = np.abs(r1.astype(float) - r2.astype(float)).mean() / 255.0
            pixel_diffs.append(round(float(diff), 3))

        result["pairwise_ssim"] = ssim_vals
        result["avg_ssim"] = round(float(np.mean(ssim_vals)), 3) if ssim_vals else None
        result["pairwise_hist_diff"] = hist_diffs
        result["pairwise_pixel_diff"] = pixel_diffs

        # --- Face detection across frames ---
        # Use Haar cascade (built into OpenCV, no extra model)
        face_counts = []
        face_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            pass

        if face_cascade is not None and not face_cascade.empty():
            for img in frames:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (640, 480))
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                face_counts.append(len(faces))
            result["face_detection_counts"] = face_counts
        else:
            self._record_tool("VisualEvidence-FaceDetect", ToolStatus.FALLBACK,
                              "Haar cascade unavailable",
                              affects=["temporal_coherence"])

        elapsed = (time.perf_counter() - t0) * 1000
        self._record_tool("VisualEvidence-OpenCV", ToolStatus.SUCCESS,
                          f"Computed pairwise metrics for {len(frames)} frames",
                          elapsed_ms=elapsed,
                          affects=["temporal_coherence", "lighting_consistency"])

        # --- Build human-readable summary ---
        lines = ["--- Algorithmic Visual Evidence (objective measurements) ---"]

        avg_ssim = result["avg_ssim"]
        if avg_ssim is not None:
            # Interpret SSIM for the VLM
            if avg_ssim > 0.85:
                ssim_interp = "very similar frames (high visual continuity)"
            elif avg_ssim > 0.6:
                ssim_interp = "moderately similar frames"
            else:
                ssim_interp = "LOW similarity — frames look quite different from each other"
            lines.append(
                f"Average pairwise SSIM: {avg_ssim:.3f} ({ssim_interp})"
            )
            lines.append(
                f"Per-pair SSIM: {ssim_vals}"
            )

        if pixel_diffs:
            avg_pd = np.mean(pixel_diffs)
            if avg_pd > 0.15:
                pd_interp = "HIGH pixel differences — significant visual changes between consecutive frames"
            elif avg_pd > 0.08:
                pd_interp = "moderate pixel differences"
            else:
                pd_interp = "small pixel differences (frames are visually similar)"
            lines.append(
                f"Average pixel difference: {avg_pd:.3f} ({pd_interp})"
            )

        if face_counts:
            lines.append(f"Faces detected per frame: {face_counts}")
            unique_counts = set(face_counts)
            if len(unique_counts) > 1 or (face_counts and max(face_counts) == 0):
                lines.append(
                    "NOTE: Face count varies across frames — this may indicate "
                    "different characters or inconsistent face generation."
                )

        lines.append(
            "Use these measurements as OBJECTIVE GROUNDING for your evaluation. "
            "Low SSIM between consecutive same-scene frames strongly suggests "
            "visual inconsistency. Varying face counts suggest character changes."
        )
        lines.append("--- End Algorithmic Evidence ---")

        result["summary"] = "\n".join(lines)

        logger.info(
            f"[VideoAgent] Visual evidence computed in {elapsed:.0f}ms: "
            f"avg_ssim={avg_ssim}, face_counts={face_counts}"
        )
        return result

    # ------------------------------------------------------------------
    # Sub-metric 2: Temporal Coherence
    # ------------------------------------------------------------------

    def _eval_temporal_coherence(self, state: GraphState) -> EvalResult | None:
        """
        Checkpoint-based temporal coherence evaluation.

        Uses dynamic rubric: builds a ContentProfile, filters applicable
        checkpoints, scores them in a single batched VLM call with anchored
        rubric descriptions, then aggregates via weighted average.

        This replaces the old "single score per transition pair" approach
        with fine-grained, anchored sub-questions that produce better
        score differentiation between good and bad videos.
        """
        prep = state.preprocessing
        shots = prep.shots if prep else []

        if len(shots) < 2:
            logger.info("[VideoAgent] temporal_coherence skipped: too few shots")
            return None

        # --- Step 1: Content profile (dynamic rubric) ---
        profile = self._build_content_profile(state)

        # --- Step 2: Filter applicable checkpoints ---
        all_cps = CHECKPOINTS.get("temporal_coherence", [])
        active_cps = self._filter_applicable(all_cps, profile)

        if not active_cps:
            logger.warning("[VideoAgent] temporal_coherence: no applicable checkpoints")
            return None

        # --- Step 3: Collect frames for VLM ---
        thumbnails = [s.thumbnail_path for s in shots if s.thumbnail_path]

        # --- Step 3b: Compute algorithmic visual evidence ---
        visual_evidence = self._compute_visual_evidence(thumbnails)
        algo_summary = visual_evidence.get("summary", "")

        shot_timeline = "\n".join(
            f"Shot {s.index}: [{s.start_sec:.1f}s-{s.end_sec:.1f}s]" for s in shots
        )
        extra_ctx = (
            f"Video has {len(shots)} shots. Timeline:\n{shot_timeline}\n\n"
            f"Content profile: characters={profile.has_characters} "
            f"(count={profile.character_count}), "
            f"held_objects={profile.has_held_objects}, "
            f"scene_changes={profile.has_scene_changes}\n\n"
            f"{algo_summary}\n\n"
            f"Evaluate temporal coherence across these consecutive frames:"
        )

        # --- Step 4: Batched checkpoint evaluation ---
        cp_results = self._evaluate_checkpoints_batched(
            checkpoints=active_cps,
            state=state,
            image_paths=thumbnails[:16],
            extra_context=extra_ctx,
        )

        # --- Step 4b: Algorithm-VLM divergence detection ---
        # If algorithmic evidence strongly disagrees with VLM scores,
        # re-evaluate with explicit warning about the divergence.
        avg_ssim = visual_evidence.get("avg_ssim")
        if avg_ssim is not None:
            for i, r in enumerate(cp_results):
                if r.checkpoint_id != "char_face_consistency":
                    continue
                # Very low SSIM (< 0.3) means frames are very different,
                # but VLM gave 3+ (acceptable or better) → strong divergence
                if avg_ssim < 0.3 and r.raw_value >= 3:
                    logger.warning(
                        f"[VideoAgent] DIVERGENCE: char_face_consistency "
                        f"VLM={r.raw_value} but avg_ssim={avg_ssim:.3f} "
                        f"(very low — frames look very different). "
                        f"Re-evaluating with emphasis on algorithmic evidence."
                    )
                    cp_def = next(
                        (c for c in active_cps if c.id == r.checkpoint_id), None
                    )
                    if cp_def:
                        divergence_ctx = (
                            f"{extra_ctx}\n\n"
                            f"=== ALGORITHM-VLM DIVERGENCE WARNING ===\n"
                            f"Algorithmic analysis shows avg_ssim={avg_ssim:.3f} "
                            f"between consecutive frames. SSIM < 0.3 means frames "
                            f"are VERY different from each other.\n"
                            f"Face counts per frame: {visual_evidence.get('face_counts', [])}\n"
                            f"This strongly suggests faces are NOT consistent.\n"
                            f"A score of 3+ is NOT justified when SSIM is this low.\n"
                            f"Re-evaluate carefully and give a score that matches "
                            f"the algorithmic evidence.\n"
                            f"=== END WARNING ===\n"
                        )
                        new_result = self._evaluate_single_checkpoint(
                            cp_def, state,
                            image_paths=thumbnails[:16],
                            extra_context=divergence_ctx,
                        )
                        logger.info(
                            f"[VideoAgent] Divergence re-eval: "
                            f"char_face_consistency {r.raw_value} → {new_result.raw_value}"
                        )
                        cp_results[i] = new_result
                    break

        # --- Step 5: Aggregate ---
        final_score = self._aggregate_checkpoint_score(active_cps, cp_results)

        # Build evidence from low-scoring checkpoints
        evidence = []
        for r in cp_results:
            if r.normalised < 0.5:
                cp_def = next((c for c in active_cps if c.id == r.checkpoint_id), None)
                evidence.append(Evidence(
                    type="temporal_checkpoint_fail",
                    issue=f"[{r.checkpoint_id}] {r.reasoning}" if r.reasoning else r.checkpoint_id,
                    severity=Severity.HIGH if r.normalised < 0.25 else Severity.MEDIUM,
                ))

        # Confidence based on how many checkpoints were evaluable
        confidence = min(0.9, 0.5 + 0.05 * len(cp_results))

        skipped_ids = [c.id for c in all_cps if c not in active_cps]

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="temporal_coherence",
            score=final_score,
            confidence=confidence,
            granularity="shot-level",
            evidence=evidence,
            checkpoint_results=cp_results,
            intermediate_repr={
                "shot_count": len(shots),
                "active_checkpoints": [c.id for c in active_cps],
                "skipped_checkpoints": skipped_ids,
                "content_profile": profile.model_dump(),
                "visual_evidence": {
                    k: v for k, v in visual_evidence.items() if k != "summary"
                },
            },
            suggestions=[
                f"Fix: {e.issue}" for e in evidence if e.severity == Severity.HIGH
            ],
        )

    # ------------------------------------------------------------------
    # Sub-metric 3: Lighting/Shadow Consistency
    # ------------------------------------------------------------------

    def _eval_lighting_consistency(self, state: GraphState) -> EvalResult | None:
        """
        Checkpoint-based lighting/shadow consistency evaluation.

        Uses anchored rubric checkpoints for: light direction, shadow
        plausibility, colour temperature, and exposure stability.
        """
        prep = state.preprocessing
        shots = prep.shots if prep else []
        thumbnails = [s.thumbnail_path for s in shots if s.thumbnail_path]

        if len(thumbnails) < 2:
            logger.info("[VideoAgent] lighting_consistency skipped: too few frames")
            return None

        profile = self._build_content_profile(state)
        all_cps = CHECKPOINTS.get("lighting_consistency", [])
        active_cps = self._filter_applicable(all_cps, profile)

        if not active_cps:
            return None

        extra_ctx = (
            f"These are {len(thumbnails)} consecutive shots from a minute-long video.\n"
            f"Evaluate lighting and shadow consistency:"
        )

        cp_results = self._evaluate_checkpoints_batched(
            checkpoints=active_cps,
            state=state,
            image_paths=thumbnails[:12],
            extra_context=extra_ctx,
        )

        final_score = self._aggregate_checkpoint_score(active_cps, cp_results)

        evidence = []
        for r in cp_results:
            if r.normalised < 0.5:
                evidence.append(Evidence(
                    type="lighting_checkpoint_fail",
                    issue=f"[{r.checkpoint_id}] {r.reasoning}" if r.reasoning else r.checkpoint_id,
                    severity=Severity.HIGH if r.normalised < 0.25 else Severity.MEDIUM,
                ))

        confidence = min(0.9, 0.5 + 0.05 * len(cp_results))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="lighting_consistency",
            score=final_score,
            confidence=confidence,
            granularity="shot-level",
            evidence=evidence,
            checkpoint_results=cp_results,
            intermediate_repr={
                "active_checkpoints": [c.id for c in active_cps],
            },
        )

    # ------------------------------------------------------------------
    # Sub-metric 4: Transition / Splice Quality
    # ------------------------------------------------------------------

    def _eval_transition_quality(self, state: GraphState) -> EvalResult | None:
        """
        Detect problematic clip splices in the final concatenated video.

        Two-layer approach (zero local models):
          Layer 1 — Algorithmic (SSIM, histogram diff, optical flow) on
                    boundary frames pre-computed in preprocessing.
          Layer 2 — VLM semantic verification: for each flagged boundary,
                    send the two boundary frames to GPT-4o Vision to
                    classify as "intentional scene change" vs "bad splice".

        The key distinction:
          • If boundary frames belong to the SAME scene but SSIM drops →
            bad splice (frame jump / flicker).
          • If boundary frames are clearly different scenes → intentional
            transition, not a defect.
        """
        prep = state.preprocessing
        transitions = prep.transitions if prep else []

        if not transitions:
            logger.info("[VideoAgent] transition_quality skipped: no transition boundaries")
            return None

        # ---- Layer 1: Algorithmic anomaly detection ----
        flagged: list[dict] = []     # transitions with suspicious metrics
        all_scores: list[float] = []

        for t in transitions:
            # Compute a composite anomaly score from pre-computed metrics.
            # Low SSIM + high histogram diff + high optical flow = likely bad splice
            ssim = t.ssim if t.ssim is not None else 0.9
            hist = t.histogram_diff if t.histogram_diff is not None else 0.0
            flow = t.optical_flow_magnitude if t.optical_flow_magnitude is not None else 0.0

            # Normalize to a 0-1 quality score (higher = better)
            # SSIM is already [0,1]; hist and flow need empirical thresholds
            ssim_score = max(0.0, min(1.0, ssim))
            hist_score = max(0.0, 1.0 - hist / 5.0)   # chi-sq > 5 → very different
            flow_score = max(0.0, 1.0 - flow / 30.0)   # mean flow > 30 → large motion

            composite = 0.5 * ssim_score + 0.25 * hist_score + 0.25 * flow_score
            all_scores.append(composite)

            # Flag if composite quality is low (possible bad splice)
            if composite < 0.6:
                flagged.append({
                    "transition": t,
                    "composite": composite,
                    "ssim": ssim, "hist": hist, "flow": flow,
                })

        # ---- Layer 2: VLM verification of flagged boundaries ----
        evidence = []
        verified_bad = 0

        for item in flagged[:8]:  # limit API calls
            t = item["transition"]
            if not (t.frame_before_path and t.frame_after_path):
                continue

            system_prompt_raw = """You are analysing a potential splice artefact in an AI-generated video.
You are given two frames: the LAST frame of one clip and the FIRST frame of the next clip.

Determine:
1. Do these two frames belong to the SAME scene or DIFFERENT scenes?
   - Same scene indicators: same background, same characters, continuous action
   - Different scene indicators: completely new location, different characters, obvious scene change
2. If SAME scene: is there a visible discontinuity (jump cut, colour shift, character position jump, flickering)?
3. If DIFFERENT scene: is the transition acceptable (natural cut) or jarring?

Return JSON:
{
  "same_scene": <bool>,
  "has_discontinuity": <bool>,
  "discontinuity_type": "<none|frame_jump|colour_shift|character_teleport|flicker|motion_break>",
  "severity": "<none|low|medium|high>",
  "reasoning": "<str>"
}"""

            system_prompt = self.maybe_add_confidence_guidance(system_prompt_raw, state)

            result = self.llm.vision_evaluate(
                system_prompt=system_prompt,
                text_prompt=(
                    f"Transition at t={t.timestamp_sec:.1f}s "
                    f"(shot {t.from_shot_index}→{t.to_shot_index}). "
                    f"Algorithmic metrics: SSIM={item['ssim']:.3f}, "
                    f"histogram_diff={item['hist']:.2f}, "
                    f"optical_flow={item['flow']:.1f}. "
                    f"Frame BEFORE the cut (left) and AFTER the cut (right):"
                ),
                image_paths=[t.frame_before_path, t.frame_after_path],
            )

            same_scene = result.get("same_scene", False)
            has_disc = result.get("has_discontinuity", False)
            disc_type = result.get("discontinuity_type", "none")
            severity_str = result.get("severity", "none")

            # Update classification on the TransitionBoundary object
            t.is_scene_change = not same_scene

            # A real problem: same scene + discontinuity detected
            if same_scene and has_disc and disc_type != "none":
                verified_bad += 1
                evidence.append(Evidence(
                    type="splice_artifact",
                    timestamp=f"{t.timestamp_sec:.1f}s",
                    shot_index=t.from_shot_index,
                    issue=f"Clip splice artifact ({disc_type}) at shot {t.from_shot_index}→{t.to_shot_index}",
                    severity=Severity(severity_str) if severity_str in ("low", "medium", "high") else Severity.MEDIUM,
                ))

        # ---- Aggregate score ----
        if not all_scores:
            logger.warning("[VideoAgent] transition_quality skipped: no transition metrics available")
            return None
        algo_score = sum(all_scores) / len(all_scores)

        # Penalise for verified bad splices
        penalty = verified_bad * 0.08
        final_score = max(0.0, min(1.0, algo_score - penalty))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="transition_quality",
            score=final_score,
            confidence=0.85 if flagged else 0.6,
            granularity="frame-level",
            evidence=evidence,
            intermediate_repr={
                "total_transitions": len(transitions),
                "algorithmically_flagged": len(flagged),
                "vlm_verified_bad": verified_bad,
                "per_transition_scores": [
                    {
                        "from": t.from_shot_index,
                        "to": t.to_shot_index,
                        "timestamp": t.timestamp_sec,
                        "ssim": t.ssim,
                        "histogram_diff": t.histogram_diff,
                        "optical_flow": t.optical_flow_magnitude,
                        "composite_score": s,
                    }
                    for t, s in zip(transitions, all_scores)
                ],
            },
            suggestions=[
                f"Add cross-fade or smooth interpolation at {e.timestamp}"
                for e in evidence
            ],
        )

    # ------------------------------------------------------------------
    # Main evaluate
    # ------------------------------------------------------------------

    def evaluate(self, state: GraphState) -> list[EvalResult]:
        raw = [
            self._eval_user_demand(state),
            self._eval_temporal_coherence(state),
            self._eval_lighting_consistency(state),
            self._eval_transition_quality(state),
        ]
        return [r for r in raw if r is not None]
