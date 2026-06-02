"""
stability_agent.py — Generation Stability Evaluation Agent

Sub-metrics:
  1. Generation Stability (>1 min) — quality maintenance over full duration
  2. Quality Degradation Rate — first-half vs second-half comparison

This is a minute-long-video-specific concern: many generation models
degrade in quality as generation progresses.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..schemas import AgentID, ContentProfile, EvalResult, Evidence, GraphState, Severity, ToolStatus
from ..checkpoints import CHECKPOINTS
from .base import BaseEvalAgent, CONFIDENCE_TOOL_GUIDANCE

logger = logging.getLogger(__name__)


class StabilityEvalAgent(BaseEvalAgent):
    agent_id = AgentID.STABILITY_EVAL

    def _eval_generation_stability(self, state: GraphState) -> EvalResult:
        """
        Evaluate whether the model maintains quality over the full minute.

        Strategy:
          1. Split video into temporal segments (first half / second half)
          2. Compare per-segment image quality (BRISQUE / NIQE)
          3. Check for mid-generation artifacts (sudden degradation, crashes)
          4. Verify actual duration meets target

        Full implementation: use BRISQUE (blind/referenceless image quality)
        Skeleton: VLM-based comparison of early vs late frames.
        """
        prep = state.preprocessing
        shots = prep.shots if prep else []
        total_duration = prep.total_duration_sec if prep else 0

        # Split shots into first half and second half
        mid_time = total_duration / 2
        first_half = [s for s in shots if s.end_sec <= mid_time]
        second_half = [s for s in shots if s.start_sec >= mid_time]

        # Collect thumbnails from each half
        first_thumbnails = [s.thumbnail_path for s in first_half if s.thumbnail_path]
        second_thumbnails = [s.thumbnail_path for s in second_half if s.thumbnail_path]

        evidence = []

        # --- Duration check ---
        target_duration = 60.0  # 1 minute target
        duration_ratio = total_duration / target_duration if target_duration > 0 else 0
        if duration_ratio < 0.9:
            evidence.append(Evidence(
                type="duration_shortfall",
                issue=f"Video is {total_duration:.1f}s, target was {target_duration:.0f}s ({duration_ratio:.0%})",
                severity=Severity.HIGH if duration_ratio < 0.7 else Severity.MEDIUM,
            ))

        # --- Checkpoint-based quality evaluation ---
        all_images = first_thumbnails[:4] + second_thumbnails[:4]

        if all_images:
            profile = self._build_content_profile(state)
            n_first = min(4, len(first_thumbnails))

            extra_ctx = (
                f"First {n_first} frames are from the FIRST HALF, "
                f"remaining are from the SECOND HALF of the video "
                f"(total {total_duration:.0f}s).\n"
                f"Evaluate generation stability and quality:"
            )

            score, confidence, cp_results, active_cps, _ = self._checkpoint_evaluate(
                metric_name="generation_stability",
                state=state,
                image_paths=all_images,
                extra_context=extra_ctx,
                profile=profile,
            )

            for r in cp_results:
                if r.normalised < 0.5:
                    evidence.append(Evidence(
                        type="stability_checkpoint_fail",
                        issue=f"[{r.checkpoint_id}] {r.reasoning}" if r.reasoning else r.checkpoint_id,
                        severity=Severity.HIGH if r.normalised < 0.25 else Severity.MEDIUM,
                    ))
        else:
            # Fallback: try BRISQUE if available
            brisque_result = self._compute_brisque_scores(
                first_thumbnails, second_thumbnails
            )
            score = brisque_result[0] if brisque_result is not None else 0.5
            confidence = 0.4
            cp_results = []
            active_cps = []

        # Apply duration penalty
        if duration_ratio < 0.9:
            duration_penalty = 0.15 if duration_ratio < 0.7 else 0.08
            score = max(0.0, score - duration_penalty)

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="generation_stability",
            score=score,
            confidence=confidence,
            granularity="video-level",
            evidence=evidence,
            checkpoint_results=cp_results,
            intermediate_repr={
                "total_duration_sec": total_duration,
                "duration_ratio": duration_ratio,
                "active_checkpoints": [c.id for c in active_cps],
            },
            suggestions=[
                f"Duration: {e.issue}" for e in evidence if e.type == "duration_shortfall"
            ],
        )

    # ------------------------------------------------------------------
    # BRISQUE-based quality scoring (OpenCV contrib)
    # ------------------------------------------------------------------

    def _compute_brisque_scores(
        self,
        first_half_images: list[str],
        second_half_images: list[str],
    ) -> tuple[float, float] | None:
        """
        Compute BRISQUE (blind image quality) scores for each half.
        Returns (overall_quality_score, degradation_rate) or None when
        BRISQUE is unavailable.
        """
        try:
            import cv2
            import numpy as np

            repo_root = Path(__file__).resolve().parents[2]
            brisque_model = str(repo_root / "brisque_model_live.yml")
            brisque_range = str(repo_root / "brisque_range_live.yml")

            def brisque_score(image_path: str) -> float:
                img = cv2.imread(image_path)
                if img is None:
                    return 50.0  # default
                # OpenCV's quality module
                score = cv2.quality.QualityBRISQUE_compute(
                    img,
                    brisque_model,
                    brisque_range,
                )
                # OpenCV versions may return scalar, tuple, or nested arrays.
                arr = np.asarray(score, dtype=float).reshape(-1)
                if arr.size == 0:
                    return 50.0
                return float(arr[0])

            first_scores = [brisque_score(p) for p in first_half_images if p]
            second_scores = [brisque_score(p) for p in second_half_images if p]

            avg_first = sum(first_scores) / len(first_scores) if first_scores else 50
            avg_second = sum(second_scores) / len(second_scores) if second_scores else 50

            # BRISQUE: lower is better; normalize to 0-1 (inverted)
            quality = 1.0 - (avg_first + avg_second) / 200
            degradation = (avg_second - avg_first) / 100  # positive = degradation

            self._record_tool("BRISQUE", ToolStatus.SUCCESS,
                              f"Computed quality scores for {len(first_scores)+len(second_scores)} frames",
                              affects=["generation_stability"])
            return max(0, min(1, quality)), degradation

        except Exception as e:
            logger.warning(f"[StabilityAgent] BRISQUE unavailable: {e}")
            self._record_tool("BRISQUE", ToolStatus.FAILED, str(e),
                              affects=["generation_stability"])
            return None

    # ------------------------------------------------------------------
    # Main evaluate
    # ------------------------------------------------------------------

    def evaluate(self, state: GraphState) -> list[EvalResult]:
        return [self._eval_generation_stability(state)]
