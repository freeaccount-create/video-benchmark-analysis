"""
diagnosis.py — Phase 3: Diagnosis Synthesizer

Collects all EvalResults from all agents, applies user-profile weights,
identifies bottlenecks, and generates a structured DiagnosisReport with
actionable insights.
"""

from __future__ import annotations

import logging

from ..config import EvalConfig
from ..llm_utils import LLMClient
from ..schemas import (
    AgentID, BottleneckItem, ContentProfile, DiagnosisReport, EvalResult,
    GraphState, ToolStatus, UserProfile,
)
from ..checkpoints import CHECKPOINTS

logger = logging.getLogger(__name__)


class DiagnosisSynthesizer:
    """Final phase: aggregate scores and generate diagnosis report."""

    def __init__(self, config: EvalConfig | None = None):
        self.config = config or EvalConfig()
        self.llm = LLMClient(self.config.llm)

    def __call__(self, state: GraphState) -> dict:
        """LangGraph node interface."""
        logger.info("[DiagnosisSynthesizer] Generating diagnosis report...")

        try:
            report = self.synthesize(state)
            return {
                "diagnosis": report,
                "execution_log": state.execution_log + [
                    f"DiagnosisSynthesizer: OK - overall={report.overall_score:.2f} grade={report.overall_grade}"
                ],
            }
        except Exception as e:
            logger.error(f"[DiagnosisSynthesizer] Failed: {e}")
            return {
                "errors": state.errors + [f"DiagnosisSynthesizer: {e}"],
                "execution_log": state.execution_log + [f"DiagnosisSynthesizer: FAILED - {e}"],
            }

    def synthesize(self, state: GraphState) -> DiagnosisReport:
        """Main synthesis logic."""
        profile = state.user_profile

        # 1. Collect all results
        all_results = (
            state.script_results + state.video_results +
            state.audio_results + state.stability_results +
            state.crossmodal_results
        )

        # 2. Compute per-dimension scores
        dimension_scores = self._compute_dimension_scores(all_results, profile)

        # 3. Compute weighted overall score
        overall_score = self._compute_overall_score(dimension_scores, profile)

        # 4. Assign grade
        overall_grade = self._assign_grade(overall_score)

        # 5. Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(all_results)

        # 6. Flag low-confidence items for human review
        needs_review = [
            r for r in all_results
            if r.confidence < self.config.confidence_threshold
        ]

        # 7. Build radar chart data
        radar_data = {
            "Script Quality": dimension_scores.get("script", 0),
            "Visual Quality": dimension_scores.get("video", 0),
            "Audio Quality": dimension_scores.get("audio", 0),
            "Cross-Modal Alignment": dimension_scores.get("crossmodal", 0),
            "Generation Stability": dimension_scores.get("stability", 0),
        }

        # 8. Build checkpoint registry snapshot (active checkpoints per metric)
        checkpoint_snapshot = self._build_checkpoint_snapshot(all_results)

        # 9. Generate narrative summary via LLM
        summary, detailed = self._generate_narrative(
            all_results, dimension_scores, bottlenecks, overall_score, overall_grade, state
        )

        return DiagnosisReport(
            overall_score=overall_score,
            overall_grade=overall_grade,
            dimension_scores=dimension_scores,
            all_results=all_results,
            bottlenecks=bottlenecks,
            needs_human_review=needs_review,
            radar_data=radar_data,
            summary=summary,
            detailed_analysis=detailed,
            content_profile=state.content_profile,
            checkpoint_registry_snapshot=checkpoint_snapshot,
        )

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_dimension_scores(
        self, results: list[EvalResult], profile: UserProfile
    ) -> dict[str, float]:
        """Average scores within each dimension.

        Dimensions whose agents returned zero results (e.g. audio when the
        video is silent) are **omitted** from the dict so they do not drag
        down the overall score.
        """
        dimension_map = {
            AgentID.SCRIPT_EVAL: "script",
            AgentID.VIDEO_EVAL: "video",
            AgentID.AUDIO_EVAL: "audio",
            AgentID.STABILITY_EVAL: "stability",
            AgentID.CROSSMODAL_EVAL: "crossmodal",
        }

        # Each bucket stores (score, confidence) pairs for weighted averaging
        buckets: dict[str, list[tuple[float, float]]] = {d: [] for d in dimension_map.values()}
        for r in results:
            dim = dimension_map.get(r.agent_id)
            if dim:
                buckets[dim].append((r.score, r.confidence))

        scores = {}
        for dim, pairs in buckets.items():
            if not pairs:
                continue  # omit — dimension had no results (N/A)
            total_conf = sum(c for _, c in pairs)
            if total_conf > 0:
                # Confidence-weighted average: high-confidence metrics
                # contribute more, but low-confidence ones aren't crushed.
                scores[dim] = sum(s * c for s, c in pairs) / total_conf
            else:
                # All confidences are 0 — fall back to simple average
                scores[dim] = sum(s for s, _ in pairs) / len(pairs)

        return scores

    # Map internal dimension keys → profiles.jsonl PriorityWeights fields.
    # stability has no dedicated weight in profiles.jsonl; it is folded into
    # visual_camera since stability is a visual-generation concern.
    _DIM_TO_WEIGHT_FIELD: dict[str, str] = {
        "script":     "text_story_arc",
        "video":      "visual_camera",
        "audio":      "audio_emotion",
        "crossmodal": "cross_modal_sync",
        "stability":  "visual_camera",     # shared with video
    }

    def _compute_overall_score(
        self, dim_scores: dict[str, float], profile: UserProfile
    ) -> float:
        """Weighted aggregation using profile priority_weights.

        Only dimensions that actually have scores participate.  If a
        dimension was skipped (not in dim_scores), its weight is excluded
        from the total so the remaining dimensions are fairly normalised.
        """
        pw = profile.priority_weights

        # Build weight for each dimension that has a score
        active_weights: dict[str, float] = {}
        for dim, score in dim_scores.items():
            field = self._DIM_TO_WEIGHT_FIELD.get(dim)
            if field:
                active_weights[dim] = getattr(pw, field, 0.25)

        total_weight = sum(active_weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            dim_scores[dim] * w
            for dim, w in active_weights.items()
        )
        return weighted_sum / total_weight

    def _assign_grade(self, score: float) -> str:
        """Map score to letter grade."""
        for grade, threshold in sorted(
            self.config.grade_boundaries.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if score >= threshold:
                return grade
        return "F"

    # ------------------------------------------------------------------
    # Bottleneck detection
    # ------------------------------------------------------------------

    def _identify_bottlenecks(self, results: list[EvalResult]) -> list[BottleneckItem]:
        """Find metrics that score below the bottleneck threshold."""
        bottlenecks = []
        for r in results:
            if r.score < self.config.bottleneck_threshold:
                bottlenecks.append(BottleneckItem(
                    metric_name=r.metric_name,
                    score=r.score,
                    agent_id=r.agent_id,
                    description=f"{r.metric_name} scored {r.score:.2f}, below threshold {self.config.bottleneck_threshold}",
                    suggestions=r.suggestions,
                ))

        # Sort by score ascending (worst first)
        bottlenecks.sort(key=lambda b: b.score)
        return bottlenecks

    # ------------------------------------------------------------------
    # Checkpoint registry snapshot
    # ------------------------------------------------------------------

    @staticmethod
    def _build_checkpoint_snapshot(results: list[EvalResult]) -> dict[str, list[dict]]:
        """Capture the checkpoint definitions that were actually used.

        For each metric that produced checkpoint_results, look up the
        corresponding definitions from the registry and serialise them.
        This makes results fully reproducible — you can see exactly which
        rubric anchors and weights were applied.
        """
        snapshot: dict[str, list[dict]] = {}
        metrics_seen = set()
        for r in results:
            if r.checkpoint_results and r.metric_name not in metrics_seen:
                metrics_seen.add(r.metric_name)
                # Get the active checkpoint IDs from results
                active_ids = {cr.checkpoint_id for cr in r.checkpoint_results}
                # Look up full definitions from the registry
                all_defs = CHECKPOINTS.get(r.metric_name, [])
                active_defs = [d for d in all_defs if d.id in active_ids]
                snapshot[r.metric_name] = [d.model_dump() for d in active_defs]
        return snapshot

    # ------------------------------------------------------------------
    # Narrative generation
    # ------------------------------------------------------------------

    def _generate_narrative(
        self,
        results: list[EvalResult],
        dim_scores: dict[str, float],
        bottlenecks: list[BottleneckItem],
        overall_score: float,
        grade: str,
        state: GraphState,
    ) -> tuple[str, str]:
        """Use LLM to generate human-readable summary and detailed analysis."""

        results_summary = "\n".join(
            f"- [{r.agent_id.value}] {r.metric_name}: {r.score:.2f} (conf={r.confidence:.2f})"
            for r in results
        )

        # Build checkpoint-level detail for each metric
        checkpoint_detail_lines = []
        for r in results:
            if r.checkpoint_results:
                checkpoint_detail_lines.append(f"\n  [{r.metric_name}] Checkpoint Breakdown:")
                for cr in r.checkpoint_results:
                    status = "PASS" if cr.normalised >= 0.5 else "FAIL"
                    checkpoint_detail_lines.append(
                        f"    - {cr.checkpoint_id}: {cr.raw_value}/5 "
                        f"(norm={cr.normalised:.2f}, {status}) — {cr.reasoning}"
                    )
        checkpoint_summary = "\n".join(checkpoint_detail_lines) if checkpoint_detail_lines else "No checkpoint data."

        bottleneck_info = "\n".join(
            f"- {b.metric_name}: {b.score:.2f} — {b.description}"
            for b in bottlenecks
        ) or "None identified."

        evidence_summary = "\n".join(
            f"- [{r.metric_name}] {e.issue} (severity: {e.severity.value})"
            for r in results
            for e in r.evidence[:2]
        )

        # Summarize tool failures for the narrative
        tool_issues = [
            r for r in state.tool_context
            if r.status != ToolStatus.SUCCESS
        ]
        tool_issue_summary = ""
        if tool_issues:
            lines = ["Tool Issues (may affect confidence of some scores):"]
            for r in tool_issues:
                lines.append(f"  [{r.status.value.upper()}] {r.tool_name}: {r.detail}")
            tool_issue_summary = "\n".join(lines)

        system_prompt = """You are writing a professional video evaluation diagnosis report.

Generate two sections:
1. "summary": A concise 2-3 sentence executive summary of the evaluation.
2. "detailed_analysis": A thorough analysis (3-5 paragraphs) covering:
   - Overall quality assessment
   - Strongest dimensions and what works well
   - Key bottlenecks and their impact
   - Checkpoint-level insights: reference specific checkpoint scores to
     pinpoint exactly what works and what fails (e.g. "character face
     consistency scored 2/5 while background continuity scored 4/5")
   - If any evaluation tools failed or fell back to degraded mode,
     note which scores may have lower confidence as a result
   - Specific actionable recommendations tied to low-scoring checkpoints
   - Priority fixes (ranked by impact)

Write in a professional, constructive tone.

Return JSON:
{
  "summary": "<2-3 sentence summary>",
  "detailed_analysis": "<3-5 paragraphs>"
}"""

        user_content = f"""Overall: {overall_score:.2f} (Grade: {grade})

Dimension Scores:
{chr(10).join(f'  {k}: {v:.2f}' for k, v in dim_scores.items())}

Individual Metrics:
{results_summary}

Checkpoint-Level Details:
{checkpoint_summary}

Bottlenecks:
{bottleneck_info}

Key Evidence:
{evidence_summary}"""

        if tool_issue_summary:
            user_content += f"\n\n{tool_issue_summary}"

        result = self.llm.evaluate(system_prompt, user_content)

        return (
            result.get("summary", f"Overall score: {overall_score:.2f} (Grade: {grade})"),
            result.get("detailed_analysis", "Detailed analysis unavailable."),
        )
