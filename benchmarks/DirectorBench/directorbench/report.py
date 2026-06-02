"""
report.py — Evaluation record persistence and rendering.

All evaluation runs are **appended** to a single JSONL file
(``eval_outputs/results.jsonl`` by default).  Each line is a complete,
self-contained record that captures:

    ┌──────────────────────────────────────────────────┐
    │  inputs   — video_path, user_prompt, script_text,│
    │             profile (full), storyboard           │
    │  outputs  — the entire DiagnosisReport           │
    └──────────────────────────────────────────────────┘

A companion CLI script ``render_report.py`` (in the project root) can
convert any record to Markdown by ``report_id``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

from .schemas import ContentProfile, DiagnosisReport, UserProfile


# ======================================================================
# Single-record data structure persisted to JSONL
# ======================================================================

def _build_record(
    report: DiagnosisReport,
    *,
    video_path: str = "",
    user_prompt: str = "",
    script_text: str = "",
    audio_path: Optional[str] = None,
    storyboard: Optional[list[dict]] = None,
    user_profile: Optional[UserProfile] = None,
) -> dict[str, Any]:
    """Build a flat dict that captures both inputs and outputs."""
    return {
        # ---- identity ----
        "report_id": report.report_id,
        "timestamp": report.timestamp,

        # ---- inputs ----
        "inputs": {
            "video_path": video_path,
            "user_prompt": user_prompt,
            "script_text": script_text,
            "audio_path": audio_path,
            "storyboard": storyboard,
            "profile": user_profile.model_dump() if user_profile else None,
        },

        # ---- outputs (the full DiagnosisReport) ----
        "results": report.model_dump(),

        # ---- reproducibility metadata ----
        # Content profile and checkpoint registry are also inside
        # report.model_dump(), but we duplicate them at top level for
        # easy access by analysis tools.
        "content_profile": (
            report.content_profile.model_dump()
            if report.content_profile else None
        ),
        "checkpoint_registry": report.checkpoint_registry_snapshot,
    }


# ======================================================================
# ReportWriter — append-only JSONL persistence
# ======================================================================

class ReportWriter:
    """Append evaluation records to a JSONL file."""

    def __init__(self, output_dir: str = "./eval_outputs", filename: str = "results.jsonl"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(output_dir, filename)

    def append(
        self,
        report: DiagnosisReport,
        *,
        video_path: str = "",
        user_prompt: str = "",
        script_text: str = "",
        audio_path: Optional[str] = None,
        storyboard: Optional[list[dict]] = None,
        user_profile: Optional[UserProfile] = None,
    ) -> str:
        """Append one evaluation record.  Returns the report_id."""
        record = _build_record(
            report,
            video_path=video_path,
            user_prompt=user_prompt,
            script_text=script_text,
            audio_path=audio_path,
            storyboard=storyboard,
            user_profile=user_profile,
        )
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        return report.report_id

    # ------------------------------------------------------------------
    # Console summary  (unchanged from before)
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(report: DiagnosisReport) -> None:
        """Print a concise summary to the console."""
        print("\n" + "=" * 70)
        print(f"  DirectorBench — Diagnosis Report")
        print(f"  Report ID: {report.report_id}")
        print(f"  Timestamp: {report.timestamp}")
        print("=" * 70)

        print(f"\n  Overall Score: {report.overall_score:.2f}  |  Grade: {report.overall_grade}")
        print("-" * 70)

        print("\n  Dimension Scores:")
        for dim, score in report.dimension_scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"    {dim:20s}  {bar}  {score:.2f}")

        if report.bottlenecks:
            print(f"\n  Bottlenecks ({len(report.bottlenecks)}):")
            for b in report.bottlenecks:
                print(f"    ⚠ {b.metric_name}: {b.score:.2f} — {b.description}")
                for s in b.suggestions[:2]:
                    print(f"      → {s}")

        # Checkpoint breakdown per metric
        has_checkpoints = any(r.checkpoint_results for r in report.all_results)
        if has_checkpoints:
            print(f"\n  Checkpoint Breakdown:")
            for r in report.all_results:
                if r.checkpoint_results:
                    print(f"    [{r.metric_name}]")
                    for cr in r.checkpoint_results:
                        icon = "✓" if cr.normalised >= 0.5 else "✗"
                        print(f"      {icon} {cr.checkpoint_id}: {cr.raw_value}/5 (norm={cr.normalised:.2f})")
                        if cr.reasoning:
                            print(f"        {cr.reasoning}")

        if report.needs_human_review:
            print(f"\n  Needs Human Review ({len(report.needs_human_review)} items):")
            for r in report.needs_human_review:
                print(f"    ? {r.metric_name}: score={r.score:.2f}, confidence={r.confidence:.2f}")

        # Content profile summary
        if report.content_profile:
            cp = report.content_profile
            traits = []
            if cp.has_characters:
                traits.append(f"characters({cp.character_count})")
            if cp.has_dialogue:
                traits.append("dialogue")
            if cp.has_scene_changes:
                traits.append(f"scenes({cp.scene_count})")
            if cp.has_background_music:
                traits.append("BGM")
            if cp.has_camera_movement:
                traits.append("camera_movement")
            if cp.is_animation_style:
                traits.append("animation")
            if cp.is_live_action_style:
                traits.append("live_action")
            if traits:
                print(f"\n  Content Profile: {', '.join(traits)}")

        print(f"\n  Summary:\n  {report.summary}")
        print("=" * 70 + "\n")


# ======================================================================
# Markdown rendering  (operates on a raw record dict)
# ======================================================================

def record_to_markdown(record: dict) -> str:
    """Convert a single JSONL record dict into a full Markdown report."""
    inputs = record.get("inputs", {})
    results = record.get("results", {})

    dim_scores = results.get("dimension_scores", {})
    bottlenecks = results.get("bottlenecks", [])
    needs_review = results.get("needs_human_review", [])
    all_results = results.get("all_results", [])
    profile = inputs.get("profile") or {}

    lines: list[str] = []

    # --- Header ---
    lines += [
        "# DirectorBench Diagnosis Report",
        "",
        f"**Report ID:** `{record.get('report_id', 'N/A')}`  ",
        f"**Timestamp:** {record.get('timestamp', 'N/A')}  ",
        f"**Overall Score:** {results.get('overall_score', 0):.2f}  ",
        f"**Grade:** {results.get('overall_grade', 'N/A')}",
        "",
        "---",
        "",
    ]

    # --- Input info ---
    lines += ["## Evaluation Inputs", ""]
    lines.append(f"**Video:** `{inputs.get('video_path', 'N/A')}`  ")
    if inputs.get("audio_path"):
        lines.append(f"**Audio:** `{inputs['audio_path']}`  ")
    if inputs.get("user_prompt"):
        lines.append(f"**User Prompt:** {inputs['user_prompt']}  ")
    if profile:
        pname = profile.get("name", "default")
        pid = profile.get("profile_id", "?")
        lines.append(f"**Profile:** {pname} (ID {pid})  ")
        pw = profile.get("priority_weights", {})
        if pw:
            weights_str = ", ".join(f"{k}={v}" for k, v in pw.items())
            lines.append(f"**Weights:** {weights_str}  ")
        hc = profile.get("hard_constraints", [])
        if hc:
            lines.append(f"**Hard Constraints:** {', '.join(hc)}  ")
        # Show prompt-generation characteristics if present
        expertise = profile.get("expertise_level", "")
        expr_style = profile.get("expression_style", "")
        if expertise:
            lines.append(f"**Expertise:** {expertise}  ")
        if expr_style:
            lines.append(f"**Expression Style:** {expr_style}  ")
    lines.append("")

    if inputs.get("script_text"):
        lines += [
            "<details><summary>Script (click to expand)</summary>",
            "",
            "```",
            inputs["script_text"].strip(),
            "```",
            "</details>",
            "",
        ]

    # --- Dimension Scores ---
    lines += ["## Dimension Scores", ""]
    if dim_scores:
        lines.append("| Dimension | Score |")
        lines.append("|-----------|-------|")
        for dim, score in dim_scores.items():
            lines.append(f"| {dim} | {score:.2f} |")
    else:
        lines.append("_No dimension scores available._")
    lines.append("")

    # --- Executive Summary ---
    lines += [
        "## Executive Summary",
        "",
        results.get("summary", "_No summary available._"),
        "",
    ]

    # --- Detailed Analysis ---
    if results.get("detailed_analysis"):
        lines += [
            "## Detailed Analysis",
            "",
            results["detailed_analysis"],
            "",
        ]

    # --- Bottlenecks ---
    if bottlenecks:
        lines += ["## Bottlenecks", ""]
        for b in bottlenecks:
            lines.append(f"### {b.get('metric_name', '?')} (score: {b.get('score', 0):.2f})")
            lines.append("")
            lines.append(b.get("description", ""))
            sugg = b.get("suggestions", [])
            if sugg:
                lines.append("")
                lines.append("**Suggestions:**")
                lines.append("")
                for s in sugg:
                    lines.append(f"- {s}")
            lines.append("")

    # --- Human Review ---
    if needs_review:
        lines += ["## Items Needing Human Review", ""]
        for r in needs_review:
            agent = r.get("agent_id", "?")
            lines.append(
                f"- **{r.get('metric_name', '?')}**: "
                f"score={r.get('score', 0):.2f}, "
                f"confidence={r.get('confidence', 0):.2f} "
                f"(from {agent})"
            )
        lines.append("")

    # --- All Results Table ---
    if all_results:
        lines += ["## All Evaluation Results", ""]
        lines.append("| Agent | Metric | Score | Confidence | Evidence Count | Checkpoints |")
        lines.append("|-------|--------|-------|------------|----------------|-------------|")
        for r in all_results:
            ev_count = len(r.get("evidence", []))
            cp_count = len(r.get("checkpoint_results", []))
            cp_str = str(cp_count) if cp_count else "-"
            lines.append(
                f"| {r.get('agent_id', '?')} | {r.get('metric_name', '?')} "
                f"| {r.get('score', 0):.2f} | {r.get('confidence', 0):.2f} "
                f"| {ev_count} | {cp_str} |"
            )
        lines.append("")

    # --- Checkpoint Breakdown ---
    has_checkpoints = any(r.get("checkpoint_results") for r in all_results)
    if has_checkpoints:
        lines += ["## Checkpoint Breakdown", ""]
        for r in all_results:
            cp_results = r.get("checkpoint_results", [])
            if not cp_results:
                continue
            metric = r.get("metric_name", "?")
            lines.append(f"### {metric}")
            lines.append("")
            lines.append("| Checkpoint | Raw | Normalised | Status | Reasoning |")
            lines.append("|------------|-----|------------|--------|-----------|")
            for cr in cp_results:
                norm = cr.get("normalised", 0)
                status = "PASS" if norm >= 0.5 else "FAIL"
                reasoning = cr.get("reasoning", "").replace("|", "\\|")
                lines.append(
                    f"| {cr.get('checkpoint_id', '?')} "
                    f"| {cr.get('raw_value', '?')}/5 "
                    f"| {norm:.2f} "
                    f"| {status} "
                    f"| {reasoning} |"
                )
            lines.append("")

    # --- Content Profile ---
    content_profile = record.get("content_profile") or results.get("content_profile")
    if content_profile:
        lines += ["## Content Profile", ""]
        lines.append("The following content traits were detected and used to dynamically")
        lines.append("select applicable evaluation checkpoints:")
        lines.append("")
        # Render as key-value pairs, skip defaults/false
        for key, val in content_profile.items():
            if key == "extra":
                if val:
                    lines.append(f"- **extra**: {val}")
            elif val and val is not True:
                lines.append(f"- **{key}**: {val}")
            elif val is True:
                lines.append(f"- **{key}**: yes")
        lines.append("")

    # --- Checkpoint Registry (rubric definitions used) ---
    checkpoint_registry = record.get("checkpoint_registry") or results.get("checkpoint_registry_snapshot", {})
    if checkpoint_registry:
        lines += ["## Checkpoint Definitions Used", ""]
        lines.append(
            "<details><summary>Click to expand full rubric definitions "
            f"({sum(len(v) for v in checkpoint_registry.values())} checkpoints across "
            f"{len(checkpoint_registry)} metrics)</summary>"
        )
        lines.append("")
        for metric_name, cp_defs in checkpoint_registry.items():
            lines.append(f"#### {metric_name}")
            lines.append("")
            for cp in cp_defs:
                lines.append(f"**{cp.get('id', '?')}** (weight={cp.get('weight', 1.0):.2f})")
                lines.append(f"> {cp.get('question', '')}")
                lines.append("")
                if cp.get("applicable_when"):
                    conds = ", ".join(f"{k}={v}" for k, v in cp["applicable_when"].items())
                    lines.append(f"*Applicable when:* {conds}")
                    lines.append("")
                rubric = cp.get("rubric", [])
                if rubric:
                    for anchor in sorted(rubric, key=lambda a: a.get("value", 0), reverse=True):
                        lines.append(
                            f"- **{anchor.get('value', '?')} ({anchor.get('label', '?')})**: "
                            f"{anchor.get('description', '')}"
                        )
                    lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


# ======================================================================
# JSONL reader utilities
# ======================================================================

def load_record_by_id(jsonl_path: str, report_id: str) -> dict | None:
    """Find a record in the JSONL file by report_id."""
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("report_id") == report_id:
                return record
    return None


def load_all_records(jsonl_path: str) -> list[dict]:
    """Load every record from the JSONL file."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
