"""
main.py — Entry point for the DirectorBench evaluation framework.

Usage:
    # As a library
    from directorbench.main import evaluate_video
    report = evaluate_video("path/to/video.mp4", user_prompt="A dramatic chase scene...")

    # As CLI
    python -m directorbench.main --video path/to/video.mp4 --prompt "..." --script "..."
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import EvalConfig
from .graph import build_eval_graph, create_initial_state
from .report import ReportWriter
from .schemas import DiagnosisReport, GraphState

logger = logging.getLogger(__name__)


def _append_tool_trace(
    output_dir: str,
    report_id: str,
    video_path: str,
    final_state: dict,
) -> str:
    """Persist per-run tool call trace (with elapsed timings) to JSONL."""
    tool_calls_raw = final_state.get("tool_context", []) or []
    tool_calls: list[dict] = []
    for item in tool_calls_raw:
        if hasattr(item, "model_dump"):
            tool_calls.append(item.model_dump())
        elif isinstance(item, dict):
            tool_calls.append(item)
        else:
            tool_calls.append({"tool_name": str(item)})

    trace_record = {
        "report_id": report_id,
        "timestamp": datetime.now().isoformat(),
        "video_path": video_path,
        "execution_log": final_state.get("execution_log", []) or [],
        "tool_calls": tool_calls,
    }

    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, "tool_traces.jsonl")
    with open(trace_path, "a") as f:
        f.write(json.dumps(trace_record, default=str, ensure_ascii=False) + "\n")
    return trace_path


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def evaluate_video(
    video_path: str,
    user_prompt: str = "",
    script_text: str = "",
    audio_path: str | None = None,
    storyboard: list[dict] | None = None,
    user_profile: dict | None = None,
    config: EvalConfig | None = None,
    output_dir: str = "./eval_outputs",
    verbose: bool = True,
) -> DiagnosisReport:
    """
    Evaluate a minute-long generated video using the multi-agent framework.

    Args:
        video_path:     Path to the generated video file.
        user_prompt:    The original user prompt/instruction.
        script_text:    The generated script/storyboard text.
        audio_path:     Optional separate audio file path.
        storyboard:     Optional list of shot intention dicts.
        user_profile:   A UserProfile object, a profiles.jsonl dict (with
                        "personalization" key), or None for defaults.
                        Use ``--profile-id N`` on CLI to load from
                        data/profiles.jsonl.
        config:         Optional EvalConfig to override defaults.
        output_dir:     Directory to save reports.
        verbose:        Print summary to console.

    Returns:
        DiagnosisReport with scores, bottlenecks, and recommendations.

    Example:
        report = evaluate_video(
            video_path="output.mp4",
            user_prompt="Create a 1-minute dramatic chase scene through a neon-lit city",
            script_text="Scene 1: A detective runs through rain-soaked streets...",
            user_profile={"weight_video": 2.0, "weight_audio": 1.5},
        )
        print(f"Grade: {report.overall_grade}")
        print(f"Score: {report.overall_score:.2f}")
        for b in report.bottlenecks:
            print(f"Bottleneck: {b.metric_name} = {b.score:.2f}")
    """
    config = config or EvalConfig(output_dir=output_dir)

    # Setup file-based debug logging into output_dir/eval_debug.log
    os.makedirs(output_dir, exist_ok=True)
    debug_log_path = os.path.join(output_dir, "eval_debug.log")
    _file_handler = logging.FileHandler(debug_log_path, mode="w", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )
    # Attach to the root 'directorbench' logger so all sub-modules are captured
    _db_logger = logging.getLogger("directorbench")
    _db_logger.addHandler(_file_handler)
    _db_logger.setLevel(min(_db_logger.level or logging.DEBUG, logging.DEBUG))
    logger.info(f"[main] Debug log → {debug_log_path}")

    # Build the evaluation graph
    graph = build_eval_graph(config)

    # Create initial state
    initial_state = create_initial_state(
        video_path=video_path,
        user_prompt=user_prompt,
        script_text=script_text,
        audio_path=audio_path,
        storyboard=storyboard,
        user_profile=user_profile,
    )

    # Run the evaluation pipeline
    logger.info(f"[main] Starting evaluation for: {video_path}")
    final_state = graph.invoke(initial_state)

    # Extract diagnosis report
    report = final_state.get("diagnosis") or DiagnosisReport()

    # Persist to JSONL (append-only, never overwrites)
    writer = ReportWriter(output_dir=output_dir)
    rid = writer.append(
        report,
        video_path=video_path,
        user_prompt=user_prompt,
        script_text=script_text,
        audio_path=audio_path,
        storyboard=storyboard,
        user_profile=initial_state.user_profile,
    )
    logger.info(f"[main] Result appended to {writer.jsonl_path} (report_id={rid})")
    trace_path = _append_tool_trace(
        output_dir=output_dir,
        report_id=rid,
        video_path=video_path,
        final_state=final_state,
    )
    logger.info(f"[main] Tool trace appended to {trace_path} (report_id={rid})")

    if verbose:
        writer.print_summary(report)

    # Cleanup file handler to avoid accumulation across repeated calls
    _db_logger.removeHandler(_file_handler)
    _file_handler.close()
    logger.info(f"[main] Full debug log saved to {debug_log_path}")

    return report


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    items: list[dict],
    config: EvalConfig | None = None,
    output_dir: str = "./eval_outputs",
) -> list[DiagnosisReport]:
    """
    Evaluate multiple videos in sequence.

    Args:
        items: List of dicts, each with keys matching evaluate_video params
               (video_path, user_prompt, script_text, etc.)
        config: Shared config.
        output_dir: Output directory.

    Returns:
        List of DiagnosisReport objects.
    """
    reports = []
    for i, item in enumerate(items):
        logger.info(f"[batch] Evaluating item {i+1}/{len(items)}: {item.get('video_path', '?')}")
        try:
            report = evaluate_video(
                **item,
                config=config,
                output_dir=os.path.join(output_dir, f"item_{i:03d}"),
            )
            reports.append(report)
        except Exception as e:
            logger.error(f"[batch] Item {i+1} failed: {e}")
            reports.append(DiagnosisReport(summary=f"Evaluation failed: {e}"))

    return reports


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

def _load_profile_by_id(
    profile_id: int,
    profiles_path: str | None = None,
) -> dict:
    """Load a single profile from profiles.jsonl by its profile_id.

    Args:
        profile_id: The ``profile_id`` field to look up.
        profiles_path: Path to the JSONL file.  Falls back to
                       ``data/profiles.jsonl`` relative to the repo root.

    Returns:
        The raw profile dict (will be parsed by ``create_initial_state``).
    """
    if profiles_path is None:
        # Try common relative locations
        for candidate in [
            os.path.join(os.path.dirname(__file__), "..", "data", "profiles.jsonl"),
            os.path.join("data", "profiles.jsonl"),
        ]:
            if os.path.isfile(candidate):
                profiles_path = candidate
                break

    if profiles_path is None or not os.path.isfile(profiles_path):
        raise FileNotFoundError(
            f"profiles.jsonl not found. Provide --profiles-jsonl or place it in data/profiles.jsonl"
        )

    for entry in _iter_json_objects(profiles_path):
        if entry.get("profile_id") == profile_id:
            logger.info(f"[main] Loaded profile {profile_id}: {entry.get('name', '?')}")
            return entry

    raise ValueError(f"Profile ID {profile_id} not found in {profiles_path}")


def _iter_json_objects(filepath: str):
    """Yield top-level JSON objects from a file that may contain either
    strict JSONL (one object per line) or concatenated / pretty-printed
    JSON objects (as in profiles.jsonl)."""
    with open(filepath) as f:
        content = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        # Skip whitespace and stray commas between objects
        while idx < len(content) and content[idx] in " \t\n\r,":
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, idx)
            yield obj
            idx = end
        except json.JSONDecodeError:
            idx += 1


def load_all_profiles(profiles_path: str | None = None) -> list[dict]:
    """Load every profile from profiles.jsonl — useful for batch evaluation."""
    if profiles_path is None:
        profiles_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "profiles.jsonl"
        )
    return list(_iter_json_objects(profiles_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DirectorBench: Multi-Agent Evaluation for Minute-Long Video Generation"
    )
    parser.add_argument("--video", required=True, help="Path to the generated video file")
    parser.add_argument("--prompt", default="", help="Original user prompt")
    parser.add_argument("--script", default="", help="Generated script text")
    parser.add_argument("--audio", default=None, help="Optional separate audio file path")
    parser.add_argument("--storyboard", default=None, help="JSON file with storyboard data")
    parser.add_argument("--profile", default=None, help="JSON file with user profile overrides")
    parser.add_argument(
        "--profile-id", type=int, default=None,
        help="Profile ID to load from data/profiles.jsonl (e.g. 1, 2, 3)"
    )
    parser.add_argument(
        "--profiles-jsonl", default=None,
        help="Path to profiles.jsonl file (default: data/profiles.jsonl)"
    )
    parser.add_argument("--output-dir", default="./eval_outputs", help="Output directory")
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load optional JSON files
    storyboard = None
    if args.storyboard:
        with open(args.storyboard) as f:
            storyboard = json.load(f)

    user_profile = None
    if args.profile_id is not None:
        user_profile = _load_profile_by_id(
            args.profile_id,
            args.profiles_jsonl,
        )
    elif args.profile:
        with open(args.profile) as f:
            user_profile = json.load(f)

    # Load script from file if it's a path
    script_text = args.script
    if script_text and os.path.isfile(script_text):
        with open(script_text) as f:
            script_text = f.read()

    # Build config
    config = EvalConfig()
    config.llm.model = args.model
    config.output_dir = args.output_dir

    # Run evaluation
    report = evaluate_video(
        video_path=args.video,
        user_prompt=args.prompt,
        script_text=script_text,
        audio_path=args.audio,
        storyboard=storyboard,
        user_profile=user_profile,
        config=config,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    # Exit with code based on grade
    if report.overall_grade in ("D", "F"):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
