#!/usr/bin/env python3
"""
render_report.py — Export a Markdown report from results.jsonl by report_id.

Usage:
    # Render a specific report by ID
    python render_report.py --id abc123def456

    # Render the most recent report
    python render_report.py --latest

    # List all available report IDs
    python render_report.py --list

    # Specify a custom JSONL path
    python render_report.py --id abc123 --jsonl path/to/results.jsonl

    # Write to a specific output file (default: <report_id>.md)
    python render_report.py --id abc123 --out report.md
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Allow running from the project root without installing the package
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Try importing from the package; fall back to a minimal standalone
# implementation so the script works even without pydantic installed.
try:
    from directorbench.report import (
        load_all_records,
        load_record_by_id,
        record_to_markdown,
    )
except ImportError:
    # Standalone fallback — only needs stdlib json
    def load_record_by_id(jsonl_path: str, report_id: str):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("report_id") == report_id:
                    return rec
        return None

    def load_all_records(jsonl_path: str) -> list:
        recs = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return recs

    def record_to_markdown(record: dict) -> str:
        """Minimal standalone Markdown renderer."""
        inputs = record.get("inputs", {})
        results = record.get("results", {})
        dim_scores = results.get("dimension_scores", {})
        bottlenecks = results.get("bottlenecks", [])
        needs_review = results.get("needs_human_review", [])
        all_results = results.get("all_results", [])
        profile = inputs.get("profile") or {}
        lines = [
            "# DirectorBench Diagnosis Report", "",
            f"**Report ID:** `{record.get('report_id', 'N/A')}`  ",
            f"**Timestamp:** {record.get('timestamp', 'N/A')}  ",
            f"**Overall Score:** {results.get('overall_score', 0):.2f}  ",
            f"**Grade:** {results.get('overall_grade', 'N/A')}", "", "---", "",
            "## Evaluation Inputs", "",
            f"**Video:** `{inputs.get('video_path', 'N/A')}`  ",
        ]
        if inputs.get("audio_path"):
            lines.append(f"**Audio:** `{inputs['audio_path']}`  ")
        if inputs.get("user_prompt"):
            lines.append(f"**User Prompt:** {inputs['user_prompt']}  ")
        if profile:
            lines.append(f"**Profile:** {profile.get('name','default')} (ID {profile.get('profile_id','?')})  ")
            pw = profile.get("priority_weights", {})
            if pw:
                lines.append(f"**Weights:** {', '.join(f'{k}={v}' for k,v in pw.items())}  ")
            hc = profile.get("hard_constraints", [])
            if hc:
                lines.append(f"**Hard Constraints:** {', '.join(hc)}  ")
        lines.append("")
        if inputs.get("script_text"):
            lines += ["<details><summary>Script (click to expand)</summary>", "",
                       "```", inputs["script_text"].strip(), "```", "</details>", ""]
        lines += ["## Dimension Scores", ""]
        if dim_scores:
            lines.append("| Dimension | Score |")
            lines.append("|-----------|-------|")
            for dim, score in dim_scores.items():
                lines.append(f"| {dim} | {score:.2f} |")
        lines.append("")
        lines += ["## Executive Summary", "", results.get("summary", "_N/A_"), ""]
        if results.get("detailed_analysis"):
            lines += ["## Detailed Analysis", "", results["detailed_analysis"], ""]
        if bottlenecks:
            lines += ["## Bottlenecks", ""]
            for b in bottlenecks:
                lines.append(f"### {b.get('metric_name','?')} (score: {b.get('score',0):.2f})")
                lines += ["", b.get("description", "")]
                for s in b.get("suggestions", []):
                    lines.append(f"- {s}")
                lines.append("")
        if needs_review:
            lines += ["## Items Needing Human Review", ""]
            for r in needs_review:
                lines.append(f"- **{r.get('metric_name','?')}**: score={r.get('score',0):.2f}, "
                             f"confidence={r.get('confidence',0):.2f} (from {r.get('agent_id','?')})")
            lines.append("")
        if all_results:
            lines += ["## All Evaluation Results", ""]
            lines.append("| Agent | Metric | Score | Confidence | Evidence Count |")
            lines.append("|-------|--------|-------|------------|----------------|")
            for r in all_results:
                lines.append(f"| {r.get('agent_id','?')} | {r.get('metric_name','?')} "
                             f"| {r.get('score',0):.2f} | {r.get('confidence',0):.2f} "
                             f"| {len(r.get('evidence',[]))} |")
            lines.append("")
        return "\n".join(lines)


def _default_jsonl_path() -> str:
    """Return the default results.jsonl location."""
    for candidate in [
        os.path.join("reports", "results.jsonl"),
        os.path.join(_project_root, "reports", "results.jsonl"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return os.path.join("reports", "results.jsonl")


def _default_md_filename(record: dict, jsonl_path: str) -> str:
    """Return default markdown file path next to the JSONL file."""
    rid = record.get("report_id") or "report"
    out_dir = os.path.dirname(os.path.abspath(jsonl_path)) or "."
    return os.path.join(out_dir, f"{rid}.md")


def cmd_list(jsonl_path: str) -> None:
    """Print all available report IDs with summary info."""
    if not os.path.isfile(jsonl_path):
        print(f"No results file found at {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    records = load_all_records(jsonl_path)
    if not records:
        print("No records found.")
        return

    print(f"{'#':>4}  {'Report ID':<14}  {'MD Filename':<20}  {'Grade':>5}  {'Score':>6}  {'Profile':<22}  {'Video Path'}")
    print(f"{'─'*4}  {'─'*14}  {'─'*20}  {'─'*5}  {'─'*6}  {'─'*22}  {'─'*40}")

    for i, rec in enumerate(records, 1):
        rid = rec.get("report_id", "?")
        results = rec.get("results", {})
        inputs = rec.get("inputs", {})
        grade = results.get("overall_grade", "?")
        score = results.get("overall_score", 0)
        profile = inputs.get("profile") or {}
        pname = profile.get("name", "default")
        pid = profile.get("profile_id", "?")
        md_name = os.path.basename(_default_md_filename(rec, jsonl_path))
        vpath = inputs.get("video_path", "?")
        # Truncate long paths
        if len(vpath) > 40:
            vpath = "..." + vpath[-37:]

        print(f"{i:>4}  {rid:<14}  {md_name:<20}  {grade:>5}  {score:>6.2f}  {pname} (#{pid}){'':<{max(0, 18-len(pname))}}  {vpath}")


def cmd_render(jsonl_path: str, report_id: str | None, latest: bool, out: str | None) -> None:
    """Render a single record to Markdown."""
    if not os.path.isfile(jsonl_path):
        print(f"No results file found at {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    if latest:
        records = load_all_records(jsonl_path)
        if not records:
            print("No records found.", file=sys.stderr)
            sys.exit(1)
        record = records[-1]
    elif report_id:
        record = load_record_by_id(jsonl_path, report_id)
        if record is None:
            print(f"Report ID '{report_id}' not found in {jsonl_path}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Specify --id <report_id> or --latest", file=sys.stderr)
        sys.exit(1)

    md = record_to_markdown(record)

    out_path = out or _default_md_filename(record, jsonl_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"Markdown report written to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render DirectorBench evaluation reports from results.jsonl"
    )
    parser.add_argument(
        "--id", dest="report_id", default=None,
        help="Report ID to render (the 12-char hex string)"
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Render the most recent report"
    )
    parser.add_argument(
        "--list", dest="list_all", action="store_true",
        help="List all report IDs in the JSONL file"
    )
    parser.add_argument(
        "--jsonl", default=None,
        help="Path to results.jsonl (default: reports/results.jsonl)"
    )
    parser.add_argument(
        "--out", "-o", default=None,
        help="Output file path for Markdown (default: <report_id>.md)"
    )

    args = parser.parse_args()
    jsonl_path = args.jsonl or _default_jsonl_path()

    if args.list_all:
        cmd_list(jsonl_path)
    elif args.report_id or args.latest:
        cmd_render(jsonl_path, args.report_id, args.latest, args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
