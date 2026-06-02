#!/usr/bin/env python3
"""
Batch-evaluate Jimeng seed2pro cases (mp4 + prompt from CSV).

This follows the same "video + prompt + output-dir + profile-id" intent as
`run.sh`, but runs a range of case_ids from a CSV file and writes each case's
artifacts into its own output subdirectory (via ``directorbench.main.evaluate_video``,
so each folder gets ``results.jsonl``, ``tool_traces.jsonl``, and ``eval_debug.log``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CaseRow:
    case_id: str
    prompt: str


def _normalize_prompt(text: str) -> str:
    return " ".join((text or "").split())


def _decode_csv_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    raise RuntimeError("Failed to decode CSV (tried utf-8-sig/utf-8/gb18030)")


def _load_cases_from_csv(csv_path: Path) -> dict[str, CaseRow]:
    text = _decode_csv_bytes(csv_path.read_bytes())
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return {}

    # Find columns (tolerant matching)
    cols = list(rows[0].keys())
    case_col = next((c for c in cols if (c or "").strip().lower() == "case_id"), None)
    prompt_col = next((c for c in cols if (c or "").strip().lower() == "prompt"), None)
    if case_col is None:
        raise RuntimeError(f"CSV missing case_id column. Columns={cols}")
    if prompt_col is None:
        raise RuntimeError(f"CSV missing prompt column. Columns={cols}")

    out: dict[str, CaseRow] = {}
    for r in rows:
        cid = (r.get(case_col) or "").strip()
        if not cid:
            continue
        prompt = (r.get(prompt_col) or "").strip()
        out[cid] = CaseRow(case_id=cid, prompt=prompt)
    return out


def _load_personalized_profiles(jsonl_path: Path) -> dict[str, dict]:
    """
    Build a prompt -> profile mapping from instruction_personalized JSONL.
    """
    if not jsonl_path.is_file():
        return {}

    mapping: dict[str, dict] = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = _normalize_prompt(item.get("generated_prompt", ""))
            profile_id = item.get("profile_id")
            personalization = item.get("profile_personalization")
            if not prompt or profile_id is None or not isinstance(personalization, dict):
                continue
            mapping[prompt] = {
                "source_case_id": str(item.get("_case_id", "")),
                "profile_id": int(profile_id),
                "profile_name": item.get("profile_name", ""),
                "user_profile": {
                    "profile_id": int(profile_id),
                    "name": item.get("profile_name", ""),
                    "personalization": personalization,
                },
            }
    return mapping


def _iter_case_ids(spec: str) -> list[str]:
    """
    Parse "1-10" / "1,2,3" / "001-010" into a list of string case_ids.
    """
    spec = spec.strip()
    if not spec:
        return []
    if "," in spec:
        return [s.strip().lstrip("0") or "0" for s in spec.split(",") if s.strip()]
    if "-" in spec:
        a, b = [x.strip() for x in spec.split("-", 1)]
        if not (a.isdigit() and b.isdigit()):
            raise RuntimeError(f"Invalid --case-range: {spec}")
        start = int(a)
        end = int(b)
        step = 1 if end >= start else -1
        return [str(i) for i in range(start, end + step, step)]
    return [spec.lstrip("0") or "0"]


def _evaluate_one(
    video_path: Path,
    prompt: str,
    output_dir: Path,
    matched_profile: dict,
) -> str:
    """
    Delegate to ``evaluate_video`` so logging/trace behavior matches CLI/main.
    """
    from directorbench.main import evaluate_video

    output_dir.mkdir(parents=True, exist_ok=True)
    user_profile = matched_profile["user_profile"]

    report = evaluate_video(
        video_path=str(video_path),
        user_prompt=prompt,
        user_profile=user_profile,
        output_dir=str(output_dir),
        verbose=True,
    )
    return report.report_id


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch eval Jimeng mp4s (case_id 1-10) using prompts from CSV.")
    parser.add_argument(
        "--csv",
        default="data/jimeng/Benchmark 测试集 - 即梦-seed2pro.csv",
        help="CSV containing columns: case_id, prompt, visualization_url, ...",
    )
    parser.add_argument(
        "--mp4-dir",
        default="data/jimeng",
        help="Directory containing downloaded mp4s named <case_id>.mp4",
    )
    parser.add_argument(
        "--case-range",
        default="1-10",
        help='Case ids to run, e.g. "1-10" or "1,2,3". Default: 1-10',
    )
    parser.add_argument(
        "--output-root",
        default="reports/jimeng_seed2pro",
        help="Root directory to write per-case outputs into (one subdir per case_id).",
    )
    parser.add_argument(
        "--instruction-jsonl",
        default="data/instruction_personalized.jsonl",
        help="JSONL containing generated_prompt -> profile mappings for personalized eval.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print what would run.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parents[1]
    csv_path = (repo_root / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv)
    mp4_dir = (repo_root / args.mp4_dir).resolve() if not Path(args.mp4_dir).is_absolute() else Path(args.mp4_dir)
    output_root = (repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    instruction_jsonl = (
        (repo_root / args.instruction_jsonl).resolve()
        if not Path(args.instruction_jsonl).is_absolute()
        else Path(args.instruction_jsonl)
    )

    cases = _load_cases_from_csv(csv_path)
    prompt_to_profile = _load_personalized_profiles(instruction_jsonl)
    wanted = _iter_case_ids(args.case_range)
    if not wanted:
        print("[ERROR] empty --case-range", file=sys.stderr)
        return 2

    missing_prompt = [cid for cid in wanted if cid not in cases]
    if missing_prompt:
        print(f"[ERROR] Missing prompts for case_ids: {missing_prompt}", file=sys.stderr)
        return 2

    missing_profile_match = [
        cid for cid in wanted if _normalize_prompt(cases[cid].prompt) not in prompt_to_profile
    ]
    if missing_profile_match:
        print(
            f"[ERROR] Missing personalized profile match for case_ids: {missing_profile_match}",
            file=sys.stderr,
        )
        return 2

    # Run sequentially (safer for rate limits); user can parallelize externally if needed.
    for cid in wanted:
        row = cases[cid]
        video_path = mp4_dir / f"{cid}.mp4"
        if not video_path.exists():
            print(f"[ERROR] Missing mp4: {video_path}", file=sys.stderr)
            return 2
        out_dir = output_root / f"case_{cid}"
        matched_profile = prompt_to_profile[_normalize_prompt(row.prompt)]

        if args.dry_run:
            print(
                f"[DRY] case_id={cid} source_case_id={matched_profile['source_case_id']} "
                f"profile_id={matched_profile['profile_id']} profile_name={matched_profile['profile_name']} "
                f"video={video_path} out={out_dir}"
            )
            continue

        print(
            f"[RUN] case_id={cid} source_case_id={matched_profile['source_case_id']} "
            f"profile_id={matched_profile['profile_id']} "
            f"profile_name={matched_profile['profile_name']} -> {out_dir}"
        )
        try:
            report_id = _evaluate_one(
                video_path,
                row.prompt,
                out_dir,
                matched_profile,
            )
            print(
                f"[OK]  case_id={cid} source_case_id={matched_profile['source_case_id']} "
                f"profile_id={matched_profile['profile_id']} "
                f"profile_name={matched_profile['profile_name']} report_id={report_id}"
            )
        except Exception as e:
            print(
                f"[FAIL] case_id={cid} source_case_id={matched_profile['source_case_id']} "
                f"profile_id={matched_profile['profile_id']} "
                f"profile_name={matched_profile['profile_name']} error={e}",
                file=sys.stderr,
            )
            # Continue with next case instead of aborting the whole batch.
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

