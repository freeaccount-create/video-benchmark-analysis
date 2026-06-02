"""Aggregate per-case results.jsonl into dimension-level performance and score
distribution statistics for one or more report directories.

Usage:
  python scripts/analyze_reports.py \
      --reports reports/jimeng_seed2pro:jimeng_seed2pro \
      --reports reports/kling:kling \
      --output-dir reports/_analysis

Each --reports argument accepts ``<path>:<label>``. The script writes:
  - per_case.csv   : flat per-case records (overall + dimension + sub-metric scores)
  - summary.csv    : aggregate stats (mean / median / std / min / max / pass-rate)
  - bottlenecks.csv: how often each metric is flagged as a bottleneck
  - report.md      : human-readable Markdown summary
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

DIMENSIONS = ["script", "video", "audio", "stability", "crossmodal"]
PASS_THRESHOLD = 0.5
GRADE_ORDER = ["A", "B", "C", "D", "F"]


def _parse_arg_pair(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"--reports expects <path>:<label>, got {raw!r}"
        )
    path, label = raw.split(":", 1)
    return path, label


def _natural_key(name: str) -> tuple:
    nums = [int(x) for x in re.findall(r"\d+", name)]
    return (nums or [0], name)


def _load_latest_record(jsonl_path: Path) -> dict[str, Any] | None:
    rec = None
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
    return rec


def _flatten_record(label: str, case_name: str, record: dict[str, Any]) -> dict[str, Any]:
    results = record.get("results", {}) or {}
    dim = results.get("dimension_scores", {}) or {}
    flat: dict[str, Any] = {
        "system": label,
        "case": case_name,
        "report_id": results.get("report_id"),
        "overall_score": results.get("overall_score"),
        "overall_grade": results.get("overall_grade"),
    }
    for d in DIMENSIONS:
        flat[f"dim_{d}"] = dim.get(d)
    for er in results.get("all_results", []) or []:
        metric = er.get("metric_name")
        if not metric:
            continue
        flat[f"m_{metric}"] = er.get("score")
        flat[f"c_{metric}"] = er.get("confidence")
    return flat


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(mean(vals), 4),
        "median": round(median(vals), 4),
        "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "pass_rate": round(sum(1 for v in vals if v >= PASS_THRESHOLD) / len(vals), 4),
    }


def _system_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall = _stats([r.get("overall_score") for r in records])
    grade_counts = Counter(r.get("overall_grade") for r in records if r.get("overall_grade"))
    dims: dict[str, dict[str, float | int]] = {}
    for d in DIMENSIONS:
        dims[d] = _stats([r.get(f"dim_{d}") for r in records])

    metric_keys = sorted({k for r in records for k in r if k.startswith("m_")})
    metrics: dict[str, dict[str, float | int]] = {}
    for k in metric_keys:
        metrics[k[2:]] = _stats([r.get(k) for r in records])
    return {
        "n_cases": len(records),
        "overall": overall,
        "grades": dict(sorted(grade_counts.items(), key=lambda x: GRADE_ORDER.index(x[0]) if x[0] in GRADE_ORDER else 99)),
        "dimensions": dims,
        "metrics": metrics,
    }


def _bottleneck_counter(label: str, root: Path) -> Counter:
    counter: Counter = Counter()
    for case_dir in sorted(root.iterdir(), key=lambda p: _natural_key(p.name)):
        results_path = case_dir / "results.jsonl"
        if not results_path.exists():
            continue
        rec = _load_latest_record(results_path)
        if not rec:
            continue
        for b in (rec.get("results", {}) or {}).get("bottlenecks", []) or []:
            metric = b.get("metric_name")
            if metric:
                counter[metric] += 1
    return counter


def _collect(label: str, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_dir in sorted(root.iterdir(), key=lambda p: _natural_key(p.name)):
        results_path = case_dir / "results.jsonl"
        if not results_path.exists():
            continue
        rec = _load_latest_record(results_path)
        if not rec:
            continue
        rows.append(_flatten_record(label, case_dir.name, rec))
    return rows


def _csv_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    lines = ["|" + "|".join(headers) + "|", sep]
    for r in rows:
        lines.append("|" + "|".join(str(x) for x in r) + "|")
    return "\n".join(lines)


def _format_pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.3f}"


def _build_markdown(per_label: dict[str, list[dict[str, Any]]],
                    summaries: dict[str, dict[str, Any]],
                    bottlenecks: dict[str, Counter]) -> str:
    out: list[str] = []
    out.append(f"# Report Analysis (pass threshold = {PASS_THRESHOLD})\n")

    out.append("## 1. Overall Score Summary\n")
    headers = ["system", "n", "mean", "median", "std", "min", "max", "pass_rate"]
    rows = []
    for label, s in summaries.items():
        o = s["overall"]
        rows.append([label, o.get("n", 0), _format_pct(o.get("mean")),
                     _format_pct(o.get("median")), _format_pct(o.get("std")),
                     _format_pct(o.get("min")), _format_pct(o.get("max")),
                     _format_pct(o.get("pass_rate"))])
    out.append(_md_table(headers, rows) + "\n")

    out.append("\n## 2. Grade Distribution\n")
    all_grades = sorted({g for s in summaries.values() for g in s["grades"]},
                        key=lambda g: GRADE_ORDER.index(g) if g in GRADE_ORDER else 99)
    headers = ["system"] + all_grades
    rows = []
    for label, s in summaries.items():
        row = [label] + [s["grades"].get(g, 0) for g in all_grades]
        rows.append(row)
    out.append(_md_table(headers, rows) + "\n")

    out.append("\n## 3. Dimension-level Mean Scores\n")
    headers = ["system"] + DIMENSIONS
    rows = []
    for label, s in summaries.items():
        row = [label]
        for d in DIMENSIONS:
            row.append(_format_pct(s["dimensions"][d].get("mean")))
        rows.append(row)
    out.append(_md_table(headers, rows) + "\n")

    out.append("\n## 4. Dimension Pass Rate (score >= 0.5)\n")
    rows = []
    for label, s in summaries.items():
        row = [label]
        for d in DIMENSIONS:
            row.append(_format_pct(s["dimensions"][d].get("pass_rate")))
        rows.append(row)
    out.append(_md_table(headers, rows) + "\n")

    out.append("\n## 5. Sub-metric Mean Scores\n")
    metric_names = sorted({m for s in summaries.values() for m in s["metrics"]})
    headers = ["metric"] + list(summaries.keys())
    rows = []
    for m in metric_names:
        row = [m]
        for label in summaries:
            row.append(_format_pct(summaries[label]["metrics"].get(m, {}).get("mean")))
        rows.append(row)
    out.append(_md_table(headers, rows) + "\n")

    out.append("\n## 6. Sub-metric Pass Rate (score >= 0.5)\n")
    rows = []
    for m in metric_names:
        row = [m]
        for label in summaries:
            row.append(_format_pct(summaries[label]["metrics"].get(m, {}).get("pass_rate")))
        rows.append(row)
    out.append(_md_table(headers, rows) + "\n")

    out.append("\n## 7. Bottleneck Frequency (times flagged)\n")
    all_bottlenecks = sorted({m for c in bottlenecks.values() for m in c})
    headers = ["metric"] + list(bottlenecks.keys())
    rows = []
    for m in all_bottlenecks:
        row = [m] + [bottlenecks[label].get(m, 0) for label in bottlenecks]
        rows.append(row)
    out.append(_md_table(headers, rows) + "\n")

    out.append("\n## 8. Score Histogram (overall_score)\n")
    bin_edges = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
    bin_labels = ["[0.0,0.3)", "[0.3,0.4)", "[0.4,0.5)", "[0.5,0.6)",
                  "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0]"]
    headers = ["system"] + bin_labels
    rows = []
    for label, recs in per_label.items():
        scores = [r.get("overall_score") for r in recs if r.get("overall_score") is not None]
        counts = [0] * (len(bin_edges) - 1)
        for s in scores:
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= s < bin_edges[i + 1]:
                    counts[i] += 1
                    break
        rows.append([label] + counts)
    out.append(_md_table(headers, rows) + "\n")

    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reports", action="append", type=_parse_arg_pair, required=True,
                   help="Pair of <path>:<label>, repeatable.")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_label: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    bottlenecks: dict[str, Counter] = {}

    for path_str, label in args.reports:
        root = Path(path_str)
        if not root.exists():
            print(f"[WARN] {root} does not exist; skipping")
            continue
        rows = _collect(label, root)
        per_label[label] = rows
        summaries[label] = _system_summary(rows)
        bottlenecks[label] = _bottleneck_counter(label, root)
        print(f"[OK] {label}: {len(rows)} cases")

    flat_rows: list[dict[str, Any]] = []
    for rows in per_label.values():
        flat_rows.extend(rows)
    _csv_dump(args.output_dir / "per_case.csv", flat_rows)

    summary_rows = []
    for label, s in summaries.items():
        base = {"system": label, "n_cases": s["n_cases"]}
        for k, v in s["overall"].items():
            base[f"overall_{k}"] = v
        for d in DIMENSIONS:
            for k, v in s["dimensions"][d].items():
                base[f"{d}_{k}"] = v
        summary_rows.append(base)
    _csv_dump(args.output_dir / "summary.csv", summary_rows)

    bottleneck_rows = []
    all_b = sorted({m for c in bottlenecks.values() for m in c})
    for m in all_b:
        row = {"metric": m}
        for label, c in bottlenecks.items():
            row[label] = c.get(m, 0)
        bottleneck_rows.append(row)
    _csv_dump(args.output_dir / "bottlenecks.csv", bottleneck_rows)

    md = _build_markdown(per_label, summaries, bottlenecks)
    (args.output_dir / "report.md").write_text(md, encoding="utf-8")

    print(f"Wrote analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
