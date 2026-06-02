#!/usr/bin/env python3
"""
Simple AudioShake connectivity test script.

Example:
    python scripts/test_audioshake.py --input /path/to/video.mp4
    python scripts/test_audioshake.py --input /path/to/audio.wav --download-dir /tmp/stems
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


DEFAULT_BASE_URL = "https://api.audioshake.ai"


def _load_dotenv_if_available(dotenv_path: Path) -> None:
    """Best-effort .env loader, so local dev can run without exporting vars."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)


def _guess_mime(file_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(file_path))
    return mime or "application/octet-stream"


def _extract_wav(input_path: Path, out_dir: Path) -> Path:
    """Extract mono 16k WAV from media via ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    out_path = out_dir / "audioshake_input.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.is_file():
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg extract wav failed: {err or 'unknown error'}")
    return out_path


def _prepare_audio_input(input_path: Path, out_dir: Path) -> tuple[Path, bool]:
    """
    Prepare upload audio.
    Returns (audio_path, is_temp_file_to_cleanup).
    """
    suffix = input_path.suffix.lower()
    if suffix == ".wav":
        return input_path, False
    return _extract_wav(input_path, out_dir), True


def _upload_audio(session: requests.Session, base_url: str, headers: dict[str, str], audio_path: Path) -> str:
    with audio_path.open("rb") as f:
        resp = session.post(
            f"{base_url}/assets",
            headers=headers,
            files={"file": (audio_path.name, f, _guess_mime(audio_path))},
            timeout=120,
        )
    resp.raise_for_status()
    data = resp.json()
    asset_id = data.get("id")
    if not asset_id:
        raise RuntimeError(f"Upload succeeded but missing asset id: {data}")
    return asset_id


def _create_task(session: requests.Session, base_url: str, headers: dict[str, str], asset_id: str) -> str:
    resp = session.post(
        f"{base_url}/tasks",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "assetId": asset_id,
            "targets": [
                {"model": "vocals", "formats": ["wav"]},
                {"model": "instrumental", "formats": ["wav"]},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get("id")
    if not task_id:
        raise RuntimeError(f"Create task succeeded but missing task id: {data}")
    return task_id


def _poll_job(
    session: requests.Session,
    base_url: str,
    headers: dict[str, str],
    task_id: str,
    timeout_sec: int,
    interval_sec: int,
) -> dict:
    def _task_status(task_data: dict) -> str:
        top = str(
            task_data.get("status")
            or task_data.get("state")
            or task_data.get("taskStatus")
            or ""
        ).lower()
        if top in {"completed", "failed", "processing", "queued", "running"}:
            return top

        target_statuses: list[str] = []
        for target in task_data.get("targets", []) or []:
            s = str(target.get("status", "")).lower()
            if s:
                target_statuses.append(s)
        if target_statuses:
            if all(s == "completed" for s in target_statuses):
                return "completed"
            if any(s == "failed" for s in target_statuses):
                return "failed"
            if any(s in {"queued", "processing", "running", "pending"} for s in target_statuses):
                return "processing"
        return "unknown"

    elapsed = 0
    consecutive_failures = 0
    while elapsed < timeout_sec:
        try:
            resp = session.get(f"{base_url}/tasks/{task_id}", headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            consecutive_failures = 0
        except requests.RequestException as e:
            consecutive_failures += 1
            print(
                f"[poll] transient network error (#{consecutive_failures}): {e} "
                f"(elapsed={elapsed}s), retrying..."
            )
            time.sleep(interval_sec)
            elapsed += interval_sec
            continue
        status = _task_status(data)
        print(f"[poll] status={status} elapsed={elapsed}s")
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"AudioShake job failed: {data}")
        time.sleep(interval_sec)
        elapsed += interval_sec
    raise TimeoutError(f"AudioShake job timed out after {timeout_sec}s")


def _collect_outputs(task_data: dict) -> list[dict]:
    outputs: list[dict] = []
    for target in task_data.get("targets", []) or []:
        model_name = str(target.get("model", "")).lower()
        for out_item in target.get("output", []) or []:
            out_item = dict(out_item)
            out_item["_target_model"] = model_name
            outputs.append(out_item)
    if not outputs:
        for item in task_data.get("outputAssets", []) or []:
            item = dict(item)
            item["_target_model"] = str(item.get("model", "")).lower()
            outputs.append(item)
    return outputs


def _download_assets(output_assets: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, asset in enumerate(output_assets, start=1):
        name = asset.get("name", f"asset_{i}")
        url = asset.get("link") or asset.get("url")
        if not url:
            print(f"[download] skip '{name}': no url")
            continue
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
        out_path = out_dir / f"{i:02d}_{safe_name}.wav"
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"[download] {name} -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test AudioShake API end-to-end")
    parser.add_argument(
        "--input",
        "--audio",
        dest="input_path",
        required=True,
        help="Path to input media file (wav/mp3/m4a/mp4/mov etc). Non-wav is converted to wav via ffmpeg.",
    )
    parser.add_argument("--api-key", default=None, help="AudioShake API key override")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="AudioShake API base URL")
    parser.add_argument("--timeout", type=int, default=300, help="Polling timeout in seconds")
    parser.add_argument("--interval", type=int, default=5, help="Polling interval in seconds")
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Optional directory to download output stems",
    )
    parser.add_argument(
        "--dotenv",
        default=".env",
        help="Path to .env (used only if python-dotenv is installed)",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Ignore HTTP(S)_PROXY environment variables for this run",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.is_file():
        print(f"[error] input file not found: {input_path}", file=sys.stderr)
        return 2

    dotenv_path = Path(args.dotenv).expanduser().resolve()
    _load_dotenv_if_available(dotenv_path)

    api_key = (args.api_key or os.environ.get("AUDIOSHAKE_API_KEY") or "").strip()
    if not api_key:
        print("[error] AUDIOSHAKE_API_KEY is empty. Export it or pass --api-key.", file=sys.stderr)
        return 2

    headers = {"x-api-key": api_key, "Accept": "application/json"}
    print(f"[info] base_url={args.base_url}")
    print(f"[info] input={input_path}")

    session = requests.Session()
    if args.no_proxy:
        session.trust_env = False
        print("[info] proxy disabled (session.trust_env=False)")

    temp_dir = tempfile.TemporaryDirectory(prefix="audioshake-test-")
    try:
        audio_path, is_temp = _prepare_audio_input(input_path, Path(temp_dir.name))
        if is_temp:
            print(f"[info] extracted wav={audio_path}")

        asset_id = _upload_audio(session, args.base_url, headers, audio_path)
        print(f"[ok] upload success, asset_id={asset_id}")

        task_id = _create_task(session, args.base_url, headers, asset_id)
        print(f"[ok] task created, task_id={task_id}")

        result = _poll_job(session, args.base_url, headers, task_id, args.timeout, args.interval)
        output_assets = _collect_outputs(result)
        print(f"[ok] job completed, output_assets={len(output_assets)}")
        for asset in output_assets:
            name = asset.get("name", "unknown")
            url = asset.get("link") or asset.get("url") or ""
            print(f"  - {name}: {url}")

        if args.download_dir:
            _download_assets(output_assets, Path(args.download_dir).expanduser().resolve())

        print("[success] AudioShake is reachable and working for this audio.")
        return 0
    except Exception as e:
        print(f"[failed] AudioShake test failed: {e}", file=sys.stderr)
        return 1
    finally:
        temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
