#!/usr/bin/env python3
"""
test_tool_latency.py — Runtime connectivity check for actual DirectorBench model flows.

Unlike synthetic micro-benchmarks, this script calls the same internal methods used
by the production agents:
  - CrossModalEvalAgent._compute_viclip_similarity (MobileViCLIP-Small)
  - CrossModalEvalAgent._compute_lipsync_score
  - CrossModalEvalAgent._compute_text_similarity
  - AudioEvalAgent._extract_bgm_features
  - StabilityEvalAgent._compute_brisque_scores

Input can come from:
  1) --video <mp4>  (extract frame/audio from the video), or
  2) random-initialized image/wav samples when --video is not provided.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from directorbench.agents.audio_agent import AudioEvalAgent
from directorbench.agents.crossmodal_agent import CrossModalEvalAgent
from directorbench.agents.stability_agent import StabilityEvalAgent
from directorbench.schemas import ASRSegment, AudioSegment, GraphState, PreprocessingOutput


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")


def _print_row(name: str, status: str, elapsed_ms: float, detail: str = "") -> None:
    icon = "✅" if status == "PASS" else "⚠️" if status == "SKIP" else "❌"
    print(f"{icon} {name:22s} {status:5s}  {elapsed_ms:8.1f} ms  {detail}")


def _run_step(name: str, fn):
    t0 = time.perf_counter()
    try:
        status, detail = fn()
    except Exception as e:
        status, detail = "FAIL", f"{type(e).__name__}: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _print_row(name, status, elapsed_ms, detail)
    return status


def _make_random_image(out_path: Path) -> Path:
    import numpy as np
    from PIL import Image
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(arr).save(out_path)
    return out_path


def _make_random_wav(out_path: Path, duration_sec: float = 3.0, sr: int = 16000) -> Path:
    n_samples = int(duration_sec * sr)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n_samples):
            v = 0.3 * math.sin(2 * math.pi * 440.0 * (i / sr))
            iv = int(max(-1.0, min(1.0, v)) * 32767)
            frames.extend(iv.to_bytes(2, byteorder="little", signed=True))
        wf.writeframes(frames)
    return out_path


def _extract_frame_and_audio(video_path: Path, out_dir: Path) -> tuple[Path, Path]:
    frame_path = out_dir / "frame.jpg"
    wav_path = out_dir / "audio.wav"

    cmd_frame = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ss", "1", "-frames:v", "1", str(frame_path),
    ]
    cmd_audio = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_path),
    ]
    p1 = subprocess.run(cmd_frame, capture_output=True, text=True)
    if p1.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extract failed: {p1.stderr.strip()}")
    p2 = subprocess.run(cmd_audio, capture_output=True, text=True)
    if p2.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extract failed: {p2.stderr.strip()}")
    return frame_path, wav_path


def _build_state(video_path: str, with_asr: bool = True) -> GraphState:
    prep = PreprocessingOutput(video_path=video_path)
    if with_asr:
        prep.asr_segments = [ASRSegment(start_sec=0.0, end_sec=3.0, text="hello world")]
    return GraphState(video_path=video_path, script_text="A person says hello", preprocessing=prep)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DirectorBench real-flow model runtime checker",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional mp4 input. If provided, frame/audio are extracted from it.",
    )
    args = parser.parse_args()

    _print_header("DirectorBench Real Flow Runtime Check")
    print(f"Python: {sys.version.split()[0]}")

    with tempfile.TemporaryDirectory(prefix="directorbench-flow-check-") as td:
        tmp = Path(td)
        video_path = Path(args.video).expanduser().resolve() if args.video else None

        if video_path and video_path.exists():
            frame_path, wav_path = _extract_frame_and_audio(video_path, tmp)
            print(f"Input mode: mp4 ({video_path})")
        else:
            frame_path = _make_random_image(tmp / "random_frame.png")
            wav_path = _make_random_wav(tmp / "random_audio.wav")
            print("Input mode: random-initialized image + wav")

        print(f"Frame: {frame_path}")
        print(f"Audio: {wav_path}")

        cross = CrossModalEvalAgent()
        audio = AudioEvalAgent()
        stability = StabilityEvalAgent()

        state = _build_state(str(video_path) if video_path else "")

        statuses: list[str] = []

        statuses.append(_run_step(
            "crossmodal.mobileviclip",
            lambda: (
                ("PASS", f"score={s:.4f}") if (s := cross._compute_viclip_similarity(
                    "A person standing in a room", [str(frame_path)]
                )) is not None else ("FAIL", "returned None (MobileViCLIP unavailable or inference failed)")
            ),
        ))

        def _lipsync_step():
            if not video_path or not video_path.exists():
                return "SKIP", "needs --video for LipSyncProxy (optical-flow + audio-energy)"
            s = cross._compute_lipsync_score(state)
            if s is None:
                return "FAIL", "returned None (LipSyncProxy unavailable or failed)"
            return "PASS", f"score={s:.4f}"

        statuses.append(_run_step("crossmodal.lipsync", _lipsync_step))

        statuses.append(_run_step(
            "crossmodal.sbert",
            lambda: (
                ("PASS", f"score={s:.4f}") if (s := cross._compute_text_similarity(
                    "A man enters a dark room.",
                    "Someone walks into a dimly lit room.",
                )) is not None else ("FAIL", "returned None (model unavailable or failed)")
            ),
        ))

        statuses.append(_run_step(
            "audio.librosa",
            lambda: (
                ("PASS", f"tempo={d.get('tempo_bpm')} energy={d.get('mean_energy'):.4f}")
                if "tempo_bpm" in (d := audio._extract_bgm_features(
                    AudioSegment(track_type="bgm", path=str(wav_path), duration_sec=3.0)
                )) else ("FAIL", f"features unavailable: {d}")
            ),
        ))

        statuses.append(_run_step(
            "stability.brisque",
            lambda: (
                ("PASS", f"quality={r[0]:.4f} degradation={r[1]:.4f}")
                if (r := stability._compute_brisque_scores([str(frame_path)], [str(frame_path)])) is not None
                else ("FAIL", "returned None (opencv-contrib/yaml missing)")
            ),
        ))

    _print_header("Summary")
    n_pass = sum(1 for s in statuses if s == "PASS")
    n_skip = sum(1 for s in statuses if s == "SKIP")
    n_fail = sum(1 for s in statuses if s == "FAIL")
    print(f"PASS={n_pass}  SKIP={n_skip}  FAIL={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
