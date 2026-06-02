"""
preprocessing.py — Phase 0: Video/Audio preprocessing pipeline.

Handles:
  - Shot detection (via PySceneDetect / TransNetV2)
  - Audio separation
  - ASR transcription (via Whisper)
  - Representative frame extraction

All heavy tools are wrapped with try/except so the framework degrades
gracefully if a tool is not installed (records failure and returns fallback data).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .config import PreprocessConfig
from .schemas import (
    ASRSegment, AudioSegment, PreprocessingOutput, ShotSegment,
    ToolCallRecord, ToolStatus, TransitionBoundary,
)

logger = logging.getLogger(__name__)


# Module-level cache for the faster-whisper model so multiple Preprocessor
# instances (one per case in a batch) share a single loaded copy of the
# weights. Keyed by (model_name, device, compute_type, download_root).
_Preprocessor_FW_CACHE: dict[tuple[str, str, str, str], object] = {}


# Circuit breaker for ASR/OpenAI in auto mode. When the OpenAI Transcriptions
# endpoint is unreachable (e.g. corporate network blocks api.openai.com), we
# don't want every case in a batch to pay the per-attempt timeout. After the
# first connection-style failure we "trip" the breaker for a configurable
# cooldown, during which the auto dispatcher skips OpenAI and goes straight
# to faster-whisper.
_OPENAI_ASR_BREAKER_OPEN_UNTIL: float = 0.0


def _openai_asr_breaker_is_open() -> bool:
    return time.time() < _OPENAI_ASR_BREAKER_OPEN_UNTIL


def _trip_openai_asr_breaker(reason: str) -> None:
    """Open the breaker for OPENAI_ASR_BREAKER_COOLDOWN_SEC seconds."""
    global _OPENAI_ASR_BREAKER_OPEN_UNTIL
    cooldown = float(os.environ.get("OPENAI_ASR_BREAKER_COOLDOWN_SEC", "300"))
    _OPENAI_ASR_BREAKER_OPEN_UNTIL = time.time() + cooldown
    logger.warning(
        f"[Preprocessor] OpenAI ASR circuit breaker OPENED for {cooldown:.0f}s "
        f"({reason}). All subsequent cases in this process will skip OpenAI "
        f"and go straight to faster-whisper."
    )


def _retry_call(
    fn,
    *,
    label: str,
    max_attempts: int = 3,
    backoff_initial_sec: float = 2.0,
    backoff_factor: float = 2.0,
    retryable: tuple[type[BaseException], ...] = (Exception,),
    fatal: tuple[type[BaseException], ...] = (),
):
    """Invoke ``fn()`` up to ``max_attempts`` times with exponential backoff.

    - ``retryable``: exception types that should trigger a retry.
    - ``fatal``: exception types that should abort immediately (e.g. auth
      errors that won't be fixed by waiting). Raised verbatim.
    Re-raises the last exception once attempts are exhausted.
    """
    delay = backoff_initial_sec
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except fatal as e:  # type: ignore[misc]
            logger.warning(f"[retry] {label}: fatal error, not retrying ({e})")
            raise
        except retryable as e:  # type: ignore[misc]
            last_exc = e
            if attempt >= max_attempts:
                logger.warning(
                    f"[retry] {label}: giving up after {attempt} attempts ({e})"
                )
                raise
            logger.info(
                f"[retry] {label}: attempt {attempt}/{max_attempts} failed "
                f"({type(e).__name__}: {e}); retrying in {delay:.1f}s"
            )
            time.sleep(delay)
            delay *= backoff_factor
    # Should be unreachable, but keep mypy/runtime sane.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"_retry_call({label}) exited without result")


class Preprocessor:
    """Orchestrator's preprocessing engine."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()
        self._temp_dir = tempfile.mkdtemp(prefix="directorbench_")
        self._tool_records: list[ToolCallRecord] = []

    def _record_tool(
        self,
        tool_name: str,
        status: ToolStatus,
        detail: str = "",
        elapsed_ms: float | None = None,
        affects: list[str] | None = None,
    ) -> None:
        """Append a tool-call record for downstream agents to consume."""
        self._tool_records.append(ToolCallRecord(
            tool_name=tool_name,
            status=status,
            detail=detail,
            elapsed_ms=elapsed_ms,
            affects=affects or [],
        ))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str,
        audio_path: str | None = None,
        script_text: str | None = None,
        storyboard: list[dict] | None = None,
    ) -> PreprocessingOutput:
        """Execute the full preprocessing pipeline."""
        logger.info(f"[Preprocessor] Starting preprocessing for: {video_path}")

        # 1) Probe video metadata
        t = time.perf_counter()
        duration, fps, resolution = self._probe_video(video_path)
        self._record_tool("Stage/ProbeVideo", ToolStatus.SUCCESS, elapsed_ms=(time.perf_counter() - t) * 1000.0)

        # 2) Shot detection
        t = time.perf_counter()
        shots = self._detect_shots(video_path, duration)
        self._record_tool("Stage/ShotDetection", ToolStatus.SUCCESS, elapsed_ms=(time.perf_counter() - t) * 1000.0)

        # 3) Extract representative frames per shot
        t = time.perf_counter()
        self._extract_frames(video_path, shots)
        self._record_tool("Stage/FrameExtraction", ToolStatus.SUCCESS, elapsed_ms=(time.perf_counter() - t) * 1000.0)

        # 3.5) Extract boundary frames & compute transition metrics
        t = time.perf_counter()
        transitions = self._analyze_transitions(video_path, shots, fps)
        self._record_tool("Stage/TransitionAnalysis", ToolStatus.SUCCESS, elapsed_ms=(time.perf_counter() - t) * 1000.0)

        # 4) Audio separation
        audio_segments = []
        effective_audio_path = audio_path
        if audio_path is None:
            t = time.perf_counter()
            effective_audio_path = self._extract_audio(video_path)
            self._record_tool("Stage/AudioExtract", ToolStatus.SUCCESS, elapsed_ms=(time.perf_counter() - t) * 1000.0)
        if effective_audio_path:
            t = time.perf_counter()
            audio_segments = self._separate_audio(effective_audio_path)
            self._record_tool("Stage/AudioSeparate", ToolStatus.SUCCESS, elapsed_ms=(time.perf_counter() - t) * 1000.0)

        # 5) ASR transcription
        asr_segments = []
        if effective_audio_path:
            t = time.perf_counter()
            asr_segments = self._run_asr(effective_audio_path, duration_sec=duration)
            self._record_tool("Stage/ASR", ToolStatus.SUCCESS, elapsed_ms=(time.perf_counter() - t) * 1000.0)

        return PreprocessingOutput(
            video_path=video_path,
            audio_path=effective_audio_path,
            script_text=script_text,
            storyboard=storyboard,
            shots=shots,
            transitions=transitions,
            audio_segments=audio_segments,
            asr_segments=asr_segments,
            total_duration_sec=duration,
            fps=fps,
            resolution=resolution,
            tool_records=self._tool_records,
        )

    # ------------------------------------------------------------------
    # Video probing
    # ------------------------------------------------------------------

    def _probe_video(self, video_path: str) -> tuple[float, float, tuple[int, int]]:
        """Use ffprobe to get video metadata."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            import json
            data = json.loads(result.stdout)

            duration = float(data.get("format", {}).get("duration", 60.0))
            video_stream = next(
                (s for s in data.get("streams", []) if s["codec_type"] == "video"),
                {}
            )
            fps_str = video_stream.get("r_frame_rate", "24/1")
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den else 24.0
            width = int(video_stream.get("width", 1920))
            height = int(video_stream.get("height", 1080))

            self._record_tool("ffprobe", ToolStatus.SUCCESS,
                              f"duration={duration:.1f}s, fps={fps:.1f}, {width}x{height}")
            return duration, fps, (width, height)
        except Exception as e:
            logger.warning(f"[Preprocessor] ffprobe failed: {e}, using defaults")
            self._record_tool("ffprobe", ToolStatus.FAILED, str(e),
                              affects=["all — video metadata defaults used"])
            return 60.0, 24.0, (1920, 1080)

    # ------------------------------------------------------------------
    # Shot detection
    # ------------------------------------------------------------------

    def _detect_shots(self, video_path: str, total_duration: float) -> list[ShotSegment]:
        """
        Detect shot boundaries using PySceneDetect.

        We try the configured threshold first, and if it returns 0 scenes
        we retry with a small ladder of more permissive thresholds before
        falling back to uniform segmentation. Some videos have soft / fade
        transitions that don't trip ContentDetector at threshold=27.0 but
        are clearly visible at 18.0–12.0.
        """
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
        except ImportError:
            logger.warning("[Preprocessor] PySceneDetect not installed, using uniform segmentation")
            self._record_tool("PySceneDetect", ToolStatus.FALLBACK,
                              "Not installed — uniform segmentation used instead",
                              affects=["temporal_coherence", "transition_quality"])
            return self._uniform_shots(total_duration)

        primary_thr = float(self.config.shot_detection_threshold)
        # Ladder: configured → 18 → 12 (skipping any that aren't strictly lower).
        ladder = [primary_thr] + [t for t in (18.0, 12.0) if t < primary_thr]

        scene_list: list = []
        used_threshold = primary_thr
        for thr in ladder:
            try:
                video = open_video(video_path)
                scene_manager = SceneManager()
                scene_manager.add_detector(ContentDetector(threshold=thr))
                scene_manager.detect_scenes(video)
                scene_list = scene_manager.get_scene_list()
            except Exception as e:
                logger.warning(
                    f"[Preprocessor] PySceneDetect raised at threshold={thr}: {e}"
                )
                scene_list = []
            used_threshold = thr
            if scene_list:
                break

        shots = []
        for i, (start, end) in enumerate(scene_list):
            start_sec = start.get_seconds()
            end_sec = end.get_seconds()
            shots.append(ShotSegment(
                index=i,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=end_sec - start_sec,
            ))

        if shots:
            detail = f"Detected {len(shots)} shots (threshold={used_threshold:g})"
            if used_threshold != primary_thr:
                detail += f"; primary threshold={primary_thr:g} returned 0 scenes"
            logger.info(f"[Preprocessor] {detail}")
            self._record_tool("PySceneDetect", ToolStatus.SUCCESS, detail,
                              affects=["temporal_coherence", "transition_quality", "generation_stability"])
            return shots

        logger.info("[Preprocessor] PySceneDetect found 0 shots at any threshold")
        self._record_tool(
            "PySceneDetect", ToolStatus.FALLBACK,
            f"No shots detected (tried thresholds {ladder}) — uniform segmentation used",
            affects=["temporal_coherence", "transition_quality"],
        )
        return self._uniform_shots(total_duration)

    def _uniform_shots(self, total_duration: float, segment_sec: float = 10.0) -> list[ShotSegment]:
        """Fallback: split video into uniform segments."""
        shots = []
        t = 0.0
        i = 0
        while t < total_duration:
            end = min(t + segment_sec, total_duration)
            shots.append(ShotSegment(index=i, start_sec=t, end_sec=end, duration_sec=end - t))
            t = end
            i += 1
        return shots

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(self, video_path: str, shots: list[ShotSegment]) -> None:
        """Extract representative frames for each shot using ffmpeg."""
        frames_dir = os.path.join(self._temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for shot in shots:
            mid = (shot.start_sec + shot.end_sec) / 2
            out_path = os.path.join(
                frames_dir,
                f"shot_{shot.index:03d}.{self.config.frame_output_format}"
            )
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-ss", str(mid), "-i", video_path,
                        "-frames:v", "1", "-q:v", "2", out_path
                    ],
                    capture_output=True, timeout=15,
                )
                shot.thumbnail_path = out_path
            except Exception as e:
                logger.warning(f"[Preprocessor] Frame extraction failed for shot {shot.index}: {e}")

    # ------------------------------------------------------------------
    # Transition boundary analysis (no local models — pure OpenCV)
    # ------------------------------------------------------------------

    def _analyze_transitions(
        self, video_path: str, shots: list[ShotSegment], fps: float
    ) -> list[TransitionBoundary]:
        """For each pair of consecutive shots, extract boundary frames and
        compute algorithmic splice-quality metrics using only OpenCV.

        Returns a list of TransitionBoundary objects with SSIM, histogram
        difference, and optical flow magnitude pre-computed.
        """
        if len(shots) < 2:
            return []

        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.warning("[Preprocessor] OpenCV not available, skipping transition analysis")
            self._record_tool("OpenCV/TransitionAnalysis", ToolStatus.FAILED,
                              "OpenCV not installed — no SSIM/histogram/optical-flow metrics",
                              affects=["transition_quality"])
            return []

        boundaries_dir = os.path.join(self._temp_dir, "boundaries")
        os.makedirs(boundaries_dir, exist_ok=True)

        transitions: list[TransitionBoundary] = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"[Preprocessor] Cannot open video: {video_path}")
            return []

        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps

        # SSIM and Farneback optical-flow on full-resolution (e.g. 1080p)
        # frames cost ~2-3 s per boundary on CPU; for a typical 25-shot
        # video that's ~60-70 s of pure OpenCV time. Downscaling the gray
        # frames to ~360p before metric computation has no meaningful
        # impact on the *relative* signal we feed downstream agents and
        # gives a 4-6× speedup. Histogram diff stays at full colour
        # resolution because it's already cheap and we want raw colour
        # statistics. The boundary thumbnails saved to disk also stay at
        # full resolution since downstream VLM agents read them.
        metric_height = int(os.environ.get("TRANSITION_METRIC_HEIGHT", "360"))

        def _downscale(frame):
            if frame is None:
                return frame
            h = frame.shape[0]
            if h <= metric_height:
                return frame
            scale = metric_height / float(h)
            new_w = max(1, int(round(frame.shape[1] * scale)))
            return cv2.resize(frame, (new_w, metric_height), interpolation=cv2.INTER_AREA)

        for i in range(len(shots) - 1):
            shot_a, shot_b = shots[i], shots[i + 1]

            # --- Extract boundary frames ---
            # Last frame of shot A (1 frame before the cut)
            frame_a_time = max(0, shot_a.end_sec - 1.0 / actual_fps)
            frame_a = self._read_frame_at(cap, frame_a_time, actual_fps)

            # First frame of shot B
            frame_b_time = shot_b.start_sec
            frame_b = self._read_frame_at(cap, frame_b_time, actual_fps)

            if frame_a is None or frame_b is None:
                continue

            # Save full-resolution boundary frames for downstream VLM use
            path_before = os.path.join(boundaries_dir, f"trans_{i:03d}_before.jpg")
            path_after = os.path.join(boundaries_dir, f"trans_{i:03d}_after.jpg")
            cv2.imwrite(path_before, frame_a)
            cv2.imwrite(path_after, frame_b)

            shot_a.last_frame_path = path_before
            shot_b.first_frame_path = path_after

            # Downscaled copies feed only the algorithmic metrics (SSIM and
            # optical flow) — these are scale-tolerant and ~5× faster at
            # 360p than at 1080p.
            frame_a_small = _downscale(frame_a)
            frame_b_small = _downscale(frame_b)

            ssim_val = self._compute_ssim(frame_a_small, frame_b_small, cv2, np)
            hist_diff = self._compute_histogram_diff(frame_a, frame_b, cv2, np)
            flow_mag = self._compute_optical_flow_magnitude(
                frame_a_small, frame_b_small, cv2, np
            )

            transitions.append(TransitionBoundary(
                from_shot_index=i,
                to_shot_index=i + 1,
                timestamp_sec=shot_a.end_sec,
                frame_before_path=path_before,
                frame_after_path=path_after,
                ssim=ssim_val,
                histogram_diff=hist_diff,
                optical_flow_magnitude=flow_mag,
            ))

        cap.release()
        logger.info(
            f"[Preprocessor] Analyzed {len(transitions)} transition boundaries "
            f"(metrics @ {metric_height}p)"
        )
        if transitions:
            self._record_tool("OpenCV/TransitionAnalysis", ToolStatus.SUCCESS,
                              f"Computed SSIM/histogram/flow for {len(transitions)} boundaries",
                              affects=["transition_quality"])
        return transitions

    @staticmethod
    def _read_frame_at(cap, time_sec: float, fps: float):
        """Seek to a timestamp and read one frame."""
        frame_idx = int(time_sec * fps)
        cap.set(1, frame_idx)  # cv2.CAP_PROP_POS_FRAMES
        ret, frame = cap.read()
        return frame if ret else None

    @staticmethod
    def _compute_ssim(frame_a, frame_b, cv2, np) -> float:
        """Compute structural similarity between two frames.
        Uses a simplified SSIM formula (luminance channel only) to avoid
        external dependencies beyond OpenCV and NumPy."""
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Resize to same dimensions if needed
        if gray_a.shape != gray_b.shape:
            h = min(gray_a.shape[0], gray_b.shape[0])
            w = min(gray_a.shape[1], gray_b.shape[1])
            gray_a = cv2.resize(gray_a, (w, h))
            gray_b = cv2.resize(gray_b, (w, h))

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)

        mu_a_sq = mu_a ** 2
        mu_b_sq = mu_b ** 2
        mu_ab = mu_a * mu_b

        sigma_a_sq = cv2.GaussianBlur(gray_a ** 2, (11, 11), 1.5) - mu_a_sq
        sigma_b_sq = cv2.GaussianBlur(gray_b ** 2, (11, 11), 1.5) - mu_b_sq
        sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_ab

        numerator = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
        denominator = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)

        ssim_map = numerator / denominator
        return float(np.mean(ssim_map))

    @staticmethod
    def _compute_histogram_diff(frame_a, frame_b, cv2, np) -> float:
        """Chi-square distance between colour histograms of two frames.
        Higher value = larger colour shift at the boundary."""
        hist_a = cv2.calcHist([frame_a], [0, 1, 2], None, [8, 8, 8],
                              [0, 256, 0, 256, 0, 256])
        hist_b = cv2.calcHist([frame_b], [0, 1, 2], None, [8, 8, 8],
                              [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        return float(cv2.compareHist(hist_a.flatten(), hist_b.flatten(),
                                     cv2.HISTCMP_CHISQR))

    @staticmethod
    def _compute_optical_flow_magnitude(frame_a, frame_b, cv2, np) -> float:
        """Mean Farneback optical flow magnitude at the boundary.
        A spike in magnitude signals abrupt motion discontinuity."""
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        if gray_a.shape != gray_b.shape:
            h = min(gray_a.shape[0], gray_b.shape[0])
            w = min(gray_a.shape[1], gray_b.shape[1])
            gray_a = cv2.resize(gray_a, (w, h))
            gray_b = cv2.resize(gray_b, (w, h))

        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(magnitude))

    # ------------------------------------------------------------------
    # Audio extraction & separation
    # ------------------------------------------------------------------

    def _extract_audio(self, video_path: str) -> str | None:
        """Extract audio track from video via ffmpeg."""
        audio_out = os.path.join(self._temp_dir, "audio.wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", audio_out],
                capture_output=True, timeout=60,
            )
            if os.path.exists(audio_out):
                self._record_tool("ffmpeg/AudioExtract", ToolStatus.SUCCESS, "",
                                  affects=["narration_reasonableness", "bgm_consistency", "video_audio_consistency", "text_audio_consistency"])
                return audio_out
            self._record_tool("ffmpeg/AudioExtract", ToolStatus.FAILED, "Output file not created",
                              affects=["narration_reasonableness", "bgm_consistency", "video_audio_consistency", "text_audio_consistency"])
            return None
        except Exception as e:
            logger.warning(f"[Preprocessor] Audio extraction failed: {e}")
            self._record_tool("ffmpeg/AudioExtract", ToolStatus.FAILED, str(e),
                              affects=["narration_reasonableness", "bgm_consistency", "video_audio_consistency", "text_audio_consistency"])
            return None

    def _separate_audio(self, audio_path: str) -> list[AudioSegment]:
        """
        Separate audio into dialogue / BGM tracks via AudioShake API.

        AudioShake Tasks API workflow:
          1. Upload audio file via /assets → get asset_id
          2. Create task via /tasks (vocals + instrumental) → get task_id
          3. Poll /tasks/{task_id} until completed
          4. Download output stems

        Falls back to returning the mixed track if the API is unavailable.

        Required env var: AUDIOSHAKE_API_KEY
        """
        if not self.config.separate_audio_tracks:
            return [AudioSegment(track_type="mixed", path=audio_path, duration_sec=0)]

        import requests as _requests

        api_key = os.environ.get("AUDIOSHAKE_API_KEY")
        if not api_key:
            logger.warning("[Preprocessor] AUDIOSHAKE_API_KEY not set, skipping audio separation")
            self._record_tool("AudioShake", ToolStatus.SKIPPED,
                              "AUDIOSHAKE_API_KEY not set — using mixed audio",
                              affects=["bgm_consistency", "narration_reasonableness"])
            return [AudioSegment(track_type="mixed", path=audio_path, duration_sec=0)]

        base_url = "https://api.audioshake.ai"
        headers = {
            "x-api-key": api_key,
            "Accept": "application/json",
        }
        use_no_proxy = (os.environ.get("AUDIOSHAKE_NO_PROXY", "").strip().lower()
                        in {"1", "true", "yes", "on"})
        session = _requests.Session()
        if use_no_proxy:
            session.trust_env = False
            logger.info("[Preprocessor] AudioShake requests proxy disabled via AUDIOSHAKE_NO_PROXY")

        max_attempts = int(os.environ.get("AUDIOSHAKE_MAX_ATTEMPTS", "3"))
        backoff_sec = float(os.environ.get("AUDIOSHAKE_BACKOFF_SEC", "5.0"))
        retryable_exc: tuple[type[BaseException], ...] = (
            _requests.ConnectionError, _requests.Timeout, _requests.HTTPError,
            ConnectionError, TimeoutError, OSError,
        )

        try:
            # --- Step 1: Upload asset (retry on transient network errors) ---
            def _do_upload():
                with open(audio_path, "rb") as f:
                    r = session.post(
                        f"{base_url}/assets",
                        headers=headers,
                        files={"file": (os.path.basename(audio_path), f, "audio/wav")},
                        timeout=120,
                    )
                # Retry only on 5xx / 429; bail on 4xx (auth, payload).
                if r.status_code >= 500 or r.status_code == 429:
                    r.raise_for_status()
                r.raise_for_status()
                return r

            upload_resp = _retry_call(
                _do_upload,
                label="AudioShake/upload",
                max_attempts=max_attempts,
                backoff_initial_sec=backoff_sec,
                retryable=retryable_exc,
            )
            asset_id = upload_resp.json().get("id")
            logger.info(f"[Preprocessor] AudioShake upload OK, asset_id={asset_id}")

            # --- Step 2: Create separation task ---
            def _do_create_task():
                r = session.post(
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
                r.raise_for_status()
                return r

            task_resp = _retry_call(
                _do_create_task,
                label="AudioShake/create_task",
                max_attempts=max_attempts,
                backoff_initial_sec=backoff_sec,
                retryable=retryable_exc,
            )
            task_id = task_resp.json().get("id")
            logger.info(f"[Preprocessor] AudioShake task created, task_id={task_id}")

            # --- Step 3: Poll until complete ---
            import time
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

            max_wait, interval = 300, 5
            elapsed = 0
            result_data = None
            while elapsed < max_wait:
                try:
                    status_resp = session.get(
                        f"{base_url}/tasks/{task_id}",
                        headers=headers,
                        timeout=30,
                    )
                    status_resp.raise_for_status()
                    result_data = status_resp.json()
                except _requests.RequestException as e:
                    logger.warning(f"[Preprocessor] AudioShake poll transient error: {e}, retrying...")
                    time.sleep(interval)
                    elapsed += interval
                    continue
                status = _task_status(result_data)
                if status == "completed":
                    break
                elif status == "failed":
                    logger.warning(f"[Preprocessor] AudioShake job failed: {result_data}")
                    self._record_tool("AudioShake", ToolStatus.FAILED,
                                      "Separation job failed — using mixed audio",
                                      affects=["bgm_consistency", "narration_reasonableness"])
                    return [AudioSegment(track_type="mixed", path=audio_path, duration_sec=0)]
                time.sleep(interval)
                elapsed += interval

            if not result_data or _task_status(result_data) != "completed":
                logger.warning("[Preprocessor] AudioShake task timed out")
                self._record_tool("AudioShake", ToolStatus.FAILED,
                                  "Separation task timed out — using mixed audio",
                                  affects=["bgm_consistency", "narration_reasonableness"])
                return [AudioSegment(track_type="mixed", path=audio_path, duration_sec=0)]

            # --- Step 4: Download stems ---
            out_dir = os.path.join(self._temp_dir, "separated")
            os.makedirs(out_dir, exist_ok=True)

            def _collect_outputs(task_data: dict) -> list[dict]:
                outputs: list[dict] = []
                for target in task_data.get("targets", []) or []:
                    model_name = str(target.get("model", "")).lower()
                    for out_item in target.get("output", []) or []:
                        out_item = dict(out_item)
                        out_item["_target_model"] = model_name
                        outputs.append(out_item)
                # Backward-compatible fallback if API shape changes.
                if not outputs:
                    for item in task_data.get("outputAssets", []) or []:
                        item = dict(item)
                        item["_target_model"] = str(item.get("model", "")).lower()
                        outputs.append(item)
                return outputs

            segments = []
            output_assets = _collect_outputs(result_data)
            for asset in output_assets:
                stem_name = asset.get("name", "unknown")
                target_model = str(asset.get("_target_model", "")).lower()
                download_url = asset.get("link") or asset.get("url", "")
                if not download_url:
                    continue

                stem_name_l = stem_name.lower()
                if "vocal" in stem_name_l or target_model == "vocals":
                    track_type = "dialogue"
                elif "instrument" in stem_name_l or target_model == "instrumental":
                    track_type = "bgm"
                else:
                    track_type = "bgm"
                stem_path = os.path.join(out_dir, f"{track_type}.wav")

                try:
                    def _do_download():
                        r = session.get(download_url, timeout=120, stream=True)
                        r.raise_for_status()
                        with open(stem_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        return True

                    _retry_call(
                        _do_download,
                        label=f"AudioShake/download[{track_type}]",
                        max_attempts=max_attempts,
                        backoff_initial_sec=backoff_sec,
                        retryable=retryable_exc,
                    )
                except _requests.RequestException as e:
                    logger.warning(f"[Preprocessor] AudioShake stem download failed ({track_type}): {e}")
                    continue

                segments.append(AudioSegment(track_type=track_type, path=stem_path, duration_sec=0))
                logger.info(f"[Preprocessor] AudioShake stem downloaded: {track_type} → {stem_path}")

            if segments:
                self._record_tool("AudioShake", ToolStatus.SUCCESS,
                                  f"Separated {len(segments)} stems",
                                  affects=["bgm_consistency", "narration_reasonableness"])
                return segments
            self._record_tool("AudioShake", ToolStatus.FAILED,
                              "No output stems — using mixed audio",
                              affects=["bgm_consistency", "narration_reasonableness"])
            return [AudioSegment(track_type="mixed", path=audio_path, duration_sec=0)]

        except Exception as e:
            logger.warning(f"[Preprocessor] AudioShake separation failed: {e}, using mixed")
            self._record_tool("AudioShake", ToolStatus.FAILED, str(e),
                              affects=["bgm_consistency", "narration_reasonableness"])
            return [AudioSegment(track_type="mixed", path=audio_path, duration_sec=0)]

    # ------------------------------------------------------------------
    # ASR — dispatch between OpenAI Transcriptions API and local
    # faster-whisper. Controlled by env var ``ASR_BACKEND``:
    #   - "openai"          : remote OpenAI API only (no local fallback)
    #   - "faster_whisper"  : local faster-whisper only (no API call)
    #   - "auto" (default)  : try OpenAI first; on connection / timeout /
    #                         rate-limit failure, transparently fall back
    #                         to faster-whisper. SKIP states (empty audio,
    #                         genuine no-speech) are NOT retried locally.
    # ------------------------------------------------------------------

    def _run_asr(self, audio_path: str, duration_sec: float | None = None) -> list[ASRSegment]:
        backend = (os.environ.get("ASR_BACKEND") or "auto").strip().lower()

        if backend == "faster_whisper":
            return self._run_asr_faster_whisper(audio_path, duration_sec=duration_sec)

        if backend == "openai":
            return self._run_asr_openai(audio_path, duration_sec=duration_sec)

        # ---- auto mode ----
        # Fast path: if the breaker has tripped earlier in this process, skip
        # the OpenAI attempt entirely. This matters in batch runs on networks
        # that can't reach api.openai.com — without the breaker, every case
        # would pay the full per-request timeout (~8-20 s) before falling back.
        if _openai_asr_breaker_is_open():
            self._record_tool(
                "ASR/OpenAI", ToolStatus.SKIPPED,
                "Circuit breaker open — using faster-whisper directly",
                affects=["narration_reasonableness", "text_audio_consistency"],
            )
            return self._run_asr_faster_whisper(audio_path, duration_sec=duration_sec)

        segs = self._run_asr_openai(audio_path, duration_sec=duration_sec)
        if segs:
            return segs

        # Look at the most recent ASR/OpenAI tool record to decide whether
        # the local backend can rescue this case.
        last = next(
            (r for r in reversed(self._tool_records) if r.tool_name == "ASR/OpenAI"),
            None,
        )
        if last is None:
            return []

        # SKIPPED with "Empty transcript" means OpenAI heard the audio and
        # returned nothing — there is no speech, so don't burn local CPU
        # to confirm. SKIPPED with "API_KEY not set" means we never tried
        # OpenAI — we should still try the local backend.
        if last.status == ToolStatus.SKIPPED:
            if "API_KEY not set" in (last.detail or ""):
                logger.info("[Preprocessor] OpenAI key missing — trying faster-whisper locally")
                return self._run_asr_faster_whisper(audio_path, duration_sec=duration_sec)
            return []

        if last.status == ToolStatus.FAILED:
            # If the failure looks like a network/connectivity issue (not
            # auth / bad-request / model-permission), open the breaker so
            # the rest of the batch can skip OpenAI altogether.
            detail_l = (last.detail or "").lower()
            networky = any(
                tok in detail_l for tok in (
                    "connection error", "connect error", "timeout",
                    "network is unreachable", "name or service not known",
                    "temporary failure", "ssl",
                )
            )
            if networky:
                _trip_openai_asr_breaker(last.detail or "connection failure")

            logger.info(
                f"[Preprocessor] OpenAI ASR failed ({last.detail[:80]}) — "
                f"falling back to faster-whisper"
            )
            return self._run_asr_faster_whisper(audio_path, duration_sec=duration_sec)

        return []

    # ------------------------------------------------------------------
    # ASR backend: OpenAI Audio Transcriptions API
    # ------------------------------------------------------------------

    def _run_asr_openai(self, audio_path: str, duration_sec: float | None = None) -> list[ASRSegment]:
        """
        Run ASR transcription via OpenAI official Audio Transcriptions API.

        Prefers:
          - gpt-4o-mini-transcribe (fast/cheap)
          - gpt-4o-transcribe (higher quality)
          - gpt-4o-transcribe-diarize (speaker-aware)

        Env vars:
          OPENAI_API_KEY (required for OpenAI API)
          OPENAI_BASE_URL (optional)
          OPENAI_TRANSCRIBE_MODEL (optional, default: gpt-4o-mini-transcribe)
        """
        openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
        model = (os.environ.get("OPENAI_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe").strip()

        if not openai_key:
            logger.warning("[Preprocessor] OPENAI_API_KEY not set, skipping ASR")
            self._record_tool("ASR/OpenAI", ToolStatus.SKIPPED,
                              "OPENAI_API_KEY not set — no speech transcription",
                              affects=["narration_reasonableness", "text_audio_consistency"])
            return []

        try:
            from openai import OpenAI
            try:
                from openai import (
                    APIConnectionError,
                    APITimeoutError,
                    RateLimitError,
                    InternalServerError,
                    AuthenticationError,
                    BadRequestError,
                    PermissionDeniedError,
                )
            except ImportError:
                # Older openai versions — fall back to generic Exception.
                APIConnectionError = APITimeoutError = RateLimitError = Exception  # type: ignore
                InternalServerError = Exception  # type: ignore
                AuthenticationError = BadRequestError = PermissionDeniedError = Exception  # type: ignore

            # Disable the openai-python client's own retry loop — we have our
            # own outer _retry_call below, and stacking both retry layers
            # multiplies the worst-case wait time (~85 s before fallback to
            # faster-whisper). Also clamp the per-request timeout so a single
            # hung connection can't stall the pipeline.
            request_timeout = float(os.environ.get("OPENAI_ASR_TIMEOUT_SEC", "20"))
            _opts = dict(api_key=openai_key, max_retries=0, timeout=request_timeout)
            if base_url:
                _opts["base_url"] = base_url
            # Inject an explicit httpx.Client when a proxy is configured.
            # See directorbench/_openai_proxy.py — the openai SDK's default
            # httpx client does NOT always honour HTTPS_PROXY (depends on
            # SDK version + sandbox), so we always go through the helper.
            from ._openai_proxy import build_openai_http_client_kwargs
            _opts.update(build_openai_http_client_kwargs(
                mode="sync", timeout_sec=request_timeout,
            ))
            client = OpenAI(**_opts)

            def _do_transcribe():
                with open(audio_path, "rb") as audio_file:
                    return client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format="json",
                    )

            result = _retry_call(
                _do_transcribe,
                label=f"ASR/OpenAI[{model}]",
                # Default lowered from 4 → 2: with `auto` mode we have a
                # local faster-whisper fallback, so burning lots of attempts
                # on a known-broken endpoint just delays the rescue.
                max_attempts=int(os.environ.get("ASR_MAX_ATTEMPTS", "2")),
                backoff_initial_sec=float(os.environ.get("ASR_BACKOFF_SEC", "2.0")),
                retryable=(
                    APIConnectionError, APITimeoutError, RateLimitError,
                    InternalServerError, ConnectionError, TimeoutError, OSError,
                ),
                fatal=(AuthenticationError, BadRequestError, PermissionDeniedError),
            )

            text = (getattr(result, "text", None) or "").strip()
            if not text:
                # Genuine silent video — record as SKIPPED, not a hard failure.
                logger.info("[Preprocessor] OpenAI ASR returned empty transcript (likely no speech)")
                self._record_tool(
                    "ASR/OpenAI", ToolStatus.SKIPPED,
                    "Empty transcript — no detectable speech in audio",
                    affects=["narration_reasonableness", "text_audio_consistency"],
                )
                return []

            seg = ASRSegment(
                start_sec=0.0,
                end_sec=float(duration_sec or 0.0),
                text=text,
                speaker=None,
                confidence=1.0,
            )
            logger.info("[Preprocessor] OpenAI Transcriptions API produced 1 segment")
            self._record_tool("ASR/OpenAI", ToolStatus.SUCCESS,
                              f"Transcribed {len(text)} chars via {model}",
                              affects=["narration_reasonableness", "text_audio_consistency"])
            return [seg]

        except Exception as e:
            logger.warning(f"[Preprocessor] OpenAI Transcriptions API failed: {e}")
            self._record_tool("ASR/OpenAI", ToolStatus.FAILED, str(e),
                              affects=["narration_reasonableness", "text_audio_consistency"])
            return []

    # ------------------------------------------------------------------
    # ASR backend: local faster-whisper (CPU-friendly, no network)
    #
    # Env vars:
    #   FASTER_WHISPER_MODEL         model name (tiny / base / small /
    #                                medium / large-v3). Default: base.
    #                                For Chinese / multilingual content,
    #                                small or medium gives noticeably
    #                                better quality than base.
    #   FASTER_WHISPER_DEVICE        "cpu" | "cuda" | "auto" (default: cpu)
    #   FASTER_WHISPER_COMPUTE_TYPE  "int8" / "int8_float16" / "float16" /
    #                                "float32". Default: int8 on CPU,
    #                                float16 on CUDA.
    #   FASTER_WHISPER_LANGUAGE      ISO code (e.g. "zh", "en") to skip
    #                                language detection. Default: auto.
    #   FASTER_WHISPER_BEAM_SIZE     Default 1 (greedy, fastest). Bump to
    #                                5 for slightly higher accuracy at ~3x
    #                                latency.
    #   FASTER_WHISPER_DOWNLOAD_ROOT Override HuggingFace cache directory.
    # ------------------------------------------------------------------

    def _run_asr_faster_whisper(
        self, audio_path: str, duration_sec: float | None = None
    ) -> list[ASRSegment]:
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError as e:
            logger.warning(f"[Preprocessor] faster-whisper not installed: {e}")
            self._record_tool(
                "ASR/FasterWhisper", ToolStatus.FAILED,
                "faster-whisper not installed — pip install faster-whisper",
                affects=["narration_reasonableness", "text_audio_consistency"],
            )
            return []

        model_name = (os.environ.get("FASTER_WHISPER_MODEL") or "base").strip()
        device = (os.environ.get("FASTER_WHISPER_DEVICE") or "cpu").strip().lower()
        if device == "auto":
            try:
                import torch  # noqa: F401
                device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        compute_type = (
            os.environ.get("FASTER_WHISPER_COMPUTE_TYPE")
            or ("int8" if device == "cpu" else "float16")
        ).strip()
        language = (os.environ.get("FASTER_WHISPER_LANGUAGE") or "").strip() or None
        beam_size = int(os.environ.get("FASTER_WHISPER_BEAM_SIZE", "1"))
        download_root = (os.environ.get("FASTER_WHISPER_DOWNLOAD_ROOT") or "").strip() or None

        try:
            model = self._load_faster_whisper_model(model_name, device, compute_type, download_root)
        except Exception as e:
            logger.warning(f"[Preprocessor] faster-whisper model load failed: {e}")
            self._record_tool(
                "ASR/FasterWhisper", ToolStatus.FAILED,
                f"Model load failed: {e}",
                affects=["narration_reasonableness", "text_audio_consistency"],
            )
            return []

        try:
            t0 = time.perf_counter()
            segments_iter, info = model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
            )

            segs: list[ASRSegment] = []
            for s in segments_iter:
                text = (s.text or "").strip()
                if not text:
                    continue
                segs.append(ASRSegment(
                    start_sec=float(s.start),
                    end_sec=float(s.end),
                    text=text,
                    speaker=None,
                    confidence=float(getattr(s, "avg_logprob", 0.0) or 0.0),
                ))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if not segs:
                logger.info("[Preprocessor] faster-whisper produced no segments (no speech)")
                self._record_tool(
                    "ASR/FasterWhisper", ToolStatus.SKIPPED,
                    "Empty transcript — no detectable speech in audio",
                    elapsed_ms=elapsed_ms,
                    affects=["narration_reasonableness", "text_audio_consistency"],
                )
                return []

            total_chars = sum(len(s.text) for s in segs)
            detected_lang = getattr(info, "language", None) or "?"
            lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
            lang_str = (
                f"{detected_lang}({lang_prob:.2f})" if language is None else detected_lang
            )
            logger.info(
                f"[Preprocessor] faster-whisper {model_name} ({device}/{compute_type}): "
                f"{len(segs)} segments, {total_chars} chars, lang={lang_str}, "
                f"{elapsed_ms / 1000:.1f}s"
            )
            self._record_tool(
                "ASR/FasterWhisper", ToolStatus.SUCCESS,
                f"Transcribed {total_chars} chars / {len(segs)} segments via "
                f"{model_name} ({device}/{compute_type}, lang={lang_str})",
                elapsed_ms=elapsed_ms,
                affects=["narration_reasonableness", "text_audio_consistency"],
            )
            return segs

        except Exception as e:
            logger.warning(f"[Preprocessor] faster-whisper transcription failed: {e}")
            self._record_tool(
                "ASR/FasterWhisper", ToolStatus.FAILED, str(e),
                affects=["narration_reasonableness", "text_audio_consistency"],
            )
            return []

    @staticmethod
    def _load_faster_whisper_model(
        model_name: str, device: str, compute_type: str, download_root: str | None,
    ):
        """Module-level cache so repeated cases in a batch share weights.

        Loading ``base`` on CPU takes ~2-4 s the first time and uses ~150 MB
        of RAM; ``small`` is ~5-10 s and ~500 MB. Without this cache we'd
        pay that cost on every single case.
        """
        cache = _Preprocessor_FW_CACHE
        key = (model_name, device, compute_type, download_root or "")
        model = cache.get(key)
        if model is not None:
            return model
        from faster_whisper import WhisperModel
        logger.info(
            f"[Preprocessor] Loading faster-whisper model={model_name} "
            f"device={device} compute_type={compute_type} ..."
        )
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        cache[key] = model
        return model
