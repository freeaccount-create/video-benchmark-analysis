"""
crossmodal_agent.py — Cross-Modal Alignment Agent

Phase 2 agent: depends on all Phase 1 agents (Script, Video, Audio, Stability).
Consumes intermediate representations from upstream agents.

Sub-metrics:
  1. Text ↔ Video Consistency
  2. Video ↔ Audio Consistency
  3. Text ↔ Audio Consistency
  4. Overall Multimodal Harmony
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from ..schemas import AgentID, ContentProfile, EvalResult, Evidence, GraphState, Severity, ToolStatus
from ..checkpoints import CHECKPOINTS
from .base import BaseEvalAgent, CONFIDENCE_TOOL_GUIDANCE

logger = logging.getLogger(__name__)


class CrossModalEvalAgent(BaseEvalAgent):
    agent_id = AgentID.CROSSMODAL_EVAL

    # ------------------------------------------------------------------
    # Sub-metric 1: Text ↔ Video Consistency
    # ------------------------------------------------------------------

    def _eval_text_video(self, state: GraphState) -> EvalResult:
        """
        Ensure video visuals match the script/storyboard.

        Approach:
          - Use ViCLIP similarity between video frames and script text
          - Plus VLM questions for narrative fit per shot

        Consumes: script_results.intermediate_repr (event chain),
                  video_results.intermediate_repr (shot_embeddings, shot_scores)
        """
        # In reference-free mode `state.script_text` is "" and
        # ScriptEvalAgent populates `state.extracted_script_text` with
        # the narrative reconstructed from VLM + ASR.
        script = state.script_text or state.extracted_script_text or ""
        prep = state.preprocessing
        shots = prep.shots if prep else []
        thumbnails = [s.thumbnail_path for s in shots if s.thumbnail_path]

        if not script or not thumbnails:
            return EvalResult(
                agent_id=self.agent_id,
                metric_name="text_video_consistency",
                score=0.5, confidence=0.3,
                metadata={"note": "Insufficient data for text-video alignment"},
            )

        # --- ViCLIP-based similarity (None when model unavailable) ---
        viclip_score = self._compute_viclip_similarity(script, thumbnails)

        # --- Checkpoint-based VLM evaluation ---
        profile = self._build_content_profile(state)

        extra_ctx = (
            f"Script:\n{script}\n\n"
            f"Content: characters={profile.has_characters}, "
            f"dialogue={profile.has_dialogue}\n\n"
            f"The following frames are from the generated video (in temporal order). "
            f"Evaluate text-video consistency:"
        )

        cp_score, confidence, cp_results, active_cps, _ = self._checkpoint_evaluate(
            metric_name="text_video_consistency",
            state=state,
            image_paths=thumbnails[:12],
            extra_context=extra_ctx,
            profile=profile,
        )

        # Combine checkpoint score with ViCLIP score (if available)
        if viclip_score is not None:
            combined_score = 0.6 * cp_score + 0.4 * viclip_score
        else:
            combined_score = cp_score

        evidence = []
        for r in cp_results:
            if r.normalised < 0.5:
                evidence.append(Evidence(
                    type="text_video_mismatch",
                    issue=f"[{r.checkpoint_id}] {r.reasoning}" if r.reasoning else r.checkpoint_id,
                    severity=Severity.HIGH if r.normalised < 0.25 else Severity.MEDIUM,
                ))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="text_video_consistency",
            score=combined_score,
            confidence=confidence,
            granularity="shot-level",
            evidence=evidence,
            checkpoint_results=cp_results,
            intermediate_repr={
                "viclip_score": viclip_score,
                "checkpoint_score": cp_score,
                "active_checkpoints": [c.id for c in active_cps],
            },
        )

    # ------------------------------------------------------------------
    # Sub-metric 2: Video ↔ Audio Consistency
    # ------------------------------------------------------------------

    def _eval_video_audio(self, state: GraphState) -> EvalResult:
        """
        Check audio sync with video:
          - Lip-sync detection (optical-flow + audio-energy correlation)
          - Event-audio alignment (explosion → boom, door → click)
          - BGM mood ↔ visual mood matching

        Consumes: video_results (shot info), audio_results (ASR, BGM features)
        """
        prep = state.preprocessing
        asr_segments = prep.asr_segments if prep else []
        shots = prep.shots if prep else []

        # --- Lip-sync score (None when unavailable) ---
        lipsync_score = self._compute_lipsync_score(state)

        # --- Checkpoint-based evaluation ---
        profile = self._build_content_profile(state)

        asr_timeline = "\n".join(
            f"[{seg.start_sec:.1f}s-{seg.end_sec:.1f}s] {seg.text}" for seg in asr_segments
        )
        shot_timeline = "\n".join(
            f"Shot {s.index}: [{s.start_sec:.1f}s-{s.end_sec:.1f}s]" for s in shots
        )

        extra_ctx = (
            f"Shot timeline:\n{shot_timeline}\n\n"
            f"ASR transcript:\n{asr_timeline}\n\n"
            f"Content: characters={profile.has_characters}, "
            f"dialogue={profile.has_dialogue}\n\n"
            f"Evaluate video-audio synchronization:"
        )

        cp_score, confidence, cp_results, active_cps, _ = self._checkpoint_evaluate(
            metric_name="video_audio_consistency",
            state=state,
            extra_context=extra_ctx,
            profile=profile,
        )

        # Combine with lip-sync tool score (if available)
        if lipsync_score is not None:
            combined = 0.5 * cp_score + 0.5 * lipsync_score
        else:
            combined = cp_score

        evidence = []
        for r in cp_results:
            if r.normalised < 0.5:
                evidence.append(Evidence(
                    type="video_audio_sync",
                    issue=f"[{r.checkpoint_id}] {r.reasoning}" if r.reasoning else r.checkpoint_id,
                    severity=Severity.HIGH if r.normalised < 0.25 else Severity.MEDIUM,
                ))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="video_audio_consistency",
            score=combined,
            confidence=confidence,
            granularity="shot-level",
            evidence=evidence,
            checkpoint_results=cp_results,
            intermediate_repr={
                "lipsync_score": lipsync_score,
                "checkpoint_score": cp_score,
                "active_checkpoints": [c.id for c in active_cps],
            },
        )

    # ------------------------------------------------------------------
    # Sub-metric 3: Text ↔ Audio Consistency
    # ------------------------------------------------------------------

    def _eval_text_audio(self, state: GraphState) -> EvalResult:
        """
        Verify that audio narration/dialogue aligns with the script.
        ASR transcript → LLM text matching.
        """
        # Use extracted narrative as fallback in reference-free mode.
        script = state.script_text or state.extracted_script_text or ""
        prep = state.preprocessing
        asr_segments = prep.asr_segments if prep else []

        if not script or not asr_segments:
            # Make the skip explicit in the log so trace readers don't have
            # to wonder whether text_audio_consistency was silently dropped.
            if not asr_segments and not script:
                reason = "no ASR transcript and no script"
            elif not asr_segments:
                reason = "no ASR transcript (silent video or ASR skipped)"
            else:
                reason = "no script text or extracted narrative"
            logger.info(
                f"[CrossModal] Skipping text_audio_consistency LLM check: "
                f"{reason} — returning fallback score=0.5 confidence=0.3"
            )
            return EvalResult(
                agent_id=self.agent_id,
                metric_name="text_audio_consistency",
                score=0.5, confidence=0.3,
            )

        asr_text = " ".join(seg.text for seg in asr_segments)

        # --- Semantic similarity (None when Sentence-BERT unavailable) ---
        semantic_sim = self._compute_text_similarity(script, asr_text)

        # --- LLM-based reasoning ---
        system_prompt_raw = """You are evaluating text-audio consistency.
Compare the original script with the actual ASR transcript from the generated video.

Check:
1. Content alignment: Does the spoken audio follow the script?
2. Completeness: Are there scripted lines that were not spoken?
3. Ad-libbing: Are there spoken lines not in the script?
4. Order: Is the dialogue order consistent with the script?

Return JSON:
{
  "score": <float 0-1>,
  "reasoning": "<str>",
  "missing_from_audio": ["<scripted line not spoken>"],
  "extra_in_audio": ["<spoken line not scripted>"],
  "confidence": <float 0-1>
}"""
        system_prompt = self.maybe_add_confidence_guidance(system_prompt_raw, state)
        tool_ctx = self.format_tool_context(
            state, extra_records=self._local_tool_records,
            relevant_metrics=["text_audio_consistency"],
        )

        result = self.llm.evaluate(
            system_prompt,
            f"Original script:\n{script}\n\nASR transcript:\n{asr_text}{tool_ctx}",
        )

        llm_score = float(result.get("score", 0.5))
        if semantic_sim is not None:
            combined = 0.5 * llm_score + 0.5 * semantic_sim
        else:
            combined = llm_score  # pure LLM when Sentence-BERT unavailable

        evidence = []
        for line in result.get("missing_from_audio", []):
            evidence.append(Evidence(
                type="script_not_spoken", issue=f"Scripted line missing in audio: {line}",
                severity=Severity.MEDIUM,
            ))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="text_audio_consistency",
            score=combined,
            confidence=float(result.get("confidence", 0.6)),
            granularity="video-level",
            evidence=evidence,
            intermediate_repr={"semantic_similarity": semantic_sim},
        )

    # ------------------------------------------------------------------
    # Sub-metric 4: Overall Multimodal Harmony
    # ------------------------------------------------------------------

    def _eval_overall_harmony(self, state: GraphState) -> EvalResult:
        """
        Global check: no modality conflicts.
        Aggregates the three cross-modal sub-metrics + holistic LLM rating.
        """
        # Collect all Phase 1 and Phase 2 results
        all_results = (
            state.script_results + state.video_results +
            state.audio_results + state.stability_results
        )

        # Build a summary for the LLM
        summary_lines = []
        for r in all_results:
            summary_lines.append(f"[{r.agent_id.value}] {r.metric_name}: {r.score:.2f} (conf={r.confidence:.2f})")
            for e in r.evidence[:2]:  # top 2 evidence items
                summary_lines.append(f"  - {e.issue} ({e.severity.value})")

        system_prompt_raw = """You are performing a holistic multimodal harmony evaluation.

Given evaluation results from individual agents (script, video, audio, stability),
provide an overall assessment of how well all modalities work together.

Focus on:
1. Cross-modal conflicts (e.g., upbeat BGM in a sad script scene)
2. Modality drift: Do different modalities tell the same story?
3. Weakest link: Which modality is dragging down the overall quality?
4. Synergy: Do modalities enhance each other?

Return JSON:
{
  "score": <float 0-1>,
  "reasoning": "<holistic analysis>",
  "conflicts": [{"modalities": "<str>", "issue": "<str>", "severity": "low|medium|high"}],
  "weakest_dimension": "<str>",
  "strongest_dimension": "<str>",
  "confidence": <float 0-1>
}"""
        system_prompt = self.maybe_add_confidence_guidance(system_prompt_raw, state)
        tool_ctx = self.format_tool_context(state, extra_records=self._local_tool_records)

        result = self.llm.evaluate(
            system_prompt,
            f"Individual agent results:\n" + "\n".join(summary_lines) + tool_ctx,
        )

        evidence = []
        for conflict in result.get("conflicts", []):
            evidence.append(Evidence(
                type="multimodal_conflict",
                issue=f"{conflict.get('modalities', '')}: {conflict.get('issue', '')}",
                severity=Severity(conflict.get("severity", "medium")),
            ))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="overall_multimodal_harmony",
            score=float(result.get("score", 0.6)),
            confidence=float(result.get("confidence", 0.7)),
            granularity="video-level",
            evidence=evidence,
            metadata={
                "weakest_dimension": result.get("weakest_dimension", ""),
                "strongest_dimension": result.get("strongest_dimension", ""),
            },
        )

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    # Class-level model cache to avoid reloading on every call
    _mobileviclip_model = None

    def _compute_viclip_similarity(self, text: str, image_paths: list[str]) -> float | None:
        """
        Compute text-video similarity using MobileViCLIP-Small.
        Returns a float in [0, 1] or None when the model is unavailable.
        """
        try:
            import torch
            import numpy as np
            from PIL import Image
            import contextlib

            # Lazy-load model (cached at class level across calls).
            if CrossModalEvalAgent._mobileviclip_model is None:
                workspace_root = Path(__file__).resolve().parents[3]
                default_repo = workspace_root / "models" / "MobileViCLIP"
                mobileviclip_repo = Path(
                    os.environ.get("MOBILEVICLIP_REPO", str(default_repo))
                ).resolve()
                small_ckpt_default = workspace_root / "models" / "mobileviclip_small.pt"
                small_ckpt = Path(
                    os.environ.get("MOBILEVICLIP_SMALL_CKPT", str(small_ckpt_default))
                ).resolve()
                vision_ckpt = Path(
                    os.environ.get("MOBILEVICLIP_VISION_CKPT", str(small_ckpt))
                ).resolve()
                text_ckpt = Path(
                    os.environ.get("MOBILEVICLIP_TEXT_CKPT", str(small_ckpt))
                ).resolve()
                extra_ckpt = Path(
                    os.environ.get("MOBILEVICLIP_EXTRA_CKPT", str(small_ckpt))
                ).resolve()

                if not mobileviclip_repo.exists():
                    raise FileNotFoundError(f"MobileViCLIP repo not found: {mobileviclip_repo}")
                if not small_ckpt.exists():
                    raise FileNotFoundError(f"MobileViCLIP small checkpoint not found: {small_ckpt}")

                import sys
                if str(mobileviclip_repo) not in sys.path:
                    sys.path.insert(0, str(mobileviclip_repo))

                # MobileViCLIP model code uses relative file loads from CWD.
                # Temporarily chdir to the repo root while constructing model.
                old_cwd = Path.cwd()
                with contextlib.ExitStack() as stack:
                    os.chdir(str(mobileviclip_repo))
                    from models.mobileviclip_small import MobileViCLIP_Small  # type: ignore

                    # MobileViCLIP_Small calls torch.load() three times during
                    # __init__ (vision_ckpt + text_ckpt + extra_ckpt). When the
                    # three paths point at the same file on a slow shared
                    # filesystem (the common case — they default to the same
                    # mobileviclip_small.pt), this re-deserializes ~500 MB of
                    # weights twice for nothing. Wrap torch.load with a
                    # path-keyed in-memory cache for the duration of model
                    # construction so each unique file is read only once.
                    _orig_torch_load = torch.load
                    _ckpt_cache: dict[str, Any] = {}

                    def _cached_torch_load(f, *args, **kwargs):
                        if isinstance(f, (str, Path)):
                            key = str(Path(f).resolve())
                            cached = _ckpt_cache.get(key)
                            if cached is not None:
                                logger.info(
                                    f"[CrossModal] torch.load cache HIT for "
                                    f"{Path(key).name}"
                                )
                                return cached
                            obj = _orig_torch_load(f, *args, **kwargs)
                            _ckpt_cache[key] = obj
                            return obj
                        return _orig_torch_load(f, *args, **kwargs)

                    torch.load = _cached_torch_load
                    stack.callback(setattr, torch, "load", _orig_torch_load)
                    # Free the dedup cache once we leave the with-block so
                    # we don't permanently hold ~500 MB of duplicated weights.
                    stack.callback(_ckpt_cache.clear)

                    class _CfgObj(dict):
                        def __getattr__(self, k):
                            v = self[k]
                            if isinstance(v, dict):
                                return _CfgObj(v)
                            return v

                    cfg = _CfgObj(
                        model={
                            "vision_encoder": {
                                "name": "mobileclip_s2",
                                "img_size": 256,
                                "head_drop_path_rate": 0.0,
                                "attn_pool_num_heads": 16,
                                "clip_embed_dim": 512,
                                "align_dim": 512,
                            },
                            "text_encoder": {"name": "mobileclip_s2"},
                            "temp": 1 / 100.0,
                            "temp_min": 1 / 100.0,
                            "freeze_vision": False,
                            "open_vision_clip_projector": True,
                            "freeze_text": True,
                            "open_text_projection": False,
                            "vision_ckpt_path": str(vision_ckpt),
                            "text_ckpt_path": str(text_ckpt),
                            "extra_ckpt_path": str(extra_ckpt),
                            "load_vision_ckpt_from_internvideo2_stage2": False,
                        }
                    )

                    logger.info(
                        f"[CrossModal] Loading MobileViCLIP-Small "
                        f"(vision={vision_ckpt.name}, text={text_ckpt.name}, extra={extra_ckpt.name}) …"
                    )
                    model = MobileViCLIP_Small(config=cfg, is_pretrain=False).eval()
                    os.chdir(str(old_cwd))
                CrossModalEvalAgent._mobileviclip_model = model

            model = CrossModalEvalAgent._mobileviclip_model

            # The MobileViCLIP-Small architecture hard-codes T=8 in temporal
            # reshape and temporal_embedding, so num_frames must stay 8.
            num_frames = max(1, int(os.environ.get("MOBILEVICLIP_NUM_FRAMES", "8")))

            # Optional CPU bfloat16 inference (Ice Lake / Sapphire Rapids+).
            # Default on for CPU; disable via MOBILEVICLIP_PRECISION=fp32.
            precision = os.environ.get(
                "MOBILEVICLIP_PRECISION",
                "bf16" if not torch.cuda.is_available() else "fp32",
            ).lower()
            autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

            # Encode text (tokenizer is built into MobileViCLIP model).
            text_tokens = model.tokenizer([text])
            with torch.inference_mode():
                if autocast_dtype is not None:
                    with torch.autocast(device_type="cpu", dtype=autocast_dtype):
                        text_emb = model.encode_text(text_tokens)
                else:
                    text_emb = model.encode_text(text_tokens)
                text_emb = text_emb.float()
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            # Pre-process all frames once and batch them into a single forward.
            tensors: list[Any] = []
            valid_paths: list[str] = []
            for path in image_paths:
                try:
                    img = Image.open(path).convert("RGB")
                except Exception as e:
                    logger.warning(f"[CrossModal] failed to open thumbnail {path}: {e}")
                    continue
                arr = np.array(img)
                t = torch.from_numpy(arr).permute(2, 0, 1).float()  # C,H,W in [0,255]
                t = model.transform(t)  # C,H,W
                t = t.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # T,C,H,W
                tensors.append(t)
                valid_paths.append(path)

            if not tensors:
                self._record_tool("MobileViCLIP-Small", ToolStatus.FALLBACK,
                                  "No valid frames could be encoded",
                                  affects=["text_video_consistency"])
                return None

            # Stack to (B, T, C, H, W) and run batched forward passes.
            batch_size = max(1, int(os.environ.get("MOBILEVICLIP_BATCH_SIZE", "4")))
            similarities: list[float] = []
            with torch.inference_mode():
                for start in range(0, len(tensors), batch_size):
                    chunk = torch.stack(tensors[start:start + batch_size], dim=0)
                    if autocast_dtype is not None:
                        with torch.autocast(device_type="cpu", dtype=autocast_dtype):
                            img_emb = model.encode_vision(chunk, test=True)
                    else:
                        img_emb = model.encode_vision(chunk, test=True)
                    img_emb = img_emb.float()
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                    sims = (img_emb @ text_emb.T).squeeze(-1)  # (B,)
                    similarities.extend(sims.flatten().tolist())

            avg_sim = float(np.mean(similarities))
            score = max(0.0, min(1.0, (avg_sim + 1.0) / 2.0))

            self._record_tool("MobileViCLIP-Small", ToolStatus.SUCCESS,
                              f"Computed text-frame similarity for {len(similarities)} frames (avg={avg_sim:.3f})",
                              affects=["text_video_consistency"])
            return score

        except ImportError as e:
            logger.warning(f"[CrossModal] MobileViCLIP unavailable (missing dependency): {e}")
            self._record_tool("MobileViCLIP-Small", ToolStatus.FALLBACK,
                              f"Not installed ({e}) — text-video similarity unavailable, relying on VLM only",
                              affects=["text_video_consistency"])
            return None
        except Exception as e:
            logger.warning(f"[CrossModal] MobileViCLIP computation failed: {e}")
            self._record_tool("MobileViCLIP-Small", ToolStatus.FAILED,
                              str(e),
                              affects=["text_video_consistency"])
            return None

    def _compute_lipsync_score(self, state: GraphState) -> float | None:
        """
        Lightweight audio-visual sync score (no heavy model required).

        Approach — "mouth-motion / audio-energy correlation":
          1. Decode the first speech segment (≤5 s) from the video.
          2. Compute per-frame optical-flow magnitude in the lower-third of
             each frame (proxy for mouth / jaw movement).
          3. Compute per-frame RMS audio energy at matching frame timestamps.
          4. Return the Pearson correlation between the two signals.
             High correlation ⟹ lips move when audio is loud ⟹ good sync.

        Runs purely on OpenCV + numpy (+ optionally librosa for audio
        loading).  Typical latency: 200-800 ms on CPU for a 5 s clip,
        vs 30-60 s for full SyncNet.
        """
        try:
            import cv2
            import numpy as np

            prep = state.preprocessing
            video_path = prep.video_path if prep else None
            asr_segments = prep.asr_segments if prep else []

            if not video_path:
                self._record_tool("LipSyncProxy", ToolStatus.SKIPPED,
                                  "No video path available",
                                  affects=["video_audio_consistency"])
                return None

            if not asr_segments:
                self._record_tool("LipSyncProxy", ToolStatus.SKIPPED,
                                  "No speech segments — lip-sync evaluation not applicable",
                                  affects=["video_audio_consistency"])
                return None

            t0 = time.perf_counter()

            # ---- Step 1: extract per-frame mouth-region optical flow ----
            seg = asr_segments[0]
            start_sec = float(seg.start_sec)
            end_sec = float(min(seg.end_sec, seg.start_sec + 5.0))

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self._record_tool("LipSyncProxy", ToolStatus.FAILED,
                                  "Cannot open video file",
                                  affects=["video_audio_consistency"])
                return None

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))

            prev_gray = None
            flow_magnitudes: list[float] = []
            max_frames = int((end_sec - start_sec) * fps)

            for _ in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                # Lower third of the frame — rough mouth region proxy
                roi = frame[int(h * 0.66):, :]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=2, winsize=15,
                        iterations=2, poly_n=5, poly_sigma=1.1,
                        flags=0,
                    )
                    mag = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                    flow_magnitudes.append(mag)

                prev_gray = gray

            cap.release()

            if len(flow_magnitudes) < 4:
                self._record_tool("LipSyncProxy", ToolStatus.FALLBACK,
                                  f"Too few frames decoded ({len(flow_magnitudes)})",
                                  affects=["video_audio_consistency"])
                return None

            # ---- Step 2: extract per-frame audio energy ----
            audio_energies: list[float] = []
            audio_path = None
            if prep and hasattr(prep, "audio_tracks"):
                for trk in (prep.audio_tracks or []):
                    if trk.track_type in ("vocal", "dialogue", "original"):
                        audio_path = trk.path
                        break

            if audio_path:
                try:
                    import librosa
                    y, sr = librosa.load(audio_path, sr=16000,
                                         offset=start_sec,
                                         duration=end_sec - start_sec)
                    hop = int(sr / fps)
                    for i in range(len(flow_magnitudes)):
                        start_sample = i * hop
                        end_sample = start_sample + hop
                        chunk = y[start_sample:end_sample] if end_sample <= len(y) else y[start_sample:]
                        if len(chunk) > 0:
                            audio_energies.append(float(np.sqrt(np.mean(chunk ** 2))))
                        else:
                            audio_energies.append(0.0)
                except Exception:
                    pass  # fall through to ffmpeg fallback

            if not audio_energies:
                # Fallback: decode audio via ffmpeg subprocess
                try:
                    import subprocess
                    import struct

                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(start_sec),
                        "-t", str(end_sec - start_sec),
                        "-i", video_path,
                        "-vn", "-acodec", "pcm_s16le",
                        "-ar", "16000", "-ac", "1",
                        "-f", "s16le", "pipe:1",
                    ]
                    proc = subprocess.run(cmd, capture_output=True, timeout=10)
                    if proc.returncode == 0 and len(proc.stdout) > 0:
                        raw = proc.stdout
                        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        hop = int(16000 / fps)
                        for i in range(len(flow_magnitudes)):
                            s = i * hop
                            e = s + hop
                            chunk = samples[s:e] if e <= len(samples) else samples[s:]
                            if len(chunk) > 0:
                                audio_energies.append(float(np.sqrt(np.mean(chunk ** 2))))
                            else:
                                audio_energies.append(0.0)
                except Exception:
                    pass

            if len(audio_energies) < len(flow_magnitudes):
                self._record_tool("LipSyncProxy", ToolStatus.FALLBACK,
                                  "Could not extract audio energy — returning motion-only heuristic",
                                  affects=["video_audio_consistency"])
                # Fallback: just use motion variance as a crude heuristic
                # High variance → likely animated speech; low → static
                var = float(np.std(flow_magnitudes))
                score = max(0.0, min(1.0, var / 2.0))
                return score

            # ---- Step 3: Pearson correlation ----
            n = min(len(flow_magnitudes), len(audio_energies))
            motion = np.array(flow_magnitudes[:n])
            energy = np.array(audio_energies[:n])

            # Normalise to zero-mean, unit-variance
            m_std = float(np.std(motion))
            e_std = float(np.std(energy))

            if m_std < 1e-8 or e_std < 1e-8:
                # Flat signal — cannot compute meaningful correlation
                self._record_tool("LipSyncProxy", ToolStatus.FALLBACK,
                                  "Flat motion or energy signal — correlation undefined",
                                  affects=["video_audio_consistency"])
                return 0.5  # neutral score

            corr = float(np.corrcoef(motion, energy)[0, 1])

            # Map Pearson r ∈ [-1, 1] → score ∈ [0, 1]
            #   r=1.0 → perfect sync → 1.0
            #   r=0.0 → uncorrelated → 0.5
            #   r=-1.0 → anti-correlated → 0.0
            score = max(0.0, min(1.0, (corr + 1.0) / 2.0))

            elapsed = (time.perf_counter() - t0) * 1000.0
            self._record_tool("LipSyncProxy", ToolStatus.SUCCESS,
                              f"motion-energy correlation r={corr:.3f}, score={score:.3f} "
                              f"({n} frames, {elapsed:.0f}ms)",
                              elapsed_ms=elapsed,
                              affects=["video_audio_consistency"])
            return score

        except ImportError as e:
            logger.warning(f"[CrossModal] LipSyncProxy deps missing: {e}")
            self._record_tool("LipSyncProxy", ToolStatus.FALLBACK,
                              f"Missing dependency ({e})",
                              affects=["video_audio_consistency"])
            return None
        except Exception as e:
            logger.warning(f"[CrossModal] LipSyncProxy failed: {e}")
            self._record_tool("LipSyncProxy", ToolStatus.FAILED,
                              str(e),
                              affects=["video_audio_consistency"])
            return None

    # Class-level Sentence-BERT cache
    _sbert_model = None

    def _compute_text_similarity(self, text_a: str, text_b: str) -> float | None:
        """
        Compute semantic similarity using Sentence-BERT (all-MiniLM-L6-v2).

        Attempts ONNX-accelerated inference via ``optimum`` first for ~2-3x
        CPU speed-up.  Falls back to standard sentence-transformers if
        optimum is not installed.

        Returns a float in [0, 1] or None when the library is unavailable.
        """
        try:
            workspace_root = Path(__file__).resolve().parents[3]
            alt_local_sbert_dir = workspace_root / "models" / "all-MiniLM-L6-v2"
            model_ref = os.environ.get("SENTENCE_BERT_MODEL_PATH")
            if model_ref:
                model_ref = str(Path(model_ref).expanduser().resolve())
            elif alt_local_sbert_dir.exists():
                model_ref = str(alt_local_sbert_dir.resolve())
            else:
                model_ref = "sentence-transformers/all-MiniLM-L6-v2"

            # Try ONNX-optimized path first
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer
                import numpy as np

                if CrossModalEvalAgent._sbert_model is None:
                    logger.info("[CrossModal] Loading Sentence-BERT (ONNX optimized) …")
                    tokenizer = AutoTokenizer.from_pretrained(model_ref)
                    ort_model = ORTModelForFeatureExtraction.from_pretrained(
                        model_ref,
                        export=True,
                    )
                    CrossModalEvalAgent._sbert_model = ("onnx", tokenizer, ort_model)

                tag, tokenizer, ort_model = CrossModalEvalAgent._sbert_model

                def _mean_pool(text: str):
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    outputs = ort_model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
                    return emb

                emb_a = _mean_pool(text_a)
                emb_b = _mean_pool(text_b)
                score = float(np.dot(emb_a, emb_b.T).squeeze())

                self._record_tool("SentenceBERT-ONNX", ToolStatus.SUCCESS,
                                  "Computed semantic similarity (ONNX-accelerated)",
                                  affects=["text_audio_consistency"])
                return max(0.0, min(1.0, score))

            except ImportError:
                # Optimum not available; fall through to standard path
                pass

            # Standard sentence-transformers path
            from sentence_transformers import SentenceTransformer, util

            if CrossModalEvalAgent._sbert_model is None:
                logger.info("[CrossModal] Loading Sentence-BERT (standard) …")
                CrossModalEvalAgent._sbert_model = ("std", SentenceTransformer(model_ref))

            tag, model = CrossModalEvalAgent._sbert_model[0], CrossModalEvalAgent._sbert_model[1]
            emb_a = model.encode(text_a, convert_to_tensor=True)
            emb_b = model.encode(text_b, convert_to_tensor=True)
            score = float(util.cos_sim(emb_a, emb_b)[0][0])

            self._record_tool("SentenceBERT", ToolStatus.SUCCESS,
                              "Computed semantic similarity",
                              affects=["text_audio_consistency"])
            return max(0.0, min(1.0, score))

        except ImportError as e:
            logger.warning(f"[CrossModal] sentence-transformers not available: {e}")
            self._record_tool("SentenceBERT", ToolStatus.FALLBACK,
                              f"Not installed ({e}) — semantic similarity unavailable, relying on LLM only",
                              affects=["text_audio_consistency"])
            return None
        except Exception as e:
            logger.warning(f"[CrossModal] text similarity failed: {e}")
            self._record_tool("SentenceBERT", ToolStatus.FAILED,
                              str(e),
                              affects=["text_audio_consistency"])
            return None

    # ------------------------------------------------------------------
    # Main evaluate
    # ------------------------------------------------------------------

    def evaluate(self, state: GraphState) -> list[EvalResult]:
        results = [
            self._eval_text_video(state),
            self._eval_video_audio(state),
            self._eval_text_audio(state),
            self._eval_overall_harmony(state),
        ]
        return results
