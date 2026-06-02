"""
audio_agent.py — Audio Evaluation Agent (Audio-Modal)

Sub-metrics:
  1. Narration/Dialogue Reasonableness — natural timing, fits narrative
  2. BGM Consistency — background music matches mood/pace
"""

from __future__ import annotations

import logging

from ..schemas import (
    AgentID, ContentProfile, EvalResult, Evidence, GraphState, Severity, ToolStatus,
)
from ..checkpoints import CHECKPOINTS
from .base import BaseEvalAgent, CONFIDENCE_TOOL_GUIDANCE

logger = logging.getLogger(__name__)


class AudioEvalAgent(BaseEvalAgent):
    agent_id = AgentID.AUDIO_EVAL

    # ------------------------------------------------------------------
    # Sub-metric 1: Narration/Dialogue Reasonableness
    # ------------------------------------------------------------------

    def _eval_narration(self, state: GraphState) -> EvalResult:
        """Checkpoint-based narration/dialogue evaluation."""
        prep = state.preprocessing
        asr_segments = prep.asr_segments if prep else []
        shots = prep.shots if prep else []
        script = state.script_text or ""

        if not asr_segments:
            logger.info("[AudioAgent] No ASR data — skipping narration evaluation")
            return None

        profile = self._build_content_profile(state)

        asr_timeline = "\n".join(
            f"[{seg.start_sec:.1f}s - {seg.end_sec:.1f}s] "
            f"{'(' + seg.speaker + ') ' if seg.speaker else ''}{seg.text}"
            for seg in asr_segments
        )
        shot_timeline = "\n".join(
            f"Shot {s.index}: [{s.start_sec:.1f}s - {s.end_sec:.1f}s]" for s in shots
        )

        extra_ctx = f"ASR Transcript:\n{asr_timeline}\n\nShot Timeline:\n{shot_timeline}"
        if script:
            extra_ctx += f"\n\nOriginal Script:\n{script}"
        extra_ctx += "\n\nEvaluate narration/dialogue quality:"

        score, confidence, cp_results, active_cps, _ = self._checkpoint_evaluate(
            metric_name="narration_reasonableness",
            state=state,
            extra_context=extra_ctx,
            profile=profile,
        )

        evidence = []
        for r in cp_results:
            if r.normalised < 0.5:
                evidence.append(Evidence(
                    type="narration_issue",
                    issue=f"[{r.checkpoint_id}] {r.reasoning}" if r.reasoning else r.checkpoint_id,
                    severity=Severity.HIGH if r.normalised < 0.25 else Severity.MEDIUM,
                ))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="narration_reasonableness",
            score=score,
            confidence=confidence,
            granularity="shot-level",
            evidence=evidence,
            checkpoint_results=cp_results,
            intermediate_repr={"asr_transcript": asr_timeline},
            suggestions=[f"Fix: {e.issue}" for e in evidence if e.severity == Severity.HIGH],
        )

    # ------------------------------------------------------------------
    # Sub-metric 2: BGM Consistency
    # ------------------------------------------------------------------

    def _eval_bgm(self, state: GraphState) -> EvalResult:
        """Check if background music matches mood / pacing / transitions / volume.

        Pipeline:
          1. Find separated BGM stem (and optional dialogue stem).
          2. Compute *per-shot* BGM features (energy, spectral centroid,
             local tempo) so the LLM can compare music behaviour to each
             scene rather than to a single global average.
          3. Compute *per-boundary* BGM transition metrics (energy delta,
             spectral delta in a ±0.5 s window around each cut) so the
             LLM can decide whether a music cut is harsh.
          4. If the dialogue stem is available, compute the BGM/dialogue
             RMS ratio per shot so the volume-balance checkpoint has
             concrete evidence to look at.
          5. Format everything into compact ASCII tables that drop into the
             LLM prompt — the previous implementation only passed one
             tempo/energy/spectral_centroid for the entire video, which
             caused the LLM to answer "no evidence -> 3/5" on every
             checkpoint (≈0.5 fallback) for ~90% of cases.
        """
        prep = state.preprocessing
        audio_segments = prep.audio_segments if prep else []
        shots = prep.shots if prep else []

        bgm_track = next((a for a in audio_segments if a.track_type == "bgm"), None)
        dialogue_track = next(
            (a for a in audio_segments if a.track_type == "dialogue"),
            None,
        )

        if not bgm_track and not audio_segments:
            logger.info("[AudioAgent] No BGM data — skipping BGM evaluation")
            return None

        # ---- Feature extraction ----
        bgm_features = self._extract_bgm_features(
            bgm_track,
            shots=shots,
            dialogue_track=dialogue_track,
        )

        profile = self._build_content_profile(state)

        # ---- Build the LLM context block ----
        narrative = (
            (state.script_text or "").strip()
            or (state.extracted_script_text or "").strip()
        )
        extra_ctx = self._format_bgm_evidence(
            shots=shots,
            features=bgm_features,
            narrative=narrative,
            profile=profile,
        )

        score, confidence, cp_results, active_cps, _ = self._checkpoint_evaluate(
            metric_name="bgm_consistency",
            state=state,
            extra_context=extra_ctx,
            profile=profile,
        )

        evidence = []
        for r in cp_results:
            if r.normalised < 0.5:
                evidence.append(Evidence(
                    type="bgm_issue",
                    issue=f"[{r.checkpoint_id}] {r.reasoning}" if r.reasoning else r.checkpoint_id,
                    severity=Severity.HIGH if r.normalised < 0.25 else Severity.MEDIUM,
                ))

        return EvalResult(
            agent_id=self.agent_id,
            metric_name="bgm_consistency",
            score=score,
            confidence=confidence,
            granularity="shot-level",
            evidence=evidence,
            checkpoint_results=cp_results,
            intermediate_repr={"bgm_features": bgm_features},
        )

    # ------------------------------------------------------------------
    # Audio feature extraction (Librosa)
    # ------------------------------------------------------------------

    def _extract_bgm_features(
        self,
        bgm_track,
        shots: list | None = None,
        dialogue_track=None,
    ) -> dict:
        """Extract structured BGM features the LLM can actually reason about.

        Returns a dict with these top-level keys (any may be absent if data
        is missing):

          ``global``         — single tempo / energy / spectral_centroid /
                               duration, plus an "energy_variability" coefficient
                               of variation that hints at dynamic range.
          ``per_shot``       — list of {shot_index, t0, t1, mean_energy,
                               peak_energy, energy_std, spectral_centroid,
                               local_tempo_bpm, mood_label}. Lets the LLM
                               judge per-scene mood-match and pacing.
          ``per_boundary``   — list of {from_shot, to_shot, t_cut,
                               energy_pre, energy_post, energy_delta_pct,
                               spectral_pre, spectral_post,
                               spectral_delta_pct, transition_label}. Lets
                               the LLM judge transition smoothness.
          ``volume_balance`` — when dialogue is separated: per-shot ratios
                               and a global summary (good_ducking /
                               competing / inaudible / no_dialogue). Lets
                               the LLM judge BGM-vs-dialogue mixing.
          ``status``         — set when extraction is impossible (no
                               librosa, file load fails, etc.).
        """
        if bgm_track is None:
            return {"status": "no_bgm_track",
                    "note": "Audio source separation did not produce a BGM stem; "
                            "the only audio track available is the unseparated mix."}

        try:
            import librosa
            import numpy as np
        except ImportError:
            logger.warning("[AudioAgent] Librosa not available, no BGM features extracted")
            self._record_tool("Librosa", ToolStatus.FAILED,
                              "Not installed — no audio features available",
                              affects=["bgm_consistency"])
            return {
                "status": "librosa_not_available",
                "note": "Librosa is not installed — no audio features could "
                        "be extracted. Score this checkpoint with low confidence.",
            }

        try:
            sr = 22050
            y, _ = librosa.load(bgm_track.path, sr=sr)
            duration = float(len(y)) / sr if sr else 0.0

            global_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if hasattr(global_tempo, "__len__"):
                global_tempo = float(global_tempo[0])
            else:
                global_tempo = float(global_tempo)
            rms_full = librosa.feature.rms(y=y).flatten()
            mean_rms = float(np.mean(rms_full)) if rms_full.size else 0.0
            std_rms = float(np.std(rms_full)) if rms_full.size else 0.0
            energy_cv = float(std_rms / mean_rms) if mean_rms > 1e-6 else 0.0
            spec_full = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
            global_spec = float(np.mean(spec_full)) if spec_full.size else 0.0

            features: dict = {
                "global": {
                    "tempo_bpm": global_tempo,
                    "mean_energy": mean_rms,
                    "energy_std": std_rms,
                    "energy_variability_cv": energy_cv,
                    "spectral_centroid": global_spec,
                    "duration_sec": duration,
                },
            }
            features["per_shot"] = self._compute_per_shot_features(
                y, sr, shots or [], librosa, np
            )
            features["per_boundary"] = self._compute_per_boundary_features(
                y, sr, shots or [], librosa, np
            )

            if dialogue_track is not None:
                try:
                    y_dlg, _ = librosa.load(dialogue_track.path, sr=sr)
                    features["volume_balance"] = self._compute_volume_balance(
                        y_bgm=y, y_dlg=y_dlg, sr=sr,
                        shots=shots or [], librosa=librosa, np=np,
                    )
                except Exception as e:
                    features["volume_balance"] = {
                        "status": "load_failed", "error": str(e),
                    }
            else:
                features["volume_balance"] = {
                    "status": "no_dialogue_stem",
                    "note": "AudioShake did not produce a separated dialogue track; "
                            "BGM/dialogue ratio cannot be measured directly.",
                }

            self._record_tool(
                "Librosa", ToolStatus.SUCCESS,
                f"Extracted global + {len(features['per_shot'])} per-shot + "
                f"{len(features['per_boundary'])} per-boundary BGM features"
                + (" + volume_balance" if dialogue_track else ""),
                affects=["bgm_consistency"],
            )
            return features

        except Exception as e:
            logger.warning(f"[AudioAgent] Feature extraction failed: {e}")
            self._record_tool("Librosa", ToolStatus.FAILED, str(e),
                              affects=["bgm_consistency"])
            return {
                "status": "extraction_failed",
                "error": str(e),
                "note": "Feature extraction failed — score this checkpoint with low confidence.",
            }

    @staticmethod
    def _slice_audio(y, sr: int, t0: float, t1: float):
        i0 = max(0, int(t0 * sr))
        i1 = min(len(y), int(t1 * sr))
        if i1 <= i0:
            return y[i0:i0]
        return y[i0:i1]

    @staticmethod
    def _classify_energy(rms: float) -> str:
        """Coarse mood label so the LLM can spot scene/music mismatches."""
        if rms < 0.005:
            return "very_quiet"
        if rms < 0.015:
            return "quiet"
        if rms < 0.04:
            return "moderate"
        if rms < 0.10:
            return "loud"
        return "very_loud"

    def _compute_per_shot_features(
        self, y, sr: int, shots: list, librosa, np
    ) -> list[dict]:
        out: list[dict] = []
        for s in shots:
            seg = self._slice_audio(y, sr, s.start_sec, s.end_sec)
            if seg.size < int(0.2 * sr):
                continue
            rms = librosa.feature.rms(y=seg).flatten()
            spec = librosa.feature.spectral_centroid(y=seg, sr=sr).flatten()
            mean_e = float(np.mean(rms)) if rms.size else 0.0
            peak_e = float(np.max(rms)) if rms.size else 0.0
            std_e = float(np.std(rms)) if rms.size else 0.0
            mean_spec = float(np.mean(spec)) if spec.size else 0.0
            # Local tempo only meaningful on segments >= 4s.
            local_tempo: float | None
            if seg.size >= 4 * sr:
                try:
                    t, _ = librosa.beat.beat_track(y=seg, sr=sr)
                    local_tempo = float(t if not hasattr(t, "__len__") else t[0])
                except Exception:
                    local_tempo = None
            else:
                local_tempo = None
            out.append({
                "shot_index": s.index,
                "t0": round(float(s.start_sec), 2),
                "t1": round(float(s.end_sec), 2),
                "duration_sec": round(float(s.end_sec - s.start_sec), 2),
                "mean_energy": round(mean_e, 5),
                "peak_energy": round(peak_e, 5),
                "energy_std": round(std_e, 5),
                "spectral_centroid": round(mean_spec, 1),
                "local_tempo_bpm": round(local_tempo, 1) if local_tempo else None,
                "energy_label": self._classify_energy(mean_e),
            })
        return out

    def _compute_per_boundary_features(
        self, y, sr: int, shots: list, librosa, np,
        window_sec: float = 0.5,
    ) -> list[dict]:
        """Energy & spectral comparison in a ±window_sec slice around each cut.

        A clean musical transition keeps energy and spectral centroid roughly
        flat across the boundary; a hard cut produces a large delta. We let
        the LLM make the qualitative call but pre-compute the deltas so it
        actually has a number to look at.
        """
        out: list[dict] = []
        if len(shots) < 2:
            return out
        for i in range(len(shots) - 1):
            a, b = shots[i], shots[i + 1]
            t_cut = float(a.end_sec)
            pre = self._slice_audio(y, sr, t_cut - window_sec, t_cut)
            post = self._slice_audio(y, sr, t_cut, t_cut + window_sec)
            if pre.size < int(0.05 * sr) or post.size < int(0.05 * sr):
                continue
            e_pre = float(np.mean(librosa.feature.rms(y=pre).flatten()))
            e_post = float(np.mean(librosa.feature.rms(y=post).flatten()))
            denom = max(e_pre, e_post, 1e-6)
            energy_delta_pct = float(abs(e_post - e_pre) / denom * 100.0)
            sp_pre = float(np.mean(librosa.feature.spectral_centroid(y=pre, sr=sr).flatten()))
            sp_post = float(np.mean(librosa.feature.spectral_centroid(y=post, sr=sr).flatten()))
            sp_denom = max(sp_pre, sp_post, 1e-6)
            spectral_delta_pct = float(abs(sp_post - sp_pre) / sp_denom * 100.0)
            # Heuristic label, kept conservative — final call is the LLM's.
            if energy_delta_pct < 25 and spectral_delta_pct < 30:
                label = "smooth"
            elif energy_delta_pct < 60 and spectral_delta_pct < 60:
                label = "noticeable"
            else:
                label = "harsh"
            out.append({
                "from_shot": a.index,
                "to_shot": b.index,
                "t_cut": round(t_cut, 2),
                "energy_pre": round(e_pre, 5),
                "energy_post": round(e_post, 5),
                "energy_delta_pct": round(energy_delta_pct, 1),
                "spectral_pre": round(sp_pre, 1),
                "spectral_post": round(sp_post, 1),
                "spectral_delta_pct": round(spectral_delta_pct, 1),
                "transition_label": label,
            })
        return out

    def _compute_volume_balance(
        self, y_bgm, y_dlg, sr: int, shots: list, librosa, np,
    ) -> dict:
        """Compare BGM vs dialogue RMS per shot. The classic "is the music
        drowning out the speech?" question — but turned into a number.

        We only count shots where dialogue is actually present (RMS above
        a tiny floor). We surface a per-shot list AND a coarse global
        verdict so the LLM has the full picture.
        """
        per_shot: list[dict] = []
        ratios: list[float] = []
        # Trim/pad bgm and dialogue to the same length for safe slicing.
        n = min(len(y_bgm), len(y_dlg))
        y_bgm = y_bgm[:n]
        y_dlg = y_dlg[:n]
        dlg_floor = 1e-3  # below this, treat as "no dialogue in this window"
        for s in shots:
            bgm_seg = self._slice_audio(y_bgm, sr, s.start_sec, s.end_sec)
            dlg_seg = self._slice_audio(y_dlg, sr, s.start_sec, s.end_sec)
            if bgm_seg.size < int(0.2 * sr):
                continue
            bgm_rms = float(np.mean(librosa.feature.rms(y=bgm_seg).flatten())) if bgm_seg.size else 0.0
            dlg_rms = float(np.mean(librosa.feature.rms(y=dlg_seg).flatten())) if dlg_seg.size else 0.0
            has_dlg = dlg_rms > dlg_floor
            ratio = (bgm_rms / dlg_rms) if dlg_rms > 1e-6 else None
            per_shot.append({
                "shot_index": s.index,
                "bgm_rms": round(bgm_rms, 5),
                "dialogue_rms": round(dlg_rms, 5),
                "bgm_over_dialogue": round(ratio, 2) if ratio is not None else None,
                "has_dialogue": has_dlg,
                "verdict": (
                    "no_dialogue" if not has_dlg
                    else "bgm_drowns_speech" if (ratio or 0.0) > 1.5
                    else "competing" if (ratio or 0.0) > 0.7
                    else "well_ducked"
                ),
            })
            if has_dlg and ratio is not None:
                ratios.append(ratio)

        # Global verdict
        if not ratios:
            global_v = "no_dialogue_in_video"
        else:
            mean_ratio = sum(ratios) / len(ratios)
            if mean_ratio > 1.5:
                global_v = "bgm_drowns_speech"
            elif mean_ratio > 0.7:
                global_v = "competing"
            else:
                global_v = "well_ducked"
        return {
            "per_shot": per_shot,
            "shots_with_dialogue": len(ratios),
            "mean_bgm_over_dialogue": round(sum(ratios) / len(ratios), 2) if ratios else None,
            "global_verdict": global_v,
        }

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _format_bgm_evidence(
        self,
        shots: list,
        features: dict,
        narrative: str,
        profile: ContentProfile,
    ) -> str:
        """Render the structured BGM evidence into a compact text block.

        Designed to be dropped straight into ``_checkpoint_evaluate``'s
        ``extra_context``. Per-shot and per-boundary tables are rendered in
        a fixed-width ASCII format the LLM can read without ambiguity.
        """
        lines: list[str] = []
        lines.append("=== BGM EVIDENCE ===")

        # Status / fallback messages first so the LLM knows when data is missing.
        if "status" in features:
            lines.append(f"BGM feature extraction status: {features['status']}")
            note = features.get("note") or features.get("error")
            if note:
                lines.append(f"  note: {note}")
            lines.append("")

        if narrative:
            snippet = narrative.strip().replace("\n", " ")
            if len(snippet) > 1200:
                snippet = snippet[:1200] + "…"
            lines.append("Narrative / scene description (use this as the visual mood reference):")
            lines.append(snippet)
            lines.append("")
        else:
            lines.append("Narrative / scene description: (none extracted — judge mood "
                         "match conservatively from shot pacing alone)")
            lines.append("")

        if shots:
            lines.append("Shot timeline:")
            for s in shots:
                lines.append(f"  Shot {s.index}: {s.start_sec:.2f}s → {s.end_sec:.2f}s "
                             f"(dur {s.end_sec - s.start_sec:.2f}s)")
            lines.append("")

        g = features.get("global") or {}
        if g:
            lines.append("Global BGM features:")
            lines.append(f"  tempo_bpm={g.get('tempo_bpm', '?'):.1f}  "
                         f"mean_energy={g.get('mean_energy', 0):.4f}  "
                         f"energy_variability_cv={g.get('energy_variability_cv', 0):.2f}  "
                         f"spectral_centroid={g.get('spectral_centroid', 0):.0f}Hz  "
                         f"duration={g.get('duration_sec', 0):.1f}s")
            lines.append("")

        per_shot = features.get("per_shot") or []
        if per_shot:
            lines.append("Per-shot BGM features (use for mood-match & tempo-pacing checkpoints):")
            lines.append("  shot |   t0..t1   | mean_E | peak_E | spec_Hz | tempo | label")
            for r in per_shot:
                tempo_cell = (
                    f"{r['local_tempo_bpm']:.0f}" if r["local_tempo_bpm"] else "  - "
                )
                lines.append(
                    f"  {r['shot_index']:>4} | "
                    f"{r['t0']:>5.1f}..{r['t1']:<5.1f} | "
                    f"{r['mean_energy']:>6.4f} | "
                    f"{r['peak_energy']:>6.4f} | "
                    f"{r['spectral_centroid']:>7.0f} | "
                    f"{tempo_cell:>5} | "
                    f"{r['energy_label']}"
                )
            lines.append(
                "  Interpret energy_label as a coarse intensity proxy. Compare against "
                "the narrative's emotional beats: a calm narrative shot paired with a "
                "'loud'/'very_loud' BGM shot is a mood mismatch; a tense shot paired "
                "with 'very_quiet' BGM is also a mismatch."
            )
            lines.append("")

        per_b = features.get("per_boundary") or []
        if per_b:
            lines.append("Per-boundary BGM transition metrics (use for transition-smoothness checkpoint):")
            lines.append("  cut@t  | A→B | dE %  | dSpec % | label")
            harsh_count = 0
            for r in per_b:
                if r["transition_label"] == "harsh":
                    harsh_count += 1
                lines.append(
                    f"  {r['t_cut']:>5.1f}s | {r['from_shot']:>2}→{r['to_shot']:<2} | "
                    f"{r['energy_delta_pct']:>5.1f} | {r['spectral_delta_pct']:>6.1f}  | "
                    f"{r['transition_label']}"
                )
            lines.append(
                f"  Heuristic: dE<25% AND dSpec<30% → smooth; >60%/>60% → harsh. "
                f"Harsh cuts in this video: {harsh_count}/{len(per_b)}."
            )
            lines.append("")

        vb = features.get("volume_balance") or {}
        if vb.get("per_shot"):
            lines.append("BGM vs Dialogue volume balance (use for volume-balance checkpoint):")
            lines.append("  shot | bgm_rms | dlg_rms | ratio | verdict")
            for r in vb["per_shot"]:
                ratio_cell = (
                    f"{r['bgm_over_dialogue']:.2f}"
                    if r["bgm_over_dialogue"] is not None else "  - "
                )
                lines.append(
                    f"  {r['shot_index']:>4} | "
                    f"{r['bgm_rms']:>7.4f} | "
                    f"{r['dialogue_rms']:>7.4f} | "
                    f"{ratio_cell:>5} | "
                    f"{r['verdict']}"
                )
            lines.append(
                f"  Summary: shots_with_dialogue={vb.get('shots_with_dialogue', 0)}, "
                f"mean_BGM_over_dialogue="
                f"{vb.get('mean_bgm_over_dialogue') if vb.get('mean_bgm_over_dialogue') is not None else '-'}, "
                f"global_verdict={vb.get('global_verdict')}"
            )
            lines.append(
                "  Heuristic: ratio>1.5 means BGM is louder than the speech "
                "(usually drowns it out); 0.7-1.5 means competing; <0.7 means "
                "BGM is well-ducked under speech."
            )
            lines.append("")
        elif vb.get("status"):
            lines.append(f"BGM/Dialogue volume balance: {vb['status']} — "
                         f"{vb.get('note', '')}")
            lines.append("")

        lines.append(
            "When evaluating, ground each rubric anchor in the concrete numbers above. "
            "Cite specific shot indices or boundary timestamps in your reasoning."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main evaluate
    # ------------------------------------------------------------------

    def evaluate(self, state: GraphState) -> list[EvalResult]:
        # Sub-metrics return None when their data is absent (e.g. no ASR,
        # no BGM).  Filter them out so the dimension score is computed only
        # from metrics that actually ran.
        raw = [
            self._eval_narration(state),
            self._eval_bgm(state),
        ]
        return [r for r in raw if r is not None]
