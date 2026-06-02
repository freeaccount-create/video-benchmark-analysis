"""
base.py — Abstract base class for all evaluation agents.

Every Specialist Agent inherits from BaseEvalAgent and implements
the `evaluate()` method, which returns a list of EvalResult objects.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
import time

from ..config import EvalConfig
from ..llm_utils import LLMClient
from ..schemas import (
    AgentID, CheckpointDef, CheckpointResult, CheckpointType,
    ContentProfile, EvalResult, GraphState, RubricAnchor,
    ToolCallRecord, ToolStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence-guidance paragraph injected into every LLM evaluation prompt.
# This is NOT a hard rule — it asks the LLM to consider tool availability
# when deciding its own confidence score.
# ---------------------------------------------------------------------------
CONFIDENCE_TOOL_GUIDANCE = """
IMPORTANT — Confidence & tool availability:
The "confidence" field you return should reflect how certain you are about
your evaluation.  If any of the tools or data sources listed in the
"Tool Availability Context" below FAILED or used a FALLBACK, and that
missing information is relevant to your current evaluation, you should
lower your confidence accordingly.  For example:
  • A critical tool failed (e.g. ASR failed → you have no dialogue text)
    and the metric depends on it → significantly lower confidence.
  • A non-critical tool used a fallback (e.g. uniform segmentation instead
    of real shot detection) → moderately lower confidence.
  • All relevant tools succeeded → assign confidence based solely on your
    evaluation certainty.
Do NOT hard-set confidence to any specific value — use your judgment about
how much the missing data actually impacts this particular evaluation.
""".strip()


class BaseEvalAgent(ABC):
    """Base class for all evaluation agents in the framework."""

    agent_id: AgentID

    def __init__(self, config: EvalConfig | None = None):
        self.config = config or EvalConfig()
        self.llm = LLMClient(self.config.llm)
        # Per-evaluation-run local tool records (e.g. librosa, BRISQUE).
        # Populated during evaluate() and merged into state on return.
        self._local_tool_records: list[ToolCallRecord] = []

    @abstractmethod
    def evaluate(self, state: GraphState) -> list[EvalResult]:
        """
        Run evaluation and return a list of EvalResult objects.
        Each sub-metric produces one EvalResult.
        """
        ...

    def __call__(self, state: GraphState) -> dict:
        """
        LangGraph node interface.
        Called by the graph; returns a dict of state updates.
        """
        agent_name = self.__class__.__name__
        logger.info(f"[{agent_name}] Starting evaluation...")

        # Reset per-run local tool records and content profile cache
        self._local_tool_records = []
        self._last_content_profile = None
        t0 = time.perf_counter()

        try:
            results = self.evaluate(state)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._record_tool(
                tool_name=f"{agent_name}/evaluate",
                status=ToolStatus.SUCCESS,
                detail=f"Produced {len(results)} results",
                elapsed_ms=elapsed_ms,
            )
            logger.info(f"[{agent_name}] Produced {len(results)} eval results")
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._record_tool(
                tool_name=f"{agent_name}/evaluate",
                status=ToolStatus.FAILED,
                detail=str(e),
                elapsed_ms=elapsed_ms,
            )
            logger.error(f"[{agent_name}] Failed: {e}")
            results = []
            return {
                self._state_key(): results,
                "errors": state.errors + [f"{agent_name}: {str(e)}"],
                "execution_log": state.execution_log + [f"{agent_name}: FAILED - {e}"],
                "tool_context": self._local_tool_records,
            }

        update = {
            self._state_key(): results,
            "execution_log": state.execution_log + [
                f"{agent_name}: OK - {len(results)} results"
            ],
            # Merge any tool records this agent produced into the shared context
            "tool_context": self._local_tool_records,
        }

        # Propagate content_profile to GraphState from a single writer only.
        # Multiple parallel phase-1 agents may all build the profile and try to
        # update the same key in one LangGraph step, which triggers:
        # InvalidUpdateError ("Can receive only one value per step").
        # Restricting writes to SCRIPT_EVAL avoids concurrent channel updates.
        if (
            self.agent_id == AgentID.SCRIPT_EVAL
            and self._last_content_profile is not None
            and state.content_profile is None
        ):
            update["content_profile"] = self._last_content_profile

        return update

    # ------------------------------------------------------------------
    # Tool-context helpers
    # ------------------------------------------------------------------

    def _record_tool(
        self,
        tool_name: str,
        status: ToolStatus,
        detail: str = "",
        elapsed_ms: float | None = None,
        affects: list[str] | None = None,
    ) -> None:
        """Register a tool call outcome (local to this agent's current run)."""
        self._local_tool_records.append(ToolCallRecord(
            tool_name=tool_name,
            status=status,
            detail=detail,
            elapsed_ms=elapsed_ms,
            affects=affects or [],
        ))

    @staticmethod
    def format_tool_context(
        state: GraphState,
        extra_records: list[ToolCallRecord] | None = None,
        relevant_metrics: list[str] | None = None,
    ) -> str:
        """Build a human-readable "Tool Availability Context" block.

        This is injected into LLM evaluation prompts so the model can see
        which upstream tools succeeded or failed, and adjust its confidence.

        Args:
            state: Current graph state (contains preprocessing tool_records
                   and accumulated tool_context from previous agents).
            extra_records: Additional records from the current agent's own
                           tool calls (e.g. librosa, BRISQUE).
            relevant_metrics: If provided, only include records whose
                              ``affects`` list overlaps with these metric names.
        """
        # Collect all records
        all_records: list[ToolCallRecord] = list(state.tool_context)
        if extra_records:
            all_records.extend(extra_records)

        if not all_records:
            return ""

        # Optional filtering by relevance
        if relevant_metrics:
            metric_set = set(relevant_metrics)
            filtered = [
                r for r in all_records
                if not r.affects or metric_set & set(r.affects)
            ]
        else:
            filtered = all_records

        if not filtered:
            return ""

        lines = ["", "--- Tool Availability Context ---"]
        for r in filtered:
            status_icon = {
                ToolStatus.SUCCESS: "OK",
                ToolStatus.FAILED: "FAILED",
                ToolStatus.FALLBACK: "FALLBACK",
                ToolStatus.SKIPPED: "SKIPPED",
            }.get(r.status, str(r.status.value))

            line = f"[{status_icon}] {r.tool_name}"
            if r.detail:
                line += f" — {r.detail}"
            if r.elapsed_ms is not None:
                line += f" [elapsed={r.elapsed_ms:.1f}ms]"
            if r.affects:
                line += f"  (affects: {', '.join(r.affects)})"
            lines.append(line)

        lines.append("--- End Tool Context ---")
        return "\n".join(lines)

    @staticmethod
    def maybe_add_confidence_guidance(system_prompt: str, state: GraphState) -> str:
        """Append confidence/tool guidance to the system prompt if any tools
        failed, fell back, or were skipped.  When all tools succeeded, the
        prompt is returned unchanged to avoid unnecessary verbosity."""
        has_issues = any(
            r.status != ToolStatus.SUCCESS for r in state.tool_context
        )
        if has_issues:
            return system_prompt + "\n\n" + CONFIDENCE_TOOL_GUIDANCE
        return system_prompt

    # ------------------------------------------------------------------
    # Content profiling (dynamic rubric foundation)
    # ------------------------------------------------------------------

    def _build_content_profile(self, state: GraphState) -> ContentProfile:
        """Generate a ContentProfile for the video via a fast VLM pass.

        If the profile already exists on ``state`` (built by a prior agent or
        by the orchestrator), return it directly to avoid duplicate API calls.
        The profile is also cached on ``self._last_content_profile`` so it can
        be propagated back into GraphState via ``__call__``.
        """
        if state.content_profile is not None:
            self._last_content_profile = state.content_profile
            return state.content_profile

        prep = state.preprocessing
        thumbnails = []
        if prep and prep.shots:
            thumbnails = [s.thumbnail_path for s in prep.shots if s.thumbnail_path]

        asr_present = bool(prep and prep.asr_segments)
        shot_count = len(prep.shots) if prep else 0

        if not thumbnails:
            # Minimal profile when no frames available
            profile = ContentProfile(
                has_dialogue=asr_present,
                scene_count=max(1, shot_count),
                is_single_shot=(shot_count <= 1),
            )
            self._last_content_profile = profile
            return profile

        system_prompt = """You are a video content analyser. Given representative frames from a video,
identify what the video contains. Return ONLY a JSON object (no commentary):
{
  "has_characters": <bool>,
  "character_count": <int>,
  "has_held_objects": <bool>,
  "has_animals": <bool>,
  "has_scene_changes": <bool>,
  "has_text_overlay": <bool>,
  "has_special_effects": <bool>,
  "is_live_action_style": <bool>,
  "is_animation_style": <bool>,
  "has_camera_movement": <bool>,
  "has_fast_motion": <bool>
}
Be precise. Only set true if clearly visible."""

        result = self.llm.vision_evaluate(
            system_prompt=system_prompt,
            text_prompt=f"Analyse these {len(thumbnails[:8])} representative frames:",
            image_paths=thumbnails[:8],
        )

        # --- Supplement character_count with ASR speaker info ---
        vlm_char_count = int(result.get("character_count", 0))
        asr_speaker_count = 0
        if prep and prep.asr_segments:
            speakers = set()
            for seg in prep.asr_segments:
                if seg.speaker:
                    speakers.add(seg.speaker)
            asr_speaker_count = len(speakers)
            # Also try to detect distinct speakers/characters from dialogue
            if asr_speaker_count <= 1:
                import re
                all_text = " ".join(seg.text for seg in prep.asr_segments)
                name_matches: set[str] = set()

                # Strategy 1: "Name:" — name followed by colon (script format)
                name_matches |= set(re.findall(
                    r'(?:^|\n|[.!?]\s+)\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)(?:\s*[:：])',
                    all_text,
                ))
                name_matches |= set(re.findall(
                    r'\b([A-Z][a-z]{1,15})\s*[:：]\s',
                    all_text,
                ))

                # Strategy 2: Vocative address — names in dialogue context.
                # Patterns: "..., Name." / "..., Name," / "Name, ..."
                # e.g. "Not another step, Marco." or "Victor, I don't..."
                # Match capitalized words (2-15 chars) adjacent to commas
                # or at sentence boundaries that look like person names.
                vocative_matches = set(re.findall(
                    r',\s+([A-Z][a-z]{1,15})[.,!?\s]',
                    all_text,
                ))
                vocative_matches |= set(re.findall(
                    r'(?:^|[.!?]\s+)([A-Z][a-z]{1,15}),\s',
                    all_text,
                ))
                # Filter out common non-name words that could be capitalized
                # at sentence starts
                _common_words = {
                    "The", "This", "That", "What", "When", "Where", "Why",
                    "How", "But", "And", "Not", "Yes", "No", "Well", "Now",
                    "Just", "Like", "Here", "There", "Maybe", "Please",
                    "Stop", "Wait", "Look", "Come", "Let", "Get", "Got",
                    "Feels", "Never", "Always", "Still", "Also", "Even",
                }
                vocative_matches -= _common_words
                name_matches |= vocative_matches

                # Strategy 3: Names mentioned with possessives or as subjects
                # in dialogue — e.g. "Marco's plan" or "Victor ran"
                possessive_matches = set(re.findall(
                    r"\b([A-Z][a-z]{1,15})'s\b",
                    all_text,
                ))
                possessive_matches -= _common_words
                name_matches |= possessive_matches

                if len(name_matches) > asr_speaker_count:
                    asr_speaker_count = len(name_matches)
                    logger.info(
                        f"[ContentProfile] Detected {asr_speaker_count} "
                        f"characters from dialogue: {name_matches}"
                    )

        # Take the max of VLM detection and ASR speaker count
        final_char_count = max(vlm_char_count, asr_speaker_count)
        if final_char_count > vlm_char_count:
            logger.info(
                f"[ContentProfile] Overriding VLM character_count "
                f"({vlm_char_count}) with ASR-based count ({final_char_count})"
            )

        profile = ContentProfile(
            has_characters=bool(result.get("has_characters", False)) or final_char_count > 0,
            character_count=final_char_count,
            has_dialogue=asr_present,
            has_held_objects=bool(result.get("has_held_objects", False)),
            has_animals=bool(result.get("has_animals", False)),
            scene_count=max(1, shot_count),
            has_scene_changes=bool(result.get("has_scene_changes", False)),
            is_single_shot=(shot_count <= 1),
            has_text_overlay=bool(result.get("has_text_overlay", False)),
            has_special_effects=bool(result.get("has_special_effects", False)),
            is_live_action_style=bool(result.get("is_live_action_style", False)),
            is_animation_style=bool(result.get("is_animation_style", False)),
            has_background_music=bool(
                prep and any(t.track_type in ("bgm", "music") for t in (prep.audio_segments or []))
            ),
            has_camera_movement=bool(result.get("has_camera_movement", False)),
            has_fast_motion=bool(result.get("has_fast_motion", False)),
        )
        self._last_content_profile = profile
        return profile

    # ------------------------------------------------------------------
    # Checkpoint-based evaluation engine
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_applicable(
        checkpoints: list[CheckpointDef],
        profile: ContentProfile,
    ) -> list[CheckpointDef]:
        """Return only those checkpoints whose conditions match the profile."""
        return [
            cp for cp in checkpoints
            if not cp.applicable_when or profile.matches(cp.applicable_when)
        ]

    @staticmethod
    def _build_rubric_text(cp: CheckpointDef) -> str:
        """Format a single checkpoint's rubric into a text block for the LLM."""
        lines = [f"Question: {cp.question}"]
        if cp.checkpoint_type == CheckpointType.BINARY:
            lines.append(
                "Answer: This is a BINARY (yes/no) question. "
                "You MUST return value=1 for Yes/Pass or value=0 for No/Fail. "
                "Do NOT return any other number (not 2, 3, 4, 5). ONLY 0 or 1."
            )
        elif cp.checkpoint_type == CheckpointType.LIKERT and cp.rubric:
            lines.append("Score using this rubric:")
            for anchor in sorted(cp.rubric, key=lambda a: a.value, reverse=True):
                lines.append(f"  {anchor.value} ({anchor.label}): {anchor.description}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Robust JSON extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_from_text(text: str):
        """Try hard to pull a JSON array or object out of *text*.

        Handles:
          - Plain JSON
          - Wrapped in markdown fences (```json ... ``` or ``` ... ```)
          - Embedded in conversational text (finds first [ ... ] or { ... })
          - Python-style True/False/None literals
        Returns the parsed Python object, or *None* on failure.
        """
        import json, re

        if not isinstance(text, str):
            return text  # already parsed

        cleaned = text.strip()

        # 1. Strip markdown code fences
        fence_re = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
        m = fence_re.search(cleaned)
        if m:
            cleaned = m.group(1).strip()

        # 2. Normalise Python literals → JSON
        cleaned_json = (
            cleaned
            .replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
        )

        # 3. Try direct parse
        for candidate in (cleaned_json, cleaned):
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass

        # 4. Try to locate the first JSON array [ ... ]
        arr_re = re.compile(r"\[.*\]", re.DOTALL)
        m = arr_re.search(cleaned_json)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        # 5. Try to locate the first JSON object { ... }
        obj_re = re.compile(r"\{.*\}", re.DOTALL)
        m = obj_re.search(cleaned_json)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    # ------------------------------------------------------------------
    # Reasoning ↔ Score consistency check
    # ------------------------------------------------------------------

    def _check_reasoning_score_consistency(
        self,
        cp_id: str,
        value: int,
        reasoning: str,
        checkpoint_type: CheckpointType = CheckpointType.LIKERT,
    ) -> bool:
        """Detect obvious contradictions between reasoning sentiment and score.

        Uses a lightweight heuristic (no extra LLM call). The polarity of
        ``value`` depends on ``checkpoint_type``:

          * LIKERT (1–5):  1 = worst, 5 = best
              - Score ≤ 2 (bad) but reasoning clearly positive → inconsistent
              - Score ≥ 4 (good) but reasoning clearly negative → inconsistent
              - Score = 5 (perfect) with any negative signal → inconsistent
              - Score = 1 (worst) with only positive signals → inconsistent
              - Score = 3 → always consistent (benefit of the doubt)

          * BINARY (0/1):  0 = FAIL (worst), 1 = PASS (best)
              - Score = 0 with reasoning clearly positive → inconsistent
              - Score = 1 with reasoning clearly negative → inconsistent
              We do NOT flag "value=1 with positives" or "value=0 with
              negatives" because for binary metrics those are perfectly
              normal pairings (PASS + "no defects", FAIL + "missing").

        Returns True if consistent, False if contradictory.
        """
        if not reasoning:
            return True
        # LIKERT only: middle score is always plausible (no analogue for binary).
        if checkpoint_type == CheckpointType.LIKERT and value == 3:
            return True

        text = reasoning.lower()

        import re as _re

        # Strong positive indicators (expanded)
        positive_phrases = [
            "perfect", "excellent", "flawless", "no issues", "no noticeable",
            "no visible", "consistently", "well-maintained", "seamless",
            "crystal clear", "persist consistently", "no vanishing",
            "no disappearing", "remains consistent", "well-rendered",
            "no artifacts", "no distortion", "coherent throughout",
            "perfectly balanced", "every word is easily understood",
            # Expanded coverage:
            "correctly", "without any", "no instances of", "high quality",
            "well matched", "well-matched", "natural", "realistic",
            "appropriate", "convincing", "no mismatch", "no inconsistenc",
            "maintains", "preserved", "intact", "stable", "uniform",
            "persist correctly", "no problems", "no defects",
            "no discrepancies", "well synchronized", "well-synchronized",
            "strong", "impressive", "solid", "good quality",
        ]
        # Strong negative indicators (expanded)
        negative_phrases = [
            "significant", "severe", "major issue", "completely broken",
            "unintelligible", "constant flickering", "fails to",
            "no coherence", "contradictions", "disrupts", "distracting",
            "incoherent", "lacks any", "fundamentally flawed",
            # Expanded coverage:
            "inconsistent", "mismatch", "different face", "different character",
            "changes between", "varies across", "not maintained",
            "poor", "broken", "jarring", "abrupt", "unnatural",
            "misaligned", "out of sync", "does not match", "doesn't match",
            "noticeab", "obvious", "glaring", "problematic",
            "low quality", "degraded", "distorted", "blurry",
            "missing", "absent", "lacking", "insufficient",
            "fluctuat", "unstable", "erratic", "jumpy",
        ]

        # Regex-based negative patterns (e.g. "no ... consistency")
        neg_patterns = [
            r"\black(?:s|ing)?\b",       # lacks, lacking
            r"\bnot\s+\w+\s*consisten",  # not very consistent, not consistent
            r"\bdoes\s+not\b",           # does not maintain
            r"\bfail",                   # fails, failed, failure
            r"\bpoor(?:ly)?\b",          # poor, poorly
        ]

        # Count positive/negative hits, but be careful about negation.
        # "no noticeable" is positive — we don't want "noticeab" to also
        # trigger as negative in that context.
        #
        # Models often respond with template prefixes like
        #     "Defects checked first: no major issues found ..."
        #     "Issues identified: none observed in the sequence."
        # The old logic only inspected the 4–9 chars BEFORE each negative
        # keyword, so it missed negations that appear AFTER the keyword
        # within the same sentence and produced false-positive INCONSISTENCY
        # retries that wasted ~30 s per case (see eval_debug.log
        # `setting_accuracy` retries 1/2 and 2/2). The fix below considers
        # the full enclosing sentence in BOTH directions.

        _SENT_END_CHARS = ".;!?\n"
        _NEG_TOKENS = (
            " no ", " not ", " without ", " never ", " none ", " nothing ",
            "n't ", " neither ", " nor ", "no major", "no meaningful",
            "no noticeable", "no significant", "no obvious", "no apparent",
            "no clear", "no visible", "no defect", "no issue", "no problem",
            "no inconsisten", "no mismatch", "no discrepan",
        )

        def _enclosing_sentence(t: str, idx: int) -> str:
            sent_start = 0
            for i in range(idx - 1, -1, -1):
                if t[i] in _SENT_END_CHARS:
                    sent_start = i + 1
                    break
            sent_end = len(t)
            for i in range(idx, len(t)):
                if t[i] in _SENT_END_CHARS:
                    sent_end = i
                    break
            return t[sent_start:sent_end]

        def _is_negated_in_sentence(t: str, kw: str) -> bool:
            """True if every occurrence of `kw` in `t` lies in a sentence
            that also contains a negation token.  We pad with a leading
            space so single-word matches like ` no ` work at sentence start.
            """
            occurrences = []
            start = 0
            while True:
                idx = t.find(kw, start)
                if idx < 0:
                    break
                occurrences.append(idx)
                start = idx + len(kw)
            if not occurrences:
                return False
            for idx in occurrences:
                sentence = " " + _enclosing_sentence(t, idx) + " "
                if not any(neg in sentence for neg in _NEG_TOKENS):
                    return False  # at least one un-negated occurrence
            return True

        pos_hits = sum(1 for p in positive_phrases if p in text)
        neg_hits = 0
        for p in negative_phrases:
            if p not in text:
                continue
            if _is_negated_in_sentence(text, p):
                continue  # every occurrence is negated within its sentence
            neg_hits += 1

        # Also count regex pattern matches, but skip ones whose match falls
        # in a negated sentence.
        for pat in neg_patterns:
            for m in _re.finditer(pat, text):
                sentence = " " + _enclosing_sentence(text, m.start()) + " "
                if not any(neg in sentence for neg in _NEG_TOKENS):
                    neg_hits += 1
                    break  # count each pattern at most once

        if checkpoint_type == CheckpointType.BINARY:
            # 0 = FAIL (worst), 1 = PASS (best). Mirror the LIKERT rules but
            # WITHOUT the "value=1 with positives" / "value=0 with negatives"
            # patterns — those are the *consistent* cases for binary metrics.
            #
            # We require ≥2 hits on the contradicting side to avoid
            # over-flagging single keyword occurrences (the same threshold
            # used by the LIKERT bad-with-positives rule).
            if value == 0 and pos_hits >= 2 and neg_hits == 0:
                logger.debug(
                    f"[Consistency] '{cp_id}' FAIL (binary): value=0 (FAIL) "
                    f"but pos_hits={pos_hits}, neg_hits={neg_hits}"
                )
                return False
            if value == 1 and neg_hits >= 2 and pos_hits == 0:
                logger.debug(
                    f"[Consistency] '{cp_id}' FAIL (binary): value=1 (PASS) "
                    f"but neg_hits={neg_hits}, pos_hits={pos_hits}"
                )
                return False
            return True

        # ----- LIKERT path (1=worst, 5=best) -----
        # Score ≤ 2 but reasoning is positive-dominant
        if value <= 2 and pos_hits >= 2 and neg_hits == 0:
            logger.debug(
                f"[Consistency] '{cp_id}' FAIL: value={value} but "
                f"pos_hits={pos_hits}, neg_hits={neg_hits}"
            )
            return False
        # Score ≥ 4 but reasoning is negative-dominant
        if value >= 4 and neg_hits >= 2 and pos_hits == 0:
            logger.debug(
                f"[Consistency] '{cp_id}' FAIL: value={value} but "
                f"neg_hits={neg_hits}, pos_hits={pos_hits}"
            )
            return False
        # Score = 5 (perfect) but ANY negative signal → suspicious
        if value == 5 and neg_hits >= 1:
            logger.debug(
                f"[Consistency] '{cp_id}' FAIL: value=5 (perfect) but "
                f"neg_hits={neg_hits} — perfect score with negatives"
            )
            return False
        # Score = 1 (worst) but ANY positive signal → suspicious
        if value == 1 and pos_hits >= 1 and neg_hits == 0:
            logger.debug(
                f"[Consistency] '{cp_id}' FAIL: value=1 but "
                f"pos_hits={pos_hits} — worst score with only positives"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Checkpoint-specific factual data injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_checkpoint_facts(
        cp: CheckpointDef,
        state: GraphState,
        user_prompt: str,
    ) -> str:
        """Inject objective, factual data into the user prompt for specific
        checkpoints that the LLM tends to answer incorrectly without evidence.

        This grounds the LLM on the actual question and prevents it from
        drifting to answer a different (related) checkpoint.
        """
        prep = state.preprocessing

        if cp.id == "duration_completeness":
            total_dur = prep.total_duration_sec if prep else 0
            target_dur = 60.0
            ratio = total_dur / target_dur if target_dur > 0 else 0
            passes = "YES (PASS=1)" if ratio >= 0.9 else "NO (FAIL=0)"
            fact_block = (
                f"\n\n=== FACTUAL DATA for '{cp.id}' ===\n"
                f"This checkpoint asks ONLY about VIDEO DURATION.\n"
                f"Measured video duration: {total_dur:.1f} seconds\n"
                f"Expected duration: ~{target_dur:.0f} seconds\n"
                f"Duration ratio: {ratio:.1%}\n"
                f"Based on this data, the answer should be: {passes}\n"
                f"You MUST base your answer on these measurements.\n"
                f"=== END FACTUAL DATA ===\n"
            )
            return user_prompt + fact_block

        if cp.id == "color_temperature":
            fact_block = (
                f"\n\n=== CHECKPOINT FOCUS ===\n"
                f"You are evaluating ONLY '{cp.id}' — colour temperature "
                f"consistency (warm/cool shifts within scenes).\n"
                f"Do NOT evaluate lighting direction, shadows, or exposure.\n"
                f"Your response id MUST be \"{cp.id}\".\n"
                f"=== END FOCUS ===\n"
            )
            return user_prompt + fact_block

        return user_prompt

    @staticmethod
    def _factual_override_for_checkpoint(
        cp: CheckpointDef,
        state: GraphState,
    ) -> tuple[int, str] | None:
        """For checkpoints that can be answered from objective data, compute
        the answer directly. Returns (value, reasoning) or None if not applicable.

        This is used as a last resort when the LLM repeatedly returns an
        answer for the wrong checkpoint (id mismatch after all retries).
        """
        prep = state.preprocessing

        if cp.id == "duration_completeness" and cp.checkpoint_type == CheckpointType.BINARY:
            total_dur = prep.total_duration_sec if prep else 0
            target_dur = 60.0
            ratio = total_dur / target_dur if target_dur > 0 else 0
            if ratio >= 0.9:
                return (1, f"PASS — Video duration {total_dur:.1f}s meets target "
                           f"~{target_dur:.0f}s (ratio={ratio:.1%}). "
                           f"[Factual override: LLM answered wrong checkpoint]")
            else:
                return (0, f"FAIL — Video duration {total_dur:.1f}s is too short "
                           f"for target ~{target_dur:.0f}s (ratio={ratio:.1%}). "
                           f"[Factual override: LLM answered wrong checkpoint]")

        return None

    # ------------------------------------------------------------------
    # Single checkpoint evaluation (with consistency retry)
    # ------------------------------------------------------------------

    # Max retries when reasoning ↔ score consistency check fails
    _CONSISTENCY_MAX_RETRIES = 2

    def _evaluate_single_checkpoint(
        self,
        cp: CheckpointDef,
        state: GraphState,
        image_paths: list[str] | None = None,
        extra_context: str = "",
        _retry_count: int = 0,
    ) -> CheckpointResult:
        """Evaluate a single checkpoint with a dedicated LLM/VLM call.

        After the LLM responds, a lightweight consistency check verifies
        that the reasoning sentiment matches the score.  If contradictory
        (e.g. positive reasoning + score 1), the call is retried once with
        an explicit correction hint.
        """
        rubric_text = self._build_rubric_text(cp)

        consistency_hint = ""
        if _retry_count > 0:
            consistency_hint = (
                f"\n\nCRITICAL: Your previous response had a problem. "
                f"You MUST evaluate EXACTLY the checkpoint with id=\"{cp.id}\". "
                f"Do NOT evaluate any other checkpoint or topic. "
                f"The question you MUST answer is: \"{cp.question}\"\n"
                f"Also ensure your numeric score MATCHES the sentiment "
                f"of your reasoning. If the quality is good, score high (4-5). "
                f"If the quality is poor, score low (1-2).\n"
            )

        # Determine if this is a visual/VLM checkpoint (images provided)
        is_visual = bool(image_paths)

        # --- Skeptical prompting (defect-first) ---
        # Applied to BOTH visual and text-only checkpoints to counter
        # the general positivity / leniency bias observed in GPT-4o.
        if is_visual:
            skeptical_block = """
CRITICAL — Defect-first evaluation protocol:
This is an AI-GENERATED video. AI video generators commonly produce these
defects: inconsistent faces across shots, morphing/changing clothing,
flickering backgrounds, objects that appear/disappear, unnatural motion,
temporal inconsistencies, and visual artifacts.

You MUST follow this evaluation order:
1. FIRST, carefully examine each frame for defects and inconsistencies.
   Compare faces, clothing, backgrounds, and objects ACROSS frames.
   If characters look different between frames, that IS an inconsistency.
2. LIST the specific defects you observe (or explicitly state "no defects
   found after careful examination" if truly none).
3. ONLY THEN assign a score based on the severity of observed defects.

Scoring calibration for AI-generated video:
• Score 5: Virtually indistinguishable from real footage. Zero defects.
  This is EXTREMELY rare for AI-generated video.
• Score 4: Minor imperfections only visible on close inspection.
• Score 3: Some noticeable issues but overall acceptable quality.
• Score 2: Clear inconsistencies that are immediately apparent.
• Score 1: Severe defects — e.g. different faces across shots, major
  flickering, objects morphing, completely broken continuity.

IMPORTANT: Do NOT give a score of 4 or 5 unless you have carefully
verified there are NO inconsistencies across all provided frames.
When in doubt, score LOWER rather than higher.
"""
        else:
            skeptical_block = """
CRITICAL — Issue-first evaluation protocol:
This is AI-GENERATED content. AI generators commonly produce these issues:
unnatural dialogue timing, robotic intonation, mismatched emotional tone,
abrupt audio cuts, silence gaps, garbled speech, desynchronisation between
audio and visual events, logical inconsistencies, and poor narrative pacing.

You MUST follow this evaluation order:
1. FIRST, identify all problems, flaws, and issues in the content.
2. LIST the specific issues you found (or explicitly state "no issues
   found after careful examination" if truly none).
3. ONLY THEN assign a score based on the severity of issues found.

Scoring calibration for AI-generated content:
• Score 5: Professional-grade quality. Zero issues. EXTREMELY rare.
• Score 4: Minor issues only noticeable on close inspection.
• Score 3: Some noticeable problems but overall acceptable.
• Score 2: Clear quality issues that are immediately apparent.
• Score 1: Severe problems making the content unusable or incoherent.

IMPORTANT: Do NOT default to high scores. A score of 5 means PERFECT
with zero flaws — this is extremely rare for AI-generated content.
When in doubt, score LOWER rather than higher.
"""

        # Value range hint for JSON output
        if cp.checkpoint_type == CheckpointType.BINARY:
            value_hint = "0 or 1"
            reasoning_hint = "explain why pass or fail"
        else:
            value_hint = "int 1-5"
            reasoning_hint = "list issues first, then justify score"

        system_prompt_raw = f"""You are a meticulous, CRITICAL quality evaluator.
Your job is to find problems, not to confirm quality.
Evaluate the following checkpoint and return your assessment.

IMPORTANT — scoring guidance:
• Read each rubric anchor carefully. Pick the level that BEST matches what
  you observe — do NOT default to the middle.
• A perfect video should get 5s; a clearly broken video should get 1s.
• Your score MUST be consistent with your reasoning. If your reasoning
  describes good quality, your score should be high. If your reasoning
  describes poor quality, your score should be low.
• Be STRICT and SKEPTICAL. Err on the side of lower scores when uncertain.
{skeptical_block}{consistency_hint}
{rubric_text}

Return ONLY a JSON object (no markdown fences, no extra text):
{{"id": "{cp.id}", "value": <{value_hint}>, "reasoning": "<{reasoning_hint}>"}}"""

        system_prompt = self.maybe_add_confidence_guidance(system_prompt_raw, state)
        tool_ctx = self.format_tool_context(state, extra_records=self._local_tool_records)
        user_prompt = extra_context + tool_ctx if extra_context else tool_ctx

        # --- Inject factual data for specific checkpoints ---
        # This anchors the LLM to the correct question with objective evidence.
        user_prompt = self._inject_checkpoint_facts(cp, state, user_prompt or "")

        try:
            logger.debug(
                f"[Checkpoints] Single-call for '{cp.id}' "
                f"(retry={_retry_count}) — "
                f"system_prompt length={len(system_prompt)}, "
                f"user_prompt length={len(user_prompt or '')}, "
                f"images={len(image_paths) if image_paths else 0}"
            )

            if image_paths:
                raw = self.llm.vision_evaluate(
                    system_prompt=system_prompt,
                    text_prompt=user_prompt or "Evaluate this frame:",
                    image_paths=image_paths[:16],
                )
            else:
                raw = self.llm.evaluate(system_prompt, user_prompt or "Evaluate:")

            logger.debug(
                f"[Checkpoints] Single-call '{cp.id}' raw response "
                f"(type={type(raw).__name__}): "
                f"{str(raw)[:300]}"
            )

            # Parse — may be a dict directly or wrapped in parse_error
            result_dict = raw
            if isinstance(raw, dict) and raw.get("parse_error"):
                logger.warning(
                    f"[Checkpoints] Single-call '{cp.id}' got parse_error, "
                    f"attempting extraction from raw text"
                )
                extracted = self._extract_json_from_text(raw.get("reasoning", ""))
                if isinstance(extracted, dict):
                    result_dict = extracted

            # --- Validate returned id matches expected checkpoint ---
            returned_id = result_dict.get("id", "")
            if returned_id and returned_id != cp.id:
                logger.warning(
                    f"[Checkpoints] ID MISMATCH for '{cp.id}': "
                    f"LLM returned id='{returned_id}'. "
                    f"Response may be for the wrong checkpoint — "
                    f"retrying if possible."
                )
                if _retry_count < self._CONSISTENCY_MAX_RETRIES:
                    return self._evaluate_single_checkpoint(
                        cp, state, image_paths, extra_context,
                        _retry_count=_retry_count + 1,
                    )
                # All retries exhausted — for BINARY checkpoints with factual
                # answers, compute the answer directly from data instead of
                # trusting the wrong-topic LLM response.
                factual_override = self._factual_override_for_checkpoint(cp, state)
                if factual_override is not None:
                    logger.warning(
                        f"[Checkpoints] ID MISMATCH for '{cp.id}' persists "
                        f"after {_retry_count} retries — using factual override "
                        f"value={factual_override[0]}"
                    )
                    return CheckpointResult(
                        checkpoint_id=cp.id,
                        raw_value=factual_override[0],
                        normalised=float(factual_override[0]),
                        reasoning=factual_override[1],
                        applicable=True,
                    )
                # For non-factual checkpoints, use the result but fix the id
                logger.warning(
                    f"[Checkpoints] ID MISMATCH for '{cp.id}' persists — "
                    f"using response but correcting id"
                )

            raw_val = result_dict.get("value", result_dict.get("score", 3))
            try:
                raw_val = int(float(str(raw_val)))
            except (ValueError, TypeError):
                raw_val = 3

            reasoning = str(result_dict.get("reasoning", result_dict.get("explanation", "")))

            logger.info(
                f"[Checkpoints] '{cp.id}' → value={raw_val}, "
                f"reasoning='{reasoning[:80]}'"
            )

            # --- Consistency check ---
            if _retry_count < self._CONSISTENCY_MAX_RETRIES:
                consistent = self._check_reasoning_score_consistency(
                    cp.id, raw_val, reasoning,
                    checkpoint_type=cp.checkpoint_type,
                )
                if not consistent:
                    logger.warning(
                        f"[Checkpoints] INCONSISTENCY detected for '{cp.id}': "
                        f"value={raw_val} but reasoning is contradictory. "
                        f"Retrying ({_retry_count + 1}/{self._CONSISTENCY_MAX_RETRIES})..."
                    )
                    return self._evaluate_single_checkpoint(
                        cp, state, image_paths, extra_context,
                        _retry_count=_retry_count + 1,
                    )

        except Exception as e:
            logger.warning(f"[Checkpoints] Single-call failed for '{cp.id}': {e}")
            raw_val = 3 if cp.checkpoint_type == CheckpointType.LIKERT else 0
            reasoning = f"Evaluation call failed: {e}"

        # Normalise
        if cp.checkpoint_type == CheckpointType.BINARY:
            # Clamp to 0/1 — LLMs sometimes return 2-5 for binary checkpoints
            if raw_val > 1:
                logger.warning(
                    f"[Checkpoints] BINARY '{cp.id}' returned value={raw_val}, "
                    f"clamping to 1. Reasoning: {reasoning[:80]}"
                )
                raw_val = 1
            raw_val = max(0, min(1, raw_val))
            # Reasoning-based override for BINARY: if LLM says PASS but
            # reasoning is clearly negative, flip to FAIL (and vice versa).
            if raw_val == 1 and reasoning:
                # Check if reasoning contradicts PASS verdict.
                # Key insight: phrases like "does not contain contradictions"
                # are POSITIVE (supporting PASS), so we must exclude negated
                # contexts before counting negative keywords.
                _r_lower = reasoning.lower()

                # Negation patterns that flip the meaning of negative keywords
                _negation_prefixes = [
                    "no ", "not ", "does not ", "doesn't ", "do not ",
                    "don't ", "without ", "free of ", "absence of ",
                    "no sign of ", "no evidence of ", "no indication of ",
                    "never ", "none of ",
                ]
                # Strip sentences that contain negation + negative keyword
                # (these are actually positive statements)
                _neg_defect_kw = [
                    "contradiction", "inconsisten", "mismatch",
                    "fails to", "not maintained", "absent", "missing",
                    "breaks", "broken", "vanish", "disappear",
                    "incoherent", "implausible", "illogical",
                ]
                # Count only genuinely negative hits — skip if preceded
                # by a negation within the same clause (rough: 60 chars)
                _neg_count = 0
                for kw in _neg_defect_kw:
                    idx = _r_lower.find(kw)
                    while idx != -1:
                        # Look back up to 60 chars for a negation prefix
                        window = _r_lower[max(0, idx - 60):idx]
                        if any(neg in window for neg in _negation_prefixes):
                            # Negated context — this is a positive statement
                            pass
                        else:
                            _neg_count += 1
                        idx = _r_lower.find(kw, idx + len(kw))

                if _neg_count >= 2:
                    logger.warning(
                        f"[Checkpoints] BINARY '{cp.id}' value=1 (PASS) but "
                        f"reasoning contains {_neg_count} genuine negative "
                        f"keywords — overriding to FAIL (0). "
                        f"Reasoning: {reasoning[:100]}"
                    )
                    raw_val = 0
            normalised = float(raw_val)
        elif cp.checkpoint_type == CheckpointType.LIKERT:
            raw_val = max(1, min(5, raw_val))
            normalised = (raw_val - 1) / 4.0
        else:
            normalised = float(raw_val)

        return CheckpointResult(
            checkpoint_id=cp.id,
            raw_value=raw_val,
            normalised=normalised,
            reasoning=reasoning,
            applicable=True,
        )

    # Maximum number of checkpoints to attempt in a single batched LLM call.
    # When the count exceeds this, we skip the batch entirely and evaluate
    # each checkpoint individually — empirically, LLMs (especially GPT-4o)
    # struggle to return >2 structured items reliably in one response.
    _BATCH_THRESHOLD = 2

    def _evaluate_checkpoints_batched(
        self,
        checkpoints: list[CheckpointDef],
        state: GraphState,
        image_paths: list[str] | None = None,
        extra_context: str = "",
    ) -> list[CheckpointResult]:
        """Evaluate checkpoints via LLM/VLM calls.

        Strategy:
          • ≤ _BATCH_THRESHOLD checkpoints → try a single batched call,
            with per-checkpoint fallback for any that the batch misses.
          • > _BATCH_THRESHOLD → go straight to individual calls (avoids
            a wasted batch call that empirically returns only 1 item).
        """
        if not checkpoints:
            return []

        if len(checkpoints) > self._BATCH_THRESHOLD:
            # --- Direct individual evaluation (skip batch) ---
            logger.info(
                f"[Checkpoints] {len(checkpoints)} checkpoints > batch threshold "
                f"({self._BATCH_THRESHOLD}), evaluating individually: "
                f"{[cp.id for cp in checkpoints]}"
            )
            return [
                self._evaluate_single_checkpoint(cp, state, image_paths, extra_context)
                for cp in checkpoints
            ]

        # --- Batched evaluation for small checkpoint sets ---
        return self._evaluate_checkpoints_batch_inner(
            checkpoints, state, image_paths, extra_context
        )

    def _evaluate_checkpoints_batch_inner(
        self,
        checkpoints: list[CheckpointDef],
        state: GraphState,
        image_paths: list[str] | None = None,
        extra_context: str = "",
    ) -> list[CheckpointResult]:
        """Batched evaluation for ≤ _BATCH_THRESHOLD checkpoints."""
        logger.info(
            f"[Checkpoints] Batch call for {len(checkpoints)} checkpoint(s): "
            f"{[cp.id for cp in checkpoints]}"
        )

        rubric_block = "\n\n".join(
            f"[Checkpoint {i+1}: {cp.id}]\n{self._build_rubric_text(cp)}"
            for i, cp in enumerate(checkpoints)
        )

        example_items = ",\n  ".join(
            f'{{"id": "{cp.id}", "value": 3, "reasoning": "..."}}'
            for cp in checkpoints
        )

        # Add skeptical prompting for VLM-based batch calls
        is_visual = bool(image_paths)
        skeptical_block = ""
        if is_visual:
            skeptical_block = """
CRITICAL — Defect-first evaluation protocol:
This is an AI-GENERATED video. AI generators commonly produce defects like
inconsistent faces, morphing clothing, flickering backgrounds, objects that
appear/disappear, and unnatural motion.
For EACH checkpoint: first list defects, then score based on severity.
Score 5 = zero defects (extremely rare). When in doubt, score LOWER.

"""

        system_prompt_raw = f"""You are a meticulous, CRITICAL video quality evaluator.
Your job is to find problems, not to confirm quality.
You will be given {len(checkpoints)} evaluation checkpoints, each with a
specific question and a scoring rubric. You MUST evaluate ALL of them.

IMPORTANT — scoring guidance:
• Read each rubric anchor carefully. Pick the level that BEST matches what
  you observe — do NOT default to the middle.
• A perfect video should get 5s; a clearly broken video should get 1s.
  Use the full range.
• Be STRICT and SKEPTICAL. Err on the side of lower scores when uncertain.
{skeptical_block}
{rubric_block}

Return ONLY a JSON array with EXACTLY {len(checkpoints)} objects.
Expected format:
[
  {example_items}
]

Replace the placeholder values (3, "...") with your actual scores and reasoning.
Do NOT wrap in markdown code fences. Do NOT add text before or after."""

        system_prompt = self.maybe_add_confidence_guidance(system_prompt_raw, state)
        tool_ctx = self.format_tool_context(state, extra_records=self._local_tool_records)

        user_prompt = extra_context + tool_ctx if extra_context else tool_ctx

        logger.debug(
            f"[Checkpoints] Batch system_prompt length={len(system_prompt)}, "
            f"user_prompt length={len(user_prompt or '')}, "
            f"images={len(image_paths) if image_paths else 0}"
        )

        if image_paths:
            raw = self.llm.vision_evaluate(
                system_prompt=system_prompt,
                text_prompt=user_prompt or "Evaluate these frames:",
                image_paths=image_paths[:16],
            )
        else:
            raw = self.llm.evaluate(system_prompt, user_prompt or "Evaluate:")

        logger.debug(
            f"[Checkpoints] Batch raw response (type={type(raw).__name__}): "
            f"{str(raw)[:500]}"
        )

        # ----------------------------------------------------------
        # Robust parsing
        # ----------------------------------------------------------
        items: list[dict] = []

        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            if raw.get("parse_error"):
                extracted = self._extract_json_from_text(raw.get("reasoning", ""))
                if isinstance(extracted, list):
                    items = extracted
                elif isinstance(extracted, dict):
                    for key in ("checkpoints", "results", "evaluations"):
                        if key in extracted and isinstance(extracted[key], list):
                            items = extracted[key]
                            break
                    if not items:
                        items = [extracted]
            else:
                for key in ("checkpoints", "results", "evaluations"):
                    if key in raw and isinstance(raw[key], list):
                        items = raw[key]
                        break
                if not items:
                    items = [raw]
        elif isinstance(raw, str):
            extracted = self._extract_json_from_text(raw)
            if isinstance(extracted, list):
                items = extracted
            elif isinstance(extracted, dict):
                items = [extracted]

        # ----------------------------------------------------------
        # Match checkpoints → items, fallback for unmatched
        # ----------------------------------------------------------
        id_map: dict[str, dict] = {}
        id_map_fuzzy: dict[str, dict] = {}
        for it in items:
            if isinstance(it, dict) and "id" in it:
                id_map[it["id"]] = it
                id_map_fuzzy[it["id"].replace("_", "").lower()] = it

        results: list[CheckpointResult] = []
        unmatched_indices: list[int] = []

        for idx, cp in enumerate(checkpoints):
            match = id_map.get(cp.id)
            if match is None:
                match = id_map_fuzzy.get(cp.id.replace("_", "").lower())
            if match is None and idx < len(items) and isinstance(items[idx], dict):
                match = items[idx]

            if match is not None:
                raw_val = match.get("value", match.get("score"))
                if raw_val is None:
                    raw_val = 3
                try:
                    raw_val = int(float(str(raw_val)))
                except (ValueError, TypeError):
                    raw_val = 3

                reasoning = str(match.get("reasoning", match.get("explanation", "")))

                if cp.checkpoint_type == CheckpointType.BINARY:
                    if raw_val > 1:
                        logger.warning(
                            f"[Checkpoints] BINARY '{cp.id}' returned value={raw_val}, "
                            f"clamping to 1"
                        )
                        raw_val = 1
                    raw_val = max(0, min(1, raw_val))
                    normalised = float(raw_val)
                elif cp.checkpoint_type == CheckpointType.LIKERT:
                    raw_val = max(1, min(5, raw_val))
                    normalised = (raw_val - 1) / 4.0
                else:
                    normalised = float(raw_val)

                results.append(CheckpointResult(
                    checkpoint_id=cp.id,
                    raw_value=raw_val,
                    normalised=normalised,
                    reasoning=reasoning,
                    applicable=True,
                ))
            else:
                results.append(None)  # type: ignore[arg-type]
                unmatched_indices.append(idx)

        if unmatched_indices:
            n_total = len(checkpoints)
            n_miss = len(unmatched_indices)
            logger.info(
                f"[Checkpoints] Batch returned {n_total - n_miss}/{n_total} matches. "
                f"Falling back to individual calls for {n_miss} checkpoint(s): "
                f"{[checkpoints[i].id for i in unmatched_indices]}"
            )
            for i in unmatched_indices:
                results[i] = self._evaluate_single_checkpoint(
                    checkpoints[i], state, image_paths, extra_context
                )

        return results

    @staticmethod
    def _aggregate_checkpoint_score(
        checkpoints: list[CheckpointDef],
        results: list[CheckpointResult],
    ) -> float:
        """Weighted average of checkpoint normalised scores.

        Weights are renormalised to sum to 1.0 over the *active* checkpoints.
        """
        if not results:
            return 0.5

        result_map = {r.checkpoint_id: r for r in results if r.applicable}
        total_weight = 0.0
        weighted_sum = 0.0
        for cp in checkpoints:
            r = result_map.get(cp.id)
            if r is None:
                continue
            total_weight += cp.weight
            weighted_sum += cp.weight * r.normalised

        if total_weight < 1e-9:
            return 0.5
        return weighted_sum / total_weight

    def _checkpoint_evaluate(
        self,
        metric_name: str,
        state: GraphState,
        image_paths: list[str] | None = None,
        extra_context: str = "",
        profile: ContentProfile | None = None,
    ) -> tuple[float, float, list["CheckpointResult"], list["CheckpointDef"], ContentProfile]:
        """One-call convenience: profile → filter → batch-evaluate → aggregate.

        Returns (score, confidence, checkpoint_results, active_checkpoints, profile).
        Agents can call this and just build the EvalResult from the tuple.
        """
        from ..checkpoints import CHECKPOINTS

        if profile is None:
            profile = self._build_content_profile(state)

        all_cps = CHECKPOINTS.get(metric_name, [])
        active_cps = self._filter_applicable(all_cps, profile)

        logger.info(
            f"[Checkpoints] _checkpoint_evaluate('{metric_name}'): "
            f"{len(all_cps)} total → {len(active_cps)} active after profile filter"
        )
        if active_cps:
            logger.info(
                f"[Checkpoints]   active IDs: {[cp.id for cp in active_cps]}"
            )

        if not active_cps:
            return 0.5, 0.3, [], [], profile

        cp_results = self._evaluate_checkpoints_batched(
            checkpoints=active_cps,
            state=state,
            image_paths=image_paths,
            extra_context=extra_context,
        )

        score = self._aggregate_checkpoint_score(active_cps, cp_results)
        logger.info(
            f"[Checkpoints] _checkpoint_evaluate('{metric_name}'): "
            f"aggregated score={score:.3f}, "
            f"results=[{', '.join(f'{r.checkpoint_id}={r.raw_value}' for r in cp_results)}]"
        )
        confidence = min(0.9, 0.5 + 0.05 * len(cp_results))

        return score, confidence, cp_results, active_cps, profile

    def _state_key(self) -> str:
        """Map agent_id to the corresponding GraphState field name."""
        mapping = {
            AgentID.SCRIPT_EVAL: "script_results",
            AgentID.VIDEO_EVAL: "video_results",
            AgentID.AUDIO_EVAL: "audio_results",
            AgentID.STABILITY_EVAL: "stability_results",
            AgentID.CROSSMODAL_EVAL: "crossmodal_results",
        }
        return mapping.get(self.agent_id, "errors")
