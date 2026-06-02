"""
llm_utils.py — Shared LLM calling utilities for all agents.

Provides a unified interface for calling GPT-4o (or compatible) models,
with structured output parsing and retry logic.

Supports both standard OpenAI and Azure OpenAI endpoints.
If AZURE_OPENAI_ENDPOINT is set, uses AzureOpenAI; otherwise falls back
to standard OpenAI.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from .config import LLMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

# Substrings that indicate a transient, retryable upstream condition. We match
# on the stringified exception so we don't depend on any particular SDK version
# raising the "right" subclass — Azure / OpenAI / corporate gateways all wrap
# 429s and 5xxs differently.
_RETRYABLE_SUBSTRINGS = (
    "429",
    "qpm limit",
    "rate limit",
    "rate_limit",
    "too many requests",
    "503",
    "502",
    "504",
    "timeout",
    "timed out",
    "connection error",
    "connection reset",
    "temporarily unavailable",
    "service unavailable",
    "internal server error",
    "overloaded",
    "unavailableerror",
)

# These almost certainly won't recover by retrying — fail fast.
_FATAL_SUBSTRINGS = (
    "401",
    "403",
    "authentication",
    "permission",
    "no model permission",
    "invalid_api_key",
    "invalid request",
    "context_length_exceeded",
    "content_filter",
    "model_not_found",
)

# 429 sometimes carries a "Retry-After: <seconds>" hint in the message; honour
# it when we can find one. Look for either explicit "retry-after" headers or
# patterns like "wait 7 s" the gateway might mention.
_RETRY_AFTER_PAT = re.compile(
    r"retry[- _]?after[^0-9]*(\d+(?:\.\d+)?)\s*(s|sec|seconds)?", re.IGNORECASE
)


def _classify_chat_error(exc: BaseException) -> str:
    """Return ``"retryable"`` / ``"fatal"`` / ``"unknown"`` for an LLM error."""
    msg = str(exc).lower()
    if any(tok in msg for tok in _FATAL_SUBSTRINGS):
        return "fatal"
    if any(tok in msg for tok in _RETRYABLE_SUBSTRINGS):
        return "retryable"
    return "unknown"


def _suggested_retry_delay(exc: BaseException) -> float | None:
    m = _RETRY_AFTER_PAT.search(str(exc))
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def _maybe_load_repo_dotenv() -> None:
    """Load repo-level .env into os.environ (best-effort).

    Cursor/CLI runs don't automatically source .env. We do a minimal parse here
    to support AZURE_OPENAI_* and X_TT_LOGID configuration without extra deps.
    Existing environment variables are not overwritten.
    """
    try:
        repo_root = Path(__file__).resolve().parent.parent  # DirectorBench/
        env_path = repo_root / ".env"
        if not env_path.exists():
            return
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        # Never fail evaluation due to dotenv parsing.
        return


def _build_client(config: LLMConfig, proxy: str | bool | None = None):
    """Create an OpenAI or AzureOpenAI client based on environment variables.

    ``proxy`` semantics (see ``directorbench._openai_proxy``):
      * ``None``  → auto from ``DIRECTORBENCH_PROXY`` / ``HTTPS_PROXY`` / ``HTTP_PROXY``
      * ``False`` or ``""`` → explicitly disable proxy, even if env vars are set
      * any URL string → use verbatim
    """
    _maybe_load_repo_dotenv()
    from ._openai_proxy import build_openai_http_client_kwargs

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    tt_logid = (os.getenv("X_TT_LOGID") or "").strip()

    # Long timeout because some Azure deployments (especially gpt-5.x) can
    # take 90+ s for a single response when the gateway is busy. The retry
    # layer in ``LLMClient.chat()`` handles 429 / network blips separately.
    proxy_kwargs = build_openai_http_client_kwargs(
        proxy=proxy, mode="sync", timeout_sec=300.0
    )

    if azure_endpoint and azure_key:
        from openai import AzureOpenAI
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        default_headers = {"X-TT-LOGID": tt_logid} if tt_logid else None
        client = AzureOpenAI(
            api_key=azure_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            default_headers=default_headers,
            **proxy_kwargs,
        )
        # For Azure, the model parameter is the deployment name
        model = os.getenv("AZURE_OPENAI_DEPLOYMENT", config.model)
        logger.info(
            f"[LLMClient] Using Azure OpenAI: endpoint={azure_endpoint}, "
            f"deployment={model}, proxy={'on' if proxy_kwargs else 'off'}"
        )
        return client, model, True
    else:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=config.base_url,
            **proxy_kwargs,
        )
        logger.info(
            f"[LLMClient] Using standard OpenAI: model={config.model}, "
            f"proxy={'on' if proxy_kwargs else 'off'}"
        )
        return client, config.model, False


class LLMClient:
    """Thin wrapper around OpenAI client with evaluation-oriented helpers."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        proxy: str | bool | None = None,
    ):
        self.config = config or LLMConfig()
        self._client, self._model, self._is_azure = _build_client(
            self.config, proxy=proxy
        )

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request and return the assistant's reply."""
        # For Azure OpenAI multimodal gateways, it's often safer to use the
        # "content parts" format for user messages, even for pure text.
        if self._is_azure:
            normalized: list[dict[str, Any]] = []
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if role == "user" and isinstance(content, str):
                    normalized.append(
                        {"role": "user", "content": [{"type": "text", "text": content}]}
                    )
                else:
                    normalized.append(m)  # keep as-is
            messages = normalized  # type: ignore[assignment]

        # Some gateways enforce that `response_format={"type":"json_object"}`
        # can only be used when the prompt explicitly mentions "json".
        if json_mode:
            def _has_json_word(msgs: list[dict[str, Any]]) -> bool:
                for m in msgs:
                    c = m.get("content")
                    if isinstance(c, str) and "json" in c.lower():
                        return True
                    if isinstance(c, list):
                        for part in c:
                            if isinstance(part, dict) and part.get("type") == "text":
                                t = (part.get("text") or "")
                                if isinstance(t, str) and "json" in t.lower():
                                    return True
                return False

            if not _has_json_word(messages):  # type: ignore[arg-type]
                # Prefer system message; fall back to first user text part.
                appended = False
                for m in messages:  # type: ignore[assignment]
                    if m.get("role") == "system" and isinstance(m.get("content"), str):
                        m["content"] = m["content"].rstrip() + "\n\nReturn a valid json object only."
                        appended = True
                        break
                if not appended:
                    for m in messages:  # type: ignore[assignment]
                        if m.get("role") == "user":
                            c = m.get("content")
                            if isinstance(c, str):
                                m["content"] = c.rstrip() + "\n\nReturn a valid json object only."
                                break
                            if isinstance(c, list) and c and isinstance(c[0], dict) and c[0].get("type") == "text":
                                c[0]["text"] = (c[0].get("text") or "").rstrip() + "\n\nReturn a valid json object only."
                                break

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Configurable retry behaviour. The Bytedance Azure gateway returns
        # 429 with `code=-2001` ("qpm limit") even at moderate parallelism;
        # without a retry layer here, an entire checkpoint is silently lost
        # (see eval_debug.log: visual_creativity dropped because of 429).
        max_attempts = max(1, int(os.environ.get("LLM_MAX_ATTEMPTS", "5")))
        backoff_initial = float(os.environ.get("LLM_BACKOFF_INITIAL_SEC", "2.0"))
        backoff_factor = float(os.environ.get("LLM_BACKOFF_FACTOR", "2.0"))
        backoff_cap = float(os.environ.get("LLM_BACKOFF_CAP_SEC", "30.0"))
        # Whether to retry on UNKNOWN (un-classifiable) errors. Default off
        # because fatal misconfigs would otherwise burn time.
        retry_unknown = os.environ.get("LLM_RETRY_UNKNOWN", "0") == "1"

        delay = backoff_initial
        last_exc: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as e:
                last_exc = e
                kind = _classify_chat_error(e)

                if kind == "fatal":
                    logger.error(
                        f"[LLMClient] Fatal error, not retrying: {e}"
                    )
                    raise
                if kind == "unknown" and not retry_unknown:
                    logger.error(
                        f"[LLMClient] Unclassified error, not retrying "
                        f"(set LLM_RETRY_UNKNOWN=1 to retry): {e}"
                    )
                    raise
                if attempt >= max_attempts:
                    logger.error(
                        f"[LLMClient] API call failed after {attempt} "
                        f"attempts: {e}"
                    )
                    raise

                # Honour upstream Retry-After hint when present; otherwise
                # exponential backoff with full jitter (avoids a thundering
                # herd if many parallel cases hit the same QPM ceiling).
                hinted = _suggested_retry_delay(e)
                if hinted is not None:
                    sleep_for = max(hinted, 0.5)
                else:
                    sleep_for = min(delay, backoff_cap) * (0.5 + random.random())
                logger.warning(
                    f"[LLMClient] {kind} error on attempt {attempt}/"
                    f"{max_attempts}; retrying in {sleep_for:.1f}s ({e})"
                )
                time.sleep(sleep_for)
                delay = min(delay * backoff_factor, backoff_cap)

        # Defensive — should be unreachable.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLMClient.chat exited without result")

    # ------------------------------------------------------------------
    # Structured evaluation call
    # ------------------------------------------------------------------

    def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
    ) -> dict[str, Any] | list:
        """
        Call the LLM for evaluation and parse the JSON response.

        Returns the parsed JSON — usually a dict, but can be a list when the
        prompt asks for an array (e.g. batched checkpoint evaluation).
        On parse failure returns ``{"score": 0.5, "reasoning": <raw_text>, "parse_error": True}``.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.chat(messages, json_mode=json_mode)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try stripping markdown fences (```json ... ```)
            import re
            fence_re = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
            m = fence_re.search(raw)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    pass
            logger.warning("[LLMClient] Failed to parse JSON, returning raw text")
            return {"score": 0.5, "reasoning": raw, "parse_error": True}

    # ------------------------------------------------------------------
    # Vision call (for VLM-based evaluation)
    # ------------------------------------------------------------------

    def vision_evaluate(
        self,
        system_prompt: str,
        text_prompt: str,
        image_paths: list[str] | None = None,
        image_urls: list[str] | None = None,
    ) -> dict[str, Any] | list:
        """
        Call GPT-4o with image inputs for visual evaluation.
        Supports both local file paths (base64) and URLs.

        Returns parsed JSON (dict or list).  On parse failure returns
        ``{"score": 0.5, "reasoning": <raw_text>, "parse_error": True}``.
        """
        import base64

        content_parts: list[dict] = [{"type": "text", "text": text_prompt}]

        # Add images from paths
        if image_paths:
            for path in image_paths:
                try:
                    with open(path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    ext = path.rsplit(".", 1)[-1].lower()
                    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}
                    })
                except Exception as e:
                    logger.warning(f"[LLMClient] Failed to load image {path}: {e}")

        # Add images from URLs
        if image_urls:
            for url in image_urls:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "high"}
                })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        raw = self.chat(messages, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try stripping markdown fences (```json ... ```)
            import re
            fence_re = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
            m = fence_re.search(raw)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    pass
            logger.warning("[LLMClient] Failed to parse VLM JSON, returning raw text")
            return {"score": 0.5, "reasoning": raw, "parse_error": True}
