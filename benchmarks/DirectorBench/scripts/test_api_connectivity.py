#!/usr/bin/env python3
"""
test_api_connectivity.py — End-to-end connectivity check for the external
APIs DirectorBench depends on.

Tests, in order:
  1. (Azure) OpenAI Chat Completions   — used by every evaluation agent
  2. OpenAI Audio Transcriptions       — used by Preprocessor for ASR
  3. AudioShake                        — used by Preprocessor for vocals/BGM
                                         separation

For each endpoint we:
  - show whether the relevant env vars are set (key values are redacted),
  - send the smallest possible real request,
  - report status / latency / a short detail string.

Usage:
    python scripts/test_api_connectivity.py
    python scripts/test_api_connectivity.py --runs 3            # repeat each probe N times
    python scripts/test_api_connectivity.py --skip audioshake   # skip a probe
    python scripts/test_api_connectivity.py --audioshake-full   # also wait for the
                                                                # AudioShake task to
                                                                # finish (slow)

Exit code is 0 only when every non-skipped probe passes.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def _hr(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")


def _row(name: str, status: str, elapsed_ms: float, detail: str = "") -> None:
    icon = {"PASS": "[OK]  ", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}.get(status, "[??]  ")
    print(f"{icon} {name:<32} {elapsed_ms:>8.1f} ms   {detail}")


def _redact(secret: str | None) -> str:
    if not secret:
        return "<unset>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]} (len={len(secret)})"


# ---------------------------------------------------------------------------
# .env loader (so the script works the same as run.sh)
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
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


# ---------------------------------------------------------------------------
# Tiny audio fixture (1 s of 440 Hz tone, mono, 16 kHz) — enough to exercise
# AudioShake & OpenAI ASR without relying on a real video.
# ---------------------------------------------------------------------------

def _make_tone_wav(out_path: Path, duration_sec: float = 1.0, sr: int = 16000) -> Path:
    n = int(duration_sec * sr)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            v = int(0.3 * math.sin(2 * math.pi * 440.0 * (i / sr)) * 32767)
            wf.writeframes(struct.pack("<h", v))
    return out_path


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe_azure_or_openai_chat(verbose: bool, proxy: str | bool | None) -> tuple[str, str]:
    """Smallest possible chat completion via the production LLMClient.

    We deliberately route through ``directorbench.llm_utils.LLMClient`` so
    this test exercises the exact same code path (Azure content-parts
    formatting, X-TT-LOGID header injection, proxy wiring, etc.) that the
    agents use at runtime. If this passes, the LLM calls inside agents
    will pass too.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not (azure_endpoint and azure_key) and not openai_key:
        return "SKIP", "neither AZURE_OPENAI_* nor OPENAI_API_KEY is set"

    from directorbench.llm_utils import LLMClient
    from directorbench._openai_proxy import resolve_proxy

    client = LLMClient(proxy=proxy)
    if verbose:
        provider = "Azure OpenAI" if client._is_azure else "OpenAI"
        resolved = resolve_proxy(proxy)
        print(f"   provider   = {provider}")
        print(f"   model/dep. = {client._model}")
        if azure_endpoint:
            print(f"   endpoint   = {azure_endpoint}")
            print(f"   api_version= {os.getenv('AZURE_OPENAI_API_VERSION', '<default>')}")
            print(f"   key        = {_redact(azure_key)}")
        else:
            print(f"   key        = {_redact(openai_key)}")
        print(f"   proxy      = {resolved or '(none)'}")

    reply = client.chat(
        messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
        max_tokens=10,
    ).strip()
    if not reply:
        return "FAIL", "empty reply from chat endpoint"
    return "PASS", f"reply={reply[:40]!r}"


def probe_openai_asr(
    audio_path: Path, verbose: bool, proxy: str | bool | None
) -> tuple[str, str]:
    """Tiny transcription request to verify the ASR endpoint is reachable.

    Routes through the same proxy helper as the production
    ``Preprocessor._run_asr_openai`` codepath.
    """
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not openai_key:
        return "SKIP", "OPENAI_API_KEY is not set"

    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    model = (os.getenv("OPENAI_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe").strip()

    from openai import OpenAI
    from directorbench._openai_proxy import build_openai_http_client_kwargs, resolve_proxy

    request_timeout = float(os.environ.get("OPENAI_ASR_TIMEOUT_SEC", "30"))
    proxy_kwargs = build_openai_http_client_kwargs(
        proxy=proxy, mode="sync", timeout_sec=request_timeout
    )

    if verbose:
        print(f"   provider   = openai.com")
        print(f"   model      = {model}")
        if base_url:
            print(f"   base_url   = {base_url}")
        print(f"   key        = {_redact(openai_key)}")
        print(f"   proxy      = {resolve_proxy(proxy) or '(none)'}")

    opts: dict = dict(api_key=openai_key, timeout=request_timeout, max_retries=0)
    if base_url:
        opts["base_url"] = base_url
    opts.update(proxy_kwargs)
    client = OpenAI(**opts)
    with audio_path.open("rb") as f:
        result = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="json",
        )
    text = (getattr(result, "text", None) or "").strip()
    return "PASS", f"text={text[:40]!r} (len={len(text)})"


def probe_audioshake(audio_path: Path, verbose: bool, full: bool) -> tuple[str, str]:
    """Upload a tiny WAV to AudioShake and (optionally) wait for the task."""
    api_key = (os.getenv("AUDIOSHAKE_API_KEY") or "").strip()
    if not api_key:
        return "SKIP", "AUDIOSHAKE_API_KEY is not set"

    base_url = "https://api.audioshake.ai"
    headers = {"x-api-key": api_key, "Accept": "application/json"}

    import requests as _requests
    use_no_proxy = (os.environ.get("AUDIOSHAKE_NO_PROXY", "").strip().lower()
                    in {"1", "true", "yes", "on"})
    session = _requests.Session()
    if use_no_proxy:
        session.trust_env = False

    if verbose:
        print(f"   base_url   = {base_url}")
        print(f"   key        = {_redact(api_key)}")
        print(f"   no_proxy   = {use_no_proxy}")
        print(f"   audio      = {audio_path} ({audio_path.stat().st_size} bytes)")

    # 1) Upload
    with audio_path.open("rb") as f:
        r = session.post(
            f"{base_url}/assets",
            headers=headers,
            files={"file": (audio_path.name, f, "audio/wav")},
            timeout=60,
        )
    r.raise_for_status()
    asset_id = r.json().get("id")
    if not asset_id:
        return "FAIL", f"upload returned no asset id: {r.text[:120]}"

    # 2) Create task
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
        timeout=30,
    )
    r.raise_for_status()
    task_id = r.json().get("id")
    if not task_id:
        return "FAIL", f"task create returned no task id: {r.text[:120]}"

    if not full:
        return "PASS", f"asset_id={asset_id}, task_id={task_id} (not awaited)"

    # 3) Poll until completed/failed (only when --audioshake-full)
    deadline = time.time() + 180
    last_status = "?"
    while time.time() < deadline:
        r = session.get(f"{base_url}/tasks/{task_id}", headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        last_status = str(data.get("status") or "").lower()
        if not last_status:
            target_statuses = [str(t.get("status", "")).lower() for t in data.get("targets", []) or []]
            if target_statuses and all(s == "completed" for s in target_statuses):
                last_status = "completed"
            elif any(s == "failed" for s in target_statuses):
                last_status = "failed"
        if last_status in {"completed", "failed"}:
            break
        time.sleep(3)

    if last_status == "completed":
        return "PASS", f"task_id={task_id} status=completed"
    return "FAIL", f"task_id={task_id} status={last_status}"


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------

def _run_probe(name: str, fn: Callable[[], tuple[str, str]], runs: int) -> str:
    """Invoke ``fn`` ``runs`` times, print each result, return the worst status."""
    statuses: list[str] = []
    latencies: list[float] = []
    for i in range(1, runs + 1):
        t0 = time.perf_counter()
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed)
        statuses.append(status)
        label = name if runs == 1 else f"{name} [{i}/{runs}]"
        _row(label, status, elapsed, detail)

    # Aggregate
    worst = "PASS"
    for s in statuses:
        if s == "FAIL":
            worst = "FAIL"
            break
        if s == "SKIP" and worst == "PASS":
            worst = "SKIP"
    if runs > 1:
        ok = [l for l, s in zip(latencies, statuses) if s == "PASS"]
        if ok:
            print(
                f"   ↳ aggregate: pass={statuses.count('PASS')}/{runs} "
                f"min={min(ok):.1f} median={statistics.median(ok):.1f} "
                f"max={max(ok):.1f} ms"
            )
    return worst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="DirectorBench external API connectivity check")
    parser.add_argument("--runs", type=int, default=1,
                        help="Repeat each probe N times to test stability")
    parser.add_argument("--skip", action="append", default=[],
                        choices=["chat", "asr", "audioshake"],
                        help="Skip a specific probe (repeatable)")
    parser.add_argument("--audioshake-full", action="store_true",
                        help="Wait for the AudioShake task to finish (slow)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print env / endpoint info before each probe")
    parser.add_argument(
        "--proxy",
        default=None,
        help=(
            "Override proxy used by OpenAI / AzureOpenAI / ASR clients. "
            "Resolution: --proxy > DIRECTORBENCH_PROXY > HTTPS_PROXY > HTTP_PROXY > none. "
            "Pass --proxy '' (empty) or --proxy off to explicitly disable."
        ),
    )
    args = parser.parse_args()

    _load_dotenv()

    # Normalise --proxy into the bool/str/None tri-state expected by the helper.
    proxy_arg: str | bool | None
    if args.proxy is None:
        proxy_arg = None  # auto-resolve from env
    elif args.proxy.strip() == "" or args.proxy.strip().lower() in {"off", "false", "no"}:
        proxy_arg = False
    else:
        proxy_arg = args.proxy.strip()

    # Resolve once for display purposes; helper does this internally too.
    from directorbench._openai_proxy import resolve_proxy
    resolved_proxy = resolve_proxy(proxy_arg)

    _hr("DirectorBench API Connectivity Check")
    print(f"Python : {sys.version.split()[0]}")
    print(f"Env    : .env loaded={bool((REPO_ROOT / '.env').exists())}")
    print(f"Probes : runs={args.runs} skip={args.skip or 'none'} "
          f"audioshake_full={args.audioshake_full}")
    print(f"Proxy  : {resolved_proxy or '(none)'}"
          + (f"  [explicit --proxy={args.proxy!r}]" if args.proxy is not None else "  [from env]"))

    skip = set(args.skip)
    overall_statuses: list[str] = []

    with tempfile.TemporaryDirectory(prefix="db-conn-test-") as td:
        tmp = Path(td)
        tone_wav = _make_tone_wav(tmp / "tone.wav", duration_sec=1.0)

        # 1) (Azure) OpenAI Chat
        _hr("1/3  (Azure) OpenAI Chat Completions")
        if "chat" in skip:
            _row("openai.chat.completions", "SKIP", 0.0, "skipped via --skip chat")
            overall_statuses.append("SKIP")
        else:
            overall_statuses.append(_run_probe(
                "openai.chat.completions",
                lambda: probe_azure_or_openai_chat(args.verbose, proxy_arg),
                args.runs,
            ))

        # 2) OpenAI Audio Transcriptions
        _hr("2/3  OpenAI Audio Transcriptions (ASR)")
        if "asr" in skip:
            _row("openai.audio.transcriptions", "SKIP", 0.0, "skipped via --skip asr")
            overall_statuses.append("SKIP")
        else:
            overall_statuses.append(_run_probe(
                "openai.audio.transcriptions",
                lambda: probe_openai_asr(tone_wav, args.verbose, proxy_arg),
                args.runs,
            ))

        # 3) AudioShake
        _hr("3/3  AudioShake (vocals / instrumental separation)")
        if "audioshake" in skip:
            _row("audioshake", "SKIP", 0.0, "skipped via --skip audioshake")
            overall_statuses.append("SKIP")
        else:
            overall_statuses.append(_run_probe(
                "audioshake",
                lambda: probe_audioshake(tone_wav, args.verbose, args.audioshake_full),
                args.runs,
            ))

    # Summary
    _hr("Summary")
    n_pass = overall_statuses.count("PASS")
    n_skip = overall_statuses.count("SKIP")
    n_fail = overall_statuses.count("FAIL")
    print(f"PASS={n_pass}  SKIP={n_skip}  FAIL={n_fail}  (out of {len(overall_statuses)})")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
