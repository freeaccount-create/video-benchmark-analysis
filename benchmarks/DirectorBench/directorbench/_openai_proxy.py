"""Shared helper for building OpenAI / AzureOpenAI clients that respect a
corporate proxy.

Why this exists
---------------
The OpenAI Python SDK (>=1.0) uses ``httpx`` underneath. ``httpx`` *does*
respect ``HTTP_PROXY`` / ``HTTPS_PROXY`` env vars when ``trust_env=True``
(its default), so in many setups simply exporting these env vars before
running DirectorBench is enough.

But there are two recurring failure modes we want to cover programmatically:

1. **Some openai-python releases / corporate forks build the underlying
   ``httpx.Client`` with ``trust_env=False``** (or with a custom
   transport that ignores env vars). On those installs, exporting
   ``HTTPS_PROXY`` looks like it's working but the SDK actually
   bypasses it and produces ``APIConnectionError: Connection error``
   from inside a sandbox that has no direct egress (this is exactly
   what burned ViMax when we wired the OpenAI image / video adapters).
2. **You want to drive the proxy from configuration**, not from a shell
   ``export``. For DirectorBench we expose ``DIRECTORBENCH_PROXY`` in
   ``.env``. For ViMax we drive it from YAML. In both cases we pass
   the resolved URL through to ``httpx`` explicitly so it doesn't depend
   on whatever ``trust_env`` value the SDK happens to use.

Resolution order
----------------

    explicit ``proxy=...`` kwarg
        > ``DIRECTORBENCH_PROXY`` env var          (project-scoped escape hatch)
        > ``HTTPS_PROXY`` env var
        > ``HTTP_PROXY`` env var
        > no proxy

``proxy=False`` or ``proxy=""`` explicitly *disables* the proxy, even when
``HTTPS_PROXY`` is set in the environment (useful for endpoints that live
inside the corporate network and would break going through the proxy).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


def resolve_proxy(proxy: Optional[Union[str, bool]] = None) -> Optional[str]:
    """Pick the proxy URL to use, given an explicit override and process env.

    See module docstring for resolution order.
    """
    if proxy is False or proxy == "":
        return None
    if isinstance(proxy, str) and proxy:
        return proxy
    return (
        os.environ.get("DIRECTORBENCH_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or None
    )


def build_openai_http_client_kwargs(
    proxy: Optional[Union[str, bool]] = None,
    *,
    mode: str = "sync",
    timeout_sec: float = 120.0,
) -> dict[str, Any]:
    """Return ``{"http_client": <httpx.[Async]Client>}`` ready to pass to
    ``OpenAI(...)`` / ``AzureOpenAI(...)`` / ``AsyncOpenAI(...)``.

    The openai-python SDK accepts a single ``http_client`` kwarg whose type
    must match the SDK class:

      * ``OpenAI``       → ``httpx.Client``
      * ``AzureOpenAI``  → ``httpx.Client``
      * ``AsyncOpenAI``  → ``httpx.AsyncClient``

    ``mode``: ``"sync"`` (default) or ``"async"``.

    If no proxy is in effect we return an empty dict ``{}`` so callers can
    splat the result with ``**`` and leave the SDK to use its built-in
    default httpx client unchanged.
    """
    proxy_url = resolve_proxy(proxy)
    if proxy_url is None:
        if proxy is False:
            logger.info("[openai_proxy] proxy explicitly disabled by config")
        return {}

    # Lazy import — httpx is a transitive dep of openai-python anyway.
    import httpx

    logger.info(f"[openai_proxy] using proxy={proxy_url} (mode={mode})")
    timeout = httpx.Timeout(timeout_sec, connect=30.0)
    cls = httpx.AsyncClient if mode == "async" else httpx.Client

    # ``proxy=`` (singular) is the modern httpx >=0.26 spelling. For older
    # httpx (0.25.x) we fall back to ``proxies=`` if needed.
    try:
        client = cls(proxy=proxy_url, timeout=timeout, trust_env=False)
    except TypeError:
        client = cls(proxies=proxy_url, timeout=timeout, trust_env=False)

    return {"http_client": client}
