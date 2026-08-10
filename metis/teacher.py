"""
Μῆτις (Metis) — Teacher API client for distillation
=====================================================
A tiny, dependency-free HTTP client that asks a frontier teacher model to
*write training text* for Metis. The teacher is reached through any
OpenAI-compatible ``/v1/chat/completions`` endpoint — e.g. a custom
"omniroute" gateway that fronts ChatGPT / DeepSeek / Claude-class models.

This is the "text distillation" path: the teacher writes prose, Metis trains
on it with its normal next-token objective (see ``metis/distill.py``).
Token-level logit distillation is impossible across different tokenizers, so
we never try to align vocabularies.

Only the Python standard library is used (``urllib``) — no ``requests``, no
``openai`` SDK — consistent with the project's dependency-lean, from-scratch
philosophy.

Configuration (env vars, mirrored by ``metis distill --teacher-*`` flags):
    METIS_TEACHER_BASE_URL   e.g. https://your-gateway.example/v1
    METIS_TEACHER_API_KEY    the gateway's API key
    METIS_TEACHER_MODEL      e.g. deepseek-chat, gpt-4o, claude-sonnet-4
    METIS_TEACHER_TIMEOUT    seconds per teacher call (default 240 — routers
                             serving 1000-token generations at ~5 tok/s need
                             ~200s, so keep the default generous)
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request

logger = logging.getLogger("metis.teacher")

ENV_BASE_URL = "METIS_TEACHER_BASE_URL"
ENV_API_KEY = "METIS_TEACHER_API_KEY"
ENV_MODEL = "METIS_TEACHER_MODEL"
ENV_TIMEOUT = "METIS_TEACHER_TIMEOUT"

DEFAULT_BASE_URL = "https://api.omniroute.example/v1"  # placeholder - override it
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 240  # see METIS_TEACHER_TIMEOUT note above

_RETRYABLE = (429, 500, 502, 503, 504)
_QUOTA_MARKER = "insufficient_quota"


class TeacherError(RuntimeError):
    """Raised when the teacher API cannot be reached or returns garbage."""


def _is_retryable(status: int, body: str) -> bool:
    """Is this a transient error worth backing off and retrying?

    Standard OpenAI retryables are 429/5xx. OmniRoute-class gateways also
    surface their per-model request queue as a 403 whose body says
    ``insufficient_quota`` — that is a rate limit, not a billing wall, and
    must be retried or a "train forever" loop dies the first time the model's
    queue is briefly saturated.
    """
    if status in _RETRYABLE:
        return True
    return status == 403 and _QUOTA_MARKER in body


def _sleep_backoff(attempt: int, status: int | None) -> None:
    """Exponential backoff between retries (longer for rate limits)."""
    base = 2.0 if status in (429, 403) else 1.0
    time.sleep(base * (2 ** attempt))


def _extract_content(payload: dict) -> str:
    """Pull the assistant text out of an OpenAI chat/completions response."""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise TeacherError(
            f"Unexpected teacher response shape: {json.dumps(payload)[:300]}"
        ) from e
    if not isinstance(content, str):
        raise TeacherError(f"Teacher content is not text: {content!r}")
    return content


class TeacherClient:
    """Minimal OpenAI-compatible chat/completions client.

    Contract (the de-facto standard for model routers):
        POST {base_url}/chat/completions
        Authorization: Bearer {api_key}
        Body: {"model", "messages", "max_tokens", "temperature"}
        Response: {"choices": [{"message": {"content": "..."}}],
                   "usage": {"total_tokens": N}}   (usage optional)

    ``complete()`` exposes the response's ``usage.total_tokens`` (when the
    gateway reports it) via ``self.last_total_tokens`` so the distillation
    loop can track API cost.
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: int = DEFAULT_TIMEOUT, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_total_tokens: int | None = None

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int, temperature: float = 0.9) -> str:
        """Ask the teacher to write text; returns the content string."""
        self.last_total_tokens = None
        url = f"{self.base_url}/chat/completions"
        # ``stream=false`` is explicit: many omniroute-class gateways default to
        # SSE streaming, which would break the plain-JSON parse below.
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # Cloudflare in front of a gateway blocks the default
            # ``Python-urllib`` User-Agent with a 1010 Browser Integrity Check.
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                usage = payload.get("usage") or {}
                self.last_total_tokens = usage.get("total_tokens")
                return _extract_content(payload)
            except urllib.error.HTTPError as e:
                last_err = e
                detail = e.read().decode("utf-8", "replace")[:300]
                if _is_retryable(e.code, detail) and attempt < self.max_retries:
                    logger.warning(
                        f"Teacher HTTP {e.code}, retry {attempt + 1}/{self.max_retries}"
                    )
                    _sleep_backoff(attempt, e.code)
                    continue
                raise TeacherError(f"Teacher HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"Teacher unreachable ({e}), retry {attempt + 1}/{self.max_retries}"
                    )
                    _sleep_backoff(attempt, None)
                    continue
                raise TeacherError(f"Teacher unreachable: {e}") from e
        raise TeacherError(f"Teacher request failed after retries: {last_err}")


class MockTeacher:
    """Offline stand-in for :class:`TeacherClient` — returns canned text.

    Used by ``--mock``, the test suite, and CI so the distillation loop runs
    with no network and no API key. ``complete()`` returns a topic-themed
    paragraph repeated enough to fill ``max_tokens``.
    """

    _SENTENCE = (
        "The {topic} is an interesting subject to study and understand. "
        "Learning about {topic} builds knowledge and curiosity. "
        "Every detail about {topic} matters for a complete picture of the world."
    )

    def __init__(self, topics: tuple[str, ...] = ("general knowledge",)):
        self._topics = list(topics)
        self._i = 0
        self.calls = 0
        self.last_total_tokens: int | None = None

    def complete(self, system_prompt: str, user_prompt: str,
                 max_tokens: int, temperature: float = 0.9) -> str:
        self.calls += 1
        topic = self._topics[self._i % len(self._topics)]
        self._i += 1
        repeats = max(1, max_tokens // 48)
        parts = []
        for r in range(repeats):
            t = topic if r % 2 == 0 else f"{topic} and related ideas"
            parts.append(self._SENTENCE.format(topic=t))
        return "\n\n".join(parts)


def build_teacher(opts) -> TeacherClient:
    """Resolve teacher settings (CLI flag > env var > default) and connect."""
    base_url = (getattr(opts, "teacher_base_url", None)
                or os.environ.get(ENV_BASE_URL)
                or DEFAULT_BASE_URL)
    api_key = (getattr(opts, "teacher_api_key", None)
               or os.environ.get(ENV_API_KEY))
    model = (getattr(opts, "teacher_model", None)
             or os.environ.get(ENV_MODEL)
             or DEFAULT_MODEL)
    if not api_key:
        raise TeacherError(
            f"Teacher API key not set. Export {ENV_API_KEY} "
            f"(or pass --teacher-api-key)."
        )
    if not base_url or base_url.endswith(".example/v1"):
        raise TeacherError(
            f"Teacher base URL looks like a placeholder: {base_url!r}. "
            f"Set {ENV_BASE_URL} to your omniroute gateway URL."
        )
    timeout = (getattr(opts, "teacher_timeout", None)
               or os.environ.get(ENV_TIMEOUT)
               or DEFAULT_TIMEOUT)
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT
    return TeacherClient(base_url, api_key, model, timeout=timeout)
