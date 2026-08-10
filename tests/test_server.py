"""
Metis — Unit Tests for the REST API Server (metis/server.py)

Focus: the streaming error-surfacing path. A mid-generation failure (CUDA OOM,
dtype/shape error) must not leave the client hanging for a ``[DONE]`` that
never arrives — it must be captured in the worker thread and re-raised in the
async frame so the SSE stream carries a visible ``error`` event.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import metis.server as server  # noqa: E402

# ── Stubs ─────────────────────────────────────────────────────────────────────

class _StubConfig:
    device = "cpu"


class _StubTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _install_globals(generate_impl):
    """Point the server's model globals + generate_text at a stub."""
    server.model_globals["model"] = object()
    server.model_globals["tokenizer"] = _StubTokenizer()
    server.model_globals["config"] = _StubConfig()
    original = server.generate_text
    server.generate_text = generate_impl
    return original


def _restore(generate_impl):
    server.generate_text = generate_impl
    server.model_globals["model"] = None
    server.model_globals["tokenizer"] = None
    server.model_globals["config"] = None


@pytest.fixture
def no_model():
    server.model_globals["model"] = None
    yield
    server.model_globals["model"] = None


def _broken_generate(*args, **kwargs):
    raise RuntimeError("mid-generation CUDA OOM")


def _collect(async_gen):
    """Run an async generator to completion, returning every yielded chunk."""

    async def _run():
        return [chunk async for chunk in async_gen]

    return asyncio.run(_run())


# ── Streaming error surfacing ────────────────────────────────────────────────

class TestStreamingErrorSurfacing:
    def test_generate_stream_emits_error_event(self):
        """A worker-thread generation failure becomes an SSE error event."""
        original = _install_globals(_broken_generate)
        try:
            req = server.GenerateRequest(prompt="hi", max_tokens=5, stream=True)
            chunks = _collect(server._stream_generator("hi", req))
        finally:
            _restore(original)
        assert any('"error"' in c and "CUDA OOM" in c for c in chunks)
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_openai_chat_stream_emits_error_event(self):
        """The OpenAI-compatible chat stream also surfaces mid-generation errors."""
        original = _install_globals(_broken_generate)
        try:
            gen = server._openai_stream_generator(
                "hi", server.GenerateRequest(prompt="hi", stream=True),
                chat=True, model_name="metis-3.0", req_id="chatcmpl-test",
            )
            chunks = _collect(gen)
        finally:
            _restore(original)
        assert any('"error"' in c and "CUDA OOM" in c for c in chunks)
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_generate_stream_finishes_on_success(self):
        """A healthy generation streams tokens and a final [DONE]."""

        def good_generate(*args, **kwargs):
            kwargs["stream_callback"]("hello")
            return "hello world"

        original = _install_globals(good_generate)
        try:
            req = server.GenerateRequest(prompt="hi", max_tokens=5, stream=True)
            chunks = _collect(server._stream_generator("hi", req))
        finally:
            _restore(original)
        assert any('"token"' in c for c in chunks)
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_stream_without_loaded_model_is_503(self, no_model):
        """Server not loaded → the stream path refuses with a clear error."""
        req = server.GenerateRequest(prompt="hi", max_tokens=5, stream=True)
        chunks = _collect(server._stream_generator("hi", req))
        assert any("Model not loaded" in c for c in chunks)


# ── Endpoint guards ───────────────────────────────────────────────────────────

class TestEndpointGuards:
    def test_generate_requires_loaded_model(self, no_model):
        """Without a loaded model, /generate must 503, not crash."""
        from fastapi.testclient import TestClient

        with TestClient(server.app) as client:
            r = client.post("/generate", json={"prompt": "hi", "max_tokens": 5})
        assert r.status_code == 503

    def test_info_not_loaded(self, no_model):
        from fastapi.testclient import TestClient

        with TestClient(server.app) as client:
            r = client.get("/info")
        assert r.status_code == 200
        assert r.json()["parameters"] == "(not loaded)"
