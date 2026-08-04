"""
Μῆτις (Metis) — Unit Tests for the Teacher API client (metis/teacher.py)

Serves canned OpenAI-compatible responses from a local HTTP server so the
client is tested without any real network or API key.
"""

import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from metis.teacher import (
    MockTeacher,
    TeacherClient,
    TeacherError,
    build_teacher,
)


class _TeacherHandler(BaseHTTPRequestHandler):
    """Serves the next entry from the class-level ``responses`` queue."""

    responses: list[tuple[int, dict]] = []
    seen_paths: list[str] = []
    seen_bodies: list[dict] = []
    seen_auth: list[str | None] = []

    def do_POST(self):  # noqa: N802 (http.server convention)
        self.__class__.seen_paths.append(self.path)
        length = int(self.headers.get("Content-Length", 0))
        self.__class__.seen_bodies.append(
            json.loads(self.rfile.read(length).decode("utf-8"))
        )
        self.__class__.seen_auth.append(self.headers.get("Authorization"))
        if not self.__class__.responses:
            self.__class__.responses = [
                (200, {"choices": [{"message": {"content": "hello world"}}]})
            ]
        status, payload = self.__class__.responses.pop(0)
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep test output clean
        pass


@pytest.fixture()
def server():
    _TeacherHandler.responses = []
    _TeacherHandler.seen_paths = []
    _TeacherHandler.seen_bodies = []
    _TeacherHandler.seen_auth = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _TeacherHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


def _client(httpd) -> TeacherClient:
    return TeacherClient(
        base_url=f"http://127.0.0.1:{httpd.server_port}/v1",
        api_key="test-key",
        model="deepseek-chat",
        max_retries=2,
    )


class TestTeacherClient:
    def test_sends_openai_shape_and_parses_content(self, server):
        client = _client(server)
        _TeacherHandler.responses = [
            (200, {
                "choices": [{"message": {"content": "hello world"}}],
                "usage": {"total_tokens": 17},
            })
        ]
        text = client.complete("sys", "user", max_tokens=64, temperature=0.7)

        assert text == "hello world"
        assert client.last_total_tokens == 17
        assert _TeacherHandler.seen_paths == ["/v1/chat/completions"]
        assert _TeacherHandler.seen_auth == ["Bearer test-key"]
        body = _TeacherHandler.seen_bodies[0]
        assert body["model"] == "deepseek-chat"
        assert body["max_tokens"] == 64
        assert body["temperature"] == 0.7
        assert [m["role"] for m in body["messages"]] == ["system", "user"]

    def test_429_retries_then_succeeds(self, server, monkeypatch):
        monkeypatch.setattr("metis.teacher._sleep_backoff", lambda *a, **k: None)
        client = _client(server)
        _TeacherHandler.responses = [
            (429, {"error": {"message": "rate limited"}}),
            (200, {"choices": [{"message": {"content": "recovered"}}]}),
        ]
        assert client.complete("sys", "user", max_tokens=10) == "recovered"
        assert len(_TeacherHandler.seen_paths) == 2

    def test_server_error_exhausts_retries(self, server, monkeypatch):
        monkeypatch.setattr("metis.teacher._sleep_backoff", lambda *a, **k: None)
        client = _client(server)
        _TeacherHandler.responses = [
            (500, {"error": {"message": "boom"}}),
            (500, {"error": {"message": "boom"}}),
            (500, {"error": {"message": "boom"}}),
        ]
        with pytest.raises(TeacherError):
            client.complete("sys", "user", max_tokens=10)
        assert len(_TeacherHandler.seen_paths) == 3  # initial + 2 retries

    def test_malformed_payload_raises(self, server):
        client = _client(server)
        _TeacherHandler.responses = [(200, {"unexpected": True})]
        with pytest.raises(TeacherError):
            client.complete("sys", "user", max_tokens=10)


class TestMockTeacher:
    def test_returns_rotating_topic_text(self):
        mock = MockTeacher(topics=("cats", "dogs"))
        first = mock.complete("sys", "user", max_tokens=100)
        second = mock.complete("sys", "user", max_tokens=100)
        assert "cats" in first
        assert "dogs" in second
        assert mock.calls == 2
        assert mock.last_total_tokens is None


class TestBuildTeacher:
    def test_requires_api_key(self, monkeypatch):
        from types import SimpleNamespace
        for env in ("METIS_TEACHER_BASE_URL", "METIS_TEACHER_API_KEY",
                    "METIS_TEACHER_MODEL"):
            monkeypatch.delenv(env, raising=False)
        with pytest.raises(TeacherError):
            build_teacher(SimpleNamespace(teacher_base_url=None,
                                          teacher_api_key=None,
                                          teacher_model=None))
