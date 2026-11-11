"""A small JSON REST API over the standard library's HTTP server.

    smart-memory serve --db memory.db --api-key "$SMART_MEMORY_API_KEY"

Endpoints (all JSON; every memory call names the scope with ``user_id`` and optional
``agent_id`` / ``thread_id``):

    GET    /health
    POST   /v1/memories          {user_id, content, memory_type?, key?, pin?, shared?, metadata?}
    POST   /v1/memories/batch    {user_id, contents: [...], memory_type?}
    POST   /v1/search            {user_id, query, k?, memory_types?, since?, until?}
    POST   /v1/context           {user_id, query, max_tokens?, since?}
    POST   /v1/feedback          {user_id, memory_ids, helpful?}
    POST   /v1/forget_user       {user_id, agent_id?}
    DELETE /v1/memories/<id>?user_id=...&agent_id=...

This is meant to sit behind your backend, which decides the ``user_id``. It refuses to
listen on a non-loopback address without an API key.
"""

from __future__ import annotations

import hmac
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .memory import Memory, ScopedMemory
from .storage.base import PermissionDenied

MAX_BODY = 1_000_000
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _scope(mem: Memory, data: Dict[str, Any]) -> ScopedMemory:
    if "user_id" not in data:
        raise ApiError(HTTPStatus.BAD_REQUEST, "user_id is required (use null for the global scope)")
    return mem.scope(data["user_id"], data.get("agent_id"), data.get("thread_id"))


def _record(r: Any) -> Dict[str, Any]:
    return r.to_dict()


def _route(mem: Memory, method: str, path: str, query: Dict[str, Any], body: Dict[str, Any]) -> Tuple[int, Any]:
    if method == "GET" and path == "/health":
        from . import __version__
        return 200, {"status": "ok", "version": __version__}
    if method == "POST" and path == "/v1/memories":
        s = _scope(mem, body)
        r = s.add(body["content"], body.get("memory_type"), key=body.get("key"), pin=bool(body.get("pin")),
                  shared=bool(body.get("shared")), metadata=body.get("metadata"), importance=body.get("importance"))
        return 201, _record(r)
    if method == "POST" and path == "/v1/memories/batch":
        s = _scope(mem, body)
        records = s.add_many(body["contents"], body.get("memory_type"))
        return 201, {"memories": [_record(r) for r in records]}
    if method == "POST" and path == "/v1/search":
        s = _scope(mem, body)
        hits = s.search(body["query"], k=int(body.get("k", 5)), memory_types=body.get("memory_types"),
                        since=body.get("since"), until=body.get("until"))
        return 200, {"results": [{**_record(h.record), "score": h.score, "signals": h.signals} for h in hits]}
    if method == "POST" and path == "/v1/context":
        s = _scope(mem, body)
        card = s.context(body["query"], max_tokens=int(body.get("max_tokens", 800)), since=body.get("since"),
                         until=body.get("until"))
        stable, dynamic = card.render_split()
        return 200, {"context": card.render(), "stable": stable, "dynamic": dynamic,
                     "memory_ids": card.memory_ids, "token_estimate": card.token_estimate}
    if method == "POST" and path == "/v1/feedback":
        s = _scope(mem, body)
        return 200, {"updated": s.feedback(body["memory_ids"], helpful=body.get("helpful", True))}
    if method == "POST" and path == "/v1/forget_user":
        return 200, {"deleted": _scope(mem, body).forget_user()}
    if method == "DELETE" and path.startswith("/v1/memories/"):
        flat = {k: v[0] for k, v in query.items()}
        if not flat.get("user_id"):
            raise ApiError(HTTPStatus.BAD_REQUEST, "user_id query parameter is required")
        deleted = _scope(mem, flat).delete(path.rsplit("/", 1)[1])
        if not deleted:
            raise ApiError(HTTPStatus.NOT_FOUND, "memory not found")
        return 200, {"deleted": True}
    raise ApiError(HTTPStatus.NOT_FOUND, f"No route for {method} {path}")


def make_handler(mem: Memory, api_key: Optional[str]) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "smart-memory"

        def _send(self, status: int, payload: Any) -> None:
            data = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _dispatch(self, method: str) -> None:
            try:
                url = urlparse(self.path)
                if url.path != "/health" and api_key is not None:
                    given = self.headers.get("Authorization", "")
                    if not hmac.compare_digest(given.encode(), f"Bearer {api_key}".encode()):
                        raise ApiError(HTTPStatus.UNAUTHORIZED, "missing or wrong API key")
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body too large")
                body: Dict[str, Any] = {}
                if length:
                    try:
                        body = json.loads(self.rfile.read(length))
                    except json.JSONDecodeError as exc:
                        raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}") from exc
                    if not isinstance(body, dict):
                        raise ApiError(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
                status, payload = _route(mem, method, url.path, parse_qs(url.query), body)
                self._send(status, payload)
            except ApiError as exc:
                self._send(exc.status, {"error": str(exc)})
            except PermissionDenied as exc:
                self._send(HTTPStatus.FORBIDDEN, {"error": str(exc)})
            except (KeyError, ValueError, TypeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:  # noqa: BLE001
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def do_DELETE(self) -> None:  # noqa: N802
            self._dispatch("DELETE")

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default; stderr only
            pass

    return Handler


def create_server(mem: Memory, host: str = "127.0.0.1", port: int = 8765, api_key: Optional[str] = None,
                  allow_insecure: bool = False) -> ThreadingHTTPServer:
    """Build (not start) the server. Call ``serve_forever()`` on the result."""
    if host not in _LOOPBACK and not api_key and not allow_insecure:
        raise ValueError(f"Refusing to listen on {host} without an API key (pass api_key, or allow_insecure=True)")
    return ThreadingHTTPServer((host, port), make_handler(mem, api_key))
