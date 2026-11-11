"""MCP server, REST API and CLI."""

import io
import json
import threading
import urllib.error
import urllib.request

import pytest

from smart_memory import Memory
from smart_memory.cli import main
from smart_memory.http_server import create_server
from smart_memory.mcp_server import MCPServer

# --- MCP ---------------------------------------------------------------------------------


def rpc(server, method, params=None, req_id=1):
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return server.handle(msg)


def test_mcp_handshake_and_tools(alice):
    server = MCPServer(alice)
    init = rpc(server, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                      "clientInfo": {"name": "test", "version": "1"}})
    assert init["result"]["protocolVersion"] == "2025-03-26"
    assert init["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert init["result"]["serverInfo"]["name"] == "smart-memory"
    unknown = rpc(server, "initialize", {"protocolVersion": "1999-01-01"})
    assert unknown["result"]["protocolVersion"] == "2025-06-18"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert rpc(server, "ping") == {"jsonrpc": "2.0", "id": 1, "result": {}}

    tools = rpc(server, "tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == ["remember", "recall", "context"]
    assert all(t["inputSchema"]["type"] == "object" for t in tools)

    saved = rpc(server, "tools/call", {"name": "remember", "arguments": {"content": "Alice prefers email"}})
    payload = json.loads(saved["result"]["content"][0]["text"])
    assert saved["result"]["isError"] is False and payload["memory_type"] == "semantic"
    ctx = rpc(server, "tools/call", {"name": "context", "arguments": {"query": "contact Alice"}})
    assert "Alice prefers email" in json.loads(ctx["result"]["content"][0]["text"])["context"]
    found = rpc(server, "tools/call", {"name": "recall", "arguments": {"query": "email"}})
    assert json.loads(found["result"]["content"][0]["text"])["results"][0]["content"] == "Alice prefers email"


def test_mcp_errors(alice):
    server = MCPServer(alice)
    assert rpc(server, "tools/call", {"name": "forget", "arguments": {"memory_id": "x"}})["error"]["code"] == -32602
    bad_args = rpc(server, "tools/call", {"name": "recall", "arguments": {"wrong": 1}})
    assert bad_args["result"]["isError"] is True
    assert rpc(server, "resources/list")["error"]["code"] == -32601
    assert server.handle({"id": 5, "method": "ping"})["error"]["code"] == -32600  # missing jsonrpc
    assert json.loads(server.handle_line("{not json"))["error"]["code"] == -32700
    assert server.handle_line("   ") is None
    batch = json.loads(server.handle_line(json.dumps([
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ])))
    assert batch == [{"jsonrpc": "2.0", "id": 1, "result": {}}]


def test_mcp_forget_is_opt_in(alice):
    server = MCPServer(alice, allow_forget=True)
    r = alice.add("temporary note about Bob")
    out = rpc(server, "tools/call", {"name": "forget", "arguments": {"memory_id": r.id}})
    assert json.loads(out["result"]["content"][0]["text"]) == {"deleted": True}


def test_mcp_stdio_loop(alice):
    lines = "\n".join(json.dumps(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "remember", "arguments": {"content": "Alice lives in Pune"}}},
    ]) + "\n"
    out = io.StringIO()
    MCPServer(alice).serve(io.StringIO(lines), out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2]
    assert alice.search("Pune")[0].content == "Alice lives in Pune"


def test_mcp_cannot_reach_other_users(mem):
    mem.scope("bob").add("Bob's secret project is Falcon")
    server = MCPServer(mem.scope("alice"))
    out = rpc(server, "tools/call", {"name": "recall", "arguments": {"query": "secret project Falcon"}})
    assert json.loads(out["result"]["content"][0]["text"])["results"] == []


# --- REST --------------------------------------------------------------------------------


@pytest.fixture
def api(clock):
    mem = Memory(clock=clock)
    server = create_server(mem, "127.0.0.1", 0, api_key="s3cret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(method, path, body=None, key="s3cret"):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    yield call
    server.shutdown()
    server.server_close()
    mem.close()


def test_rest_full_flow(api):
    assert api("GET", "/health", key=None)[0] == 200
    status, rec = api("POST", "/v1/memories", {"user_id": "alice", "agent_id": "support",
                                               "content": "Alice prefers email, never phone calls"})
    assert status == 201 and rec["memory_type"] == "semantic" and "embedding" not in rec
    status, batch = api("POST", "/v1/memories/batch", {"user_id": "alice", "agent_id": "support",
                                                       "contents": ["Alice lives in Pune", "Refund API failed today"]})
    assert status == 201 and len(batch["memories"]) == 2
    status, found = api("POST", "/v1/search", {"user_id": "alice", "agent_id": "support", "query": "contact email"})
    assert status == 200 and found["results"][0]["id"] == rec["id"] and "signals" in found["results"][0]
    status, card = api("POST", "/v1/context", {"user_id": "alice", "agent_id": "support", "query": "refund Alice"})
    assert status == 200 and card["context"].startswith("## Memory") and card["memory_ids"]
    assert api("POST", "/v1/feedback", {"user_id": "alice", "agent_id": "support",
                                        "memory_ids": [rec["id"]]}) == (200, {"updated": 1})
    status, _ = api("DELETE", f"/v1/memories/{rec['id']}?user_id=alice&agent_id=support")
    assert status == 200
    assert api("DELETE", f"/v1/memories/{rec['id']}?user_id=alice&agent_id=support")[0] == 404
    assert api("POST", "/v1/forget_user", {"user_id": "alice", "agent_id": "support"}) == (200, {"deleted": 2})


def test_rest_security_and_validation(api):
    assert api("POST", "/v1/search", {"user_id": "a", "query": "x"}, key=None)[0] == 401
    assert api("POST", "/v1/search", {"user_id": "a", "query": "x"}, key="wrong")[0] == 401
    assert api("POST", "/v1/search", {"query": "x"})[0] == 400  # user_id required
    assert api("POST", "/v1/memories", {"user_id": "a", "content": "  "})[0] == 400
    assert api("POST", "/v1/memories", {"user_id": "a"})[0] == 400
    assert api("POST", "/v1/nope", {"user_id": "a"})[0] == 404
    assert api("DELETE", "/v1/memories/abc")[0] == 400
    _, rec = api("POST", "/v1/memories", {"user_id": "alice", "content": "Alice secret"})
    assert api("DELETE", f"/v1/memories/{rec['id']}?user_id=bob")[0] == 404  # invisible to bob
    _, hits = api("POST", "/v1/search", {"user_id": "bob", "query": "Alice secret"})
    assert hits["results"] == []


def test_rest_refuses_public_bind_without_key():
    with pytest.raises(ValueError, match="API key"):
        create_server(Memory(), "0.0.0.0", 0)
    server = create_server(Memory(), "0.0.0.0", 0, allow_insecure=True)
    server.server_close()


# --- CLI ---------------------------------------------------------------------------------


def test_cli_round_trip(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["add", "--db", db, "--user", "me", "Alice prefers email, never phone calls"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "preference"
    assert main(["add", "--db", db, "--user", "me", "--pin", "Alice is a premium customer"]) == 0
    capsys.readouterr()
    assert main(["search", "--db", db, "--user", "me", "contact email"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["content"].startswith("Alice prefers email")
    assert main(["context", "--db", db, "--user", "me", "contact Alice"]) == 0
    assert "### Profile" in capsys.readouterr().out
    assert main(["stats", "--db", db]) == 0
    assert json.loads(capsys.readouterr().out)["active_by_type"] == {"semantic": 2}
    assert main(["export", "--db", db, str(tmp_path / "out.jsonl")]) == 0
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 2
    assert main(["audit", "--db", db, str(tmp_path / "audit.jsonl"), "--user", "me"]) == 0
    assert main(["search", "--db", db, "--user", "me", "x", "--since", "not-a-time"]) == 2


def test_cli_redact_and_mcp(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "cli.db")
    main(["add", "--db", db, "--user", "me", "--redact", "my email is me@example.com"])
    assert "[EMAIL]" in json.loads(capsys.readouterr().out)["content"]
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(request))
    assert main(["mcp", "--db", db, "--user", "me"]) == 0
    assert len(json.loads(capsys.readouterr().out)["result"]["tools"]) == 3
    with pytest.raises(SystemExit):
        main(["--version"])
