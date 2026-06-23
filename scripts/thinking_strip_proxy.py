#!/usr/bin/env python3
"""Thinking-block stripping middleware proxy.

Sits between Claude Code and a litellm proxy serving non-Anthropic models
(GLM-5.2, Qwen, etc.) via sglang/vLLM.

Problem:
  litellm translates Anthropic Messages API <-> OpenAI Chat Completions.
  GLM-5.2 (on sglang) always returns reasoning_content, which litellm
  converts to Anthropic {"type":"thinking",...} blocks in the response.
  On the NEXT turn, Claude Code sends these thinking blocks back as part
  of the assistant message history -> litellm forwards them to sglang's
  OpenAI endpoint -> sglang rejects thinking block type (not in OpenAI
  schema) -> HTTP 400.

Fix:
  This proxy intercepts /v1/messages and strips ALL thinking /
  redacted_thinking content blocks from:
  - Request message history (so litellm never sees them)
  - Response message content (so Claude Code never stores them)
  Both streaming (SSE) and non-streaming responses are handled.

Usage:
  python3 scripts/thinking_strip_proxy.py <upstream_base_url> [port]

  upstream_base_url  e.g. https://litellm-prod.toolsfdg.net
  port               default 18769

  Then set ANTHROPIC_BASE_URL=http://127.0.0.1:18769 in autoresearch.sh.

Pure stdlib -- no dependencies, works with system python3.
"""
import http.server
import json
import socketserver
import sys
import urllib.error
import urllib.request

DEFAULT_PORT = 18769
STRIP_TYPES = frozenset(["thinking", "redacted_thinking"])


def strip_thinking_blocks(content):
    """Remove thinking/redacted_thinking blocks from a content field."""
    if isinstance(content, str) or content is None:
        return content
    if not isinstance(content, list):
        return content
    return [b for b in content if b.get("type") not in STRIP_TYPES]


def process_request_body(body_bytes):
    """Strip thinking blocks from request body. Returns new bytes."""
    try:
        body = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body_bytes
    msgs = body.get("messages")
    if isinstance(msgs, list):
        for msg in msgs:
            if isinstance(msg, dict) and "content" in msg:
                msg["content"] = strip_thinking_blocks(msg["content"])
    return json.dumps(body).encode()


def process_response_json(data):
    """Strip thinking blocks from a parsed JSON response."""
    if isinstance(data, dict):
        content = data.get("content")
        if isinstance(content, list):
            data["content"] = strip_thinking_blocks(content)
    return data


class StripProxyHandler(http.server.BaseHTTPRequestHandler):
    upstream_base: str = ""
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        sys.stderr.write(
            f"[proxy] {self.command} {self.path} "
            f"{args[1] if len(args) > 1 else ''}\n"
        )

    def _do_forward(self):
        path = self.path
        query = ""
        if "?" in path:
            path, query = path.split("?", 1)

        upstream = self.upstream_base.rstrip("/")
        url = f"{upstream}{path}"
        if query:
            url = f"{url}?{query}"

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        is_messages = "/v1/messages" in path
        if body and is_messages:
            body = process_request_body(body)

        req = urllib.request.Request(
            url, data=body if body else None, method=self.command
        )
        skip = {"host", "content-length", "transfer-encoding", "connection"}
        for key, val in self.headers.items():
            if key.lower() not in skip:
                req.add_header(key, val)
        if body:
            req.add_header("Content-Length", str(len(body)))

        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            self._forward_error(e)
            return
        except Exception as e:
            self._send_json_error(502, f"proxy upstream error: {e}")
            return

        content_type = resp.headers.get("Content-Type", "")
        is_stream = "text/event-stream" in content_type

        if is_stream and is_messages:
            self._proxy_stream(resp)
        else:
            self._proxy_plain(resp, strip=is_messages)

    def _proxy_plain(self, resp, strip=False):
        """Non-streaming: read full body, optionally strip, send back."""
        body = resp.read()
        if strip and body:
            try:
                data = json.loads(body)
                data = process_response_json(data)
                body = json.dumps(data).encode()
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        self.send_response(resp.status)
        skip = {"content-length", "transfer-encoding", "connection"}
        for key, val in resp.headers.items():
            if key.lower() not in skip:
                self.send_header(key, val)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _proxy_stream(self, resp):
        """Streaming SSE: forward events, stripping thinking blocks."""
        self.send_response(200)
        for key in ("Content-Type", "Cache-Control", "X-Accel-Buffering"):
            val = resp.headers.get(key)
            if val:
                self.send_header(key, val)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        if self.command == "HEAD":
            return

        skip_block = False
        try:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")

                if line == "":
                    if not skip_block:
                        self._write_chunk(b"\n")
                    continue

                if not line.startswith("data: "):
                    if not skip_block:
                        self._write_chunk((line + "\n").encode())
                    continue

                payload = line[6:]
                done_marker = "[" + "DONE" + "]"
                if payload == done_marker:
                    if not skip_block:
                        self._write_chunk((line + "\n").encode())
                    continue

                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    if not skip_block:
                        self._write_chunk((line + "\n").encode())
                    continue

                etype = event.get("type", "")

                if etype == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") in STRIP_TYPES:
                        skip_block = True
                        continue
                    skip_block = False
                    self._write_chunk((line + "\n").encode())
                    continue

                if skip_block:
                    if etype in (
                        "content_block_delta",
                        "content_block_stop",
                        "ping",
                    ):
                        continue
                    skip_block = False

                if etype == "message_start":
                    msg = event.get("message", {})
                    content = msg.get("content")
                    if isinstance(content, list):
                        stripped = strip_thinking_blocks(content)
                        if len(stripped) != len(content):
                            msg["content"] = stripped
                            line = "data: " + json.dumps(event)

                self._write_chunk((line + "\n").encode())

            self._write_chunk(b"")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            sys.stderr.write(f"[proxy] stream error: {e}\n")

    def _write_chunk(self, data):
        """Write a chunk in HTTP chunked transfer encoding."""
        if data:
            self.wfile.write(f"{len(data):X}\r\n".encode())
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
        else:
            self.wfile.write(b"0\r\n\r\n")

    def _forward_error(self, http_err):
        body = http_err.read()
        self.send_response(http_err.code)
        skip = {"content-length", "transfer-encoding", "connection"}
        for key, val in http_err.headers.items():
            if key.lower() not in skip:
                self.send_header(key, val)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json_error(self, code, msg):
        body = json.dumps(
            {"error": {"type": "proxy_error", "message": msg}}
        ).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_POST(self):
        self._do_forward()

    def do_GET(self):
        self._do_forward()

    def do_PUT(self):
        self._do_forward()

    def do_DELETE(self):
        self._do_forward()

    def do_PATCH(self):
        self._do_forward()

    def do_HEAD(self):
        self._do_forward()

    def do_OPTIONS(self):
        self._do_forward()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: thinking_strip_proxy.py <upstream_base_url> [port]\n"
            "  e.g. thinking_strip_proxy.py https://litellm-prod.toolsfdg.net 18769\n"
        )
        return 1

    upstream = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT

    StripProxyHandler.upstream_base = upstream
    server = ThreadingHTTPServer(("127.0.0.1", port), StripProxyHandler)
    sys.stderr.write(
        f"[proxy] thinking-strip proxy listening on http://127.0.0.1:{port}\n"
        f"[proxy] forwarding to {upstream}\n"
        f"[proxy] stripping thinking/redacted_thinking blocks from /v1/messages\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("[proxy] shutting down\n")
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
