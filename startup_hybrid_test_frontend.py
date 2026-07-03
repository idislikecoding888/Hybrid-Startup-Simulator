#!/usr/bin/env python3
"""
Dummy local frontend + reverse proxy for the Startup Simulator backend.

Run backend first:
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

Then run this file:
    python startup_hybrid_test_frontend.py

Open:
    http://127.0.0.1:5500
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import json
import sys

BACKEND = "http://127.0.0.1:8000"
HOST = "127.0.0.1"
PORT = 5500

INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Startup Simulator Hybrid Tester</title>
  <style>
    body { font-family: Arial, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:24px; }
    .card { background:#111827; border:1px solid #334155; border-radius:14px; padding:16px; margin-bottom:16px; }
    button { margin:4px 6px 4px 0; padding:10px 14px; border:0; border-radius:10px; background:#2563eb; color:white; cursor:pointer; }
    button.secondary { background:#475569; }
    pre { white-space:pre-wrap; word-break:break-word; background:#0b1220; padding:12px; border-radius:12px; overflow:auto; }
    .row { display:flex; gap:16px; flex-wrap:wrap; }
    .col { flex:1 1 320px; }
    input { padding:10px; border-radius:10px; border:1px solid #334155; background:#0b1220; color:#e2e8f0; width:120px; }
    a { color:#7dd3fc; }
  </style>
</head>
<body>
  <h1>Startup Simulator Hybrid Tester</h1>
  <div class="card">
    <p>This page talks to the real backend through a local proxy, so you can test the hybrid model without CORS issues.</p>
    <label>Steps:
      <input id="steps" type="number" value="1" min="1" max="100" />
    </label>
    <div style="margin-top:10px;">
      <button onclick="callApi('/simulation/start?steps=' + document.getElementById('steps').value)">Start</button>
      <button onclick="callApi('/simulation/step')">Step</button>
      <button onclick="callApi('/simulation/reset')" class="secondary">Reset</button>
      <button onclick="callApi('/state/')">State</button>
      <button onclick="callApi('/metrics/current')">Metrics</button>
      <button onclick="callApi('/logs/latest')">Latest Log</button>
      <button onclick="callApi('/simulation/stop')" class="secondary">Stop</button>
    </div>
  </div>

  <div class="row">
    <div class="card col">
      <h3>Response</h3>
      <pre id="out">Click a button.</pre>
    </div>
    <div class="card col">
      <h3>Quick tips</h3>
      <pre id="tips">
• If you see LLM errors, set GROQ_API_KEY in backend/.env
• If PPO is unavailable, the hybrid code falls back gracefully
• Real outputs appear after the backend is running
      </pre>
    </div>
  </div>

<script>
async function callApi(path) {
  const out = document.getElementById('out');
  out.textContent = 'Loading...';
  try {
    const res = await fetch('/api' + path);
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); }
    catch { data = text; }
    out.textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Request failed: ' + e;
  }
}
</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, content_type="text/plain; charset=utf-8", body=b""):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            return

        if parsed.path.startswith("/api/"):
            target = BACKEND + parsed.path[4:]
            if parsed.query:
                target += "?" + parsed.query
            try:
                req = Request(target, headers={"User-Agent": "StartupHybridTester/1.0"})
                with urlopen(req, timeout=30) as resp:
                    body = resp.read()
                    ctype = resp.headers.get_content_type()
                    if ctype == "application/json":
                        ctype = "application/json; charset=utf-8"
                    self._send(resp.status, ctype, body)
            except HTTPError as e:
                body = e.read() if hasattr(e, "read") else str(e).encode("utf-8")
                self._send(e.code, "text/plain; charset=utf-8", body)
            except URLError as e:
                self._send(502, "text/plain; charset=utf-8", f"Backend unavailable: {e}".encode("utf-8"))
            return

        self._send(404, "text/plain; charset=utf-8", b"Not found")

def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Dummy frontend running on http://{HOST}:{PORT}")
    print(f"Proxying API requests to {BACKEND}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()

if __name__ == "__main__":
    main()
