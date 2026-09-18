"""
routes.py
---------
Lightweight HTTP router using only Python's built-in http.server.
No FastAPI, no external dependencies.

This is used for local demo/testing. When Module 1+2 are ready,
swap this out for FastAPI or keep it — the optimizer logic is unchanged.

Run with:
    python -m backend.main
"""

from __future__ import annotations
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler

from backend.modules.privacy.optimizer import optimize, DEFAULT_RISK_THRESHOLD

_SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "sample"


def _handle_analyze(body: dict) -> dict:
    """POST /privacy/analyze — runs the optimizer on Module 2 output."""
    threshold = body.get("threshold") or DEFAULT_RISK_THRESHOLD
    return optimize(conversation_input=body, threshold=threshold)


def _handle_demo(case: str) -> dict:
    """GET /privacy/demo/<case> — runs a pre-loaded mock (low|medium|high)."""
    file_map = {
        "low":    _SAMPLE_DIR / "mock_low_risk.json",
        "medium": _SAMPLE_DIR / "mock_medium_risk.json",
        "high":   _SAMPLE_DIR / "mock_high_risk.json",
    }
    if case not in file_map:
        return {"error": f"Unknown case '{case}'. Choose from: low, medium, high"}
    path = file_map[case]
    if not path.exists():
        return {"error": f"Mock file not found: {path.name}"}
    data = json.loads(path.read_text())
    return optimize(conversation_input=data)


class PrivacyHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — enough for demo and frontend calls."""

    def log_message(self, fmt, *args):
        # Suppress default request logs; replace with clean output
        print(f"  {self.command} {self.path}  →  {args[1] if len(args) > 1 else ''}")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # CORS — allow React dev server
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.rstrip("/")

        if path in ("/", "/health"):
            self._send_json({"status": "ok", "module": "Privacy Guardrail (Module 3)"})

        elif path.startswith("/privacy/demo/"):
            case = path.split("/privacy/demo/")[-1]
            result = _handle_demo(case)
            status = 400 if "error" in result else 200
            self._send_json(result, status)

        else:
            self._send_json({"error": f"Unknown route: {self.path}"}, 404)

    def do_POST(self):
        path = self.path.rstrip("/")

        if path == "/privacy/analyze":
            length = int(self.headers.get("Content-Length", 0))
            raw    = self.rfile.read(length)
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as e:
                self._send_json({"error": f"Invalid JSON: {e}"}, 400)
                return
            try:
                result = _handle_analyze(body)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": f"Unknown route: {self.path}"}, 404)
