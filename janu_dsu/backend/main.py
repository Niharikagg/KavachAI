"""
main.py
-------
Entry point for the Privacy Guardrail (Module 3).
Uses only Python's built-in http.server — no external packages needed.

Run with:
    python -m backend.main
        or
    python backend/main.py

Server starts on http://localhost:8000
"""

from http.server import HTTPServer
from backend.api.routes import PrivacyHandler

HOST = "localhost"
PORT = 8000


def main():
    server = HTTPServer((HOST, PORT), PrivacyHandler)
    print(f"\n  Privacy Guardrail — Module 3")
    print(f"  Running on http://{HOST}:{PORT}\n")
    print(f"  Endpoints:")
    print(f"    GET  /health")
    print(f"    GET  /privacy/demo/low")
    print(f"    GET  /privacy/demo/medium")
    print(f"    GET  /privacy/demo/high")
    print(f"    POST /privacy/analyze\n")
    print(f"  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
