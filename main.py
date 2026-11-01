#!/usr/bin/env python3
"""
RAG Studio - Main Application Launcher
Runs directly with Python and Uvicorn on 0.0.0.0 (Zero-Docker).

Usage:
    python main.py
    python main.py --port 8089
    python main.py --host 0.0.0.0 --port 8089
"""

import sys
import os
import argparse

# Ensure UTF-8 console output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch RAG Studio Web UI Server")
    parser.add_argument("port_pos", nargs="?", type=int, help="Optional port number (e.g. 8089)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0 for LAN/network access)")
    parser.add_argument("--port", "-p", type=int, help="Port to listen on (default: 8089)")

    args, unknown = parser.parse_known_args()

    target_port = args.port or args.port_pos or int(os.environ.get("PORT", 8089))
    target_host = args.host or "0.0.0.0"

    import server
    server.run(host=target_host, port=target_port)
