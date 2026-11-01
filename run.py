#!/usr/bin/env python3
"""
RAG Studio - Web UI Launcher Alias
Runs: python run.py [--port 8089]
"""
import sys
import main

if __name__ == '__main__':
    main_args = sys.argv[1:]
    import server
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("port_pos", nargs="?", type=int)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int)
    args, _ = parser.parse_known_args()
    target_port = args.port or args.port_pos or int(server.os.environ.get("PORT", 8089))
    target_host = args.host or "0.0.0.0"
    server.run(host=target_host, port=target_port)
