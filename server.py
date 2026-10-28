#!/usr/bin/env python3
"""
Operational HTTP Server & CORS Proxy for RAG Studio
- Serves static files for the RAG Studio UI
- Provides an optional /api/proxy endpoint to forward requests to any LLM Base URL (OpenAI, Ollama, OpenRouter, Groq, etc.)
  bypassing browser CORS restrictions effortlessly.
"""

import http.server
import socketserver
import urllib.request
import urllib.error
import json
import os
import sys
import webbrowser

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class RAGStudioHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_OPTIONS(self):
        # Handle CORS preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, target-url')
        self.end_headers()

    def do_POST(self):
        if self.path.startswith('/api/proxy'):
            self.handle_proxy()
        else:
            self.send_error(404, "Endpoint not found")

    def handle_proxy(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b''

        # Target URL passed via header or json body
        target_url = self.headers.get('target-url')
        auth_header = self.headers.get('Authorization')

        try:
            req_data = json.loads(body.decode('utf-8')) if body else {}
        except Exception:
            req_data = {}

        if not target_url and 'targetUrl' in req_data:
            target_url = req_data.pop('targetUrl')
            if 'authHeader' in req_data and not auth_header:
                auth_header = req_data.pop('authHeader')
            body = json.dumps(req_data.get('payload', req_data)).encode('utf-8')

        if not target_url:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'target-url header or targetUrl in body is required'}).encode('utf-8'))
            return

        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'RAG-Studio-Proxy/1.0'
        }
        if auth_header:
            headers['Authorization'] = auth_header

        try:
            req = urllib.request.Request(target_url, data=body, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=120) as resp:
                self.send_response(resp.status)
                self.send_header('Access-Control-Allow-Origin', '*')
                content_type = resp.headers.get('Content-Type', 'application/json')
                self.send_header('Content-Type', content_type)

                is_stream = 'text/event-stream' in content_type

                if is_stream:
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'keep-alive')
                    self.end_headers()
                    while True:
                        chunk = resp.read(1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    data = resp.read()
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

        except urllib.error.HTTPError as e:
            error_body = e.read()
            self.send_response(e.code)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(error_body)
        except Exception as e:
            self.send_response(500)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

def run():
    port = PORT
    for attempt in range(5):
        try:
            with socketserver.TCPServer(("", port), RAGStudioHandler) as httpd:
                url = f"http://localhost:{port}"
                print("\n" + "="*65)
                print(f"  [+] RAG Studio with Live LLM Connection is RUNNING!")
                print(f"  [*] Browser URL: {url}")
                print(f"  [*] Built-in Anti-CORS Proxy active at: {url}/api/proxy")
                print("  [*] Press Ctrl+C to stop.")
                print("="*65 + "\n")
                webbrowser.open(url)
                httpd.serve_forever()
                break
        except OSError:
            port += 1

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass
    try:
        run()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)
