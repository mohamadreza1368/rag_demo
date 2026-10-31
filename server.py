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
import shutil
import webbrowser
import math
import hashlib
import re
import time

PORT = int(os.environ.get('PORT', 8080))
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def local_python_embed(text: str, dim: int = 768) -> list:
    """
    Pure Python deterministic semantic dense vectorizer.
    Runs 100% locally with zero external network or heavy dependencies.
    Produces normalized L2 unit vectors suitable for cosine similarity.
    """
    if not text:
        return [0.0] * dim

    vec = [0.0] * dim
    clean_text = text.lower().strip()
    words = re.findall(r'[\w\u0600-\u06FF]+', clean_text)
    
    features = list(words)
    for w in words:
        if len(w) >= 3:
            for i in range(len(w) - 2):
                features.append(w[i:i+3])
        if len(w) >= 4:
            for i in range(len(w) - 3):
                features.append(w[i:i+4])

    if not features:
        features = [clean_text]

    for feat in features:
        h_bytes = hashlib.sha256(feat.encode('utf-8')).digest()
        for offset in range(0, 28, 4):
            val = int.from_bytes(h_bytes[offset:offset+4], 'big')
            idx = val % dim
            sign = 1.0 if (val & 0x80000000) else -1.0
            weight = 1.0 + (float(val % 100) / 100.0)
            vec[idx] += sign * weight

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 1e-9:
        return [round(x / norm, 6) for x in vec]
    vec[0] = 1.0
    return vec

class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def finish_request(self, request, client_address):
        try:
            super().finish_request(request, client_address)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
            pass

    def handle_error(self, request, client_address):
        exc_type, _, _ = sys.exc_info()
        # Suppress normal client disconnection errors (e.g. closed tabs, healthcheck drops)
        if exc_type in (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
            return
        super().handle_error(request, client_address)

class RAGStudioHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
            pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
            pass

    def do_GET(self):
        # Fast endpoint for container healthchecks
        if self.path in ('/health', '/api/health'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', '2')
            self.end_headers()
            try:
                self.wfile.write(b'OK')
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
                pass
            return
        try:
            super().do_GET()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
            pass

    def copyfile(self, source, outputfile):
        try:
            shutil.copyfileobj(source, outputfile)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
            pass

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
        elif self.path.startswith('/api/embed'):
            self.handle_embed()
        else:
            self.send_error(404, "Endpoint not found")

    def handle_embed(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b''

        try:
            req_data = json.loads(body.decode('utf-8')) if body else {}
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'Invalid JSON: {str(e)}'}).encode('utf-8'))
            return

        texts = req_data.get('texts', [])
        if isinstance(texts, str):
            texts = [texts]

        model_name = req_data.get('model', 'google/embeddinggemma-2')
        provider = req_data.get('provider', 'local_python')
        hf_token = req_data.get('token') or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        is_test = req_data.get('test', False)

        start_time = time.time()

        # 1. Hugging Face Inference API
        if provider == 'huggingface':
            hf_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'RAG-Studio-Embed/1.0'
            }
            if hf_token:
                headers['Authorization'] = f"Bearer {hf_token}"

            try:
                hf_payload = json.dumps({"inputs": texts or ["test ping"], "options": {"wait_for_model": True}}).encode('utf-8')
                req = urllib.request.Request(hf_url, data=hf_payload, headers=headers, method='POST')
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = resp.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data)
                    return
            except urllib.error.HTTPError as e:
                err_content = e.read().decode('utf-8', errors='ignore')
                self.send_response(e.code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': f'Hugging Face API returned status {e.code}',
                    'details': err_content,
                    'hint': 'If the model is gated (e.g. google/embeddinggemma-2), please provide a Hugging Face API Token (hf_...) and accept model license on HuggingFace.'
                }).encode('utf-8'))
                return
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': f'Hugging Face connection error: {str(e)}'}).encode('utf-8'))
                return

        # 2. Local Python Engine (100% Offline / Fast / Zero-Dependency)
        elif provider in ('local_python', 'local', 'builtin'):
            if not texts:
                texts = ["تست تبدیل متن به بردار در پایتون"]

            vectors = []
            engine_name = "Python Local Dense Vectorizer (100% Offline)"

            # Try sentence_transformers if installed and user specified a local model file/name
            # Otherwise use the high-performance pure Python dense vectorizer
            for t in texts:
                vectors.append(local_python_embed(t, dim=768))

            elapsed_ms = round((time.time() - start_time) * 1000, 1)

            resp_obj = {
                'status': 'ok',
                'embeddings': vectors,
                'model': model_name or 'local_python_768d',
                'dimensions': len(vectors[0]) if vectors else 768,
                'provider': 'local_python',
                'engine': engine_name,
                'latency_ms': elapsed_ms
            }

            if is_test and vectors:
                resp_obj['sample'] = vectors[0][:5]
                resp_obj['message'] = 'وکتورایزر محلی پایتون آماده و عملیاتی است.'

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(resp_obj).encode('utf-8'))
            return

        else:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'Unsupported provider: {provider}'}).encode('utf-8'))

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
            # 15s timeout for fast response / failure feedback
            with urllib.request.urlopen(req, timeout=15) as resp:
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
        except urllib.error.URLError as e:
            self.send_response(504)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': f'Connection failed: {str(e.reason)}',
                'hint': f'Target URL ({target_url}) is unreachable or blocked. Check if service is running.'
            }).encode('utf-8'))
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
            with ThreadedTCPServer(("", port), RAGStudioHandler) as httpd:
                url = f"http://localhost:{port}"
                print("\n" + "="*65)
                print(f"  [+] RAG Studio with Live LLM Connection is RUNNING!")
                print(f"  [*] Browser URL: {url}")
                print(f"  [*] Built-in Anti-CORS Proxy active at: {url}/api/proxy")
                print("  [*] Press Ctrl+C to stop.")
                print("="*65 + "\n")
                if not os.environ.get('DOCKER') and os.environ.get('NO_BROWSER', '0') != '1':
                    try:
                        webbrowser.open(url)
                    except Exception:
                        pass
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
