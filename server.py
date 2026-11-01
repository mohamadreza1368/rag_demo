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
import urllib.parse
import json
import os
import sys
import shutil
import webbrowser
import math
import hashlib
import re
import time
import ssl

# Unverified SSL context for internal corporate / self-signed certificate gateways
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

PORT = int(os.environ.get('PORT', 8080))
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOADED_MODELS = {}

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

def find_cached_model_path(model_id="google/embeddinggemma-2"):
    candidates = []
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_home:
        candidates.append(os.path.join(hf_home, "hub") if not hf_home.endswith("hub") else hf_home)
        candidates.append(hf_home)
    candidates.extend([
        "/root/.cache/huggingface/hub",
        "/root/.cache/huggingface",
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/huggingface"),
    ])
    clean_name = "models--" + model_id.replace("/", "--")
    for base in candidates:
        if not os.path.exists(base):
            continue
        model_dir = os.path.join(base, clean_name)
        if os.path.exists(model_dir):
            snapshots_dir = os.path.join(model_dir, "snapshots")
            if os.path.exists(snapshots_dir):
                try:
                    snaps = [os.path.join(snapshots_dir, s) for s in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, s))]
                    if snaps:
                        return snaps[0]
                except Exception:
                    pass
            return model_dir
    return None

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
        elif self.path.startswith('/api/proxy'):
            self.handle_proxy(method='GET')
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
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, target-url, X-File-Name')
        self.end_headers()

    def do_POST(self):
        if self.path.startswith('/api/proxy'):
            self.handle_proxy()
        elif self.path.startswith('/api/embed'):
            self.handle_embed()
        elif self.path.startswith('/api/parse-doc'):
            self.handle_parse_doc()
        else:
            self.send_error(404, "Endpoint not found")

    def handle_parse_doc(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b''

        raw_filename = self.headers.get('X-File-Name', 'document.docx')
        filename = urllib.parse.unquote(raw_filename)

        if not body:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'فایل ارسالی خالی است.'}, ensure_ascii=False).encode('utf-8'))
            return

        try:
            import zipfile
            import xml.etree.ElementTree as ET
            import io

            docx_file = io.BytesIO(body)
            paragraphs = []

            if zipfile.is_zipfile(docx_file):
                with zipfile.ZipFile(docx_file) as z:
                    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

                    # 1. Main document body
                    if 'word/document.xml' in z.namelist():
                        xml_bytes = z.read('word/document.xml')
                        tree = ET.fromstring(xml_bytes)
                        for p in tree.iterfind('.//w:p', ns):
                            texts = [node.text for node in p.iterfind('.//w:t', ns) if node.text]
                            if texts:
                                paragraphs.append(''.join(texts))

                    # 2. Footnotes if present
                    if 'word/footnotes.xml' in z.namelist():
                        try:
                            tree = ET.fromstring(z.read('word/footnotes.xml'))
                            for p in tree.iterfind('.//w:p', ns):
                                texts = [node.text for node in p.iterfind('.//w:t', ns) if node.text]
                                if texts:
                                    paragraphs.append(''.join(texts))
                        except Exception:
                            pass
            else:
                # Fallback to UTF-8 plain text if uploaded
                try:
                    text_content = body.decode('utf-8')
                    paragraphs = text_content.splitlines()
                except UnicodeDecodeError:
                    raise ValueError('فایل ارسالی ساختار معتبر DOCX یا متن UTF-8 ندارد.')

            extracted_text = '\n\n'.join(paragraphs).strip()
            if not extracted_text:
                extracted_text = 'متنی در سند Word یافت نشد.'

            resp_obj = {
                'status': 'ok',
                'text': extracted_text,
                'filename': filename,
                'paragraphs': len(paragraphs)
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(resp_obj, ensure_ascii=False).encode('utf-8'))
            return

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': f'خطا در استخراج متن از فایل Word: {str(e)}',
                'filename': filename
            }, ensure_ascii=False).encode('utf-8'))

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
        if not texts:
            texts = ["تست تبدیل متن به بردار با مدل کش‌شده سیستم"]

        # Exclusively dedicated to google/embeddinggemma-2 from system cache
        model_name = "google/embeddinggemma-2"
        is_test = req_data.get('test', False)
        start_time = time.time()

        cached_path = find_cached_model_path(model_name)
        model_target = cached_path if cached_path else model_name

        # 1. Primary engine: SentenceTransformer loaded from system cache
        try:
            from sentence_transformers import SentenceTransformer
            if 'gemma2' not in LOADED_MODELS:
                try:
                    LOADED_MODELS['gemma2'] = SentenceTransformer(model_target, local_files_only=True)
                except Exception:
                    LOADED_MODELS['gemma2'] = SentenceTransformer(model_target)

            st_model = LOADED_MODELS['gemma2']
            raw_vecs = st_model.encode(texts, show_progress_bar=False)
            vectors = [v.tolist() for v in raw_vecs]
            dim = len(vectors[0]) if vectors else 768
            elapsed_ms = round((time.time() - start_time) * 1000, 1)

            resp_obj = {
                'status': 'ok',
                'embeddings': vectors,
                'model': model_name,
                'dimensions': dim,
                'source': f'System Cache ({cached_path or "Hugging Face Cache"})',
                'engine': f'SentenceTransformer: google/embeddinggemma-2 ({dim}D)',
                'latency_ms': elapsed_ms
            }
            if is_test:
                resp_obj['sample'] = vectors[0][:5]
                resp_obj['message'] = f'مدل {model_name} با موفقیت از کش سیستم بارگذاری و اجرا شد.'

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(resp_obj).encode('utf-8'))
            return

        except ImportError:
            # 2. Secondary engine: Hugging Face Transformers + PyTorch loaded from system cache
            try:
                import torch
                from transformers import AutoTokenizer, AutoModel
                if 'gemma2_hf' not in LOADED_MODELS:
                    tok = AutoTokenizer.from_pretrained(model_target, local_files_only=True)
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token or '[PAD]'
                    mod = AutoModel.from_pretrained(model_target, local_files_only=True)
                    LOADED_MODELS['gemma2_hf'] = (tok, mod)

                tok, mod = LOADED_MODELS['gemma2_hf']
                inputs = tok(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
                with torch.no_grad():
                    outputs = mod(**inputs)
                    mask = inputs['attention_mask'].unsqueeze(-1)
                    embeddings = torch.sum(outputs[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    vectors = embeddings.tolist()

                dim = len(vectors[0]) if vectors else 768
                elapsed_ms = round((time.time() - start_time) * 1000, 1)
                resp_obj = {
                    'status': 'ok',
                    'embeddings': vectors,
                    'model': model_name,
                    'dimensions': dim,
                    'source': f'System Cache ({cached_path or "Hugging Face Cache"})',
                    'engine': f'Transformers PyTorch: google/embeddinggemma-2 ({dim}D)',
                    'latency_ms': elapsed_ms
                }
                if is_test:
                    resp_obj['sample'] = vectors[0][:5]
                    resp_obj['message'] = f'مدل {model_name} با موفقیت توسط موتور PyTorch از کش سیستم اجرا شد.'

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(resp_obj).encode('utf-8'))
                return

            except ImportError:
                self.send_response(501)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': f'پکیج sentence-transformers یا transformers برای اجرای مدل «{model_name}» از کش سیستم نصب نیست.',
                    'hint': 'لطفا دستور "pip install sentence-transformers torch" را اجرا فرمایید یا اطمینان حاصل کنید در کانتینر داکر نصب شده باشد.',
                    'model': model_name,
                    'source': cached_path or 'HF Cache'
                }).encode('utf-8'))
                return

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': f'خطا در بارگذاری یا اجرای مدل {model_name} از کش سیستم: {str(e)}',
                'hint': 'بررسی فرمایید فایل‌های کش شده مدل در مسیر ~/.cache/huggingface به صورت کامل دانلود و مانت شده باشند.',
                'model': model_name,
                'path': cached_path
            }).encode('utf-8'))
            return

    def handle_proxy(self, method=None):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b''

        # Target URL passed via header, query string or JSON body
        target_url = self.headers.get('target-url')
        auth_header = self.headers.get('Authorization')

        if not target_url and '?' in self.path:
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            target_url = params.get('targetUrl', [None])[0]

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
            self.wfile.write(json.dumps({'error': 'target-url header, query parameter or targetUrl in body is required'}).encode('utf-8'))
            return

        http_method = method or req_data.get('method') or getattr(self, 'command', 'POST')

        # Automatic Docker host mapping:
        # If running inside Docker and target points to localhost or 127.0.0.1,
        # redirect to host.docker.internal so it connects to services running on host (e.g. Ollama, LM Studio, vLLM)
        if os.environ.get('DOCKER') and target_url:
            target_url = target_url.replace('://localhost:', '://host.docker.internal:').replace('://127.0.0.1:', '://host.docker.internal:')

        headers = {
            'User-Agent': 'RAG-Studio-Proxy/1.0'
        }
        if http_method != 'GET':
            headers['Content-Type'] = 'application/json'

        if auth_header and auth_header.strip():
            headers['Authorization'] = auth_header.strip()

        proxy_timeout = 60  # Allow up to 60 seconds for slow LLM responses / cold starts
        try:
            post_data = body if (http_method != 'GET' and body) else None
            req = urllib.request.Request(target_url, data=post_data, headers=headers, method=http_method)
            # Use ssl_context with unverified certs so self-signed / internal corporate certs work smoothly
            with urllib.request.urlopen(req, timeout=proxy_timeout, context=ssl_context) as resp:
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
        except (TimeoutError, urllib.error.URLError) as e:
            self.send_response(504)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            reason = str(getattr(e, 'reason', e))
            self.wfile.write(json.dumps({
                'error': f'Request timeout / connection failed: {reason}',
                'target': target_url,
                'hint': 'سرور مدل زبانی در مدت ۶۰ ثانیه پاسخ نداد. بررسی فرمایید که مدل در حال اجرا باشد و پورت یا آدرس شبکه از داخل سرور قابل دسترسی باشد.'
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
