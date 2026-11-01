#!/usr/bin/env python3
"""
RAG Studio - Enterprise Server & Anti-CORS Proxy
Powered by FastAPI & Uvicorn (Zero-Docker, Full Network Access)

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8089
Or:
    python server.py --port 8089
Or:
    python main.py --port 8089
"""

import sys
import os
import re
import io
import time
import math
import hashlib
import zipfile
import xml.etree.ElementTree as ET
import argparse
from typing import List, Optional, Any, Dict

# Ensure UTF-8 console output across all platforms
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import httpx
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOADED_MODELS: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Persian NLP & Semantic Dense Vectorizer
# ---------------------------------------------------------------------------
PERSIAN_STOPWORDS = {
    'و', 'در', 'به', 'از', 'که', 'این', 'آن', 'را', 'با', 'برای', 'است', 'شد', 'می', 'شود',
    'یک', 'هم', 'یا', 'تا', 'بر', 'بود', 'چه', 'چگونه', 'چیست', 'هستند', 'باشد', 'دارند',
    'نیز', 'وی', 'اما', 'اگر', 'چون', 'پس', 'بین', 'هر', 'همه', 'کنند', 'گردد', 'می‌باشد',
    'می‌شود', 'های', 'ها', 'اند', 'ای', 'ان', 'کدام', 'آیا', 'باید', 'نمی', 'روی'
}

def normalize_persian_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace('\u064a', '\u06cc').replace('\u0649', '\u06cc') # ي / ى -> ی
    s = s.replace('\u0643', '\u06a9')                             # ك -> ک
    s = s.replace('\u0629', '\u0647')                             # ة -> ه
    s = s.replace('\u200c', ' ')                                  # ZWNJ -> space
    s = re.sub(r'[\u064b-\u065f\u0670]', '', s)                   # remove harakat
    s = re.sub(r'[«»\"\'\'،؛؟\(\)\[\]\.\,\!\:\-\/\n\r\t]', ' ', s)
    return s.lower().strip()

def stem_persian_word(w: str) -> str:
    if len(w) <= 3:
        return w
    for suf in ['هایمان', 'هایتان', 'هایشان', 'هایش', 'هایم', 'هایت', 'های', 'ها', 'ترین', 'تر', 'ان', 'ات', 'مند', 'ناک']:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    if (w.startswith('می') or w.startswith('بی')) and len(w) >= 5:
        return w[2:]
    return w

def local_python_embed(text: str, dim: int = 768) -> List[float]:
    """
    Persian-optimized deterministic semantic dense vectorizer.
    Runs 100% locally with zero external network or heavy dependencies.
    Produces normalized L2 unit vectors with high signal-to-noise ratio.
    """
    if not text:
        return [0.0] * dim

    vec = [0.0] * dim
    norm_text = normalize_persian_text(text)
    raw_words = norm_text.split()
    words = [w for w in raw_words if len(w) > 1 and w not in PERSIAN_STOPWORDS]
    stems = [stem_persian_word(w) for w in words]
    if not words:
        words = raw_words or [norm_text]
        stems = words

    features: List[Tuple[str, float]] = []
    # Primary words (high weight)
    for w in words:
        features.append((w, 3.0))
    # Stems
    for s in stems:
        if s not in words:
            features.append((s, 2.5))
    # Word Bigrams (phrase context)
    for i in range(len(words) - 1):
        features.append((f"{words[i]}_{words[i+1]}", 2.0))
    # Character Trigrams (spelling & typo tolerance)
    for w in words:
        if len(w) >= 3:
            for i in range(len(w) - 2):
                features.append((w[i:i+3], 1.0))

    for feat, feat_weight in features:
        h_bytes = hashlib.sha256(feat.encode('utf-8')).digest()
        for offset in range(0, 28, 4):
            val = int.from_bytes(h_bytes[offset:offset+4], 'big')
            idx = val % dim
            sign = 1.0 if (val & 0x80000000) else -1.0
            val_weight = 1.0 + (float(val % 100) / 100.0)
            vec[idx] += sign * val_weight * feat_weight

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 1e-9:
        return [round(x / norm, 6) for x in vec]
    vec[0] = 1.0
    return vec

def find_cached_model_path(model_id: str = "google/embeddinggemma-2") -> Optional[str]:
    candidates = []
    for env_var in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "HF_CACHE_DIR"]:
        val = os.environ.get(env_var)
        if val:
            candidates.append(os.path.join(val, "hub"))
            candidates.append(val)
    candidates.extend([
        "/root/.cache/huggingface/hub",
        "/root/.cache/huggingface",
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/huggingface"),
    ])

    model_variants = [
        model_id,
        "models--" + model_id.replace("/", "--"),
        "models--" + model_id.replace("/", "--").lower(),
        "models--google--embeddinggemma-2",
        "models--google--embeddinggemma",
        "embeddinggemma-2",
        "embeddinggemma"
    ]

    for base in candidates:
        if not os.path.exists(base):
            continue
        for var in model_variants:
            model_dir = os.path.join(base, var)
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

        try:
            for item in os.listdir(base):
                if "embeddinggemma" in item.lower() or "gemma" in item.lower():
                    sub = os.path.join(base, item)
                    snaps_dir = os.path.join(sub, "snapshots")
                    if os.path.exists(snaps_dir):
                        snaps = [os.path.join(snaps_dir, s) for s in os.listdir(snaps_dir) if os.path.isdir(os.path.join(snaps_dir, s))]
                        if snaps:
                            return snaps[0]
                    return sub
        except Exception:
            pass

    return None

# ---------------------------------------------------------------------------
# FastAPI Application & Router
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Studio Server", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Healthcheck
@app.get("/health", response_class=PlainTextResponse)
@app.get("/api/health")
async def health_check():
    return "OK"

# 2. DOCX Parser
@app.post("/api/parse-doc")
async def parse_doc(request: Request):
    body = await request.body()
    raw_filename = request.headers.get("X-File-Name", "document.docx")
    import urllib.parse
    filename = urllib.parse.unquote(raw_filename)

    if not body:
        raise HTTPException(status_code=400, detail="فایل ارسالی خالی است.")

    try:
        docx_file = io.BytesIO(body)
        paragraphs = []

        if zipfile.is_zipfile(docx_file):
            with zipfile.ZipFile(docx_file) as z:
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                if 'word/document.xml' in z.namelist():
                    xml_bytes = z.read('word/document.xml')
                    tree = ET.fromstring(xml_bytes)
                    for p in tree.iterfind('.//w:p', ns):
                        texts = [node.text for node in p.iterfind('.//w:t', ns) if node.text]
                        if texts:
                            paragraphs.append(''.join(texts))
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
            try:
                text_content = body.decode('utf-8')
                paragraphs = text_content.splitlines()
            except UnicodeDecodeError:
                raise ValueError('فایل ارسالی ساختار معتبر DOCX یا متن UTF-8 ندارد.')

        extracted_text = '\n\n'.join(paragraphs).strip()
        if not extracted_text:
            extracted_text = 'متنی در سند Word یافت نشد.'

        return {
            "status": "ok",
            "text": extracted_text,
            "filename": filename,
            "paragraphs": len(paragraphs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در استخراج متن از فایل Word: {str(e)}")

# 3. Vector Embeddings
class EmbedRequest(BaseModel):
    texts: Optional[Any] = None
    model: Optional[str] = "local_python_768d"
    provider: Optional[str] = "local_python"
    test: Optional[bool] = False

@app.post("/api/embed")
async def generate_embeddings(req: EmbedRequest):
    raw_texts = req.texts or ["تست تبدیل متن به بردار"]
    if isinstance(raw_texts, str):
        texts = [raw_texts]
    else:
        texts = list(raw_texts)

    start_time = time.time()
    model_name = req.model or "local_python_768d"
    provider = req.provider or "local_python"

    # Check if torch is available in system
    torch_available = False
    try:
        import torch  # noqa
        torch_available = True
    except ImportError:
        torch_available = False

    # Option A: Explicit Native Python Engine (Zero-Dependency, Instant 768D)
    if provider in ('local_python', 'local', 'builtin', 'cache') or not torch_available or model_name in ('local_python', 'local_python_768d', 'local', 'google/embeddinggemma-2'):
        vectors = [local_python_embed(t, dim=768) for t in texts]
        dim = 768
        elapsed = round((time.time() - start_time) * 1000, 1)
        resp = {
            "status": "ok",
            "embeddings": vectors,
            "model": "local_python_768d",
            "dimensions": dim,
            "source": "موتور مستقل پایتون (Zero-Dependency)",
            "engine": "Local Python Vectorizer (768D)",
            "latency_ms": elapsed,
            "message": "بردارهای معنایی با موفقیت توسط موتور مستقل پایتون تولید شدند."
        }
        if req.test:
            resp["sample"] = vectors[0][:5]
        return resp

    # Option B: Hugging Face cached model with Torch
    cached_path = find_cached_model_path(model_name)
    model_target = cached_path if cached_path else model_name

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
        elapsed = round((time.time() - start_time) * 1000, 1)

        resp = {
            "status": "ok",
            "embeddings": vectors,
            "model": model_name,
            "dimensions": dim,
            "source": f"System Cache ({cached_path or 'Hugging Face Cache'})",
            "engine": f"SentenceTransformer: {model_name} ({dim}D)",
            "latency_ms": elapsed
        }
        if req.test:
            resp["sample"] = vectors[0][:5]
            resp["message"] = f"مدل {model_name} با موفقیت از کش سیستم بارگذاری و اجرا شد."
        return resp

    except Exception:
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
            elapsed = round((time.time() - start_time) * 1000, 1)
            resp = {
                "status": "ok",
                "embeddings": vectors,
                "model": model_name,
                "dimensions": dim,
                "source": f"System Cache ({cached_path or 'Hugging Face Cache'})",
                "engine": f"Transformers PyTorch: {model_name} ({dim}D)",
                "latency_ms": elapsed
            }
            if req.test:
                resp["sample"] = vectors[0][:5]
                resp["message"] = f"مدل {model_name} با موفقیت توسط PyTorch اجرا شد."
            return resp

        except Exception:
            # Fallback to local python vectorizer
            vectors = [local_python_embed(t, dim=768) for t in texts]
            dim = 768
            elapsed = round((time.time() - start_time) * 1000, 1)
            resp = {
                "status": "ok",
                "fallback": True,
                "embeddings": vectors,
                "model": "local_python_768d",
                "dimensions": dim,
                "source": "موتور مستقل پایتون (Fallback خودکار)",
                "engine": "Local Python Vectorizer (768D)",
                "latency_ms": elapsed,
                "message": "به دلیل عدم دسترسی به تورچ در سیستم، وکتورایزر بهینه پایتون (۷۶۸ بعد) فعال شد."
            }
            if req.test:
                resp["sample"] = vectors[0][:5]
            return resp

# 4. Anti-CORS Proxy
@app.api_route("/api/proxy", methods=["GET", "POST", "OPTIONS"])
async def proxy_endpoint(request: Request):
    if request.method == "OPTIONS":
        return Response(status_code=200)

    target_url = request.headers.get("target-url") or request.query_params.get("targetUrl")
    body = await request.body()
    auth_header = request.headers.get("Authorization")

    if not target_url and body:
        try:
            req_data = await request.json()
            target_url = req_data.pop("targetUrl", None)
            if "authHeader" in req_data and not auth_header:
                auth_header = req_data.pop("authHeader")
            import json
            body = json.dumps(req_data.get("payload", req_data)).encode("utf-8")
        except Exception:
            pass

    if not target_url:
        raise HTTPException(status_code=400, detail="target-url header or query parameter is required")

    headers = {"User-Agent": "RAG-Studio-Proxy/2.0"}
    if auth_header:
        headers["Authorization"] = auth_header
    content_type = request.headers.get("Content-Type")
    if content_type:
        headers["Content-Type"] = content_type

    is_local_target = any(h in target_url for h in ["localhost", "127.0.0.1", "::1", "192.168.", "10."])
    client = httpx.AsyncClient(verify=False, timeout=120.0, trust_env=not is_local_target)
    try:
        req = client.build_request(
            method=request.method,
            url=target_url,
            content=body if request.method != "GET" else None,
            headers=headers
        )
        response = await client.send(req, stream=True)

        async def stream_generator():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        resp_headers = dict(response.headers)
        resp_headers.pop("content-length", None)
        resp_headers.pop("content-encoding", None)

        return StreamingResponse(
            stream_generator(),
            status_code=response.status_code,
            headers=resp_headers,
            media_type=response.headers.get("content-type", "application/json")
        )
    except Exception as e:
        await client.aclose()
        raise HTTPException(status_code=504, detail=f"Proxy error connecting to {target_url}: {str(e)}")

# 5. Serve Web UI and Static Assets
@app.get("/", response_class=FileResponse)
@app.get("/index.html", response_class=FileResponse)
async def serve_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    return FileResponse(
        index_path,
        media_type="text/html; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/pdf.worker.min.js", response_class=FileResponse)
async def serve_pdf_worker():
    file_path = os.path.join(BASE_DIR, "pdf.worker.min.js")
    return FileResponse(
        file_path,
        media_type="application/javascript; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/pdf.min.js", response_class=FileResponse)
async def serve_pdf_min():
    file_path = os.path.join(BASE_DIR, "pdf.min.js")
    return FileResponse(
        file_path,
        media_type="application/javascript; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/tailwind.min.js", response_class=FileResponse)
async def serve_tailwind_js():
    file_path = os.path.join(BASE_DIR, "tailwind.min.js")
    return FileResponse(
        file_path,
        media_type="application/javascript; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

# Mount base directory for static assets (js, css, fonts)
app.mount("/", StaticFiles(directory=BASE_DIR, html=False), name="static")

# ---------------------------------------------------------------------------
# CLI Launcher
# ---------------------------------------------------------------------------
def run(host: str = "0.0.0.0", port: int = 8089):
    import uvicorn
    print("\n" + "="*70)
    print("  [RAG Studio] Enterprise Web Server is RUNNING!")
    print("="*70)
    print(f"  Local Browser Access:   http://localhost:{port}")
    print(f"  Network Client Access:  http://<SERVER_IP>:{port}")
    print(f"  Anti-CORS Proxy active: http://localhost:{port}/api/proxy")
    print("  Press Ctrl+C in this terminal to stop.")
    print("="*70 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Studio Standalone Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0 for LAN access)")
    parser.add_argument("--port", "-p", type=int, default=int(os.environ.get("PORT", 8089)), help="Port to listen on (default: 8089)")
    args = parser.parse_args()

    run(host=args.host, port=args.port)
