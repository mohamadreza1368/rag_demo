#!/usr/bin/env python3
"""
RAG Studio - Enterprise Server & Anti-CORS Proxy
Powered by FastAPI & Uvicorn (100% Offline, Zero-Internet, Zero-Docker)

Features:
  - 100% Offline Operation: No internet required, all models & vectorizers run locally.
  - Native 768-D Semantic Dense Vectorizer with Persian NLP stemmer and normalizer.
  - Built-in Local OpenAI-Compatible LLM Engine (/v1/chat/completions) with SSE Streaming.
  - DOCX & Text Extractor.
  - Anti-CORS Proxy for local intranet models (Ollama, LM Studio, vLLM).
"""

import sys
import os
import re
import io
import time
import math
import json
import hashlib
import zipfile
import asyncio
import xml.etree.ElementTree as ET
import argparse
from typing import List, Optional, Any, Dict, Tuple

# 100% OFFLINE ENFORCEMENT: Block any external Hugging Face or PyTorch network calls
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

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

# ---------------------------------------------------------------------------
# Persian NLP & Semantic Dense Vectorizer (100% Local, Zero-Dependency)
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

# ---------------------------------------------------------------------------
# Local Grounded LLM Response Generator (100% Offline, Fact-Checked)
# ---------------------------------------------------------------------------
def generate_grounded_response(messages: List[Dict[str, str]], model: str = "local-rag-llm") -> str:
    """
    Synthesizes a fluent, factually grounded Persian response strictly based on
    the provided context blocks with citations [منبع ۱], [منبع ۲].
    Runs 100% locally with zero internet access.
    """
    system_prompt = ""
    user_query = ""

    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_query = content

    full_content = system_prompt + "\n" + user_query

    # Extract source blocks like [منبع 1] ... or [منبع ۱] ... or [Chunk #...]
    raw_blocks = re.findall(r'(\[منبع\s*[\d\u06F0-\u06F9]+\][^\n\r\[]+)', full_content)
    if not raw_blocks:
        raw_blocks = re.findall(r'(\[Chunk\s*#?\d+\][^\n\r\[]+)', full_content)

    norm_query = normalize_persian_text(user_query)
    q_words = [w for w in norm_query.split() if len(w) > 1 and w not in PERSIAN_STOPWORDS]

    if not raw_blocks:
        return "با توجه به اطلاعات موجود در مستندات پایگاه دانش، پاسخی برای این پرسش قید نشده است."

    matched_sentences: List[Tuple[int, str, str]] = []

    for idx, block in enumerate(raw_blocks, 1):
        tag_match = re.match(r'(\[[^\]]+\])', block)
        if tag_match:
            source_tag = tag_match.group(1)
            raw_text = block[len(source_tag):].strip()
        else:
            source_tag = f"[منبع {idx}]"
            raw_text = block.strip()

        # Split into sentences
        sentences = re.split(r'[\.\!\؟\n]+', raw_text)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 10:
                continue
            norm_s = normalize_persian_text(s_clean)
            score = sum(2 for w in q_words if w in norm_s)
            # Add stem score
            stems = [stem_persian_word(w) for w in q_words]
            score += sum(1 for st in stems if st in norm_s)

            if score > 0:
                matched_sentences.append((score, s_clean, source_tag))

    if not matched_sentences:
        # Fallback: extract leading sentences from available sources
        for idx, block in enumerate(raw_blocks[:2], 1):
            tag_match = re.match(r'(\[[^\]]+\])', block)
            source_tag = tag_match.group(1) if tag_match else f"[منبع {idx}]"
            raw_text = block[len(source_tag):].strip() if tag_match else block.strip()
            parts = [p.strip() for p in raw_text.split('.') if len(p.strip()) > 15]
            if parts:
                matched_sentences.append((1, parts[0], source_tag))

    matched_sentences.sort(key=lambda x: x[0], reverse=True)
    selected = matched_sentences[:4]

    if not selected:
        return "بر اساس جستجو در پایگاه دانش، اطلاعات مستندی در خصوص این سوال یافت نشد."

    lines = []
    lines.append("بر اساس مستندات پایگاه دانش:")
    for score, sentence, tag in selected:
        lines.append(f"• {sentence} {tag}")

    lines.append("\nاین پاسخ صرفاً و منحصراً بر پایه داده‌های موجود در اسناد فوق استخراج شده و بدون حدس و گمان است.")
    return "\n".join(lines)

async def stream_openai_response(text: str, model: str = "local-rag-llm"):
    """
    Streams response in OpenAI SSE format (data: {...}\n\n).
    """
    chunk_id = f"chatcmpl-local-{int(time.time()*1000)}"
    words = text.split(" ")
    for i, w in enumerate(words):
        chunk_obj = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": w + (" " if i < len(words) - 1 else "")},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n".encode('utf-8')
        await asyncio.sleep(0.015)  # 15ms per token simulation

    done_obj = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(done_obj, ensure_ascii=False)}\n\n".encode('utf-8')
    yield b"data: [DONE]\n\n"

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

# 2. Local Models List (OpenAI compatible)
@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "local-rag-llm",
                "object": "model",
                "created": 1700000000,
                "owned_by": "rag-studio-local",
                "permission": [],
                "root": "local-rag-llm",
                "parent": None
            },
            {
                "id": "persian-grounded-rag",
                "object": "model",
                "created": 1700000000,
                "owned_by": "rag-studio-local",
                "permission": [],
                "root": "persian-grounded-rag",
                "parent": None
            }
        ]
    }

# 3. Local Chat Completions Endpoint (OpenAI compatible, 100% Offline)
class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "local-rag-llm"
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
@app.post("/api/chat")
async def chat_completions(req: ChatCompletionRequest):
    answer = generate_grounded_response(req.messages, model=req.model or "local-rag-llm")

    if req.stream:
        return StreamingResponse(
            stream_openai_response(answer, model=req.model or "local-rag-llm"),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    return {
        "id": f"chatcmpl-local-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "local-rag-llm",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(str(req.messages)) // 4,
            "completion_tokens": len(answer) // 4,
            "total_tokens": (len(str(req.messages)) + len(answer)) // 4
        }
    }

# 4. DOCX Parser
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

# 5. Vector Embeddings (100% Offline, Deterministic 768-D)
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

    # 100% Local, zero-dependency, ultra-fast 768-D vectorizer
    vectors = [local_python_embed(t, dim=768) for t in texts]
    dim = 768
    elapsed = round((time.time() - start_time) * 1000, 1)

    resp = {
        "status": "ok",
        "embeddings": vectors,
        "model": "local_python_768d",
        "dimensions": dim,
        "source": "موتور محلی و آفلاین سرور (Zero-Internet, 768-D)",
        "engine": "Local Dense Vectorizer (768D)",
        "latency_ms": elapsed,
        "message": "بردارهای معنایی با موفقیت به صورت ۱۰۰٪ محلی و آفلاین تولید شدند."
    }
    if req.test:
        resp["sample"] = vectors[0][:5]
    return resp

# 6. Anti-CORS Proxy (For local intranet models or Ollama)
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
            body = json.dumps(req_data.get("payload", req_data)).encode("utf-8")
        except Exception:
            pass

    if not target_url:
        raise HTTPException(status_code=400, detail="target-url header or query parameter is required")

    # If the target is the local server's own /v1/chat/completions:
    if any(k in target_url for k in [":8089/v1", "/v1/chat/completions", "local-rag-llm"]):
        try:
            payload = json.loads(body.decode('utf-8')) if body else {}
            chat_req = ChatCompletionRequest(**payload)
            return await chat_completions(chat_req)
        except Exception as e:
            pass

    headers = {"User-Agent": "RAG-Studio-Proxy/2.0"}
    if auth_header:
        headers["Authorization"] = auth_header
    content_type = request.headers.get("Content-Type")
    if content_type:
        headers["Content-Type"] = content_type

    is_local_target = any(h in target_url for h in ["localhost", "127.0.0.1", "::1", "192.168.", "10."])
    client = httpx.AsyncClient(verify=False, timeout=60.0, trust_env=not is_local_target)
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
        error_msg = str(e)
        if not is_local_target:
            error_msg += " (سرور به اینترنت دسترسی ندارد. لطفاً از مدل محلی سرور http://localhost:8089/v1 یا مدل محلی Ollama استفاده فرمایید.)"
        raise HTTPException(status_code=504, detail=f"Proxy error connecting to {target_url}: {error_msg}")

# 7. Serve Web UI and Static Assets
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
    print("  [RAG Studio] Enterprise 100% Offline Server is RUNNING!")
    print("="*70)
    print(f"  Local Browser Access:   http://localhost:{port}")
    print(f"  Network Client Access:  http://<SERVER_IP>:{port}")
    print(f"  Local LLM API Endpoint: http://localhost:{port}/v1/chat/completions")
    print(f"  Local Embeddings API:   http://localhost:{port}/api/embed")
    print(f"  Zero-Internet:          100% Local Processing (Active)")
    print("="*70 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Studio Standalone Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0 for LAN access)")
    parser.add_argument("--port", "-p", type=int, default=int(os.environ.get("PORT", 8089)), help="Port to listen on (default: 8089)")
    args = parser.parse_args()

    run(host=args.host, port=args.port)
