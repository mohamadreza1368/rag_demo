#!/bin/sh
set -e

# Test if transformers and torch are installed
if ! python -c "import transformers, torch" >/dev/null 2>&1; then
    echo "======================================================================"
    echo "📦 [RAG Studio] Transformers/PyTorch not found in the base container."
    echo "⏳ Attempting automatic install from internal repository at startup..."
    echo "======================================================================"
    pip install --no-cache-dir transformers torch accelerate huggingface-hub || true
    if python -c "import transformers, torch" >/dev/null 2>&1; then
        echo "✅ [RAG Studio] Successfully installed Transformers and PyTorch!"
    else
        echo "⚠️ [RAG Studio] Startup install skipped or waiting for network."
        echo "💡 You can install anytime via the web UI or by running:"
        echo "   docker compose exec rag_studio pip install transformers torch accelerate"
    fi
    echo "======================================================================"
fi

exec "$@"
