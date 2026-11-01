# syntax=docker/dockerfile:1
ARG DOCKER_REGISTRY=dockerhub.ime.co.ir/
FROM ${DOCKER_REGISTRY}library/python:3.11-slim

LABEL maintainer="mohamadreza1368"
LABEL description="RAG Studio - Interactive and Operational Retrieval-Augmented Generation"

ARG DEBIAN_MIRROR="https://linux.repo.ime.co.ir/repository/debian"
ARG PYPI_INDEX="https://pypi.repo.ime.co.ir/repository/pypi-group/simple"
ARG PYPI_HOST="pypi.repo.ime.co.ir"

# Configure Debian APT mirror
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
        sed -i "s|https://deb.debian.org/debian|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i "s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" /etc/apt/sources.list; \
        sed -i "s|https://deb.debian.org/debian|${DEBIAN_MIRROR}|g" /etc/apt/sources.list; \
    fi

# Configure PyPI mirror environment variables
ENV PIP_INDEX_URL="${PYPI_INDEX}" \
    PIP_TRUSTED_HOST="${PYPI_HOST}" \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOCKER=1 \
    NO_BROWSER=1 \
    PORT=8080

WORKDIR /app

# Copy dependency configuration
COPY requirements.txt pip.conf /app/
ENV PIP_CONFIG_FILE=/app/pip.conf

# Install requirements with fallback stages (supports host network in BuildKit and fallback)
RUN --network=host pip install --no-cache-dir transformers torch accelerate huggingface-hub || \
    pip install --no-cache-dir transformers torch accelerate huggingface-hub || \
    pip install --no-cache-dir -r requirements.txt || true

# Copy all application assets
COPY . /app

# Ensure entrypoint script has Unix line endings and execution permissions
RUN sed -i 's/\r$//' /app/entrypoint.sh 2>/dev/null || true && \
    chmod +x /app/entrypoint.sh 2>/dev/null || true

EXPOSE 8080

# Native healthcheck using python standard library (no curl required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request, os; resp = urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8080)}/health', timeout=3); resp.read()" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "server.py"]
