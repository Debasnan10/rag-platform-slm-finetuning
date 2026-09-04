# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# System deps: curl for the HEALTHCHECK, build-essential because a couple
# of the ML wheels (faiss-cpu, sentence-transformers deps) occasionally
# need to compile small C extensions on install.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy just the dependency manifest first so Docker can cache this layer
# and skip the (slow) pip install when only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
