# Multi-stage Dockerfile for the RedSea GPT demo.
#
# Stage 1 (web-build): compile the React frontend to static assets.
# Stage 2 (runtime): Python image with the API + committed vector index +
#   built frontend. Serves both /api/* and the static SPA (single-origin),
#   but in the Vercel split-deploy we point Vercel at the static build and
#   only the /api routes hit this container.
#
# The demo runs baseline RAG (REDSEA_ENGINE=baseline by default), which does
# NOT load the 1.1GB reranker at boot. The agent path lazily loads it and
# falls back to dense-only retrieval if the weights are absent — so we do NOT
# bake the reranker into the image. That keeps the image ~2.5GB smaller and
# the build minutes shorter. Set REDSEA_ENGINE=agent at runtime if you want
# the CRAG path (it'll download the reranker on first agent request).

# ---------- Stage 1: build the React frontend ----------
FROM node:20-slim AS web-build
WORKDIR /build
# Install deps first for layer caching
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
# Build. VITE_API_BASE_URL is intentionally unset here: default "/api" means
# same-origin, which works because this same image serves the static files.
# For the Vercel split-deploy, Vercel builds the frontend separately with
# VITE_API_BASE_URL set to this Railway backend's URL.
RUN npm run build

# ---------- Stage 2: Python runtime ----------
FROM python:3.10-slim AS runtime
WORKDIR /app

# System build deps for scientific Python (chromadb, sentence-transformers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (layer-cached; only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY api/ ./api/
COPY generation/ ./generation/
COPY evaluation/ ./evaluation/

# Committed ChromaDB vector index (107MB) — enables retrieval on cold boot
# without re-running ingestion. PDFs are NOT included (copyright); the index
# is sufficient.
COPY chroma_redsea/ ./chroma_redsea/

# Built frontend from stage 1 (served as static files via REDSEA_WEB_DIR)
COPY --from=web-build /build/dist ./web/dist

# Runtime config
ENV REDSEA_WEB_DIR=/app/web/dist \
    REDSEA_ENGINE=baseline \
    PYTHONUNBUFFERED=1 \
    PORT=8787

EXPOSE 8787

# Healthcheck: FastAPI /api/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8787}/api/health || exit 1

# Railway/Render inject PORT; honour it. uvicorn is the ASGI server.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8787}"]
