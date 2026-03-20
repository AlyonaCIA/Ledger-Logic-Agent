FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency manifest first (layer cache)
COPY pyproject.toml .

# Install runtime dependencies into the system Python (no venv needed in container)
RUN uv pip install --system --no-cache \
    "fastapi[standard]" \
    uvicorn \
    "google-cloud-aiplatform>=1.49.0" \
    google-genai \
    requests \
    python-dotenv \
    pydantic \
    pymupdf

# Copy application source
COPY main.py .
COPY agent/ ./agent/

# Cloud Run injects PORT; default 8080
ENV PORT=8080

# Auth via GCP Application Default Credentials (Workload Identity on Cloud Run).
# GCP_PROJECT_ID and VERTEX_LOCATION are injected at runtime via --set-env-vars.
# Optional API_KEY to protect /solve is injected via --set-secrets.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1"]
