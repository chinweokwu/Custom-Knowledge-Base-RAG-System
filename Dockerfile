# --- STAGE 1: Builder ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Torch and Torchvision together to ensure compatibility
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --no-cache-dir --user --default-timeout=1000 -r requirements.txt


# --- STAGE 2: Runtime ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/models_cache \
    LOG_FILE_PATH=/app/logs/app.log \
    CHROMA_DB_PATH=/app/chroma_db \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY . .

RUN mkdir -p /app/models_cache /app/logs /app/chroma_db /app/uploads /app/media

EXPOSE 8000 9382

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
