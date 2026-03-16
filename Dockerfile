# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/models_cache \
    LOG_FILE_PATH=/app/logs/app.log \
    CHROMA_DB_PATH=/app/chroma_db

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/models_cache /app/logs /app/chroma_db /app/uploads /app/media

# Expose ports
EXPOSE 8000 9382

# The actual command will be overridden in docker-compose for each service
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
