# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Create model directory
RUN mkdir -p model

# Copy model files if they exist
COPY model/ model/

# Expose port (Render uses dynamic PORT)
EXPOSE 8000

# Health check disabled for Render (uses its own health checks)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Run the application (Render provides PORT env variable)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
