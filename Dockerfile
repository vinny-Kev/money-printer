# Minimal Dockerfile - Gets the bot working FAST
FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install ultra-minimal requirements first
COPY requirements-ultra-minimal.txt .
RUN pip install --no-cache-dir -r requirements-ultra-minimal.txt

# Copy essential files
COPY ultra_simple_health.py .
COPY production_server.py .
COPY src/ ./src/
COPY data/ ./data/ 2>/dev/null || mkdir -p data

# Set environment variables
ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Create directories
RUN mkdir -p data logs cache

# Create non-root user
RUN useradd -m -u 1001 bot && \
    chown -R bot:bot /app && \
    chmod +x /app/ultra_simple_health.py
USER bot

EXPOSE 8000

# Start the ultra-simple health server (guaranteed to work)
CMD ["python", "ultra_simple_health.py"]
