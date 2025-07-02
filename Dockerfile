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

# Copy ALL source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Create directories
RUN mkdir -p data logs cache

# Create non-root user
RUN useradd -m bot && chown -R bot:bot /app
USER bot

EXPOSE 8000

# Start the production server (handles health checks and Discord bot gracefully)
CMD ["python", "production_server.py"]
