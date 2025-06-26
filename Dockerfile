# Enhanced Discord Bot for Railway with Trading Features
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ta-lib and build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install ta-lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements and install Python packages
COPY requirements-linux.txt .
RUN pip install --no-cache-dir -r requirements-linux.txt

# Copy ALL source code - this is the key fix!
COPY . .

# Ensure the src directory structure is properly set up
RUN ls -la /app/src/ && \
    ls -la /app/src/trading_bot/ && \
    ls -la /app/src/data_collector/ && \
    ls -la /app/src/model_training/

# Create necessary directories with proper permissions
RUN mkdir -p data logs cache ohlcv_cache secrets && \
    chmod 755 data logs cache ohlcv_cache secrets

# Environment variables for Python and Railway
ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Create non-root user for security
RUN useradd -m -u 1001 bot && chown -R bot:bot /app
USER bot

# Expose health check port (Railway uses PORT env var)
EXPOSE 8000

# Health check for Docker (Railway ignores this but good practice)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Start the Discord bot directly (debug environment first)
CMD ["sh", "-c", "python debug_railway.py && python src/lightweight_discord_bot.py"]
