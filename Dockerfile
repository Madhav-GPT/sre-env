FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENABLE_WEB_INTERFACE=true \
    PORT=7860 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Install the package + Gradio UI deps (anthropic / openai / gradio for the BYOK UI).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir 'gradio>=5.0.0' anthropic openai huggingface_hub structlog pyyaml

EXPOSE 7860

# app.py exposes a module-level ``app`` object that is the FastAPI server with
# Gradio mounted at /. Single port (7860) serves both the UI at / and every
# OpenEnv / MCP / metadata route at /tasks /step /reset /mcp /health /info etc.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
