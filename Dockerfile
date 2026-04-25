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

# app.py mounts the Gradio terminal-UI inside the OpenEnv FastAPI server, so a
# single port (7860) serves both the UI at / and the OpenEnv routes at /tasks
# /step /reset /mcp /health.
CMD ["python", "app.py"]
