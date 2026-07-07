FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch to prevent downloading massive CUDA binaries
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --user -r requirements.txt
#Pre-cache the Hugging Face embedding model to avoid startup download delays
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"


FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH \
    HF_HOME=/app/cache/huggingface \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/cache /app/cache
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]

