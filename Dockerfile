FROM python:3.11-slim

WORKDIR /app

# System deps for sentencepiece + torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the model weights into the image layer so the container starts fast.
# Set HF_HOME so the cache lands inside the image at a known path.
ENV HF_HOME=/app/.hf_cache
ARG MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"

EXPOSE 8000

CMD ["python", "scripts/run_server.py", \
     "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--device", "cpu", \
     "--num-blocks", "64", \
     "--log-level", "info"]
