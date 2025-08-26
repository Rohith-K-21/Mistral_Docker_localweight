# Minimal CUDA runtime
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python + minimal system packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install required Python packages
RUN pip3 install --no-cache-dir \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121 \
    transformers accelerate runpod sentencepiece tokenizers safetensors

# Copy your worker files
COPY handler.py .
COPY runpod.toml .

# Environment variables
ENV MODEL_DIR=/runpod-volume/Mistral_weight
ENV HF_HOME=/runpod-volume/huggingface

# Start worker
CMD ["python3", "-u", "handler.py"]