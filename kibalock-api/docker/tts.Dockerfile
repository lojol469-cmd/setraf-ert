# TTS Service - NumPy 1.22.0 isolé
FROM nvidia/cuda:13.0.0-base-ubuntu22.04

LABEL maintainer="KibaLock Team"
LABEL description="TTS Service - Text-to-Speech with Coqui TTS"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip

WORKDIR /app

COPY docker/requirements-tts.txt .

# Installation TTS avec numpy 1.22.0
RUN pip install --no-cache-dir numpy==1.22.0
RUN pip install --no-cache-dir scipy==1.11.2

# PyTorch CUDA 13.0
RUN pip install --no-cache-dir --pre \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# TTS et dépendances
RUN pip install --no-cache-dir -r requirements-tts.txt

# Copie code
COPY docker/tts_api.py .

RUN mkdir -p /models/tts /app/logs

EXPOSE 8001

CMD ["python3.10", "-m", "uvicorn", "tts_api:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1"]
