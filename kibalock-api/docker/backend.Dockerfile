# Backend KibaLock - FAISS + DeepFace + Streamlit
# NumPy 2.2.6 pour compatibilité FAISS
FROM nvidia/cuda:13.0.0-base-ubuntu22.04

LABEL maintainer="KibaLock Team"
LABEL description="KibaLock Backend - Biometric Authentication with FAISS & DeepFace"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-13.0

# Installation système
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    wget \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Copie requirements spécifiques backend
COPY docker/requirements-backend.txt .

# Installation dépendances (ordre important)
RUN pip install --no-cache-dir numpy>=2.2.0

RUN pip install --no-cache-dir \
    scipy>=1.15.0 \
    scikit-learn>=1.7.0 \
    networkx>=3.4.0

# PyTorch CUDA 13.0
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# FAISS avec GPU
RUN pip install --no-cache-dir faiss-gpu

# Autres dépendances
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copie code source
COPY kibalock_faiss.py .
COPY kibalock.py .
COPY .env .

# Dossiers
RUN mkdir -p /data/faiss_indices /app/logs /models

EXPOSE 8505

# Health check
COPY docker/healthcheck-backend.py /app/

CMD ["python3.10", "-m", "streamlit", "run", "kibalock_faiss.py", \
     "--server.port=8505", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]
