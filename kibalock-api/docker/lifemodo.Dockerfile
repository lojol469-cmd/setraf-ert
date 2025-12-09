# LifeModo API - Transformers + FastAPI
# NumPy 1.23.5 pour compatibilité transformers
FROM nvidia/cuda:13.0.0-base-ubuntu22.04

LABEL maintainer="KibaLock Team"
LABEL description="LifeModo API - AI Training Service with GPU Support"

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Mise à jour pip
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Répertoire de travail
WORKDIR /app

# Copie des requirements spécifiques LifeModo
COPY docker/requirements-lifemodo.txt .

# Installation des dépendances Python (ordre important !)
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.10.1 \
    scikit-learn==1.3.2

# Installation PyTorch CUDA 13.0
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# Installation des autres dépendances
RUN pip install --no-cache-dir -r requirements-lifemodo.txt

# Copie du code source
COPY lifemodo_api.py .
COPY kibalock_agent_simple.py .
COPY .env .

# Création des dossiers nécessaires
RUN mkdir -p /models/huggingface /models/transformers /app/logs

# Health check endpoint
COPY docker/healthcheck-lifemodo.py /app/

# Exposition du port
EXPOSE 8000

# Commande de démarrage
CMD ["python3.10", "-m", "uvicorn", "lifemodo_api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
