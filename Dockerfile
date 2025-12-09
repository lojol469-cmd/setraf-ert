# =====================================================
# SETRAF - Subaquifère ERT Analysis Tool
# Dockerfile pour déploiement complet
# =====================================================
FROM python:3.10-slim

LABEL maintainer="Belikan M. <nyundumathryme@gmail.com>"
LABEL description="SETRAF - ERT Geophysical Analysis Platform with Streamlit"
LABEL version="1.0.0"

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8504 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Installation des dépendances système (PyGIMLi needs cmake, libboost, libeigen3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    cmake \
    libboost-all-dev \
    libeigen3-dev \
    libopenblas-dev \
    liblapack-dev \
    libsuitesparse-dev \
    libtrilinos-zoltan-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copie des requirements et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pygimli

# Copie du code source SETRAF
COPY ERTest.py .
COPY ERT.py .
COPY api_setraf.py .
COPY auth_module.py .
COPY logo_belikan.png .
COPY .env .

# Création des dossiers nécessaires
RUN mkdir -p /app/logs /app/data /app/uploads /app/exports

# Exposition des ports
# 8504: ERTest.py (Streamlit standalone)
# 8505: api_setraf.py (FastAPI)
# 8506: ERT.py (Kibali version)
EXPOSE 8504 8505 8506

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8504/_stcore/health || exit 1

# Script d'entrée par défaut: ERTest.py
CMD ["python", "-m", "streamlit", "run", "ERTest.py", \
     "--server.port=8504", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]
