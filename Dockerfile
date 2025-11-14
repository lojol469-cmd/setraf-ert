# ========================================
# SETRAF Frontend - Hugging Face Spaces
# Dockerfile optimisé pour PyGIMLi + Streamlit
# ========================================

FROM python:3.10-bullseye

# Éviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système pour PyGIMLi
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libgmp-dev \
    libmpfr-dev \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers de l'application
COPY . .

# Créer le répertoire .streamlit pour la configuration
RUN mkdir -p /app/.streamlit

# Créer le fichier de configuration Streamlit
RUN echo '[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
port = 7860\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = "#667eea"\n\
backgroundColor = "#ffffff"\n\
secondaryBackgroundColor = "#f0f2f6"\n\
textColor = "#262730"\n\
font = "sans serif"' > /app/.streamlit/config.toml

# Exposer le port 7860 (requis par Hugging Face Spaces)
EXPOSE 7860

# Commande de lancement
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
