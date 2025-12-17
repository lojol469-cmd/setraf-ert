# Dockerfile pour HuggingFace Spaces
# Updated: 2025-12-15
FROM python:3.10-slim

WORKDIR /app

# Installer dépendances système pour matplotlib et pygimli
RUN apt-get update && apt-get install -y \
    libgomp1 \
    fonts-dejavu-core \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers
COPY requirements.txt .
COPY ERTest.py .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Créer répertoires avec permissions pour upload de fichiers
RUN mkdir -p /tmp/streamlit && chmod 777 /tmp/streamlit

# Configurer matplotlib pour backend Agg (sans display)
ENV MPLBACKEND=Agg
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Port Streamlit
EXPOSE 7860

# Variables d'environnement
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Lancer Streamlit
CMD ["streamlit", "run", "ERTest.py", "--server.port=7860", "--server.address=0.0.0.0"]
