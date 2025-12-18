# ==========================================
# SETRAF Authentication Backend - Dockerfile
# Node.js Express + MongoDB + WebSocket
# ==========================================

FROM node:18-alpine

# Métadonnées
LABEL maintainer="BelikanM"
LABEL description="SETRAF Authentication Server with JWT, MongoDB, WebSocket"
LABEL version="1.0.0"

# Installer des outils système basiques
RUN apk add --no-cache \
    curl \
    bash \
    openssl

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY package*.json ./

# Installer les dépendances
RUN npm install --omit=dev && \
    npm cache clean --force

# Copier le code source
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p logs public

# Exposer les ports
# 5000: API REST
# 5001: WebSocket (si différent)
EXPOSE 5000

# Variables d'environnement par défaut
ENV NODE_ENV=production \
    AUTH_PORT=5000

# Healthcheck pour vérifier que le serveur répond
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Utilisateur non-root pour la sécurité
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001 && \
    chown -R nodejs:nodejs /app

USER nodejs

# Commande de démarrage
CMD ["node", "server.js"]
