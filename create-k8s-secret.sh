#!/bin/bash

###############################################################################
# SETRAF - Script de création des secrets Kubernetes depuis .env
###############################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Fichier .env non trouvé: $ENV_FILE"
    exit 1
fi

echo "🔐 Création du secret Kubernetes depuis .env..."

# Charger les variables d'environnement
source "$ENV_FILE"

# Créer le secret
kubectl create secret generic setraf-secrets \
    --namespace=setraf \
    --from-literal=mongo-uri="${MONGO_URI}" \
    --from-literal=mongo-user="${MONGO_USER}" \
    --from-literal=mongo-password="${MONGO_PASSWORD}" \
    --from-literal=mongo-cluster="${MONGO_CLUSTER}" \
    --from-literal=mongo-db-name="${MONGO_DB_NAME}" \
    --from-literal=jwt-secret="${JWT_SECRET}" \
    --from-literal=jwt-refresh-secret="${JWT_REFRESH_SECRET}" \
    --from-literal=email-user="${EMAIL_USER}" \
    --from-literal=email-pass="${EMAIL_PASS}" \
    --from-literal=public-key="${PUBLIC_KEY:-}" \
    --from-literal=private-key="${PRIVATE_KEY:-}" \
    --from-literal=hf-token="${HF_TOKEN:-}" \
    --from-literal=tavily-api-key="${TAVILY_API_KEY:-}" \
    --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Secret créé avec succès!"
