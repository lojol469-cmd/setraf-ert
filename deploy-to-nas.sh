#!/bin/bash

# =====================================================
# Script de déploiement Docker vers NAS
# Export + Transfert automatique
# =====================================================

# Configuration - À modifier selon vos besoins
IMAGE_NAME="belikanm/kibaertanalyste"
IMAGE_TAG="latest"
TAR_FILE="setraf-app.tar"
NAS_USER="admin"
NAS_IP="192.168.1.100"
NAS_PATH="/volume1/docker/images"

echo "🐋 Déploiement Docker vers NAS"
echo "================================"

# Étape 1: Vérifier que Docker est disponible
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé ou pas dans le PATH"
    exit 1
fi

# Étape 2: Vérifier que l'image existe
if ! docker images | grep -q "$IMAGE_NAME"; then
    echo "❌ Image $IMAGE_NAME:$IMAGE_TAG introuvable"
    echo "Images disponibles :"
    docker images
    exit 1
fi

echo "✅ Image trouvée : $IMAGE_NAME:$IMAGE_TAG"

# Étape 3: Exporter l'image
echo "📦 Export de l'image en .tar..."
docker save -o "$TAR_FILE" "$IMAGE_NAME:$IMAGE_TAG"

if [ $? -eq 0 ]; then
    echo "✅ Export réussi : $TAR_FILE"
    ls -lh "$TAR_FILE"
else
    echo "❌ Échec de l'export"
    exit 1
fi

# Étape 4: Transférer vers le NAS
echo "📤 Transfert vers le NAS..."
scp "$TAR_FILE" "$NAS_USER@$NAS_IP:$NAS_PATH/"

if [ $? -eq 0 ]; then
    echo "✅ Transfert réussi vers $NAS_IP:$NAS_PATH"
else
    echo "❌ Échec du transfert"
    echo "Vérifiez :"
    echo "  - Adresse IP du NAS : $NAS_IP"
    echo "  - Utilisateur : $NAS_USER"
    echo "  - Chemin NAS : $NAS_PATH"
    echo "  - Connexion SSH autorisée"
    exit 1
fi

# Étape 5: Importer sur le NAS (optionnel)
echo "🔄 Import automatique sur le NAS..."
ssh "$NAS_USER@$NAS_IP" "docker load < $NAS_PATH/$TAR_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Import réussi sur le NAS"
    echo "📋 Images disponibles sur le NAS :"
    ssh "$NAS_USER@$NAS_IP" "docker images"
else
    echo "⚠️ Import automatique échoué"
    echo "Importez manuellement via l'interface GUI du NAS"
fi

echo ""
echo "🎉 Déploiement terminé !"
echo "Suivez le guide GUI_DOCKER_NAS.md pour lancer le container"