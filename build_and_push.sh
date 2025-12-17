#!/bin/bash

# =====================================================
# SETRAF - Script de Build et Push vers Docker Hub
# Usage: ./build_and_push.sh [version]
# Exemple: ./build_and_push.sh v2.0.0
# =====================================================

set -e  # Arrêt en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_USERNAME="belikanm"
DOCKER_REGISTRY="docker.io"
IMAGE_NAME="setraf"
DOCKERFILE="Dockerfile.optimized"

# Version (par défaut: latest)
VERSION="${1:-latest}"
IMAGE_TAG="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
IMAGE_LATEST="${DOCKER_USERNAME}/${IMAGE_NAME}:latest"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🐳 SETRAF - Build & Push Docker Image${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  📦 Image: ${IMAGE_TAG}"
echo -e "  📝 Dockerfile: ${DOCKERFILE}"
echo -e "  👤 Username: ${DOCKER_USERNAME}"
echo ""

# Vérifier que le Dockerfile existe
if [ ! -f "${DOCKERFILE}" ]; then
    echo -e "${RED}❌ Erreur: ${DOCKERFILE} n'existe pas${NC}"
    exit 1
fi

# Vérifier que startup.sh existe
if [ ! -f "startup.sh" ]; then
    echo -e "${RED}❌ Erreur: startup.sh n'existe pas${NC}"
    exit 1
fi

# Rendre startup.sh exécutable
chmod +x startup.sh
echo -e "${GREEN}✅ startup.sh est exécutable${NC}"
echo ""

# Authentification Docker Hub
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}🔐 Authentification Docker Hub${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Vérifier si déjà connecté
if docker info 2>/dev/null | grep -q "Username: ${DOCKER_USERNAME}"; then
    echo -e "${GREEN}✅ Déjà connecté à Docker Hub en tant que ${DOCKER_USERNAME}${NC}"
else
    echo -e "${BLUE}🔑 Connexion à Docker Hub...${NC}"
    echo "   Username: ${DOCKER_USERNAME}"
    echo ""
    
    # Demander le token (masqué)
    read -sp "   Token: " DOCKER_TOKEN
    echo ""
    
    # Connexion
    echo "${DOCKER_TOKEN}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Authentification réussie${NC}"
    else
        echo -e "${RED}❌ Échec de l'authentification${NC}"
        exit 1
    fi
fi
echo ""

# Build de l'image
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}🔨 Build de l'image Docker${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

BUILD_START=$(date +%s)

docker build \
    -t "${IMAGE_TAG}" \
    -t "${IMAGE_LATEST}" \
    -f "${DOCKERFILE}" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VERSION="${VERSION}" \
    --progress=plain \
    .

BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))

echo ""
echo -e "${GREEN}✅ Build terminé en ${BUILD_DURATION}s${NC}"
echo ""

# Afficher les informations de l'image
echo -e "${BLUE}📊 Informations de l'image:${NC}"
docker images "${DOCKER_USERNAME}/${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo ""

# Test de l'image (optionnel)
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}🧪 Test de l'image (optionnel)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
read -p "Voulez-vous tester l'image localement avant de push ? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🚀 Démarrage du container de test...${NC}"
    docker run --rm -d \
        --name setraf-test \
        -p 8504:8504 \
        -e DOWNLOAD_MISTRAL=false \
        "${IMAGE_TAG}"
    
    echo ""
    echo -e "${GREEN}✅ Container de test démarré${NC}"
    echo -e "${BLUE}   URL: http://localhost:8504${NC}"
    echo -e "${YELLOW}   Appuyez sur ENTER pour arrêter le test et continuer...${NC}"
    read
    
    docker stop setraf-test
    echo -e "${GREEN}✅ Test terminé${NC}"
    echo ""
fi

# Push vers Docker Hub
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📤 Push vers Docker Hub${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

PUSH_START=$(date +%s)

# Push de la version spécifique
echo -e "${BLUE}📤 Push ${IMAGE_TAG}...${NC}"
docker push "${IMAGE_TAG}"

# Push de latest si ce n'est pas déjà fait
if [ "${VERSION}" != "latest" ]; then
    echo -e "${BLUE}📤 Push ${IMAGE_LATEST}...${NC}"
    docker push "${IMAGE_LATEST}"
fi

PUSH_END=$(date +%s)
PUSH_DURATION=$((PUSH_END - PUSH_START))

echo ""
echo -e "${GREEN}✅ Push terminé en ${PUSH_DURATION}s${NC}"
echo ""

# Résumé
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ DÉPLOIEMENT RÉUSSI${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Images disponibles:${NC}"
echo -e "  🐳 docker pull ${IMAGE_TAG}"
if [ "${VERSION}" != "latest" ]; then
    echo -e "  🐳 docker pull ${IMAGE_LATEST}"
fi
echo ""
echo -e "${GREEN}Lien Docker Hub:${NC}"
echo -e "  🔗 https://hub.docker.com/r/${DOCKER_USERNAME}/${IMAGE_NAME}"
echo ""
echo -e "${YELLOW}Prochaines étapes:${NC}"
echo -e "  1. Déployer avec Docker Compose:"
echo -e "     ${BLUE}docker-compose -f docker-compose.production.yml up -d${NC}"
echo ""
echo -e "  2. Ou déployer sur Kubernetes:"
echo -e "     ${BLUE}kubectl apply -f kubernetes/${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
