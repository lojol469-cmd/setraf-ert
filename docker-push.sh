#!/bin/bash

###############################################################################
# SETRAF - Docker Push Script
# Push l'image Docker vers Docker Hub
###############################################################################

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
DOCKER_USERNAME="belikanm"
IMAGE_NAME="kibaertanalyste"
VERSION="1.0.0"

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          🐋 SETRAF - Docker Push                             ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker n'est pas installé ou non accessible${NC}"
    exit 1
fi

# Vérifier que l'image existe
if ! docker images "${DOCKER_USERNAME}/${IMAGE_NAME}" | grep -q "${VERSION}"; then
    echo -e "${RED}❌ Image ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} introuvable${NC}"
    echo -e "${YELLOW}Lancez d'abord: ./docker-build.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Image trouvée${NC}"
echo ""

# Afficher les informations
echo -e "${CYAN}Configuration du push:${NC}"
echo -e "  Repository: ${GREEN}${DOCKER_USERNAME}/${IMAGE_NAME}${NC}"
echo -e "  Versions: ${GREEN}${VERSION}, latest${NC}"
echo ""

# Vérifier l'authentification Docker Hub
echo -e "${YELLOW}🔐 Vérification de l'authentification Docker Hub...${NC}"
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo -e "${YELLOW}⚠️  Non authentifié sur Docker Hub${NC}"
    echo -e "${CYAN}Connexion à Docker Hub...${NC}"
    docker login
fi

echo -e "${GREEN}✓ Authentifié${NC}"
echo ""

# Push version spécifique
echo -e "${YELLOW}📤 Push de ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}...${NC}"
docker push "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Version ${VERSION} pushée avec succès${NC}"
else
    echo -e "${RED}❌ Échec du push de la version ${VERSION}${NC}"
    exit 1
fi

echo ""

# Push latest
echo -e "${YELLOW}📤 Push de ${DOCKER_USERNAME}/${IMAGE_NAME}:latest...${NC}"
docker push "${DOCKER_USERNAME}/${IMAGE_NAME}:latest"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Tag latest pushé avec succès${NC}"
else
    echo -e "${RED}❌ Échec du push du tag latest${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ Images Docker publiées avec succès !                     ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  Repository: hub.docker.com/r/${DOCKER_USERNAME}/${IMAGE_NAME}        ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  Utilisation:                                                 ║${NC}"
echo -e "${GREEN}║    docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}                 ║${NC}"
echo -e "${GREEN}║    docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:latest                    ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  Démarrage:                                                   ║${NC}"
echo -e "${GREEN}║    docker run -d -p 8504:8504 \\                              ║${NC}"
echo -e "${GREEN}║      --name setraf \\                                         ║${NC}"
echo -e "${GREEN}║      ${DOCKER_USERNAME}/${IMAGE_NAME}:latest                            ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  Accès: http://localhost:8504                                ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
