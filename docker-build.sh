#!/bin/bash

###############################################################################
# SETRAF - Docker Build Script
# Build et tag l'image Docker pour SETRAF
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
DOCKERFILE="Dockerfile"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          🐋 SETRAF - Docker Build                            ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker n'est pas installé ou non accessible${NC}"
    echo -e "${YELLOW}Activez WSL2 Docker integration dans Docker Desktop${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker trouvé${NC}"
echo ""

# Afficher les informations de build
echo -e "${CYAN}Configuration du build:${NC}"
echo -e "  Image: ${GREEN}${DOCKER_USERNAME}/${IMAGE_NAME}${NC}"
echo -e "  Version: ${GREEN}${VERSION}${NC}"
echo -e "  Tags: ${GREEN}${VERSION}, latest${NC}"
echo -e "  Dockerfile: ${GREEN}${DOCKERFILE}${NC}"
echo ""

# Build l'image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
echo ""

docker build \
    -t "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}" \
    -t "${DOCKER_USERNAME}/${IMAGE_NAME}:latest" \
    -f "${DOCKERFILE}" \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Build réussi !${NC}"
    echo ""
    
    # Afficher les informations de l'image
    echo -e "${CYAN}Informations de l'image:${NC}"
    docker images "${DOCKER_USERNAME}/${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ Image Docker créée avec succès !                         ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║  Tags:                                                        ║${NC}"
    echo -e "${GREEN}║    - ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}                       ║${NC}"
    echo -e "${GREEN}║    - ${DOCKER_USERNAME}/${IMAGE_NAME}:latest                          ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║  Prochaines étapes:                                          ║${NC}"
    echo -e "${GREEN}║    1. Tester: docker run -p 8504:8504 ${DOCKER_USERNAME}/${IMAGE_NAME}  ║${NC}"
    echo -e "${GREEN}║    2. Push: ./docker-push.sh                                 ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
else
    echo ""
    echo -e "${RED}❌ Build échoué${NC}"
    exit 1
fi
