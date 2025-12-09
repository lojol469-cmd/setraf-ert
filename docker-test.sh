#!/bin/bash

###############################################################################
# SETRAF - Docker Test Script
# Test l'image Docker localement avant le push
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
CONTAINER_NAME="setraf-test"
PORT=8504

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          🧪 SETRAF - Docker Test                             ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Vérifier que l'image existe
if ! docker images "${DOCKER_USERNAME}/${IMAGE_NAME}" | grep -q "${VERSION}"; then
    echo -e "${RED}❌ Image ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} introuvable${NC}"
    echo -e "${YELLOW}Lancez d'abord: ./docker-build.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Image trouvée${NC}"
echo ""

# Arrêter le container de test s'il existe
if docker ps -a | grep -q "${CONTAINER_NAME}"; then
    echo -e "${YELLOW}🛑 Arrêt du container de test existant...${NC}"
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

# Démarrer le container de test
echo -e "${YELLOW}🚀 Démarrage du container de test...${NC}"
echo ""

docker run -d \
    --name "${CONTAINER_NAME}" \
    -p ${PORT}:8504 \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/data:/app/data" \
    "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Échec du démarrage du container${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Container démarré${NC}"
echo ""

# Attendre que l'application démarre
echo -e "${YELLOW}⏳ Attente du démarrage de l'application (30s max)...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:${PORT}/_stcore/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Application prête !${NC}"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Timeout - l'application n'a pas démarré${NC}"
        echo -e "${YELLOW}Logs du container:${NC}"
        docker logs "${CONTAINER_NAME}"
        docker stop "${CONTAINER_NAME}"
        docker rm "${CONTAINER_NAME}"
        exit 1
    fi
    
    echo -n "."
    sleep 1
done

echo ""
echo ""

# Afficher les informations
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ SETRAF fonctionne correctement !                        ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  🌐 URL: http://localhost:${PORT}                            ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  Container: ${CONTAINER_NAME}                                       ║${NC}"
echo -e "${GREEN}║  Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}                       ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  Commandes utiles:                                           ║${NC}"
echo -e "${GREEN}║    docker logs ${CONTAINER_NAME}      # Voir les logs               ║${NC}"
echo -e "${GREEN}║    docker stop ${CONTAINER_NAME}      # Arrêter                     ║${NC}"
echo -e "${GREEN}║    docker rm ${CONTAINER_NAME}        # Supprimer                   ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Ouvrir le navigateur automatiquement (optionnel)
echo -e "${CYAN}💡 Ouvrez http://localhost:${PORT} dans votre navigateur${NC}"
echo ""

# Afficher les logs en temps réel
echo -e "${YELLOW}📄 Logs du container (Ctrl+C pour quitter):${NC}"
echo ""
docker logs -f "${CONTAINER_NAME}"
