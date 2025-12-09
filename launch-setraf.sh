#!/bin/bash

###############################################################################
# SETRAF Launch Kernel - Lanceur automatique d'images Docker
# Trouve et lance automatiquement la dernière image SETRAF disponible
###############################################################################

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          🚀 SETRAF Launch Kernel                            ║${NC}"
echo -e "${CYAN}║          Lanceur automatique d'images Docker                ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Fonction pour trouver le port libre suivant
find_free_port() {
    local port=8501
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo $port
}

# Fonction pour nettoyer les anciens containers
cleanup_old_containers() {
    echo -e "${YELLOW}🧹 Nettoyage des anciens containers SETRAF...${NC}"

    # Arrêter les containers qui ne répondent pas
    docker ps -a --filter "name=setraf-auto" --format "{{.Names}}" | while read -r container; do
        if docker ps --filter "name=$container" --format "{{.Names}}" | grep -q "$container"; then
            # Container en cours d'exécution, vérifier s'il répond
            local port=$(docker port "$container" 2>/dev/null | head -1 | cut -d: -f2)
            if [ -n "$port" ] && ! curl -s --max-time 2 "http://localhost:$port" >/dev/null 2>&1; then
                echo -e "${YELLOW}  Arrêt du container non-répondant: $container${NC}"
                docker stop "$container" >/dev/null 2>&1 || true
            fi
        fi
    done

    # Supprimer les containers arrêtés
    docker ps -a --filter "name=setraf-auto" --filter "status=exited" --format "{{.Names}}" | while read -r container; do
        echo -e "${YELLOW}  Suppression du container arrêté: $container${NC}"
        docker rm "$container" >/dev/null 2>&1 || true
    done
}

# Recherche des images SETRAF disponibles
echo -e "${BLUE}🔍 Recherche des images SETRAF disponibles...${NC}"

# Liste des images SETRAF triées par date (plus récente en premier)
SETRAF_IMAGES=$(docker images --format "table {{.Repository}}:{{.Tag}}\t{{.CreatedAt}}" | grep -E "(setraf|kibali)" | sort -k2 -r | head -5)

if [ -z "$SETRAF_IMAGES" ]; then
    echo -e "${RED}❌ Aucune image SETRAF trouvée${NC}"
    echo -e "${YELLOW}Vérifiez que vous avez construit ou téléchargé une image SETRAF${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Images SETRAF trouvées:${NC}"
echo "$SETRAF_IMAGES" | nl -w2 -s'. '
echo ""

# Sélection de la meilleure image (priorité aux images complètes)
BEST_IMAGE=""
for img in $(echo "$SETRAF_IMAGES" | awk '{print $1}'); do
    if [[ "$img" == *"setraf-ert"* ]]; then
        BEST_IMAGE="$img"
        break
    elif [[ "$img" == *"kibaertanalyste"* ]]; then
        BEST_IMAGE="$img"
        break
    fi
done

if [ -z "$BEST_IMAGE" ]; then
    BEST_IMAGE=$(echo "$SETRAF_IMAGES" | head -1 | awk '{print $1}')
fi

echo -e "${GREEN}🎯 Image sélectionnée: ${BEST_IMAGE}${NC}"
echo ""

# Nettoyage
cleanup_old_containers

# Recherche d'un port libre
FREE_PORT=$(find_free_port)
echo -e "${GREEN}📡 Port libre trouvé: $FREE_PORT${NC}"
echo ""

# Génération du nom de container unique
CONTAINER_NAME="setraf-auto-$(date +%s)"

# Lancement du container
echo -e "${YELLOW}🚀 Lancement du container...${NC}"
echo -e "  Image: $BEST_IMAGE"
echo -e "  Container: $CONTAINER_NAME"
echo -e "  Port: $FREE_PORT"
echo ""

# Déterminer le port interne selon l'image
if [[ "$BEST_IMAGE" == *"setraf-ert"* ]]; then
    INTERNAL_PORT=7860  # Les images setraf-ert lancent sur 7860
elif [[ "$BEST_IMAGE" == *"kibaertanalyste"* ]]; then
    INTERNAL_PORT=8504
else
    INTERNAL_PORT=8501
fi

# Lancement
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$FREE_PORT:$INTERNAL_PORT" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/data:/app/data" \
    "$BEST_IMAGE"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Échec du lancement du container${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Container lancé avec succès !${NC}"
echo ""

# Attente du démarrage
echo -e "${YELLOW}⏳ Attente du démarrage de l'application...${NC}"
for i in {1..30}; do
    if curl -s --max-time 2 "http://localhost:$FREE_PORT" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Application opérationnelle !${NC}"
        break
    fi

    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Timeout - l'application n'a pas démarré${NC}"
        echo -e "${YELLOW}Logs du container:${NC}"
        docker logs "$CONTAINER_NAME" | tail -10
        exit 1
    fi

    echo -n "."
    sleep 1
done

echo ""
echo ""

# Affichage des informations finales
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  ✅ SETRAF opérationnel !                                   ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  🌐 URL: http://localhost:${FREE_PORT}                       ║${NC}"
echo -e "${GREEN}║  📦 Image: ${BEST_IMAGE}                                     ║${NC}"
echo -e "${GREEN}║  🐳 Container: ${CONTAINER_NAME}                              ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  Commandes utiles:                                           ║${NC}"
echo -e "${GREEN}║    docker logs ${CONTAINER_NAME}      # Voir les logs        ║${NC}"
echo -e "${GREEN}║    docker stop ${CONTAINER_NAME}      # Arrêter              ║${NC}"
echo -e "${GREEN}║    ./launch-setraf.sh stop          # Arrêter tous          ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Ouverture automatique dans le navigateur (optionnel)
echo -e "${CYAN}💡 Ouvrez http://localhost:${FREE_PORT} dans votre navigateur${NC}"
echo ""

# Mode monitoring si demandé
if [ "$1" = "monitor" ]; then
    echo -e "${YELLOW}📊 Mode monitoring activé (Ctrl+C pour quitter)${NC}"
    echo ""
    while true; do
        if ! docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
            echo -e "${RED}❌ Container arrêté${NC}"
            exit 1
        fi

        # Vérifier la santé
        if curl -s --max-time 2 "http://localhost:$FREE_PORT" >/dev/null 2>&1; then
            echo -e "$(date '+%H:%M:%S') - ${GREEN}✅ Application OK${NC}"
        else
            echo -e "$(date '+%H:%M:%S') - ${RED}❌ Application KO${NC}"
        fi

        sleep 10
    done
fi

# Commande stop
if [ "$1" = "stop" ]; then
    echo -e "${YELLOW}🛑 Arrêt de tous les containers SETRAF...${NC}"
    docker ps -a --filter "name=setraf-auto" --format "{{.Names}}" | while read -r container; do
        echo -e "  Arrêt de $container"
        docker stop "$container" >/dev/null 2>&1 || true
        docker rm "$container" >/dev/null 2>&1 || true
    done
    echo -e "${GREEN}✅ Tous les containers SETRAF arrêtés${NC}"
    exit 0
fi

# Commande status
if [ "$1" = "status" ]; then
    echo -e "${CYAN}📊 Statut des containers SETRAF actifs:${NC}"
    echo ""
    docker ps --filter "name=setraf-auto" --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}"
    exit 0
fi