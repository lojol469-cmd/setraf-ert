#!/bin/bash

# ==========================================
# SETRAF Stack Management Script
# Gérer le déploiement Docker complet
# ==========================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DOCKER_CMD="/mnt/c/Program Files/Docker/Docker/resources/bin/docker"
COMPOSE_FILE="docker-compose.full.yml"

# Banner
show_banner() {
    echo -e "${BLUE}"
    echo "════════════════════════════════════════════════════"
    echo "   🚀 SETRAF Docker Stack Management"
    echo "════════════════════════════════════════════════════"
    echo -e "${NC}"
}

# Status
show_status() {
    echo -e "${YELLOW}📊 Status des conteneurs SETRAF:${NC}"
    "$DOCKER_CMD" ps -a | grep -E "setraf|CONTAINER"
    echo ""
}

# Logs
show_logs() {
    SERVICE=$1
    if [ -z "$SERVICE" ]; then
        echo -e "${YELLOW}📋 Logs de tous les services:${NC}"
        "$DOCKER_CMD"-compose -f "$COMPOSE_FILE" logs --tail=50
    else
        echo -e "${YELLOW}📋 Logs de $SERVICE:${NC}"
        "$DOCKER_CMD" logs "$SERVICE" --tail=100 -f
    fi
}

# Health check
health_check() {
    echo -e "${YELLOW}🏥 Vérification de santé des services...${NC}\n"
    
    # Backend
    echo -n "Backend (port 5000): "
    if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ FAIL${NC}"
    fi
    
    # API
    echo -n "API FastAPI (port 8505): "
    if curl -s http://localhost:8505/api/status > /dev/null 2>&1; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ FAIL${NC}"
    fi
    
    # Frontend
    echo -n "Frontend Streamlit (port 8504): "
    if curl -s http://localhost:8504/_stcore/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ FAIL${NC}"
    fi
    
    echo ""
}

# URLs
show_urls() {
    echo -e "${BLUE}🔗 Accès aux services:${NC}"
    echo ""
    echo -e "  ${GREEN}Frontend Streamlit:${NC}"
    echo "    http://localhost:8504"
    echo ""
    echo -e "  ${GREEN}Backend Auth API:${NC}"
    echo "    http://localhost:5000"
    echo "    http://localhost:5000/api/health"
    echo ""
    echo -e "  ${GREEN}FastAPI ERT:${NC}"
    echo "    http://localhost:8505"
    echo "    http://localhost:8505/api/docs"
    echo ""
}

# Start
start_stack() {
    echo -e "${YELLOW}🚀 Démarrage du stack SETRAF...${NC}"
    "$DOCKER_CMD"-compose -f "$COMPOSE_FILE" up -d
    echo ""
    sleep 3
    show_status
    echo ""
    health_check
    show_urls
}

# Stop
stop_stack() {
    echo -e "${YELLOW}🛑 Arrêt du stack SETRAF...${NC}"
    "$DOCKER_CMD"-compose -f "$COMPOSE_FILE" down
    echo -e "${GREEN}✅ Stack arrêté${NC}"
}

# Restart
restart_stack() {
    echo -e "${YELLOW}🔄 Redémarrage du stack SETRAF...${NC}"
    "$DOCKER_CMD"-compose -f "$COMPOSE_FILE" restart
    echo ""
    sleep 3
    show_status
    health_check
}

# Update (pull + restart)
update_stack() {
    echo -e "${YELLOW}📥 Mise à jour des images depuis Docker Hub...${NC}"
    "$DOCKER_CMD"-compose -f "$COMPOSE_FILE" pull
    echo ""
    echo -e "${YELLOW}🔄 Redémarrage avec les nouvelles images...${NC}"
    "$DOCKER_CMD"-compose -f "$COMPOSE_FILE" up -d
    echo ""
    sleep 3
    show_status
    health_check
}

# Clean
clean_stack() {
    echo -e "${RED}🧹 Nettoyage complet (containers + volumes)...${NC}"
    read -p "⚠️  Êtes-vous sûr ? Cela supprimera tous les volumes ! (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        "$DOCKER_CMD"-compose -f "$COMPOSE_FILE" down -v
        echo -e "${GREEN}✅ Nettoyage terminé${NC}"
    else
        echo -e "${YELLOW}❌ Annulé${NC}"
    fi
}

# Menu
show_menu() {
    echo -e "${BLUE}Commandes disponibles:${NC}"
    echo "  start    - Démarrer le stack"
    echo "  stop     - Arrêter le stack"
    echo "  restart  - Redémarrer le stack"
    echo "  status   - Voir l'état des conteneurs"
    echo "  logs     - Voir les logs (logs [service])"
    echo "  health   - Vérifier la santé des services"
    echo "  urls     - Afficher les URLs d'accès"
    echo "  update   - Mettre à jour depuis Docker Hub"
    echo "  clean    - Supprimer tout (containers + volumes)"
    echo ""
}

# Main
show_banner

case "$1" in
    start)
        start_stack
        ;;
    stop)
        stop_stack
        ;;
    restart)
        restart_stack
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    health)
        health_check
        ;;
    urls)
        show_urls
        ;;
    update)
        update_stack
        ;;
    clean)
        clean_stack
        ;;
    *)
        show_menu
        echo -e "${YELLOW}Usage: $0 {start|stop|restart|status|logs|health|urls|update|clean}${NC}"
        exit 1
        ;;
esac
