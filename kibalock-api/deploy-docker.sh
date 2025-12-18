#!/bin/bash

###############################################################################
#                                                                             #
#  🔐 KibaLock Docker Deployment Script v1.0                                 #
#  Déploiement complet multi-conteneurs avec GPU                             #
#                                                                             #
###############################################################################

set -e  # Exit on error

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Bannière
print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║          🔐 KibaLock Docker Deployment                       ║
║          Architecture Multi-Conteneurs GPU                    ║
║                                                               ║
║          Services:                                           ║
║          • LifeModo API (Transformers)                       ║
║          • Backend (FAISS + DeepFace)                        ║
║          • TTS Service (Coqui)                               ║
║          • Frontend (React 3D)                               ║
║          • MongoDB                                           ║
║          • Nginx Reverse Proxy                               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Logging
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérifications préalables
check_requirements() {
    log_info "Vérification des prérequis..."
    
    # Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker n'est pas installé !"
        exit 1
    fi
    log_info "✓ Docker $(docker --version | cut -d' ' -f3)"
    
    # Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose n'est pas installé !"
        exit 1
    fi
    log_info "✓ Docker Compose disponible"
    
    # NVIDIA Docker (optionnel mais recommandé)
    if command -v nvidia-smi &> /dev/null; then
        log_info "✓ NVIDIA GPU détecté : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
        
        if docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            log_info "✓ NVIDIA Docker Runtime OK"
        else
            log_warn "NVIDIA Docker Runtime non configuré. GPU non accessible aux conteneurs."
            log_warn "Installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        fi
    else
        log_warn "Aucun GPU NVIDIA détecté. Les conteneurs tourneront en mode CPU."
    fi
    
    # Espace disque
    AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$AVAILABLE_SPACE" -lt 20 ]; then
        log_warn "Espace disque faible : ${AVAILABLE_SPACE}GB disponible. 20GB+ recommandé."
    else
        log_info "✓ Espace disque : ${AVAILABLE_SPACE}GB disponible"
    fi
}

# Création des dossiers nécessaires
create_directories() {
    log_info "Création des dossiers de données..."
    
    mkdir -p models/{huggingface,transformers,tts,faiss}
    mkdir -p data/faiss_indices
    mkdir -p logs/{lifemodo,backend,tts}
    mkdir -p ssl
    
    log_info "✓ Structure de dossiers créée"
}

# Génération .env si absent
generate_env() {
    if [ ! -f .env ]; then
        log_info "Génération du fichier .env..."
        
        # Détection IP locale
        LOCAL_IP=$(ip route get 1.1.1.1 | grep -oP 'src \K\S+' 2>/dev/null || echo "localhost")
        
        cat > .env << EOF
# KibaLock Environment Variables
# Généré automatiquement le $(date)

# Network
LOCAL_IP=${LOCAL_IP}
API_PORT=8000
BACKEND_PORT=8505
TTS_PORT=8001
FRONTEND_PORT=3000

# MongoDB
MONGO_URI=mongodb://kibalock-mongo:27017/
MONGO_DB=kibalock

# GPU
CUDA_VISIBLE_DEVICES=0

# Security (CHANGEZ CES VALEURS EN PRODUCTION!)
JWT_SECRET=$(openssl rand -hex 32)
MONGO_ROOT_PASSWORD=$(openssl rand -hex 16)

# Models
HF_HOME=/models/huggingface
TRANSFORMERS_CACHE=/models/transformers
EOF
        
        log_info "✓ Fichier .env créé avec IP=${LOCAL_IP}"
    else
        log_info "✓ Fichier .env existant utilisé"
    fi
}

# Build des images
build_images() {
    log_info "Build des images Docker..."
    
    docker-compose build --parallel || {
        log_error "Échec du build des images"
        exit 1
    }
    
    log_info "✓ Images construites avec succès"
}

# Pull des images de base (accélère le build)
pull_base_images() {
    log_info "Téléchargement des images de base..."
    
    docker pull nvidia/cuda:13.0.0-cudnn8-runtime-ubuntu22.04 &
    docker pull mongo:7.0 &
    docker pull nginx:alpine &
    docker pull node:20-alpine &
    
    wait
    log_info "✓ Images de base téléchargées"
}

# Démarrage des services
start_services() {
    log_info "Démarrage des services..."
    
    docker-compose up -d || {
        log_error "Échec du démarrage des services"
        docker-compose logs
        exit 1
    }
    
    log_info "✓ Services démarrés"
}

# Vérification santé des services
health_check() {
    log_info "Vérification de la santé des services..."
    
    sleep 10  # Attendre que les services démarrent
    
    SERVICES=("lifemodo-api" "kibalock-backend" "kibalock-mongo")
    
    for service in "${SERVICES[@]}"; do
        if docker ps | grep -q "$service"; then
            STATUS=$(docker inspect --format='{{.State.Health.Status}}' "kibalock-$service" 2>/dev/null || echo "unknown")
            if [ "$STATUS" = "healthy" ] || [ "$STATUS" = "unknown" ]; then
                log_info "✓ $service : OK"
            else
                log_warn "⚠ $service : $STATUS"
            fi
        else
            log_error "✗ $service : Non démarré"
        fi
    done
}

# Affichage des URLs
show_urls() {
    source .env 2>/dev/null || LOCAL_IP="localhost"
    
    echo ""
    log_info "═════════════════════════════════════════════════════════"
    log_info "✅ KibaLock déployé avec succès !"
    log_info "═════════════════════════════════════════════════════════"
    echo ""
    echo -e "${GREEN}📡 Services accessibles :${NC}"
    echo -e "   • LifeModo API     : ${BLUE}http://${LOCAL_IP}:8000${NC}"
    echo -e "   • API Docs         : ${BLUE}http://${LOCAL_IP}:8000/docs${NC}"
    echo -e "   • Backend KibaLock : ${BLUE}http://${LOCAL_IP}:8505${NC}"
    echo -e "   • TTS Service      : ${BLUE}http://${LOCAL_IP}:8001${NC}"
    echo -e "   • Frontend React   : ${BLUE}http://${LOCAL_IP}:3000${NC}"
    echo -e "   • MongoDB          : ${BLUE}mongodb://${LOCAL_IP}:27017${NC}"
    echo ""
    echo -e "${YELLOW}📋 Commandes utiles :${NC}"
    echo "   docker-compose ps              # État des services"
    echo "   docker-compose logs -f         # Logs en temps réel"
    echo "   docker-compose down            # Arrêter tout"
    echo "   docker-compose restart [svc]   # Redémarrer un service"
    echo ""
}

# Nettoyage en cas d'erreur
cleanup() {
    log_error "Interruption détectée. Nettoyage..."
    docker-compose down
    exit 1
}

trap cleanup INT TERM

# Menu principal
main() {
    print_banner
    
    case "${1:-deploy}" in
        deploy)
            check_requirements
            create_directories
            generate_env
            pull_base_images
            build_images
            start_services
            health_check
            show_urls
            ;;
        
        start)
            log_info "Démarrage des services existants..."
            docker-compose up -d
            show_urls
            ;;
        
        stop)
            log_info "Arrêt des services..."
            docker-compose down
            log_info "✓ Services arrêtés"
            ;;
        
        restart)
            log_info "Redémarrage des services..."
            docker-compose restart
            log_info "✓ Services redémarrés"
            ;;
        
        rebuild)
            log_info "Reconstruction des images..."
            docker-compose down
            docker-compose build --no-cache
            docker-compose up -d
            log_info "✓ Images reconstruites et services relancés"
            ;;
        
        logs)
            docker-compose logs -f "${2:-}"
            ;;
        
        status)
            docker-compose ps
            ;;
        
        clean)
            log_warn "⚠️  Suppression de tous les conteneurs, volumes et images..."
            read -p "Êtes-vous sûr ? (yes/no): " -r
            if [[ $REPLY == "yes" ]]; then
                docker-compose down -v --rmi all
                log_info "✓ Nettoyage complet effectué"
            else
                log_info "Annulé"
            fi
            ;;
        
        *)
            echo "Usage: $0 {deploy|start|stop|restart|rebuild|logs|status|clean}"
            echo ""
            echo "  deploy   - Déploiement complet (build + start)"
            echo "  start    - Démarrer les services"
            echo "  stop     - Arrêter les services"
            echo "  restart  - Redémarrer les services"
            echo "  rebuild  - Reconstruire les images"
            echo "  logs     - Afficher les logs (logs [service])"
            echo "  status   - État des services"
            echo "  clean    - Nettoyage complet"
            exit 1
            ;;
    esac
}

main "$@"
