#!/bin/bash

# ╔════════════════════════════════════════════════════════════════╗
# ║       🚀 KibaLock Universal Launcher - Start All Services     ║
# ║    Backend + LifeModo API + React Frontend (Auto-Discovery)   ║
# ╚════════════════════════════════════════════════════════════════╝

set -e  # Exit on error

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Répertoire du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Charger utilitaires de détection réseau
source utils/detect_network.sh

# Variables globales pour les PIDs
BACKEND_PID=""
API_PID=""
FRONTEND_PID=""

# Fichiers de logs
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
BACKEND_LOG="$LOG_DIR/backend_$(date +%Y%m%d_%H%M%S).log"
API_LOG="$LOG_DIR/api_$(date +%Y%m%d_%H%M%S).log"
FRONTEND_LOG="$LOG_DIR/frontend_$(date +%Y%m%d_%H%M%S).log"

# Fonction de nettoyage
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Arrêt de tous les services...${NC}"
    
    if [ ! -z "$FRONTEND_PID" ]; then
        echo -e "${YELLOW}  Arrêt du frontend React (PID: $FRONTEND_PID)${NC}"
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$BACKEND_PID" ]; then
        echo -e "${YELLOW}  Arrêt du backend KibaLock (PID: $BACKEND_PID)${NC}"
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$API_PID" ]; then
        echo -e "${YELLOW}  Arrêt de LifeModo API (PID: $API_PID)${NC}"
        kill $API_PID 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ Tous les services arrêtés${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Header
clear
echo -e "${PURPLE}${BOLD}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║           🔐 KibaLock Universal Launcher v1.0                 ║
║                                                                ║
║       🧠 LifeModo API + 🐍 Backend + ⚛️ React Frontend        ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# === ÉTAPE 1: Détection de la plateforme ===
echo -e "${CYAN}${BOLD}[ÉTAPE 1/8] 🔍 Détection de la plateforme...${NC}"
PLATFORM=$(detect_platform)
LOCAL_IP=$(get_local_ip "$PLATFORM")
echo -e "${GREEN}✓${NC} Plateforme: ${BOLD}$PLATFORM${NC}"
echo -e "${GREEN}✓${NC} IP locale: ${BOLD}$LOCAL_IP${NC}"
echo ""

# === ÉTAPE 2: Configuration automatique ===
echo -e "${CYAN}${BOLD}[ÉTAPE 2/8] ⚙️  Génération de la configuration automatique...${NC}"

# Trouver des ports libres
BACKEND_PORT=$(find_free_port 8505)
API_PORT=$(find_free_port 8000)
FRONTEND_PORT=$(find_free_port 3000)

echo -e "${GREEN}✓${NC} Port Backend KibaLock: ${BOLD}$BACKEND_PORT${NC}"
echo -e "${GREEN}✓${NC} Port LifeModo API: ${BOLD}$API_PORT${NC}"
echo -e "${GREEN}✓${NC} Port React Frontend: ${BOLD}$FRONTEND_PORT${NC}"

# Générer .env.auto
ENV_AUTO_FILE=".env.auto"
generate_env_file "$ENV_AUTO_FILE"
echo -e "${GREEN}✓${NC} Configuration générée: ${BOLD}$ENV_AUTO_FILE${NC}"
echo ""

# Charger la configuration
source "$ENV_AUTO_FILE"

# === ÉTAPE 3: Vérification Python ===
echo -e "${CYAN}${BOLD}[ÉTAPE 3/8] 🐍 Vérification de Python...${NC}"

# Utiliser Python 3.10 de l'environnement gestmodo
GESTMODO_PYTHON="$HOME/miniconda3/envs/gestmodo/bin/python"
GESTMODO_PIP="$HOME/miniconda3/envs/gestmodo/bin/pip"

if [ ! -f "$GESTMODO_PYTHON" ]; then
    echo -e "${RED}❌ Environnement gestmodo non trouvé!${NC}"
    echo -e "${YELLOW}Créer avec: conda create -n gestmodo python=3.10${NC}"
    exit 1
fi

PYTHON_VERSION=$($GESTMODO_PYTHON --version)
echo -e "${GREEN}✓${NC} $PYTHON_VERSION (environnement gestmodo)"
echo ""

# === ÉTAPE 4: Vérification Node.js ===
echo -e "${CYAN}${BOLD}[ÉTAPE 4/8] 📦 Vérification de Node.js...${NC}"

# Chercher Node.js sur Windows (WSL)
NODE_PATH="/mnt/c/Program Files/nodejs/node.exe"
NPM_PATH="/mnt/c/Program Files/nodejs/npm"

if [ -f "$NODE_PATH" ]; then
    NODE_VERSION=$("$NODE_PATH" --version 2>/dev/null)
    NPM_VERSION=$("$NPM_PATH" --version 2>/dev/null)
    echo -e "${GREEN}✓${NC} Node.js $NODE_VERSION (Windows)"
    echo -e "${GREEN}✓${NC} npm $NPM_VERSION (Windows)"
    SKIP_FRONTEND=false
elif command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓${NC} Node.js $NODE_VERSION"
    echo -e "${GREEN}✓${NC} npm $NPM_VERSION"
    SKIP_FRONTEND=false
else
    echo -e "${YELLOW}⚠${NC} Node.js non trouvé, frontend React sera ignoré"
    echo -e "${YELLOW}  Seuls les backends Python seront lancés${NC}"
    SKIP_FRONTEND=true
fi
echo ""

# === ÉTAPE 5: Environnements virtuels Python ===
echo -e "${CYAN}${BOLD}[ÉTAPE 5/8] 🔧 Configuration des environnements virtuels...${NC}"

# Note: On utilise l'environnement gestmodo existant
echo -e "${GREEN}✓${NC} Utilisation de l'environnement gestmodo"
echo -e "${BLUE}  Python:${NC} $GESTMODO_PYTHON"
echo ""

# === ÉTAPE 6: Installation des dépendances ===
echo -e "${CYAN}${BOLD}[ÉTAPE 6/8] 📚 Vérification des dépendances...${NC}"

# Backend KibaLock
echo -e "${YELLOW}  Backend KibaLock...${NC}"
if ! $GESTMODO_PYTHON -c "import streamlit" &>/dev/null; then
    echo -e "${YELLOW}    Installation des dépendances Backend...${NC}"
    $GESTMODO_PIP install -q -r requirements.txt
fi
echo -e "${GREEN}✓${NC} Backend dependencies OK"

# LifeModo API
echo -e "${YELLOW}  LifeModo API...${NC}"
if ! $GESTMODO_PYTHON -c "import fastapi" &>/dev/null; then
    echo -e "${YELLOW}    Installation des dépendances LifeModo...${NC}"
    $GESTMODO_PIP install -q fastapi uvicorn torch transformers TTS openai-whisper deepface
fi
echo -e "${GREEN}✓${NC} LifeModo dependencies OK"

# Frontend React
if [ "$SKIP_FRONTEND" = false ]; then
    echo -e "${YELLOW}  React Frontend...${NC}"
    if [ ! -d "frontend/node_modules" ]; then
        echo -e "${YELLOW}    Installation des dépendances React...${NC}"
        if command -v npm &>/dev/null; then
            cd frontend && npm install --silent && cd ..
        elif [ -f "$NPM_PATH" ]; then
            cd frontend && "$NPM_PATH" install --silent && cd ..
        fi
    fi
    echo -e "${GREEN}✓${NC} React dependencies OK"
else
    echo -e "${YELLOW}⊘${NC} React Frontend ignoré (Node.js non disponible)"
fi
echo ""

# === ÉTAPE 7: Configuration du Frontend ===
echo -e "${CYAN}${BOLD}[ÉTAPE 7/8] ⚛️  Configuration du Frontend React...${NC}"

# Créer .env pour le frontend
cat > frontend/.env << EOF
VITE_API_URL=http://${LOCAL_IP}:${API_PORT}
VITE_BACKEND_URL=http://${LOCAL_IP}:${BACKEND_PORT}
VITE_WS_URL=ws://${LOCAL_IP}:${API_PORT}
EOF
echo -e "${GREEN}✓${NC} Frontend .env configuré avec API auto-discovery"
echo ""

# === ÉTAPE 8: Lancement des services ===
echo -e "${CYAN}${BOLD}[ÉTAPE 8/8] 🚀 Lancement de tous les services...${NC}"
echo ""

# 8.1: LifeModo API
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${PURPLE}${BOLD}  🧠 LifeModo API${NC}"
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Port:${NC} $API_PORT"
echo -e "${BLUE}  URL:${NC} http://${LOCAL_IP}:${API_PORT}"
echo -e "${BLUE}  Docs:${NC} http://${LOCAL_IP}:${API_PORT}/docs"
echo -e "${BLUE}  Log:${NC} $API_LOG"
echo ""

$GESTMODO_PYTHON lifemodo_api.py --host 0.0.0.0 --port $API_PORT > "$API_LOG" 2>&1 &
API_PID=$!

echo -e "${GREEN}✓${NC} LifeModo API démarré (PID: $API_PID)"
sleep 3  # Attendre que l'API démarre

# 8.2: Backend KibaLock
echo ""
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${PURPLE}${BOLD}  🐍 Backend KibaLock (Streamlit + FAISS)${NC}"
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Port:${NC} $BACKEND_PORT"
echo -e "${BLUE}  URL:${NC} http://${LOCAL_IP}:${BACKEND_PORT}"
echo -e "${BLUE}  Log:${NC} $BACKEND_LOG"
echo ""

$GESTMODO_PYTHON -m streamlit run kibalock_faiss.py \
    --server.port=$BACKEND_PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.primaryColor="#667eea" \
    --theme.backgroundColor="#ffffff" \
    > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

echo -e "${GREEN}✓${NC} Backend KibaLock démarré (PID: $BACKEND_PID)"
sleep 3  # Attendre que Streamlit démarre

# 8.3: Frontend React
if [ "$SKIP_FRONTEND" = false ]; then
    echo ""
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}${BOLD}  ⚛️  React Frontend (3D UI)${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Port:${NC} $FRONTEND_PORT"
    echo -e "${BLUE}  URL:${NC} http://${LOCAL_IP}:${FRONTEND_PORT}"
    echo -e "${BLUE}  Log:${NC} $FRONTEND_LOG"
    echo ""

    cd frontend
    if command -v npm &>/dev/null; then
        npm run dev -- --host 0.0.0.0 --port $FRONTEND_PORT > "$FRONTEND_LOG" 2>&1 &
    elif [ -f "$NPM_PATH" ]; then
        "$NPM_PATH" run dev -- --host 0.0.0.0 --port $FRONTEND_PORT > "$FRONTEND_LOG" 2>&1 &
    fi
    FRONTEND_PID=$!
    cd ..

    echo -e "${GREEN}✓${NC} Frontend React démarré (PID: $FRONTEND_PID)"
    sleep 2
else
    echo ""
    echo -e "${YELLOW}⊘ Frontend React ignoré (Node.js non disponible)${NC}"
    FRONTEND_PID=""
fi

# === RÉSUMÉ FINAL ===
echo ""
echo ""
echo -e "${GREEN}${BOLD}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║                 ✅ TOUS LES SERVICES SONT ACTIFS !            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${CYAN}${BOLD}📊 TABLEAU DE BORD DES SERVICES${NC}"
echo ""
echo -e "${PURPLE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║ ${BOLD}Service${NC}               ${PURPLE}║ ${BOLD}Status${NC}  ${PURPLE}║ ${BOLD}URL${NC}                              ${PURPLE}║${NC}"
echo -e "${PURPLE}╠═══════════════════════════════════════════════════════════════╣${NC}"
printf "${PURPLE}║${NC} %-21s ${PURPLE}║${NC} ${GREEN}%-7s${NC} ${PURPLE}║${NC} %-32s ${PURPLE}║${NC}\n" "🧠 LifeModo API" "RUNNING" "http://${LOCAL_IP}:${API_PORT}"
printf "${PURPLE}║${NC} %-21s ${PURPLE}║${NC} ${GREEN}%-7s${NC} ${PURPLE}║${NC} %-32s ${PURPLE}║${NC}\n" "🐍 Backend KibaLock" "RUNNING" "http://${LOCAL_IP}:${BACKEND_PORT}"
printf "${PURPLE}║${NC} %-21s ${PURPLE}║${NC} ${GREEN}%-7s${NC} ${PURPLE}║${NC} %-32s ${PURPLE}║${NC}\n" "⚛️  React Frontend" "RUNNING" "http://${LOCAL_IP}:${FRONTEND_PORT}"
echo -e "${PURPLE}╚═══════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${CYAN}${BOLD}🔗 LIENS RAPIDES${NC}"
echo ""
echo -e "  ${BOLD}Frontend (Interface 3D):${NC}"
echo -e "    • Local:    ${BLUE}http://localhost:${FRONTEND_PORT}${NC}"
echo -e "    • Network:  ${BLUE}http://${LOCAL_IP}:${FRONTEND_PORT}${NC}"
echo ""
echo -e "  ${BOLD}Backend (Streamlit):${NC}"
echo -e "    • Local:    ${BLUE}http://localhost:${BACKEND_PORT}${NC}"
echo -e "    • Network:  ${BLUE}http://${LOCAL_IP}:${BACKEND_PORT}${NC}"
echo ""
echo -e "  ${BOLD}API (LifeModo):${NC}"
echo -e "    • Local:    ${BLUE}http://localhost:${API_PORT}${NC}"
echo -e "    • Network:  ${BLUE}http://${LOCAL_IP}:${API_PORT}${NC}"
echo -e "    • Docs:     ${BLUE}http://localhost:${API_PORT}/docs${NC}"
echo ""
echo -e "${CYAN}${BOLD}📝 LOGS${NC}"
echo ""
echo -e "  • API:      ${YELLOW}tail -f $API_LOG${NC}"
echo -e "  • Backend:  ${YELLOW}tail -f $BACKEND_LOG${NC}"
echo -e "  • Frontend: ${YELLOW}tail -f $FRONTEND_LOG${NC}"
echo ""
echo -e "${CYAN}${BOLD}💡 INFORMATIONS${NC}"
echo ""
echo -e "  • Plateforme:  ${BOLD}$PLATFORM${NC}"
echo -e "  • IP Locale:   ${BOLD}$LOCAL_IP${NC}"
echo -e "  • Python:      ${BOLD}$PYTHON_VERSION${NC}"
echo -e "  • Node.js:     ${BOLD}$NODE_VERSION${NC}"
echo ""
echo -e "${YELLOW}${BOLD}⏹️  Pour arrêter tous les services: Ctrl+C${NC}"
echo ""

# Attendre un signal d'arrêt
wait
