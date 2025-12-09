#!/bin/bash

# === KibaLock Launcher Script ===
# Lance l'application KibaLock avec toutes les vérifications

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PORT=8505

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║          🔐 KibaLock Biometric Authentication            ║"
echo "║          Système d'authentification multimodal            ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Fonction de log
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ❌ $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠️  $1${NC}"
}

# Vérifier Python
log "Vérification de Python..."
if ! command -v python3 &> /dev/null; then
    error "Python 3 n'est pas installé"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
success "Python ${PYTHON_VERSION} trouvé"

# Vérifier/Créer l'environnement virtuel
if [ ! -d "$VENV_DIR" ]; then
    log "Création de l'environnement virtuel..."
    python3 -m venv "$VENV_DIR"
    success "Environnement virtuel créé"
else
    log "Environnement virtuel existant trouvé"
fi

# Activer l'environnement virtuel
log "Activation de l'environnement virtuel..."
source "$VENV_DIR/bin/activate"

# Installer/Mettre à jour les dépendances
if [ ! -f "$VENV_DIR/.installed" ] || [ "$1" == "--install" ]; then
    log "Installation des dépendances..."
    pip install --upgrade pip setuptools wheel
    pip install -r "$SCRIPT_DIR/requirements.txt"
    touch "$VENV_DIR/.installed"
    success "Dépendances installées"
else
    log "Dépendances déjà installées (utilisez --install pour réinstaller)"
fi

# Vérifier le fichier .env
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    warning "Fichier .env non trouvé"
    if [ -f "$SCRIPT_DIR/.env.example" ]; then
        log "Copie de .env.example vers .env..."
        cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
        warning "Veuillez éditer .env avec vos paramètres"
    fi
fi

# Vérifier MongoDB
log "Vérification de la connexion MongoDB..."
python3 -c "
import os
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv()
try:
    client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'))
    client.server_info()
    print('✅ MongoDB connecté')
except Exception as e:
    print(f'❌ Erreur MongoDB: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    success "MongoDB opérationnel"
else
    error "Impossible de se connecter à MongoDB"
    exit 1
fi

# Créer les dossiers nécessaires
log "Création des répertoires..."
mkdir -p ~/kibalock/{embeddings,temp,logs}
success "Répertoires créés"

# Vérifier le port
log "Vérification du port $PORT..."
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    warning "Le port $PORT est déjà utilisé"
    read -p "Voulez-vous tuer le processus et continuer? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        PID=$(lsof -ti:$PORT)
        kill -9 $PID
        success "Processus $PID terminé"
    else
        error "Lancement annulé"
        exit 1
    fi
fi

# Afficher les informations
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 Lancement de KibaLock...${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}📍 URL locale:${NC}      http://localhost:$PORT"
echo -e "${BLUE}📍 URL réseau:${NC}     http://$(hostname -I | awk '{print $1}'):$PORT"
echo -e "${BLUE}📁 Répertoire:${NC}     $SCRIPT_DIR"
echo -e "${BLUE}🐍 Python:${NC}         $PYTHON_VERSION"
echo -e "${BLUE}📊 Logs:${NC}           ~/kibalock/logs/"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Lancer Streamlit
log "Démarrage de Streamlit..."
streamlit run "$SCRIPT_DIR/kibalock.py" \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base "dark" \
    --theme.primaryColor "#667eea" \
    --theme.backgroundColor "#0e1117" \
    --theme.secondaryBackgroundColor "#262730"
