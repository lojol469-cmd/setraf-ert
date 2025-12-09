#!/bin/bash

###############################################################################
# KibaLock Mini Kernel OS - Gestionnaire de services biométriques
# Lance et supervise LifeModo API (FastAPI), Backend KibaLock (Streamlit), Frontend React
###############################################################################

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
CONDA_BASE="$HOME/miniconda3"
CONDA_ENV="gestmodo"  # Environnement avec toutes les dépendances installées

# Activer automatiquement conda gestmodo
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" 2>/dev/null || {
        echo -e "${YELLOW}⚠️  Conda gestmodo non activé automatiquement${NC}"
    }
fi

# Chemins Python
GESTMODO_PYTHON="$CONDA_BASE/envs/$CONDA_ENV/bin/python"
GESTMODO_PIP="$CONDA_BASE/envs/$CONDA_ENV/bin/pip"

# Chemins Node.js (Windows via WSL)
NODE_PATH="/mnt/c/Program Files/nodejs/node.exe"
NPM_PATH="/mnt/c/Program Files/nodejs/npm"

# Fichiers PID
API_PID_FILE="/tmp/kibalock_api.pid"
BACKEND_PID_FILE="/tmp/kibalock_backend.pid"
FRONTEND_PID_FILE="/tmp/kibalock_frontend.pid"

# Logs
LOG_DIR="$SCRIPT_DIR/logs"
API_LOG="$LOG_DIR/lifemodo_api.log"
BACKEND_LOG="$LOG_DIR/kibalock_backend.log"
FRONTEND_LOG="$LOG_DIR/react_frontend.log"
KERNEL_LOG="$LOG_DIR/kernel.log"

# Ports par défaut
API_PORT=8000
BACKEND_PORT=8505
FRONTEND_PORT=3000

# Détection automatique de l'IP
LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "127.0.0.1")
if [ -z "$LOCAL_IP" ] || [ "$LOCAL_IP" = "127.0.0.1" ]; then
    LOCAL_IP=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K\S+' || echo "172.20.31.35")
fi

###############################################################################
# Fonctions utilitaires
###############################################################################

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$KERNEL_LOG"
}

print_banner() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║          🔐 KibaLock Mini Kernel OS v1.0                     ║"
    echo "║          Système d'Authentification Biométrique Multimodal    ║"
    echo "║                                                               ║"
    echo "║          Services: FastAPI + Streamlit + React 3D            ║"
    echo "║          Modèles: Phi-3.5 | Whisper | FaceNet512 | FAISS    ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_dependencies() {
    log "INFO" "Vérification des dépendances..."
    
    # Vérifier Miniconda
    if [ ! -d "$CONDA_BASE" ]; then
        log "ERROR" "Miniconda non trouvé: $CONDA_BASE"
        echo -e "${RED}❌ Miniconda non installé${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Miniconda trouvé${NC}"
    
    # Vérifier l'environnement gestmodo
    if [ ! -d "$CONDA_BASE/envs/$CONDA_ENV" ]; then
        log "ERROR" "Environnement conda '$CONDA_ENV' non trouvé"
        echo -e "${RED}❌ Environnement '$CONDA_ENV' non trouvé${NC}"
        echo -e "${YELLOW}Créez-le avec: conda create -n $CONDA_ENV python=3.10${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Environnement conda '$CONDA_ENV' trouvé${NC}"
    
    # Vérifier Python 3.10
    if [ ! -f "$GESTMODO_PYTHON" ]; then
        log "ERROR" "Python non trouvé dans l'environnement $CONDA_ENV"
        echo -e "${RED}❌ Python non trouvé dans gestmodo${NC}"
        exit 1
    fi
    local PYTHON_VERSION=$($GESTMODO_PYTHON --version)
    echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"
    
    # Vérifier requirements.txt
    if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
        log "WARN" "requirements.txt non trouvé"
        echo -e "${YELLOW}⚠ requirements.txt non trouvé${NC}"
    else
        echo -e "${CYAN}Vérification des packages Python...${NC}"
        
        # Vérifier les packages critiques
        local MISSING_PACKAGES=()
        local CRITICAL_PACKAGES=("fastapi" "uvicorn" "pymongo" "transformers" "langchain" "openai-whisper" "TTS" "faiss" "streamlit")
        
        for package in "${CRITICAL_PACKAGES[@]}"; do
            if ! $GESTMODO_PYTHON -c "import ${package//-/_}" 2>/dev/null; then
                MISSING_PACKAGES+=("$package")
            fi
        done
        
        if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
            log "WARN" "Packages manquants détectés: ${MISSING_PACKAGES[*]}"
            echo -e "${YELLOW}⚠ Packages manquants: ${MISSING_PACKAGES[*]}${NC}"
            echo ""
            read -p "Installer les dépendances depuis requirements.txt? (y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${CYAN}Installation des dépendances...${NC}"
                log "INFO" "Installation des packages depuis requirements.txt"
                
                # Installation avec pip (exclure PyTorch qui nécessite un index spécial)
                $GESTMODO_PIP install -r "$SCRIPT_DIR/requirements.txt" --upgrade 2>&1 | tee -a "$LOG_DIR/dependencies_install.log"
                
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✓ Dépendances installées avec succès${NC}"
                    log "INFO" "Installation des dépendances réussie"
                else
                    log "ERROR" "Échec de l'installation des dépendances"
                    echo -e "${RED}❌ Échec de l'installation${NC}"
                    echo -e "${YELLOW}Vérifiez le log: $LOG_DIR/dependencies_install.log${NC}"
                fi
            else
                echo -e "${YELLOW}⚠ Poursuite sans installation (certaines fonctionnalités peuvent échouer)${NC}"
                log "WARN" "Installation des dépendances ignorée par l'utilisateur"
            fi
        else
            echo -e "${GREEN}✓ Tous les packages critiques sont installés${NC}"
        fi
    fi
    
    # Vérifier PyTorch et CUDA
    echo -e "${CYAN}Vérification PyTorch et CUDA...${NC}"
    local TORCH_CHECK=$($GESTMODO_PYTHON << 'PYEOF'
import sys
try:
    import torch
    cuda_available = torch.cuda.is_available()
    torch_version = torch.__version__
    
    if cuda_available:
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        compute_cap = f"sm_{props.major}{props.minor}"
        
        # Vérifier si RTX 5090 avec sm_120/sm_130
        if "5090" in gpu_name and compute_cap in ["sm_120", "sm_130"]:
            # Vérifier si CUDA 13.0+ pour support RTX 50 series
            cuda_major = int(cuda_version.split('.')[0])
            if cuda_major < 13:
                print(f"ERROR:RTX_5090_NOT_SUPPORTED:PyTorch {torch_version} with CUDA {cuda_version} does not support {gpu_name} ({compute_cap})")
                print(f"SOLUTION:Run ./upgrade_pytorch_cuda13.sh to install PyTorch with CUDA 13.0")
                sys.exit(1)
        
        print(f"OK:CUDA:{cuda_version}:GPU:{gpu_name}:ComputeCap:{compute_cap}")
    else:
        print(f"OK:CPU_ONLY:PyTorch:{torch_version}")
    
    sys.exit(0)
except ImportError:
    print("ERROR:PYTORCH_NOT_INSTALLED")
    sys.exit(1)
except Exception as e:
    print(f"ERROR:UNKNOWN:{str(e)}")
    sys.exit(1)
PYEOF
)
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        if echo "$TORCH_CHECK" | grep -q "PYTORCH_NOT_INSTALLED"; then
            log "ERROR" "PyTorch non installé"
            echo -e "${RED}❌ PyTorch non installé${NC}"
            echo -e "${YELLOW}Installation: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130${NC}"
            read -p "Installer PyTorch avec CUDA 13.0 maintenant? (y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${CYAN}Installation de PyTorch CUDA 13.0...${NC}"
                $GESTMODO_PIP install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✓ PyTorch installé avec succès${NC}"
                else
                    echo -e "${RED}❌ Échec de l'installation PyTorch${NC}"
                    exit 1
                fi
            else
                echo -e "${YELLOW}⚠ Poursuite sans PyTorch (mode CPU uniquement)${NC}"
            fi
        elif echo "$TORCH_CHECK" | grep -q "RTX_5090_NOT_SUPPORTED"; then
            log "WARN" "RTX 5090 nécessite PyTorch avec CUDA 13.0"
            echo -e "${RED}❌ $(echo "$TORCH_CHECK" | grep "ERROR:" | cut -d: -f4-)${NC}"
            echo -e "${YELLOW}$(echo "$TORCH_CHECK" | grep "SOLUTION:" | cut -d: -f2-)${NC}"
            echo ""
            read -p "Lancer la migration PyTorch CUDA 13.0 maintenant? (y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${CYAN}Lancement de la migration...${NC}"
                ./upgrade_pytorch_cuda13.sh
                if [ $? -ne 0 ]; then
                    echo -e "${RED}❌ Migration échouée${NC}"
                    exit 1
                fi
            else
                echo -e "${YELLOW}⚠️  Mode CPU uniquement (performances réduites)${NC}"
                log "WARN" "KibaLock démarré sans support GPU"
            fi
        else
            log "ERROR" "Erreur vérification PyTorch: $TORCH_CHECK"
            echo -e "${RED}❌ Erreur PyTorch: $TORCH_CHECK${NC}"
            exit 1
        fi
    else
        # Parse résultat OK
        if echo "$TORCH_CHECK" | grep -q "CUDA"; then
            local CUDA_VER=$(echo "$TORCH_CHECK" | cut -d: -f3)
            local GPU_NAME=$(echo "$TORCH_CHECK" | cut -d: -f5)
            local COMPUTE_CAP=$(echo "$TORCH_CHECK" | cut -d: -f7)
            echo -e "${GREEN}✓ PyTorch avec CUDA $CUDA_VER${NC}"
            echo -e "${GREEN}✓ GPU: $GPU_NAME ($COMPUTE_CAP)${NC}"
            log "INFO" "GPU détecté: $GPU_NAME ($COMPUTE_CAP) - CUDA $CUDA_VER"
        else
            echo -e "${YELLOW}⚠ PyTorch en mode CPU uniquement${NC}"
            log "INFO" "PyTorch en mode CPU"
        fi
    fi
    
    # Vérifier les fichiers essentiels
    if [ ! -f "$SCRIPT_DIR/lifemodo_api.py" ]; then
        log "ERROR" "LifeModo API non trouvée: lifemodo_api.py"
        echo -e "${RED}❌ lifemodo_api.py non trouvé${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ LifeModo API trouvée${NC}"
    
    if [ ! -f "$SCRIPT_DIR/kibalock_faiss.py" ]; then
        log "ERROR" "Backend KibaLock non trouvé: kibalock_faiss.py"
        echo -e "${RED}❌ kibalock_faiss.py non trouvé${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Backend KibaLock trouvé${NC}"
    
    # Vérifier Node.js (optionnel pour frontend)
    if [ -f "$NODE_PATH" ]; then
        local NODE_VERSION=$("$NODE_PATH" --version 2>/dev/null)
        echo -e "${GREEN}✓ Node.js $NODE_VERSION (Windows)${NC}"
        SKIP_FRONTEND=false
    elif command -v node &>/dev/null; then
        local NODE_VERSION=$(node --version)
        echo -e "${GREEN}✓ Node.js $NODE_VERSION${NC}"
        SKIP_FRONTEND=false
    else
        echo -e "${YELLOW}⚠ Node.js non trouvé, frontend sera ignoré${NC}"
        SKIP_FRONTEND=true
    fi
    
    if [ ! "$SKIP_FRONTEND" = true ] && [ ! -d "$FRONTEND_DIR" ]; then
        log "WARN" "Dossier frontend non trouvé"
        echo -e "${YELLOW}⚠ Dossier frontend/ non trouvé${NC}"
        SKIP_FRONTEND=true
    fi
}

setup_environment() {
    log "INFO" "Configuration de l'environnement..."
    
    # Créer le dossier de logs
    mkdir -p "$LOG_DIR"
    echo -e "${GREEN}✓ Dossier de logs créé${NC}"
    
    # Créer dossiers FAISS si nécessaire
    mkdir -p "$HOME/kibalock/faiss_indexes"
    mkdir -p "$HOME/kibalock/audio_samples"
    mkdir -p "$HOME/kibalock/face_images"
    
    # Nettoyer les anciens logs (garder les 5 derniers)
    cd "$LOG_DIR"
    ls -t kernel.log.* 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true
    
    # Archiver le log actuel s'il existe
    if [ -f "$KERNEL_LOG" ]; then
        mv "$KERNEL_LOG" "$KERNEL_LOG.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Créer/mettre à jour .env
    create_env_file
}

create_env_file() {
    log "INFO" "Création du fichier .env..."
    
    cat > "$SCRIPT_DIR/.env" << EOF
# KibaLock Configuration - Auto-généré le $(date)
# Platform: WSL
# IP: $LOCAL_IP

# MongoDB Atlas
MONGODB_URI=mongodb+srv://SETRAF:Dieu19961991%3F%3F%21%3F%3F%21@cluster0.5tjz9v0.mongodb.net/kibalock?retryWrites=true&w=majority

# Ports des services
API_PORT=$API_PORT
BACKEND_PORT=$BACKEND_PORT
FRONTEND_PORT=$FRONTEND_PORT

# URLs auto-découvertes
VITE_API_URL=http://$LOCAL_IP:$API_PORT
VITE_BACKEND_URL=http://$LOCAL_IP:$BACKEND_PORT
VITE_API_URL_AUTO=http://$LOCAL_IP:$API_PORT

# Chemins locaux
FAISS_INDEX_DIR=$HOME/kibalock/faiss_indexes
AUDIO_SAMPLES_DIR=$HOME/kibalock/audio_samples
FACE_IMAGES_DIR=$HOME/kibalock/face_images

# Modèles IA
WHISPER_MODEL=base
FACE_MODEL=Facenet512
PHI_MODEL=microsoft/Phi-3.5-mini-instruct
XTTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2

# Sécurité
JWT_SECRET=$(openssl rand -hex 32 2>/dev/null || echo "kibalock_secret_key_$(date +%s)")
SESSION_DURATION=3600

# Performance
FAISS_NPROBE=10
FAISS_K=5
SIMILARITY_THRESHOLD=0.85

# Logs
LOG_LEVEL=INFO
EOF

    echo -e "${GREEN}✓ Fichier .env créé${NC}"
    log "INFO" ".env créé avec IP=$LOCAL_IP"
}

find_free_port() {
    local port=$1
    while netstat -an 2>/dev/null | grep -q ":$port "; do
        port=$((port + 1))
    done
    echo $port
}

start_lifemodo_api() {
    log "INFO" "Démarrage de LifeModo API (FastAPI)..."
    echo -e "${YELLOW}🚀 Lancement de LifeModo API...${NC}"
    
    cd "$SCRIPT_DIR"
    
    # Vérifier les dépendances FastAPI
    if ! $GESTMODO_PYTHON -c "import fastapi" &>/dev/null; then
        log "WARN" "FastAPI non trouvé, installation..."
        echo -e "${YELLOW}⚠️  Installation de FastAPI...${NC}"
        $GESTMODO_PIP install -q fastapi uvicorn torch transformers TTS openai-whisper deepface faiss-cpu pymongo
    fi
    
    # Trouver un port libre si nécessaire
    API_PORT=$(find_free_port $API_PORT)
    
    # Arrêter les instances existantes
    pkill -9 -f "lifemodo_api.py" 2>/dev/null || true
    sleep 1
    
    # Démarrer FastAPI
    nohup $GESTMODO_PYTHON lifemodo_api.py --host 0.0.0.0 --port $API_PORT > "$API_LOG" 2>&1 &
    local pid=$!
    echo $pid > "$API_PID_FILE"
    
    # Attendre que le serveur démarre
    sleep 3
    
    # Vérifier si le processus tourne
    if ps -p $pid > /dev/null 2>&1; then
        log "INFO" "LifeModo API démarré (PID: $pid, Port: $API_PORT)"
        echo -e "${GREEN}✓ LifeModo API démarré sur http://$LOCAL_IP:$API_PORT${NC}"
        echo -e "${BLUE}  Docs API: http://$LOCAL_IP:$API_PORT/docs${NC}"
        return 0
    else
        log "ERROR" "Échec du démarrage de LifeModo API"
        echo -e "${RED}❌ Échec du démarrage de LifeModo API${NC}"
        tail -20 "$API_LOG"
        return 1
    fi
}

start_kibalock_backend() {
    log "INFO" "Démarrage du Backend KibaLock (Streamlit + FAISS)..."
    echo -e "${YELLOW}🚀 Lancement du Backend KibaLock...${NC}"
    
    cd "$SCRIPT_DIR"
    
    # Vérifier les dépendances Streamlit
    if ! $GESTMODO_PYTHON -c "import streamlit" &>/dev/null; then
        log "WARN" "Streamlit non trouvé, installation..."
        echo -e "${YELLOW}⚠️  Installation de Streamlit...${NC}"
        $GESTMODO_PIP install -q streamlit faiss-cpu pymongo numpy pillow
    fi
    
    # Trouver un port libre si nécessaire
    BACKEND_PORT=$(find_free_port $BACKEND_PORT)
    
    # Arrêter les instances Streamlit existantes
    pkill -9 -f "streamlit run" 2>/dev/null || true
    pkill -9 -f "kibalock_faiss.py" 2>/dev/null || true
    sleep 2
    
    # Démarrer Streamlit avec l'environnement gestmodo
    nohup $GESTMODO_PYTHON -m streamlit run kibalock_faiss.py \
        --server.port=$BACKEND_PORT \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false \
        --theme.primaryColor="#667eea" \
        --theme.backgroundColor="#ffffff" \
        > "$BACKEND_LOG" 2>&1 &
    local pid=$!
    echo $pid > "$BACKEND_PID_FILE"
    
    # Attendre que le serveur démarre
    sleep 5
    
    # Vérifier si le processus tourne
    if ps -p $pid > /dev/null 2>&1; then
        log "INFO" "Backend KibaLock démarré (PID: $pid, Port: $BACKEND_PORT)"
        echo -e "${GREEN}✓ Backend KibaLock démarré sur http://$LOCAL_IP:$BACKEND_PORT${NC}"
        return 0
    else
        log "ERROR" "Échec du démarrage du Backend KibaLock"
        echo -e "${RED}❌ Échec du démarrage du Backend KibaLock${NC}"
        tail -20 "$BACKEND_LOG"
        return 1
    fi
}

start_react_frontend() {
    if [ "$SKIP_FRONTEND" = true ]; then
        log "INFO" "Frontend React ignoré (Node.js non disponible)"
        echo -e "${YELLOW}⊘ Frontend React ignoré (Node.js non disponible)${NC}"
        return 0
    fi
    
    log "INFO" "Démarrage du Frontend React (3D Interface)..."
    echo -e "${YELLOW}🚀 Lancement du Frontend React...${NC}"
    
    cd "$FRONTEND_DIR"
    
    # Installer les dépendances npm si nécessaire
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}⚠️  Installation des dépendances npm...${NC}"
        if command -v npm &>/dev/null; then
            npm install --silent
        elif [ -f "$NPM_PATH" ]; then
            "$NPM_PATH" install --silent
        fi
    fi
    
    # Trouver un port libre si nécessaire
    FRONTEND_PORT=$(find_free_port $FRONTEND_PORT)
    
    # Arrêter les instances existantes
    pkill -9 -f "vite.*--port $FRONTEND_PORT" 2>/dev/null || true
    sleep 1
    
    # Démarrer React avec Vite
    if command -v npm &>/dev/null; then
        nohup npm run dev -- --host 0.0.0.0 --port $FRONTEND_PORT > "$FRONTEND_LOG" 2>&1 &
    elif [ -f "$NPM_PATH" ]; then
        nohup "$NPM_PATH" run dev -- --host 0.0.0.0 --port $FRONTEND_PORT > "$FRONTEND_LOG" 2>&1 &
    fi
    local pid=$!
    echo $pid > "$FRONTEND_PID_FILE"
    
    # Attendre que le serveur démarre
    sleep 3
    
    # Vérifier si le processus tourne
    if ps -p $pid > /dev/null 2>&1; then
        log "INFO" "Frontend React démarré (PID: $pid, Port: $FRONTEND_PORT)"
        echo -e "${GREEN}✓ Frontend React démarré sur http://$LOCAL_IP:$FRONTEND_PORT${NC}"
        cd "$SCRIPT_DIR"
        return 0
    else
        log "ERROR" "Échec du démarrage du Frontend React"
        echo -e "${RED}❌ Échec du démarrage du Frontend React${NC}"
        tail -20 "$FRONTEND_LOG"
        cd "$SCRIPT_DIR"
        return 1
    fi
}

stop_services() {
    log "INFO" "Arrêt des services..."
    echo -e "${YELLOW}🛑 Arrêt des services KibaLock...${NC}"
    
    # Arrêter LifeModo API
    if [ -f "$API_PID_FILE" ]; then
        local api_pid=$(cat "$API_PID_FILE")
        if ps -p $api_pid > /dev/null 2>&1; then
            kill $api_pid 2>/dev/null || true
            log "INFO" "LifeModo API arrêté (PID: $api_pid)"
            echo -e "${GREEN}✓ LifeModo API arrêté${NC}"
        fi
        rm -f "$API_PID_FILE"
    fi
    
    # Arrêter Backend KibaLock
    if [ -f "$BACKEND_PID_FILE" ]; then
        local backend_pid=$(cat "$BACKEND_PID_FILE")
        if ps -p $backend_pid > /dev/null 2>&1; then
            kill $backend_pid 2>/dev/null || true
            log "INFO" "Backend KibaLock arrêté (PID: $backend_pid)"
            echo -e "${GREEN}✓ Backend KibaLock arrêté${NC}"
        fi
        rm -f "$BACKEND_PID_FILE"
    fi
    
    # Arrêter Frontend React
    if [ -f "$FRONTEND_PID_FILE" ]; then
        local frontend_pid=$(cat "$FRONTEND_PID_FILE")
        if ps -p $frontend_pid > /dev/null 2>&1; then
            kill $frontend_pid 2>/dev/null || true
            log "INFO" "Frontend React arrêté (PID: $frontend_pid)"
            echo -e "${GREEN}✓ Frontend React arrêté${NC}"
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi
    
    # Tuer tous les processus restants
    pkill -f "lifemodo_api.py" 2>/dev/null || true
    pkill -f "kibalock_faiss.py" 2>/dev/null || true
    pkill -f "streamlit run" 2>/dev/null || true
}

status_services() {
    echo -e "${CYAN}📊 Statut des services KibaLock${NC}"
    echo ""
    
    # Statut LifeModo API
    if [ -f "$API_PID_FILE" ]; then
        local api_pid=$(cat "$API_PID_FILE")
        if ps -p $api_pid > /dev/null 2>&1; then
            echo -e "${GREEN}● LifeModo API (FastAPI)${NC}"
            echo -e "  Status: ${GREEN}Running${NC} (PID: $api_pid)"
            echo -e "  URL: http://$LOCAL_IP:$API_PORT"
            echo -e "  Docs: http://$LOCAL_IP:$API_PORT/docs"
            echo -e "  Log: $API_LOG"
        else
            echo -e "${RED}● LifeModo API${NC}"
            echo -e "  Status: ${RED}Stopped${NC}"
        fi
    else
        echo -e "${RED}● LifeModo API${NC}"
        echo -e "  Status: ${RED}Not started${NC}"
    fi
    
    echo ""
    
    # Statut Backend KibaLock
    if [ -f "$BACKEND_PID_FILE" ]; then
        local backend_pid=$(cat "$BACKEND_PID_FILE")
        if ps -p $backend_pid > /dev/null 2>&1; then
            echo -e "${GREEN}● Backend KibaLock (Streamlit + FAISS)${NC}"
            echo -e "  Status: ${GREEN}Running${NC} (PID: $backend_pid)"
            echo -e "  URL: http://$LOCAL_IP:$BACKEND_PORT"
            echo -e "  Log: $BACKEND_LOG"
        else
            echo -e "${RED}● Backend KibaLock${NC}"
            echo -e "  Status: ${RED}Stopped${NC}"
        fi
    else
        echo -e "${RED}● Backend KibaLock${NC}"
        echo -e "  Status: ${RED}Not started${NC}"
    fi
    
    echo ""
    
    # Statut Frontend React
    if [ "$SKIP_FRONTEND" = true ]; then
        echo -e "${YELLOW}● Frontend React${NC}"
        echo -e "  Status: ${YELLOW}Skipped (Node.js non disponible)${NC}"
    elif [ -f "$FRONTEND_PID_FILE" ]; then
        local frontend_pid=$(cat "$FRONTEND_PID_FILE")
        if ps -p $frontend_pid > /dev/null 2>&1; then
            echo -e "${GREEN}● Frontend React (3D Interface)${NC}"
            echo -e "  Status: ${GREEN}Running${NC} (PID: $frontend_pid)"
            echo -e "  URL: http://$LOCAL_IP:$FRONTEND_PORT"
            echo -e "  Log: $FRONTEND_LOG"
        else
            echo -e "${RED}● Frontend React${NC}"
            echo -e "  Status: ${RED}Stopped${NC}"
        fi
    else
        echo -e "${RED}● Frontend React${NC}"
        echo -e "  Status: ${RED}Not started${NC}"
    fi
}

restart_services() {
    log "INFO" "Redémarrage des services..."
    stop_services
    sleep 2
    start_services
}

start_services() {
    print_banner
    
    # 🤖 LANCER L'AGENT KERNEL EN PREMIER pour auto-fix
    if [ -f "$SCRIPT_DIR/kibalock_agent_kernel.py" ]; then
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${CYAN}         🤖 AGENT KERNEL - Maintenance Autonome${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        log "INFO" "Lancement de l'Agent Kernel pour maintenance autonome"
        
        $GESTMODO_PYTHON "$SCRIPT_DIR/kibalock_agent_kernel.py" --once
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Agent Kernel: Système vérifié et corrigé${NC}"
            log "INFO" "Agent Kernel: maintenance autonome réussie"
        else
            echo -e "${YELLOW}⚠️  Agent Kernel: Problèmes détectés (voir /tmp/kibalock_agent_kernel.log)${NC}"
            log "WARN" "Agent Kernel a détecté des problèmes"
        fi
        echo ""
    fi
    
    check_dependencies
    setup_environment
    
    echo ""
    log "INFO" "Démarrage du système KibaLock..."
    echo -e "${CYAN}IP Locale détectée: ${BOLD}$LOCAL_IP${NC}"
    echo ""
    
    # Démarrer LifeModo API
    if ! start_lifemodo_api; then
        log "ERROR" "Impossible de démarrer LifeModo API"
        exit 1
    fi
    
    echo ""
    
    # Démarrer Backend KibaLock
    if ! start_kibalock_backend; then
        log "ERROR" "Impossible de démarrer Backend KibaLock"
        stop_services
        exit 1
    fi
    
    echo ""
    
    # Démarrer Frontend React (optionnel)
    start_react_frontend
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║  ✅ Système KibaLock démarré avec succès !                   ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║  🧠 API LifeModo: http://$LOCAL_IP:$API_PORT                ║${NC}"
    echo -e "${GREEN}║  🐍 Backend KibaLock: http://$LOCAL_IP:$BACKEND_PORT        ║${NC}"
    if [ "$SKIP_FRONTEND" = false ]; then
    echo -e "${GREEN}║  ⚛️  React Frontend: http://$LOCAL_IP:$FRONTEND_PORT        ║${NC}"
    fi
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║  📝 Logs: $LOG_DIR                        ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}💡 Commandes utiles:${NC}"
    echo -e "   - Status:  ${YELLOW}./kibalock-kernel.sh status${NC}"
    echo -e "   - Monitor: ${YELLOW}./kibalock-kernel.sh monitor${NC}"
    echo -e "   - Logs:    ${YELLOW}./kibalock-kernel.sh logs [api|backend|frontend|all]${NC}"
    echo -e "   - Stop:    ${YELLOW}./kibalock-kernel.sh stop${NC}"
    echo ""
    log "INFO" "Système KibaLock opérationnel sur $LOCAL_IP"
}

show_logs() {
    local service=$1
    case $service in
        api|lifemodo)
            echo -e "${CYAN}📄 Logs LifeModo API (Temps réel):${NC}"
            echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter${NC}"
            echo ""
            tail -f "$API_LOG"
            ;;
        backend|kibalock|streamlit)
            echo -e "${CYAN}📄 Logs Backend KibaLock (Temps réel):${NC}"
            echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter${NC}"
            echo ""
            tail -f "$BACKEND_LOG"
            ;;
        frontend|react)
            echo -e "${CYAN}📄 Logs Frontend React (Temps réel):${NC}"
            echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter${NC}"
            echo ""
            tail -f "$FRONTEND_LOG"
            ;;
        kernel|system)
            echo -e "${CYAN}📄 Logs Kernel (Temps réel):${NC}"
            echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter${NC}"
            echo ""
            tail -f "$KERNEL_LOG"
            ;;
        all)
            echo -e "${CYAN}📄 Logs de tous les services (Temps réel):${NC}"
            echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter${NC}"
            echo ""
            tail -f "$API_LOG" "$BACKEND_LOG" "$FRONTEND_LOG" "$KERNEL_LOG" 2>/dev/null
            ;;
        *)
            echo -e "${RED}Service inconnu. Utilisez: api, backend, frontend, kernel, ou all${NC}"
            ;;
    esac
}

monitor_services() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          📊 KibaLock - Monitoring en Temps Réel             ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter le monitoring${NC}"
    echo ""
    
    while true; do
        clear
        echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║          📊 KibaLock - Monitoring en Temps Réel             ║${NC}"
        echo -e "${CYAN}║          $(date '+%Y-%m-%d %H:%M:%S')                                  ║${NC}"
        echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        # Statut des services
        echo -e "${MAGENTA}═══ SERVICES ═══${NC}"
        echo ""
        
        # LifeModo API
        if [ -f "$API_PID_FILE" ]; then
            local api_pid=$(cat "$API_PID_FILE")
            if ps -p $api_pid > /dev/null 2>&1; then
                local api_mem=$(ps -p $api_pid -o rss= 2>/dev/null | awk '{printf "%.1f MB", $1/1024}')
                local api_cpu=$(ps -p $api_pid -o %cpu= 2>/dev/null | xargs)
                local api_time=$(ps -p $api_pid -o etime= 2>/dev/null | xargs)
                echo -e "${GREEN}● LifeModo API (FastAPI)${NC}"
                echo -e "  PID:     ${api_pid}"
                echo -e "  Status:  ${GREEN}Running${NC}"
                echo -e "  Uptime:  ${api_time}"
                echo -e "  CPU:     ${api_cpu}%"
                echo -e "  Memory:  ${api_mem}"
                echo -e "  Port:    $API_PORT"
            else
                echo -e "${RED}● LifeModo API${NC}"
                echo -e "  Status:  ${RED}Stopped${NC}"
            fi
        else
            echo -e "${RED}● LifeModo API${NC}"
            echo -e "  Status:  ${RED}Not started${NC}"
        fi
        
        echo ""
        
        # Backend KibaLock
        if [ -f "$BACKEND_PID_FILE" ]; then
            local backend_pid=$(cat "$BACKEND_PID_FILE")
            if ps -p $backend_pid > /dev/null 2>&1; then
                local backend_mem=$(ps -p $backend_pid -o rss= 2>/dev/null | awk '{printf "%.1f MB", $1/1024}')
                local backend_cpu=$(ps -p $backend_pid -o %cpu= 2>/dev/null | xargs)
                local backend_time=$(ps -p $backend_pid -o etime= 2>/dev/null | xargs)
                echo -e "${GREEN}● Backend KibaLock (Streamlit + FAISS)${NC}"
                echo -e "  PID:     ${backend_pid}"
                echo -e "  Status:  ${GREEN}Running${NC}"
                echo -e "  Uptime:  ${backend_time}"
                echo -e "  CPU:     ${backend_cpu}%"
                echo -e "  Memory:  ${backend_mem}"
                echo -e "  Port:    $BACKEND_PORT"
            else
                echo -e "${RED}● Backend KibaLock${NC}"
                echo -e "  Status:  ${RED}Stopped${NC}"
            fi
        else
            echo -e "${RED}● Backend KibaLock${NC}"
            echo -e "  Status:  ${RED}Not started${NC}"
        fi
        
        echo ""
        
        # Frontend React
        if [ "$SKIP_FRONTEND" = false ] && [ -f "$FRONTEND_PID_FILE" ]; then
            local frontend_pid=$(cat "$FRONTEND_PID_FILE")
            if ps -p $frontend_pid > /dev/null 2>&1; then
                local frontend_mem=$(ps -p $frontend_pid -o rss= 2>/dev/null | awk '{printf "%.1f MB", $1/1024}')
                local frontend_cpu=$(ps -p $frontend_pid -o %cpu= 2>/dev/null | xargs)
                local frontend_time=$(ps -p $frontend_pid -o etime= 2>/dev/null | xargs)
                echo -e "${GREEN}● Frontend React (3D Interface)${NC}"
                echo -e "  PID:     ${frontend_pid}"
                echo -e "  Status:  ${GREEN}Running${NC}"
                echo -e "  Uptime:  ${frontend_time}"
                echo -e "  CPU:     ${frontend_cpu}%"
                echo -e "  Memory:  ${frontend_mem}"
                echo -e "  Port:    $FRONTEND_PORT"
            else
                echo -e "${RED}● Frontend React${NC}"
                echo -e "  Status:  ${RED}Stopped${NC}"
            fi
        else
            echo -e "${YELLOW}● Frontend React${NC}"
            echo -e "  Status:  ${YELLOW}Skipped${NC}"
        fi
        
        echo ""
        echo -e "${MAGENTA}═══ ACTIVITÉ RÉCENTE ═══${NC}"
        echo ""
        
        # Dernières lignes des logs API
        echo -e "${CYAN}🧠 LifeModo API (derniers événements):${NC}"
        tail -3 "$API_LOG" 2>/dev/null | grep -v "^$" | sed 's/^/  /' || echo -e "  ${YELLOW}Aucune activité récente${NC}"
        echo ""
        
        # Dernières lignes des logs Backend
        echo -e "${CYAN}🐍 Backend KibaLock (derniers événements):${NC}"
        tail -3 "$BACKEND_LOG" 2>/dev/null | grep -v "^$" | sed 's/^/  /' || echo -e "  ${YELLOW}Aucune activité récente${NC}"
        echo ""
        
        # Statistiques système
        echo -e "${MAGENTA}═══ SYSTÈME ═══${NC}"
        echo ""
        
        # Charge système
        local load_avg=$(uptime | grep -oP 'load average: \K.*')
        echo -e "${CYAN}Load Average:${NC} ${load_avg}"
        
        # Mémoire
        local mem_info=$(free -h | grep "Mem:" | awk '{printf "Used: %s / Total: %s", $3, $2}')
        echo -e "${CYAN}Memory:${NC} ${mem_info}"
        
        # Connexions réseau
        local connections=$(netstat -an 2>/dev/null | grep -E ":($API_PORT|$BACKEND_PORT|$FRONTEND_PORT)" | grep ESTABLISHED | wc -l)
        echo -e "${CYAN}Active Connections:${NC} ${connections}"
        
        echo ""
        echo -e "${YELLOW}Rafraîchissement dans 5 secondes... (Ctrl+C pour quitter)${NC}"
        
        sleep 5
    done
}

###############################################################################
# Menu principal
###############################################################################

case "${1:-start}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        status_services
        ;;
    logs)
        show_logs "${2:-kernel}"
        ;;
    monitor|watch)
        monitor_services
        ;;
    *)
        echo -e "${CYAN}Usage: $0 {start|stop|restart|status|logs|monitor}${NC}"
        echo ""
        echo -e "${YELLOW}Commandes disponibles:${NC}"
        echo -e "  ${GREEN}start${NC}              - Démarrer les services KibaLock"
        echo -e "  ${GREEN}stop${NC}               - Arrêter les services"
        echo -e "  ${GREEN}restart${NC}            - Redémarrer les services"
        echo -e "  ${GREEN}status${NC}             - Voir le statut des services"
        echo -e "  ${GREEN}logs [service]${NC}     - Voir les logs (api|backend|frontend|kernel|all)"
        echo -e "  ${GREEN}monitor${NC}            - Monitoring en temps réel"
        echo ""
        echo -e "${CYAN}Exemples:${NC}"
        echo -e "  $0 start"
        echo -e "  $0 logs api"
        echo -e "  $0 logs all"
        echo -e "  $0 monitor"
        echo -e "  $0 status"
        echo ""
        echo -e "${PURPLE}${BOLD}KibaLock - Système d'Authentification Biométrique Multimodal${NC}"
        exit 1
        ;;
esac
