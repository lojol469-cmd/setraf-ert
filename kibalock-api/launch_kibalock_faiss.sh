#!/bin/bash

# === KibaLock FAISS Launcher ===
# Lance le système d'authentification biométrique avec FAISS
# Ultra-fast similarity search avec index vectoriel

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         🔐 KibaLock FAISS - Authentification Biométrique     ║"
echo "║         ⚡ Powered by FAISS Vector Database                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Répertoire du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# === Vérifications système ===
echo -e "${BLUE}[1/8]${NC} Vérification de Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 n'est pas installé${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓${NC} Python ${PYTHON_VERSION} détecté"

# === Environnement virtuel ===
echo -e "${BLUE}[2/8]${NC} Configuration de l'environnement virtuel..."
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚙${NC} Création de l'environnement virtuel..."
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Environnement virtuel créé"
else
    echo -e "${GREEN}✓${NC} Environnement virtuel existant"
fi

# Activation
source venv/bin/activate

# === Vérification .env ===
echo -e "${BLUE}[3/8]${NC} Vérification de la configuration..."
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠${NC} Fichier .env non trouvé, création depuis .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}⚠${NC} Veuillez configurer le fichier .env avant de continuer"
        exit 1
    else
        echo -e "${RED}❌ Fichier .env.example introuvable${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓${NC} Configuration trouvée"

# === Installation des dépendances ===
echo -e "${BLUE}[4/8]${NC} Vérification des dépendances..."

# Vérifier si FAISS est installé
if ! python3 -c "import faiss" &> /dev/null; then
    echo -e "${YELLOW}⚙${NC} Installation de FAISS et des dépendances..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} Dépendances installées"
else
    echo -e "${GREEN}✓${NC} FAISS déjà installé"
fi

# === Vérification MongoDB ===
echo -e "${BLUE}[5/8]${NC} Vérification de la connexion MongoDB..."
MONGO_URI=$(grep MONGO_URI .env | cut -d'=' -f2)
if [ -z "$MONGO_URI" ]; then
    echo -e "${RED}❌ MONGO_URI non configuré dans .env${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} URI MongoDB configuré"

# === Création des répertoires ===
echo -e "${BLUE}[6/8]${NC} Création de la structure des répertoires..."
mkdir -p ~/kibalock/embeddings
mkdir -p ~/kibalock/temp
mkdir -p ~/kibalock/logs
mkdir -p ~/kibalock/faiss_indexes
echo -e "${GREEN}✓${NC} Répertoires créés"

# === Vérification des ports ===
echo -e "${BLUE}[7/8]${NC} Vérification des ports..."
PORT=$(grep STREAMLIT_PORT .env | cut -d'=' -f2)
PORT=${PORT:-8505}

if lsof -i:$PORT &> /dev/null; then
    echo -e "${YELLOW}⚠${NC} Port $PORT déjà utilisé"
    PID=$(lsof -ti:$PORT)
    echo -e "${YELLOW}⚠${NC} PID du processus: $PID"
    read -p "Voulez-vous tuer le processus existant? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill -9 $PID
        echo -e "${GREEN}✓${NC} Processus arrêté"
    else
        echo -e "${RED}❌ Abandon du lancement${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓${NC} Port $PORT disponible"

# === Affichage des informations système ===
echo ""
echo -e "${PURPLE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║     📊 INFORMATIONS SYSTÈME FAISS             ║${NC}"
echo -e "${PURPLE}╠═══════════════════════════════════════════════╣${NC}"

# Vérifier les index existants
if [ -d ~/kibalock/faiss_indexes ] && [ "$(ls -A ~/kibalock/faiss_indexes)" ]; then
    VOICE_SIZE=$(stat -c%s ~/kibalock/faiss_indexes/voice_index.faiss 2>/dev/null | numfmt --to=iec || echo "0")
    FACE_SIZE=$(stat -c%s ~/kibalock/faiss_indexes/face_index.faiss 2>/dev/null | numfmt --to=iec || echo "0")
    COMBINED_SIZE=$(stat -c%s ~/kibalock/faiss_indexes/combined_index.faiss 2>/dev/null | numfmt --to=iec || echo "0")
    
    echo -e "${PURPLE}║${NC} 🎤 Index Vocal:    ${VOICE_SIZE}                    "
    echo -e "${PURPLE}║${NC} 📸 Index Facial:   ${FACE_SIZE}                    "
    echo -e "${PURPLE}║${NC} 🧬 Index Combiné:  ${COMBINED_SIZE}                    "
else
    echo -e "${PURPLE}║${NC} ⚠️  Aucun index FAISS existant                   "
    echo -e "${PURPLE}║${NC} 📝 Les index seront créés au premier utilisateur"
fi

echo -e "${PURPLE}║${NC} 🌐 Port:           $PORT                            "
echo -e "${PURPLE}║${NC} 🐍 Python:         $PYTHON_VERSION                  "
echo -e "${PURPLE}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# === Lancement de Streamlit ===
echo -e "${BLUE}[8/8]${NC} Lancement de KibaLock FAISS..."
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  🚀 KibaLock FAISS démarré avec succès !                      ║${NC}"
echo -e "${GREEN}║                                                                ║${NC}"
echo -e "${GREEN}║  📱 Interface web: http://localhost:$PORT                       ║${NC}"
echo -e "${GREEN}║  ⚡ Mode: FAISS Ultra-Fast Search                              ║${NC}"
echo -e "${GREEN}║  🔐 Authentification: Voix + Visage                           ║${NC}"
echo -e "${GREEN}║                                                                ║${NC}"
echo -e "${GREEN}║  📖 Pour arrêter: Ctrl+C                                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Lancer Streamlit avec la version FAISS
streamlit run kibalock_faiss.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.primaryColor="#667eea" \
    --theme.backgroundColor="#ffffff" \
    --theme.secondaryBackgroundColor="#f0f2f6" \
    --theme.textColor="#262730" \
    --theme.font="sans serif"
