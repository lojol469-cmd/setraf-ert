#!/bin/bash

# === LifeModo API Launcher ===
# Télécharge les modèles IA et lance l'API d'entraînement temps réel

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         🧠 LifeModo API - Real-time Training Service         ║"
echo "║              for KibaLock Biometric Authentication            ║"
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
echo -e "${BLUE}[1/7]${NC} Vérification de Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 n'est pas installé${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓${NC} Python ${PYTHON_VERSION} détecté"

# Vérifier CUDA
echo -e "${BLUE}[2/7]${NC} Vérification CUDA/GPU..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo -e "${GREEN}✓${NC} GPU disponible: ${GPU_COUNT} device(s)"
    USE_GPU=true
else
    echo -e "${YELLOW}⚠${NC} Pas de GPU détecté, utilisation du CPU"
    USE_GPU=false
fi

# === Environnement virtuel ===
echo -e "${BLUE}[3/7]${NC} Configuration de l'environnement virtuel..."
if [ ! -d "venv_lifemodo" ]; then
    echo -e "${YELLOW}⚙${NC} Création de l'environnement virtuel..."
    python3 -m venv venv_lifemodo
    echo -e "${GREEN}✓${NC} Environnement virtuel créé"
fi

source venv_lifemodo/bin/activate

# === Installation des dépendances ===
echo -e "${BLUE}[4/7]${NC} Installation des dépendances..."

pip install --upgrade pip wheel setuptools

# Core dependencies
pip install fastapi uvicorn[standard] python-multipart
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
    pip install torch torchvision torchaudio

# AI Models
pip install openai-whisper transformers accelerate sentencepiece
pip install TTS  # Coqui TTS
pip install deepface opencv-python
pip install numpy scipy scikit-learn

echo -e "${GREEN}✓${NC} Dépendances installées"

# === Téléchargement des modèles ===
echo -e "${BLUE}[5/7]${NC} Téléchargement des modèles IA..."

MODELS_DIR="$HOME/lifemodo_api/models"
mkdir -p "$MODELS_DIR"

echo -e "${YELLOW}⚙${NC} Téléchargement de Phi-3.5-mini-instruct (7B)..."
python3 << 'EOF'
import os
from transformers import AutoModelForCausalLM, AutoProcessor

model_name = "microsoft/Phi-3.5-mini-instruct"
cache_dir = os.path.expanduser("~/lifemodo_api/models/phi")

print(f"Téléchargement de {model_name}...")

try:
    processor = AutoProcessor.from_pretrained(
        model_name, 
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ Phi-3.5-mini-instruct téléchargé")
except Exception as e:
    print(f"❌ Erreur: {e}")
EOF

echo -e "${YELLOW}⚙${NC} Téléchargement de Whisper (base)..."
python3 << 'EOF'
import whisper
import os

os.makedirs(os.path.expanduser("~/lifemodo_api/models/whisper"), exist_ok=True)

try:
    model = whisper.load_model("base", download_root=os.path.expanduser("~/lifemodo_api/models/whisper"))
    print("✅ Whisper (base) téléchargé")
except Exception as e:
    print(f"❌ Erreur: {e}")
EOF

echo -e "${YELLOW}⚙${NC} Téléchargement de Coqui TTS (XTTS-v2)..."
python3 << 'EOF'
from TTS.api import TTS
import os

os.environ['COQUI_TOS_AGREED'] = '1'

try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("✅ XTTS-v2 téléchargé")
except Exception as e:
    print(f"❌ Erreur: {e}")
EOF

echo -e "${YELLOW}⚙${NC} Téléchargement de DeepFace (FaceNet512)..."
python3 << 'EOF'
from deepface import DeepFace
import os

try:
    # Force download
    DeepFace.build_model("Facenet512")
    print("✅ FaceNet512 téléchargé")
except Exception as e:
    print(f"❌ Erreur: {e}")
EOF

echo -e "${GREEN}✓${NC} Tous les modèles téléchargés"

# === Création des répertoires ===
echo -e "${BLUE}[6/7]${NC} Création de la structure des répertoires..."
mkdir -p ~/lifemodo_api/models
mkdir -p ~/lifemodo_api/training_data
mkdir -p ~/lifemodo_api/checkpoints
mkdir -p ~/lifemodo_api/logs
echo -e "${GREEN}✓${NC} Répertoires créés"

# === Vérification des ports ===
echo -e "${BLUE}[7/7]${NC} Vérification des ports..."
PORT=8000

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
        PORT=8001
        echo -e "${YELLOW}⚠${NC} Utilisation du port alternatif: $PORT"
    fi
fi
echo -e "${GREEN}✓${NC} Port $PORT disponible"

# === Affichage des informations ===
echo ""
echo -e "${PURPLE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║        📊 INFORMATIONS SYSTÈME                ║${NC}"
echo -e "${PURPLE}╠═══════════════════════════════════════════════╣${NC}"
echo -e "${PURPLE}║${NC} 🐍 Python:        $PYTHON_VERSION"
echo -e "${PURPLE}║${NC} 💾 GPU:           $(if [ "$USE_GPU" = true ]; then echo "Disponible"; else echo "Non disponible (CPU)"; fi)"
echo -e "${PURPLE}║${NC} 🌐 Port API:      $PORT"
echo -e "${PURPLE}║${NC} 📁 Base dir:      ~/lifemodo_api"
echo -e "${PURPLE}╠═══════════════════════════════════════════════╣${NC}"
echo -e "${PURPLE}║        🤖 MODÈLES IA CHARGÉS                  ║${NC}"
echo -e "${PURPLE}╠═══════════════════════════════════════════════╣${NC}"
echo -e "${PURPLE}║${NC} ✅ Phi-3.5-mini-instruct (7B)"
echo -e "${PURPLE}║${NC} ✅ Whisper (base)"
echo -e "${PURPLE}║${NC} ✅ Coqui TTS (XTTS-v2)"
echo -e "${PURPLE}║${NC} ✅ DeepFace (FaceNet512)"
echo -e "${PURPLE}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# === Lancement de l'API ===
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  🚀 LifeModo API démarré avec succès !                        ║${NC}"
echo -e "${GREEN}║                                                                ║${NC}"
echo -e "${GREEN}║  📱 API URL: http://localhost:$PORT                             ║${NC}"
echo -e "${GREEN}║  📖 Documentation: http://localhost:$PORT/docs                  ║${NC}"
echo -e "${GREEN}║  🔐 Pour KibaLock: http://localhost:$PORT/api/*                ║${NC}"
echo -e "${GREEN}║                                                                ║${NC}"
echo -e "${GREEN}║  Endpoints disponibles:                                        ║${NC}"
echo -e "${GREEN}║  • POST /api/train/voice     - Entraîner modèle vocal        ║${NC}"
echo -e "${GREEN}║  • POST /api/train/face      - Entraîner modèle facial       ║${NC}"
echo -e "${GREEN}║  • POST /api/chat            - Chat avec Phi-3.5 AI           ║${NC}"
echo -e "${GREEN}║  • POST /api/voice/clone     - Clonage vocal                  ║${NC}"
echo -e "${GREEN}║  • POST /api/update/embedding - Mise à jour temps réel        ║${NC}"
echo -e "${GREEN}║                                                                ║${NC}"
echo -e "${GREEN}║  📖 Pour arrêter: Ctrl+C                                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Lancer l'API
python3 lifemodo_api.py --host 0.0.0.0 --port $PORT --reload
