#!/bin/bash
# =====================================================
# SETRAF - Pré-téléchargement des modèles IA
# Script à exécuter AVANT le déploiement (optionnel)
# Télécharge tous les modèles pour accélérer le premier démarrage
# =====================================================

set -e

echo "🚀 =============================================="
echo "🚀 Pré-téléchargement des modèles IA SETRAF"
echo "🚀 =============================================="
echo ""

# Configuration
CACHE_DIR="${1:-/opt/setraf/huggingface-cache}"
mkdir -p "$CACHE_DIR"

echo "📁 Dossier de cache: $CACHE_DIR"
echo ""

# Fonction pour télécharger un modèle
download_model() {
    local MODEL_NAME=$1
    local MODEL_ID=$2
    local MODEL_TYPE=$3
    local MODEL_SIZE=$4
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Téléchargement: $MODEL_NAME"
    echo "   ID: $MODEL_ID"
    echo "   Taille: $MODEL_SIZE"
    echo ""
    
    python3 << PYTHON_SCRIPT
import os
os.environ['HF_HOME'] = '$CACHE_DIR'
os.environ['TRANSFORMERS_CACHE'] = '$CACHE_DIR'

print("⬇️  Téléchargement en cours...")

try:
    if '$MODEL_TYPE' == 'sentence_transformers':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('$MODEL_ID', cache_folder='$CACHE_DIR')
    elif '$MODEL_TYPE' == 'transformers':
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('$MODEL_ID', cache_dir='$CACHE_DIR')
        model = AutoModel.from_pretrained('$MODEL_ID', cache_dir='$CACHE_DIR')
    
    print("✅ Téléchargement terminé!")
except Exception as e:
    print(f"❌ Erreur: {str(e)}")
    exit(1)
PYTHON_SCRIPT

    if [ $? -eq 0 ]; then
        echo "✅ $MODEL_NAME téléchargé avec succès"
    else
        echo "❌ Échec du téléchargement de $MODEL_NAME"
        return 1
    fi
    echo ""
}

# Vérifier Python et les dépendances
echo "🔍 Vérification des dépendances..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trouvé. Installez Python 3.10+"
    exit 1
fi

# Installer les bibliothèques si nécessaire
echo "📦 Installation des bibliothèques Python..."
pip install -q transformers sentence-transformers torch

echo "✅ Dépendances OK"
echo ""

# Télécharger les modèles
download_model \
    "SentenceTransformer (Embeddings)" \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "sentence_transformers" \
    "88 MB"

download_model \
    "CLIP (Vision)" \
    "openai/clip-vit-base-patch32" \
    "transformers" \
    "600 MB"

# Mistral-7B (optionnel car très gros)
read -p "⚠️  Télécharger Mistral-7B (14 GB) ? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    download_model \
        "Mistral-7B (LLM)" \
        "mistralai/Mistral-7B-v0.1" \
        "transformers" \
        "14 GB"
else
    echo "⏭️  Mistral-7B ignoré (sera téléchargé à la première utilisation)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Tous les modèles sont téléchargés!"
echo "📁 Cache: $CACHE_DIR"
echo ""
echo "💡 Montez ce dossier dans Docker avec:"
echo "   -v $CACHE_DIR:/root/.cache/huggingface"
echo ""
du -sh "$CACHE_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
