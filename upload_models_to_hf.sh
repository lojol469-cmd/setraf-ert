#!/bin/bash

###############################################################################
# Script pour uploader les modèles locaux vers HF Spaces
# Permet d'éviter les téléchargements à chaque démarrage
###############################################################################

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  📤 UPLOAD DES MODÈLES LOCAUX VERS HF SPACES               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

SPACE_NAME="BelikanM/SETRAF-ERT"
HF_CACHE_DIR="$HOME/.cache/huggingface"

echo "📊 Analyse des modèles locaux..."
echo "• Phi-3-mini-4k-instruct: $(du -sh $HF_CACHE_DIR/hub/models--microsoft--Phi-3-mini-4k-instruct | cut -f1)"
echo "• CLIP vit-base-patch32: $(du -sh $HF_CACHE_DIR/hub/models--openai--clip-vit-base-patch32 | cut -f1)"
echo ""

# Vérifier si HF CLI est installé
if ! command -v huggingface-cli &> /dev/null && ! command -v hf &> /dev/null; then
    echo "📦 Installation de Hugging Face CLI..."
    pip install huggingface-hub --quiet
fi

# Login
echo "🔑 Connexion à Hugging Face..."
if [ -n "$HF_TOKEN" ]; then
    hf auth login --token "$HF_TOKEN"
else
    hf auth login
fi

echo ""
echo "🚀 Upload des modèles..."

# Upload Phi-3
echo "📤 Upload Phi-3-mini-4k-instruct..."
hf upload "$SPACE_NAME" "$HF_CACHE_DIR/hub/models--microsoft--Phi-3-mini-4k-instruct" models/phi3/ --repo-type=space

# Upload CLIP
echo "📤 Upload CLIP vit-base-patch32..."
hf upload "$SPACE_NAME" "$HF_CACHE_DIR/hub/models--openai--clip-vit-base-patch32" models/clip/ --repo-type=space

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ✅ UPLOAD TERMINÉ !                                       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Modèles uploadés dans:"
echo "  • models/phi3/ - Phi-3-mini-4k-instruct"
echo "  • models/clip/ - CLIP vit-base-patch32"
echo ""
echo "🔧 Pour utiliser ces modèles dans le code:"
echo "  • Modifier les chemins vers './models/phi3/' et './models/clip/'"
echo "  • Désactiver local_files_only=True"
echo ""

