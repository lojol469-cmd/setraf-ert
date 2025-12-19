#!/bin/bash

###############################################################################
# Script de déploiement SETRAF sur HF Spaces avec modèles locaux
# Utilise les modèles déjà en cache pour éviter les téléchargements
###############################################################################

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  🚀 DÉPLOIEMENT SETRAF AVEC MODÈLES LOCAUX                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Variables
SPACE_NAME="BelikanM/SETRAF-ERT"
HF_CACHE_DIR="$HOME/.cache/huggingface"

echo "📦 Préparation du déploiement..."

# 1. Préparer les fichiers de base
./prepare_hf_spaces.sh

# 2. Créer un script d'initialisation qui copie les modèles locaux
cat > hf_spaces_deploy/setup_models.py << 'EOF_SETUP'
#!/usr/bin/env python3
"""
Script d'initialisation pour copier les modèles locaux vers HF Spaces
"""
import os
import shutil
from pathlib import Path

def copy_model_to_cache(local_model_path, cache_model_name):
    """Copie un modèle local vers le cache HF"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    target_dir = cache_dir / f"models--{cache_model_name.replace('/', '--')}"
    
    if target_dir.exists():
        print(f"✅ {cache_model_name} déjà présent dans le cache")
        return True
    
    source_dir = Path(local_model_path)
    if not source_dir.exists():
        print(f"❌ Source {local_model_path} introuvable")
        return False
    
    print(f"📋 Copie {cache_model_name} vers le cache...")
    try:
        shutil.copytree(source_dir, target_dir)
        print(f"✅ {cache_model_name} copié avec succès")
        return True
    except Exception as e:
        print(f"❌ Erreur copie {cache_model_name}: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration des modèles locaux pour HF Spaces...")
    
    # Copier Phi-3-mini
    phi_success = copy_model_to_cache(
        "/home/belikan/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-mini-4k-instruct"
    )
    
    # Copier CLIP
    clip_success = copy_model_to_cache(
        "/home/belikan/.cache/huggingface/hub/models--openai--clip-vit-base-patch32",
        "openai/clip-vit-base-patch32"
    )
    
    if phi_success and clip_success:
        print("🎉 Tous les modèles configurés avec succès!")
    else:
        print("⚠️ Certains modèles n'ont pas pu être copiés")
EOF_SETUP

chmod +x hf_spaces_deploy/setup_models.py

# 3. Modifier le README pour indiquer l'utilisation des modèles locaux
cat > hf_spaces_deploy/README.md << 'EOF_README'
---
title: SETRAF - ERT Geophysical Analysis
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: ERTest.py
pinned: false
license: mit
---

# SETRAF - Subaquifère ERT Analysis Tool

Application d'analyse géophysique par tomographie de résistivité électrique (ERT).

## 🌟 Fonctionnalités

- 📊 Analyse de données ERT
- 🤖 Intelligence artificielle intégrée (RAG + LLMs locaux)
- 📈 Visualisations interactives
- 📄 Génération de rapports automatiques
- 🔍 Système de recherche avancé

## 🚀 Technologies

- **Streamlit** - Interface utilisateur
- **PyGIMLi** - Géophysique ERT
- **Transformers** - Modèles IA (CLIP, Phi-3-mini) - CACHE LOCAL
- **LangChain** - Système RAG
- **PyTorch** - Deep Learning

## ⚡ Avantages

- **Modèles locaux** : Pas de téléchargement au démarrage
- **Performance** : Démarrage ultra-rapide
- **Fiable** : Modèles toujours disponibles

## 📝 Configuration

Les secrets suivants doivent être configurés dans les Settings de l'espace:

- `HF_TOKEN` - Token Hugging Face (optionnel pour modèles locaux)

## 👨‍💻 Auteur

Belikan M. - nyundumathryme@gmail.com

## 📄 Licence

MIT License
EOF_README

echo "📤 Déploiement sur Hugging Face Spaces..."

# 4. Utiliser HF CLI pour uploader
cd hf_spaces_deploy

# Vérifier si HF CLI est installé
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installation de Hugging Face CLI..."
    pip install huggingface-hub --quiet
fi

# Login (utilise le token de l'env si disponible)
if [ -n "$HF_TOKEN" ]; then
    echo "🔑 Login avec token existant..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "🔑 Login interactif (entrez votre token HF):"
    huggingface-cli login
fi

# Créer/uploader le space
echo "🚀 Upload vers $SPACE_NAME..."
huggingface-cli upload "$SPACE_NAME" . --repo-type=space

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ✅ DÉPLOIEMENT TERMINÉ !                                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 Accès à votre app: https://huggingface.co/spaces/$SPACE_NAME"
echo ""
echo "⚡ Avantages:"
echo "  • Modèles locaux = démarrage ultra-rapide"
echo "  • Pas de téléchargement = économie de bande passante"
echo "  • Fiable = fonctionne même sans connexion internet"
echo ""

