#!/bin/bash

# =====================================================
# SETRAF - Script de démarrage avec téléchargement automatique des modèles
# Ce script télécharge les modèles IA depuis HuggingFace si nécessaire
# =====================================================

set -e  # Arrêt en cas d'erreur

echo "════════════════════════════════════════════════════════════════"
echo "🚀 SETRAF - ERT Geophysical Analysis Platform"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Fonction pour afficher le temps écoulé
start_time=$(date +%s)
show_elapsed() {
    elapsed=$(($(date +%s) - start_time))
    echo "⏱️  Temps écoulé: ${elapsed}s"
}

# Vérification de la connexion Internet
echo "🔍 Vérification de la connexion Internet..."
if curl -s --connect-timeout 5 https://huggingface.co > /dev/null; then
    echo "✅ Connexion Internet OK"
    ONLINE=true
else
    echo "⚠️  Pas de connexion Internet - Mode hors-ligne"
    ONLINE=false
fi
echo ""

# Fonction pour télécharger un modèle HuggingFace
download_model() {
    local model_type=$1
    local model_id=$2
    local display_name=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Modèle: $display_name"
    echo "   ID: $model_id"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python3 << EOF
import os
import sys
from pathlib import Path

# Configuration du cache
os.environ['HF_HOME'] = '/root/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface'

model_type = "$model_type"
model_id = "$model_id"

try:
    print(f"🔄 Chargement/téléchargement de {model_id}...")
    
    if model_type == "sentence-transformer":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id)
        print(f"✅ Sentence Transformer chargé avec succès")
        
    elif model_type == "clip":
        from transformers import CLIPProcessor, CLIPModel
        print("   - Téléchargement du processeur CLIP...")
        processor = CLIPProcessor.from_pretrained(model_id)
        print("   - Téléchargement du modèle CLIP...")
        model = CLIPModel.from_pretrained(model_id)
        print(f"✅ CLIP chargé avec succès")
        
    elif model_type == "mistral":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        print("   - Téléchargement du tokenizer Mistral...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("   - Téléchargement du modèle Mistral (cela peut prendre 10-15 min)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        print(f"✅ Mistral chargé avec succès")
    
    # Vérifier la taille du cache
    cache_path = Path("/root/.cache/huggingface")
    if cache_path.exists():
        import subprocess
        result = subprocess.run(['du', '-sh', str(cache_path)], 
                              capture_output=True, text=True)
        size = result.stdout.split()[0]
        print(f"📊 Taille du cache: {size}")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement: {str(e)}", file=sys.stderr)
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        echo "✅ $display_name prêt"
        show_elapsed
    else
        echo "❌ Échec du téléchargement de $display_name"
        exit 1
    fi
    echo ""
}

# Téléchargement des modèles si connexion Internet disponible
if [ "$ONLINE" = true ]; then
    echo "════════════════════════════════════════════════════════════════"
    echo "📥 TÉLÉCHARGEMENT DES MODÈLES IA"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Modèle 1: Sentence Transformer pour embeddings (88 MB)
    download_model "sentence-transformer" \
                   "sentence-transformers/all-MiniLM-L6-v2" \
                   "Sentence Transformer (Embeddings RAG)"
    
    # Modèle 2: CLIP pour analyse d'images (600 MB)
    download_model "clip" \
                   "openai/clip-vit-base-patch32" \
                   "CLIP Vision-Language Model"
    
    # Modèle 3: Mistral pour génération de texte (14 GB) - OPTIONNEL
    # Décommentez si vous voulez télécharger Mistral au démarrage
    if [ "${DOWNLOAD_MISTRAL:-false}" = "true" ]; then
        download_model "mistral" \
                       "mistralai/Mistral-7B-v0.1" \
                       "Mistral-7B (Génération de rapports)"
    else
        echo "⏭️  Mistral-7B sera téléchargé à la première utilisation (14 GB)"
        echo ""
    fi
    
    echo "════════════════════════════════════════════════════════════════"
    echo "✅ TOUS LES MODÈLES SONT PRÊTS"
    echo "════════════════════════════════════════════════════════════════"
    show_elapsed
    echo ""
else
    echo "⚠️  Mode hors-ligne: Les modèles doivent déjà être en cache"
    echo ""
fi

# Afficher les informations de cache
echo "════════════════════════════════════════════════════════════════"
echo "📊 ÉTAT DU CACHE"
echo "════════════════════════════════════════════════════════════════"
if [ -d "/root/.cache/huggingface" ]; then
    cache_size=$(du -sh /root/.cache/huggingface 2>/dev/null | cut -f1 || echo "0")
    model_count=$(find /root/.cache/huggingface -name "config.json" 2>/dev/null | wc -l)
    echo "   Emplacement: /root/.cache/huggingface"
    echo "   Taille totale: $cache_size"
    echo "   Modèles en cache: $model_count"
else
    echo "   ⚠️  Cache vide (premier démarrage)"
fi
echo ""

# Démarrage de l'application Streamlit
echo "════════════════════════════════════════════════════════════════"
echo "🎯 DÉMARRAGE DE L'APPLICATION SETRAF"
echo "════════════════════════════════════════════════════════════════"
echo "   URL: http://0.0.0.0:8504"
echo "   Fichier: ERTest.py"
echo ""
show_elapsed
echo "════════════════════════════════════════════════════════════════"
echo ""

# Lancer Streamlit
exec streamlit run ERTest.py \
    --server.port=8504 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true \
    --browser.gatherUsageStats=false
