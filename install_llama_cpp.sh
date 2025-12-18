#!/bin/bash
# Script d'installation de llama-cpp-python pour SETRAF
# Permet d'utiliser des modèles GGUF avec memory mapping natif

echo "🔥 Installation de llama-cpp-python pour optimisation RAM..."
echo ""

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé. Installez Python 3.8+ d'abord."
    exit 1
fi

echo "📦 Installation de llama-cpp-python (CPU uniquement)..."
pip install llama-cpp-python --upgrade

echo ""
echo "✅ Installation terminée !"
echo ""
echo "📥 Pour télécharger un modèle GGUF optimisé:"
echo "   cd models/"
echo "   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
echo ""
echo "💡 Avantages GGUF:"
echo "   - Utilise 2-3GB RAM au lieu de 14-21GB"
echo "   - Memory mapping natif (poids sur SSD)"
echo "   - Plus rapide au chargement"
echo ""
