#!/bin/bash
# Script de lancement rapide de l'application SETRAF optimisée

echo "=================================================="
echo "🚀 SETRAF - Application ERT Optimisée"
echo "=================================================="
echo ""
echo "✅ Optimisations appliquées :"
echo "   • LLM 3-4x plus rapide (15-30s au lieu de 60s)"
echo "   • Protection anti-blocage avec timeout de 45s"
echo "   • Fallback intelligent automatique"
echo "   • Correction de l'erreur accelerate"
echo ""
echo "📊 Utilisation recommandée :"
echo "   1. Charger vos données ERT (.dat)"
echo "   2. Activer 'Analyse LLM complète'"
echo "   3. Cliquer sur '🧠 Lancer l'analyse LLM'"
echo "   4. Attendre 15-30 secondes → Interprétation générée !"
echo ""
echo "=================================================="
echo "Lancement de Streamlit..."
echo "=================================================="
echo ""

cd "$(dirname "$0")"
streamlit run ERTest.py --server.maxUploadSize 500
