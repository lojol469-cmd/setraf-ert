#!/bin/bash
# Script de lancement SETRAF avec optimisations mémoire

clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🔥 SETRAF - Lancement avec optimisation mémoire              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Vérifier si Streamlit est déjà en cours d'exécution
if pgrep -f "streamlit run.*ERTest.py" > /dev/null; then
    echo "⚠️  Streamlit est déjà en cours d'exécution"
    echo "🔄 Arrêt de l'instance précédente..."
    pkill -9 -f "streamlit run.*ERTest.py"
    sleep 3
    echo "✅ Instance arrêtée"
    echo ""
fi

# Afficher la mémoire disponible
echo "📊 État de la mémoire:"
free -h | grep "Mem:"
echo ""

# Vérifier la mémoire disponible
AVAILABLE_MEM=$(free -g | awk '/^Mem:/ {print $7}')
echo "💾 Mémoire disponible: ${AVAILABLE_MEM}GB"

if [ "$AVAILABLE_MEM" -lt 4 ]; then
    echo "❌ ERREUR: Mémoire insuffisante (${AVAILABLE_MEM}GB < 4GB requis)"
    echo "💡 Fermez d'autres applications ou redémarrez le système"
    exit 1
fi

echo "✅ Mémoire suffisante pour le démarrage"
echo ""

# Variables d'optimisation mémoire
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

# Limiter l'usage mémoire Python
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=100000

echo "⚙️  Variables d'optimisation configurées"
echo "🚀 Lancement de SETRAF avec Memory Mapping..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Lancer Streamlit en arrière-plan
cd /home/belikan/KIbalione8/SETRAF
nohup streamlit run ERTest.py --server.maxUploadSize 500 > setraf_output.log 2>&1 &
STREAMLIT_PID=$!

echo "✅ SETRAF démarré (PID: $STREAMLIT_PID)"
echo ""
echo "🌐 Accès: http://localhost:8501"
echo "📝 Logs: tail -f setraf_output.log"
echo "🛑 Stop: kill $STREAMLIT_PID"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 Dans l'application, sélectionnez:"
echo "   🤖 Transformers + mmap (4-6GB RAM)"
echo "   ou"
echo "   🔥 GGUF + llama.cpp (2-3GB RAM) [si installé]"
echo ""

# Attendre quelques secondes pour voir si le démarrage réussit
sleep 5

if ps -p $STREAMLIT_PID > /dev/null; then
    echo "✅ Application lancée avec succès !"
    echo ""
    echo "📊 Surveillance mémoire en temps réel:"
    watch -n 5 "ps aux | grep streamlit | grep -v grep | awk '{print \$2, \$4\"% RAM\", \$6/1024\"MB\"}'"
else
    echo "❌ Erreur au démarrage. Consultez les logs:"
    tail -20 setraf_output.log
fi
