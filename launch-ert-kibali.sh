#!/bin/bash

###############################################################################
# Lanceur ERT.py (Kibali Analyst avec tab ERTest intégré)
# Port: 8506
###############################################################################

echo "🚀 Démarrage de ERT.py (Kibali Analyst complet avec ERTest intégré)..."
echo "📊 Port: 8506"
echo ""

# Arrêter les instances existantes
pkill -f "streamlit run ERT.py" 2>/dev/null || true
sleep 2

# Démarrer ERT.py
cd /home/belikan/KIbalione8/SETRAF
conda run -n gestmodo streamlit run ERT.py --server.port 8506 --server.address 0.0.0.0 &
ERT_PID=$!

echo ""
echo "✅ ERT.py démarré avec succès !"
echo ""
echo "📊 URL d'accès: http://localhost:8506"
echo "🔧 Process ID: $ERT_PID"
echo ""
echo "⏹️  Pour arrêter:"
echo "   pkill -f 'streamlit run ERT.py'"
echo ""
