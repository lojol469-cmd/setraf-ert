#!/bin/bash
# Script de monitoring mémoire pour SETRAF

clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  📊 SETRAF - Monitoring Mémoire en temps réel                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Fonction pour afficher l'usage mémoire
show_memory() {
    echo "⏰ $(date '+%H:%M:%S')"
    echo ""
    
    # Mémoire système globale
    echo "🖥️  MÉMOIRE SYSTÈME:"
    free -h | head -2
    echo ""
    
    # Processus Streamlit
    echo "🔥 PROCESSUS SETRAF:"
    ps aux | grep streamlit | grep -v grep | awk '{
        printf "   PID: %s\n", $2
        printf "   CPU: %s%%\n", $3
        printf "   RAM: %s%% (%s MB)\n", $4, $6/1024
        printf "   Temps: %s\n", $10
    }'
    
    if ! pgrep -f "streamlit run.*ERTest.py" > /dev/null; then
        echo "   ⚠️  Aucun processus SETRAF actif"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Affichage en boucle
while true; do
    show_memory
    sleep 5
done
