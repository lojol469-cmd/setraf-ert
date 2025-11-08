#!/bin/bash
# ========================================
# SETRAF - Subaquifère ERT Analysis Tool
# Script de lancement
# ========================================

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}   SETRAF - Subaquifère ERT Analysis Tool${NC}"
echo -e "${CYAN}   💧 Analyse géophysique avancée${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""

# Définir le répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Définir l'environnement Python
GESTMODO_PYTHON="$HOME/miniconda3/envs/gestmodo/bin/python"
GESTMODO_PIP="$HOME/miniconda3/envs/gestmodo/bin/pip"

# Vérifier que l'environnement gestmodo existe
if [ ! -f "$GESTMODO_PYTHON" ]; then
    echo -e "${RED}❌ Erreur: Environnement gestmodo non trouvé!${NC}"
    echo -e "${RED}   Chemin attendu: $GESTMODO_PYTHON${NC}"
    echo ""
    echo -e "${YELLOW}📝 Créer l'environnement avec:${NC}"
    echo -e "   conda create -n gestmodo python=3.10"
    echo -e "   conda activate gestmodo"
    echo -e "   pip install -r requirements.txt"
    exit 1
fi

# Afficher la version Python
PYTHON_VERSION=$($GESTMODO_PYTHON --version 2>&1)
echo -e "${GREEN}✅ Python trouvé: $PYTHON_VERSION${NC}"
echo -e "${GREEN}✅ Environnement: gestmodo${NC}"
echo ""

# Vérifier si requirements.txt existe et installer les dépendances
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}📦 Vérification des dépendances...${NC}"
    
    # Vérifier si streamlit est installé
    if ! $GESTMODO_PYTHON -c "import streamlit" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Installation des dépendances manquantes...${NC}"
        $GESTMODO_PIP install -r requirements.txt -q
        echo -e "${GREEN}✅ Dépendances installées${NC}"
    else
        echo -e "${GREEN}✅ Dépendances OK${NC}"
    fi
fi
echo ""

# Arrêter les instances Streamlit existantes
echo -e "${YELLOW}🔄 Arrêt des instances existantes...${NC}"
pkill -9 -f "streamlit run" 2>/dev/null || true
sleep 2

# Port par défaut
PORT=${1:-8504}

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}   📊 Fonctionnalités SETRAF${NC}"
echo -e "${CYAN}================================================${NC}"
echo -e "${GREEN}  ✅ Tab 1: Calculateur Température Ts (Ravensgate Sonic)${NC}"
echo -e "${GREEN}  ✅ Tab 2: Analyse fichiers .dat avec sections d'eau${NC}"
echo -e "${GREEN}  ✅ Tab 3: Pseudo-sections ERT 2D/3D${NC}"
echo -e "${GREEN}  ✅ Tab 4: 🪨 Stratigraphie Complète + 3D interactive${NC}"
echo -e "${GREEN}  ✅ Tab 5: 🔬 Inversion pyGIMLi - ERT géophysique avancée${NC}"
echo -e "${GREEN}  ✅ Précision millimétrique (3 décimales)${NC}"
echo -e "${GREEN}  ✅ Export PDF stratigraphique haute résolution${NC}"
echo -e "${GREEN}  ✅ Classification automatique 8 catégories géologiques${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""

# Lancer l'application
echo -e "${GREEN}🚀 Démarrage de SETRAF...${NC}"
echo -e "${BLUE}🌐 URL locale: http://localhost:$PORT${NC}"
echo -e "${BLUE}📡 URL réseau: http://$(hostname -I | awk '{print $1}'):$PORT${NC}"
echo ""
echo -e "${YELLOW}💡 Appuyez sur Ctrl+C pour arrêter l'application${NC}"
echo ""

# Lancer Streamlit
$GESTMODO_PYTHON -m streamlit run ERTest.py \
    --server.port $PORT \
    --server.headless true \
    --browser.gatherUsageStats false

# Cleanup
echo ""
echo -e "${YELLOW}🛑 Arrêt de SETRAF${NC}"
