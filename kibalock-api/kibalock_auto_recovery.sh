#!/bin/bash
# KibaLock Auto-Recovery - Version Simplifiée et Rapide

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SCRIPT="$SCRIPT_DIR/kibalock-kernel.sh"

# Couleurs
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🚀 KibaLock Auto-Recovery${NC}\n"

# Activer conda
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null
conda activate gestmodo 2>/dev/null

# Vérification rapide packages critiques
echo "� Vérification dépendances..."
python3 "$SCRIPT_DIR/kibalock_kernel_agent.py"

# Lancer services
echo -e "\n🚀 Démarrage services..."
"$KERNEL_SCRIPT" start

echo -e "\n${GREEN}✅ Terminé${NC}"
echo -e "Services: http://localhost:8000 | http://localhost:8505"

