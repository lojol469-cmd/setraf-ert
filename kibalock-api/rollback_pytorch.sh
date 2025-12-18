#!/bin/bash

###############################################################################
# Script de rollback PyTorch vers CUDA 12.1 stable
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

GESTMODO_PIP="$HOME/miniconda3/envs/gestmodo/bin/pip"

echo -e "${YELLOW}${BOLD}🔄 Rollback PyTorch vers CUDA 12.1${NC}\n"

# Trouver le dernier backup
LATEST_BACKUP=$(ls -t /tmp/gestmodo_packages_backup_*.txt 2>/dev/null | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo -e "${RED}❌ Aucun backup trouvé${NC}"
    echo -e "${YELLOW}Installation manuelle:${NC}"
    echo "  $GESTMODO_PIP install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

echo -e "${CYAN}📦 Backup trouvé: $LATEST_BACKUP${NC}\n"

# Désinstaller PyTorch actuel
echo -e "${YELLOW}Désinstallation PyTorch nightly...${NC}"
$GESTMODO_PIP uninstall -y torch torchvision torchaudio

# Réinstaller version stable
echo -e "${YELLOW}Installation PyTorch 2.5.1 + CUDA 12.1...${NC}"
$GESTMODO_PIP install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo -e "${GREEN}✅ Rollback terminé${NC}\n"
