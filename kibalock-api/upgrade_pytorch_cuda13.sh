#!/bin/bash

###############################################################################
# Script de migration PyTorch vers CUDA 13.0 nightly pour RTX 5090
# Support CUDA Capability sm_130
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

GESTMODO_PYTHON="$HOME/miniconda3/envs/gestmodo/bin/python"
GESTMODO_PIP="$HOME/miniconda3/envs/gestmodo/bin/pip"

echo -e "${CYAN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║     🔥 Migration PyTorch vers CUDA 13.0 Nightly              ║"
echo "║     Support RTX 5090 (sm_130)                                ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Vérifier l'environnement
if [ ! -f "$GESTMODO_PYTHON" ]; then
    echo -e "${RED}❌ Environnement gestmodo non trouvé${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Étape 1: Sauvegarde des packages actuels${NC}"
echo ""
$GESTMODO_PIP freeze > /tmp/gestmodo_packages_backup_$(date +%Y%m%d_%H%M%S).txt
echo -e "${GREEN}✓ Backup créé${NC}\n"

echo -e "${YELLOW}📊 Étape 2: Packages PyTorch actuels${NC}"
echo ""
$GESTMODO_PIP list | grep -E "torch|nvidia-cuda"
echo ""

echo -e "${YELLOW}⚠️  Étape 3: Désinstallation de PyTorch actuel${NC}"
echo ""
read -p "Continuer la désinstallation? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}ℹ️  Installation annulée${NC}"
    exit 0
fi

echo -e "${CYAN}Désinstallation de torch, torchvision, torchaudio...${NC}"
$GESTMODO_PIP uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Désinstaller aussi les packages CUDA NVIDIA
echo -e "${CYAN}Désinstallation des packages CUDA 12.1...${NC}"
$GESTMODO_PIP uninstall -y nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 2>/dev/null || true
$GESTMODO_PIP uninstall -y nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 2>/dev/null || true
$GESTMODO_PIP uninstall -y nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 2>/dev/null || true
$GESTMODO_PIP uninstall -y nvidia-nccl-cu12 nvidia-nvtx-cu12 2>/dev/null || true

echo -e "${GREEN}✓ Désinstallation terminée${NC}\n"

echo -e "${YELLOW}🔥 Étape 4: Installation PyTorch 2.7.0 Nightly (CUDA 13.0)${NC}"
echo ""
echo -e "${CYAN}Source: https://pytorch.org/get-started/locally/${NC}"
echo -e "${CYAN}Index: https://download.pytorch.org/whl/nightly/cu130${NC}\n"

# Installation depuis l'index nightly
$GESTMODO_PIP install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

echo -e "${GREEN}✓ PyTorch nightly installé${NC}\n"

echo -e "${YELLOW}🔍 Étape 5: Vérification de l'installation${NC}"
echo ""

# Test PyTorch et CUDA
$GESTMODO_PYTHON << 'PYEOF'
import torch
import sys

print("="*60)
print("PyTorch Configuration")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Nombre de GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: sm_{props.major}{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi-processors: {props.multi_processor_count}")
    
    # Test tensor sur GPU
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x)
        print(f"\n✅ Test GPU réussi! Tensor shape: {y.shape}")
    except Exception as e:
        print(f"\n❌ Erreur test GPU: {e}")
        sys.exit(1)
else:
    print("\n⚠️  CUDA non disponible - mode CPU uniquement")

print("="*60)
PYEOF

echo ""

echo -e "${YELLOW}📦 Étape 6: Réinstallation des dépendances cassées (si nécessaire)${NC}"
echo ""

# Vérifier et réinstaller les packages qui pourraient avoir été cassés
PACKAGES_TO_CHECK=(
    "accelerate"
    "transformers"
    "sentence-transformers"
    "openai-whisper"
    "TTS"
)

for pkg in "${PACKAGES_TO_CHECK[@]}"; do
    echo -e "${CYAN}Vérification de $pkg...${NC}"
    if ! $GESTMODO_PYTHON -c "import ${pkg//-/_}" 2>/dev/null; then
        echo -e "${YELLOW}  ⚠️  $pkg cassé, réinstallation...${NC}"
        $GESTMODO_PIP install --upgrade --no-deps $pkg
    else
        echo -e "${GREEN}  ✓ $pkg OK${NC}"
    fi
done

echo ""
echo -e "${GREEN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║     ✅ Migration PyTorch CUDA 13.0 terminée !                ║"
echo "║                                                               ║"
echo "║     Votre RTX 5090 est maintenant pleinement supporté        ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

echo -e "${CYAN}💡 Prochaines étapes:${NC}"
echo "  1. Testez avec: $GESTMODO_PYTHON -c 'import torch; print(torch.cuda.is_available())'"
echo "  2. Lancez KibaLock: ./kibalock-kernel.sh start"
echo "  3. Si problème, restaurez: $GESTMODO_PIP install -r /tmp/gestmodo_packages_backup_*.txt"
echo ""
