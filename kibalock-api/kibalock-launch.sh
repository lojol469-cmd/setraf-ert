#!/bin/bash
# KibaLock Launcher Global - Active gestmodo et lance tout
# Usage: ./kibalock-launch.sh [start|stop|restart|status]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL="$SCRIPT_DIR/kibalock-kernel.sh"

# Bannière
echo -e "\033[0;36m"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🔒 KibaLock - Système Biométrique Intelligent           ║
║     Mini OS Autonome avec GPU CUDA 13.0                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "\033[0m"

# Activer gestmodo automatiquement
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate gestmodo 2>/dev/null && {
        echo -e "\033[0;32m✅ Environnement conda 'gestmodo' activé\033[0m"
    } || {
        echo -e "\033[1;33m⚠️  Impossible d'activer gestmodo\033[0m"
    }
fi

# Passer tous les arguments au kernel
"$KERNEL" "$@"
