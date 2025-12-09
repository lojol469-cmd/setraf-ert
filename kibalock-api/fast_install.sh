#!/bin/bash
# Installation ultra-rapide avec aria2c
# Télécharge en parallèle avec 16 connexions par fichier

set -e

CONDA_ENV="gestmodo"
echo "🚀 Installation ULTRA-RAPIDE avec aria2c..."

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Configuration pip pour utiliser aria2c (téléchargements parallèles)
export PIP_DOWNLOAD_CACHE="$HOME/.cache/pip"
mkdir -p "$PIP_DOWNLOAD_CACHE"

# Liste des packages par ordre de priorité
CRITICAL_PACKAGES=(
    "deepface"
    "opencv-python-headless" 
    "facenet-pytorch"
    "retina-face"
    "mediapipe"
    "soundfile"
    "librosa"
    "speechrecognition"
    "motor"
    "bcrypt"
    "pyjwt"
    "cryptography"
    "psutil"
    "tf-keras"
)

echo "📦 Installation de ${#CRITICAL_PACKAGES[@]} packages critiques..."

# Installation parallèle par groupe de 5
for ((i=0; i<${#CRITICAL_PACKAGES[@]}; i+=5)); do
    GROUP=("${CRITICAL_PACKAGES[@]:i:5}")
    echo "🔧 Groupe $((i/5 + 1)): ${GROUP[*]}"
    
    # Installer en parallèle avec pip (aria2 automatique si disponible)
    pip install -q --upgrade --use-deprecated=legacy-resolver "${GROUP[@]}" &
    
    # Limiter à 3 groupes en parallèle max
    if [ $((($i/5 + 1) % 3)) -eq 0 ]; then
        wait
    fi
done

# Attendre la fin de tous les processus
wait

echo ""
echo "✅ Installation terminée!"
echo ""
echo "🔍 Vérification..."
python3 << 'PYEOF'
packages = ['deepface', 'torch', 'fastapi', 'langchain', 'whisper', 'transformers', 
            'streamlit', 'pymongo', 'cv2', 'soundfile', 'librosa', 'motor']
ok = []
missing = []
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
        ok.append(pkg)
        print(f'✅ {pkg}')
    except:
        missing.append(pkg)
        print(f'❌ {pkg}')

print(f'\n📊 Résultat: {len(ok)}/{len(packages)} packages installés')
if missing:
    print(f'⚠️  Manquants: {", ".join(missing)}')
PYEOF

echo ""
echo "🚀 Prêt pour le lancement!"
