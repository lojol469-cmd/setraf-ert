#!/bin/bash
# Installation rapide de TOUTES les dépendances KibaLock
# Compatible CUDA 13.0 + PyTorch 2.10

set -e

CONDA_ENV="gestmodo"
echo "🚀 Installation dépendances KibaLock..."

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Groupe 1: Vision & Face (compatible CUDA)
echo "📸 Vision & Face Recognition..."
pip install -q deepface opencv-python-headless facenet-pytorch retina-face mediapipe

# Groupe 2: Audio
echo "🎵 Audio Processing..."
pip install -q soundfile librosa speechrecognition pydub

# Groupe 3: Database & Web
echo "🗄️  Database & Web..."
pip install -q motor python-multipart aiofiles

# Groupe 4: Security
echo "🔐 Security..."
pip install -q bcrypt pyjwt cryptography

# Groupe 5: Utilities
echo "🛠️  Utilities..."
pip install -q psutil pandas scikit-learn scipy

# Groupe 6: AI & LangChain (déjà installés mais vérification)
echo "🤖 AI & LangChain..."
pip install -q --upgrade langchain langchain-huggingface langchain-community

# Groupe 7: TensorFlow/Keras (pour deepface)
echo "🧠 TensorFlow/Keras..."
pip install -q tf-keras tensorflow

echo "✅ Installation terminée!"
echo ""
echo "📦 Vérification packages critiques..."
python3 -c "
packages = ['deepface', 'torch', 'fastapi', 'langchain', 'whisper', 'transformers', 'streamlit', 'pymongo', 'cv2', 'soundfile']
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
        print(f'✓ {pkg}')
    except:
        print(f'✗ {pkg}')
"
