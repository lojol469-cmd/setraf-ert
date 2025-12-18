#!/bin/bash

# =====================================================
# SETRAF - Préparation déploiement Hugging Face Spaces
# =====================================================

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🤗 PRÉPARATION DÉPLOIEMENT HUGGING FACE SPACES               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Créer dossier de déploiement
DEPLOY_DIR="hf_spaces_deploy"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

echo "📦 Copie des fichiers nécessaires..."

# Copier les fichiers essentiels
cp ERTest.py "$DEPLOY_DIR/"
cp requirements_hf_spaces.txt "$DEPLOY_DIR/requirements.txt"
cp README_HF_SPACES.md "$DEPLOY_DIR/README.md"

# Copier logo si existe
if [ -f "logo_belikan.png" ]; then
    cp logo_belikan.png "$DEPLOY_DIR/"
    echo "  ✅ logo_belikan.png"
fi

echo "  ✅ ERTest.py"
echo "  ✅ requirements.txt"
echo "  ✅ README.md"
echo ""

# Créer .gitignore
cat > "$DEPLOY_DIR/.gitignore" << 'EOF'
__pycache__/
*.pyc
*.pyo
.env
.venv
venv/
*.log
.DS_Store
EOF
echo "  ✅ .gitignore"
echo ""

# Afficher la taille des fichiers
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Fichiers préparés:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
du -sh "$DEPLOY_DIR"/*
echo ""

# Instructions
cat << 'EOF'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 PROCHAINES ÉTAPES - DÉPLOIEMENT SUR HUGGING FACE SPACES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  Aller sur: https://huggingface.co/new-space

2️⃣  Créer un nouveau Space:
   - Owner: BelikanM
   - Space name: ERT-SETRAF
   - License: MIT
   - SDK: Streamlit
   - Hardware: CPU basic (gratuit) ou CPU upgrade (payant)

3️⃣  Une fois créé, cloner le repo:
   git clone https://huggingface.co/spaces/BelikanM/ERT-SETRAF
   cd ERT-SETRAF

4️⃣  Copier les fichiers:
   cp -r ../hf_spaces_deploy/* .

5️⃣  Configurer les secrets (dans l'interface web HF Spaces):
   Settings → Repository secrets → New secret
   
   Ajouter:
   - Name: HF_TOKEN
     Value: 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
   
   - Name: TAVILY_API_KEY
     Value: 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

6️⃣  Pusher les fichiers:
   git add .
   git commit -m "Initial commit - SETRAF ERT Analysis"
   git push

7️⃣  Attendre le build (5-10 minutes)
   HF Spaces va installer les dépendances automatiquement
   
   Les modèles IA seront téléchargés au premier démarrage

8️⃣  Accéder à votre app:
   https://huggingface.co/spaces/BelikanM/ERT-SETRAF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ MÉTHODE RAPIDE (avec Hugging Face CLI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Installer HF CLI si pas déjà fait
pip install huggingface-hub

# Login
huggingface-cli login
# Token: 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

# Créer et pusher le Space
cd hf_spaces_deploy
huggingface-cli upload BelikanM/ERT-SETRAF . --repo-type=space

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Dossier prêt: ./hf_spaces_deploy/

EOF
