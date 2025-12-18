#!/bin/bash

###############################################################################
# Script de déploiement SETRAF sur GitHub
# Auteur: BelikanM / lojol469-cmd
###############################################################################

set -e

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║     🚀 Déploiement SETRAF sur GitHub + Hugging Face         ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Variables
GITHUB_USERNAME="lojol469-cmd"
GITHUB_REPO="setraf-ert"
GITHUB_URL="https://github.com/${GITHUB_USERNAME}/${GITHUB_REPO}.git"

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "   GitHub User: $GITHUB_USERNAME"
echo "   Repository: $GITHUB_REPO"
echo "   URL: $GITHUB_URL"
echo ""

# 1. Vérifier que nous sommes dans le bon dossier
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}❌ Erreur: Dockerfile non trouvé${NC}"
    echo -e "${YELLOW}💡 Exécutez ce script depuis /home/belikan/setraf-frontend-hf${NC}"
    exit 1
fi

# 2. Initialiser Git si nécessaire
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}🔧 Initialisation du dépôt Git...${NC}"
    git init
    git config user.name "$HF_USERNAME"
    git config user.email "nyundumathryme@gmail.com"
    echo -e "${GREEN}✓ Git initialisé${NC}"
else
    echo -e "${GREEN}✓ Dépôt Git existant${NC}"
fi

# 3. Créer .gitignore
echo -e "${YELLOW}📝 Création de .gitignore...${NC}"
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Secrets
.env
*.key
*.pem

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
*.bak
EOF
echo -e "${GREEN}✓ .gitignore créé${NC}"

# 4. Ajouter tous les fichiers
echo -e "${YELLOW}📦 Ajout des fichiers...${NC}"
git add .
echo -e "${GREEN}✓ Fichiers ajoutés${NC}"

# 5. Commit
echo -e "${YELLOW}💾 Création du commit...${NC}"
git commit -m "🚀 Initial deployment of SETRAF frontend to Hugging Face Spaces

- Streamlit application with PyGIMLi for ERT analysis
- Docker configuration optimized for HF Spaces
- Authentication module integrated with Render backend
- Water type classification and visualization tools
" || echo -e "${YELLOW}⚠️  Aucun changement à commiter${NC}"
echo -e "${GREEN}✓ Commit créé${NC}"

# 6. Configurer le remote GitHub
echo -e "${YELLOW}🔗 Configuration du remote GitHub...${NC}"

# Supprimer l'ancien remote s'il existe
git remote remove origin 2>/dev/null || true

# Ajouter le nouveau remote
git remote add origin "$GITHUB_URL"
echo -e "${GREEN}✓ Remote GitHub configuré${NC}"

# 7. Push vers GitHub
echo -e "${YELLOW}🚀 Déploiement vers GitHub...${NC}"
echo -e "${BLUE}   Cela peut prendre quelques instants...${NC}"

# Vérifier si la branche main existe
if git show-ref --verify --quiet refs/heads/main; then
    echo -e "${GREEN}✓ Branche main existante${NC}"
else
    echo -e "${YELLOW}⚠️  Renommage de la branche en main${NC}"
    git branch -M main
fi

git push -u origin main --force

echo -e "${GREEN}✓ Code poussé vers GitHub${NC}"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  ✅ Déploiement GitHub réussi !                              ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}📦 Repository GitHub:${NC}"
echo -e "   https://github.com/${GITHUB_USERNAME}/${GITHUB_REPO}"
echo ""
echo -e "${YELLOW}⚙️  Prochaines étapes pour déployer sur Hugging Face:${NC}"
echo ""
echo -e "${CYAN}1️⃣  Créer un Space Hugging Face:${NC}"
echo -e "   • Aller sur: https://huggingface.co/new-space"
echo -e "   • Owner: BelikanM"
echo -e "   • Space name: setraf-ert"
echo -e "   • License: agpl-3.0"
echo -e "   • Space SDK: Docker"
echo -e "   • Space hardware: CPU (ou GPU T4 si besoin de PyGIMLi optimisé)"
echo ""
echo -e "${CYAN}2️⃣  Connecter le repository GitHub:${NC}"
echo -e "   • Dans Settings du Space"
echo -e "   • Section 'Repository'"
echo -e "   • Lier avec: https://github.com/${GITHUB_USERNAME}/${GITHUB_REPO}"
echo ""
echo -e "${CYAN}3️⃣  Configurer les Secrets (Settings > Variables and secrets):${NC}"
echo -e "   • USE_PRODUCTION_BACKEND = true"
echo -e "   • PRODUCTION_BACKEND_URL = https://setraf-auth.onrender.com"
echo ""
echo -e "${CYAN}4️⃣  Synchroniser et déployer:${NC}"
echo -e "   • Cliquer sur 'Sync' dans le Space"
echo -e "   • Le build Docker démarrera automatiquement (5-10 min)"
echo ""
echo -e "${GREEN}🎉 Votre application sera accessible sur:${NC}"
echo -e "   https://huggingface.co/spaces/BelikanM/setraf-ert"
echo ""
echo -e "${YELLOW}💡 Astuce: Chaque push sur GitHub mettra à jour automatiquement le Space !${NC}"
echo ""
