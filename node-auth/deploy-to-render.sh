#!/bin/bash

# ==========================================
# SETRAF Backend - Render Deployment Helper
# ==========================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "════════════════════════════════════════════════════"
echo "   🚀 SETRAF Backend - Déploiement Render"
echo "════════════════════════════════════════════════════"
echo -e "${NC}"

# Vérifier que l'image existe
echo -e "${YELLOW}🔍 Vérification de l'image Docker...${NC}"
if /mnt/c/Program\ Files/Docker/Docker/resources/bin/docker images | grep -q "belikanm/setraf-auth"; then
    echo -e "${GREEN}✅ Image trouvée localement${NC}"
    /mnt/c/Program\ Files/Docker/Docker/resources/bin/docker images | grep "belikanm/setraf-auth"
else
    echo -e "${RED}❌ Image non trouvée. Exécutez ./docker-build.sh d'abord${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}📦 Information de l'image:${NC}"
echo "   Image: belikanm/setraf-auth:latest"
echo "   Docker Hub: https://hub.docker.com/r/belikanm/setraf-auth"
echo "   Taille: 279MB"
echo ""

# Étapes de déploiement
echo -e "${YELLOW}📋 Étapes de déploiement sur Render:${NC}"
echo ""
echo -e "${GREEN}1. Créer un compte sur Render${NC}"
echo "   → https://dashboard.render.com/register"
echo ""

echo -e "${GREEN}2. Créer un nouveau Web Service${NC}"
echo "   → Dashboard → New + → Web Service"
echo "   → Deploy an existing image from a registry"
echo ""

echo -e "${GREEN}3. Configurer le service${NC}"
echo "   Image URL: docker.io/belikanm/setraf-auth:latest"
echo "   Name: setraf-auth-backend"
echo "   Region: Oregon (US West) ou Frankfurt (Europe)"
echo "   Instance Type: Starter (7$/mois) ou Free"
echo ""

echo -e "${GREEN}4. Configuration réseau${NC}"
echo "   Port: 5000"
echo "   Health Check Path: /api/health"
echo ""

echo -e "${GREEN}5. Variables d'environnement${NC}"
echo "   (Voir le fichier ci-dessous)"
echo ""

# Créer un fichier avec les variables d'environnement
ENV_FILE="render-env-variables.txt"
echo -e "${YELLOW}📝 Création du fichier des variables d'environnement...${NC}"

cat > "$ENV_FILE" << 'EOF'
# =========================
# SETRAF Backend - Render Environment Variables
# Copier ces variables dans Render Dashboard → Environment
# =========================

NODE_ENV=production
AUTH_PORT=5000

# MongoDB Atlas
MONGO_URI=mongodb+srv://SETRAF:Dieu19961991%3F%3F%21%3F%3F%21@cluster0.5tjz9v0.mongodb.net/myDatabase10?retryWrites=true&w=majority&appName=Cluster0
MONGO_USER=SETRAF
MONGO_PASSWORD=Dieu19961991??!??!
MONGO_CLUSTER=cluster0.5tjz9v0.mongodb.net
MONGO_DB_NAME=myDatabase10

# JWT Secrets
JWT_SECRET=Dieu19961991??!??!
JWT_REFRESH_SECRET=Dieu19961991??!??!_refresh

# Email Configuration (Nodemailer)
EMAIL_USER=nyundumathryme@gmail.com
EMAIL_PASS=zsrrymlixizhiybl

# API Keys
PUBLIC_KEY=qazghazz
PRIVATE_KEY=264419a2-cd4e-471a-81b3-04c522669052

# =========================
# IMPORTANT:
# 1. Aller sur MongoDB Atlas (https://cloud.mongodb.com)
# 2. Network Access → Add IP Address → "Allow from anywhere" (0.0.0.0/0)
# 3. Cela permettra à Render de se connecter à MongoDB
# =========================
EOF

echo -e "${GREEN}✅ Fichier créé: $ENV_FILE${NC}"
echo ""

# Instructions MongoDB
echo -e "${YELLOW}⚠️  IMPORTANT - Configuration MongoDB Atlas:${NC}"
echo ""
echo "Avant de déployer, configurer MongoDB Atlas:"
echo "1. Aller sur https://cloud.mongodb.com"
echo "2. Sélectionner votre cluster"
echo "3. Network Access → Add IP Address"
echo "4. Choisir 'Allow access from anywhere' (0.0.0.0/0)"
echo "5. Confirmer"
echo ""
echo -e "${YELLOW}Cela permet à Render (IP dynamique) de se connecter à MongoDB${NC}"
echo ""

# Test local avant déploiement
echo -e "${YELLOW}🧪 Test local de l'image (optionnel):${NC}"
echo ""
echo "Pour tester avant de déployer:"
echo "  docker run -p 5000:5000 --env-file ../.env belikanm/setraf-auth:latest"
echo ""
echo "Puis tester:"
echo "  curl http://localhost:5000/api/health"
echo ""

# Commandes utiles
echo -e "${BLUE}📚 Commandes utiles après déploiement:${NC}"
echo ""
echo "# Tester le backend déployé"
echo "curl https://VOTRE-SERVICE.onrender.com/api/health"
echo ""
echo "# Voir les logs"
echo "render logs --service setraf-auth-backend --tail"
echo ""
echo "# Redéployer après une mise à jour"
echo "render deploy --service setraf-auth-backend"
echo ""

# Résumé
echo -e "${BLUE}"
echo "════════════════════════════════════════════════════"
echo "   ✅ Prêt pour le déploiement"
echo "════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""
echo -e "${GREEN}Fichiers créés:${NC}"
echo "  ✓ render.yaml (config Render)"
echo "  ✓ RENDER_DEPLOYMENT.md (documentation complète)"
echo "  ✓ $ENV_FILE (variables d'environnement)"
echo ""
echo -e "${YELLOW}Prochaines étapes:${NC}"
echo "  1. Ouvrir https://dashboard.render.com"
echo "  2. Créer le Web Service avec l'image Docker"
echo "  3. Copier les variables depuis $ENV_FILE"
echo "  4. Configurer MongoDB Atlas IP whitelist"
echo "  5. Déployer !"
echo ""
echo -e "${BLUE}📖 Documentation complète: RENDER_DEPLOYMENT.md${NC}"
echo ""
