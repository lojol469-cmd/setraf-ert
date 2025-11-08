#!/bin/bash

# Script de test de l'API OTP SETRAF
# Usage: ./test_otp.sh [email]

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://172.20.31.35:5000/api/auth"
EMAIL="${1:-test@example.com}"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         🧪 Test de l'API OTP SETRAF                     ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Test 1 : Envoyer un OTP
echo -e "${YELLOW}📤 Test 1: Envoi du code OTP${NC}"
echo -e "   Email: ${EMAIL}"
echo ""

RESPONSE=$(curl -s -X POST "${API_URL}/send-otp" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"${EMAIL}\"}")

echo "Réponse:"
echo "${RESPONSE}" | jq '.' 2>/dev/null || echo "${RESPONSE}"
echo ""

# Vérifier si succès
SUCCESS=$(echo "${RESPONSE}" | jq -r '.success' 2>/dev/null)
if [ "$SUCCESS" = "true" ]; then
    echo -e "${GREEN}✅ OTP envoyé avec succès${NC}"
    
    # Extraire le code OTP en mode dev
    OTP_CODE=$(echo "${RESPONSE}" | jq -r '.debug.otpCode' 2>/dev/null)
    if [ "$OTP_CODE" != "null" ] && [ -n "$OTP_CODE" ]; then
        echo -e "${BLUE}🔧 Code OTP (mode dev): ${OTP_CODE}${NC}"
        echo ""
        
        # Test 2 : Vérifier l'OTP
        echo -e "${YELLOW}📥 Test 2: Vérification du code OTP${NC}"
        echo ""
        
        VERIFY_RESPONSE=$(curl -s -X POST "${API_URL}/verify-otp" \
          -H "Content-Type: application/json" \
          -d "{\"email\":\"${EMAIL}\",\"otp\":\"${OTP_CODE}\"}")
        
        echo "Réponse:"
        echo "${VERIFY_RESPONSE}" | jq '.' 2>/dev/null || echo "${VERIFY_RESPONSE}"
        echo ""
        
        VERIFY_SUCCESS=$(echo "${VERIFY_RESPONSE}" | jq -r '.success' 2>/dev/null)
        if [ "$VERIFY_SUCCESS" = "true" ]; then
            echo -e "${GREEN}✅ OTP vérifié avec succès${NC}"
            echo -e "${GREEN}✅ Authentification réussie !${NC}"
            
            # Extraire les tokens
            ACCESS_TOKEN=$(echo "${VERIFY_RESPONSE}" | jq -r '.accessToken' 2>/dev/null)
            if [ -n "$ACCESS_TOKEN" ] && [ "$ACCESS_TOKEN" != "null" ]; then
                echo ""
                echo -e "${BLUE}🔑 Access Token:${NC} ${ACCESS_TOKEN:0:50}..."
            fi
        else
            echo -e "${RED}❌ Échec de la vérification de l'OTP${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Code OTP non disponible (mode production)${NC}"
        echo -e "${YELLOW}   Vérifiez votre boîte email pour obtenir le code${NC}"
        echo ""
        echo -e "${BLUE}Pour tester la vérification, utilisez:${NC}"
        echo -e "curl -X POST ${API_URL}/verify-otp \\"
        echo -e "  -H 'Content-Type: application/json' \\"
        echo -e "  -d '{\"email\":\"${EMAIL}\",\"otp\":\"VOTRE_CODE\"}'"
    fi
else
    echo -e "${RED}❌ Échec de l'envoi de l'OTP${NC}"
    MESSAGE=$(echo "${RESPONSE}" | jq -r '.message' 2>/dev/null)
    if [ -n "$MESSAGE" ] && [ "$MESSAGE" != "null" ]; then
        echo -e "${RED}   Message: ${MESSAGE}${NC}"
    fi
fi

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         📊 Résumé du test                                ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "API URL: ${API_URL}"
echo -e "Email testé: ${EMAIL}"
echo ""
echo -e "${YELLOW}💡 Conseils:${NC}"
echo -e "1. Vérifiez les logs du serveur: ./setraf-kernel.sh logs node"
echo -e "2. Vérifiez votre email (spam inclus)"
echo -e "3. Le code expire après 10 minutes"
echo -e "4. En mode dev, le code s'affiche dans la réponse"
echo ""
