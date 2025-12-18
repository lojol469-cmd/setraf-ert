#!/bin/bash

# =====================================================
# SETRAF - Script de déploiement Kubernetes
# Usage: ./deploy-k8s.sh [apply|delete|status]
# =====================================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NAMESPACE="setraf"
ACTION="${1:-apply}"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}☸️  SETRAF - Déploiement Kubernetes${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

case "$ACTION" in
    apply)
        echo -e "${GREEN}📦 Déploiement de SETRAF sur Kubernetes...${NC}"
        echo ""
        
        # 1. Créer le namespace
        echo -e "${YELLOW}1. Création du namespace${NC}"
        kubectl apply -f kubernetes/namespace.yaml
        echo ""
        
        # 2. Créer les ConfigMaps et Secrets
        echo -e "${YELLOW}2. Configuration (ConfigMap & Secrets)${NC}"
        kubectl apply -f kubernetes/configmap.yaml
        echo ""
        
        # 3. Créer les PVC
        echo -e "${YELLOW}3. Persistent Volume Claims${NC}"
        kubectl apply -f kubernetes/pvc.yaml
        echo ""
        
        # 4. Déployer l'application
        echo -e "${YELLOW}4. Déploiement de l'application${NC}"
        kubectl apply -f kubernetes/deployment.yaml
        echo ""
        
        # 5. Créer le service
        echo -e "${YELLOW}5. Service & Ingress${NC}"
        kubectl apply -f kubernetes/service.yaml
        echo ""
        
        echo -e "${GREEN}✅ Déploiement terminé${NC}"
        echo ""
        echo -e "${BLUE}📊 État du déploiement:${NC}"
        kubectl get all -n ${NAMESPACE}
        ;;
        
    delete)
        echo -e "${RED}🗑️  Suppression de SETRAF...${NC}"
        echo ""
        read -p "Êtes-vous sûr de vouloir supprimer SETRAF ? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kubectl delete -f kubernetes/service.yaml
            kubectl delete -f kubernetes/deployment.yaml
            kubectl delete -f kubernetes/pvc.yaml
            kubectl delete -f kubernetes/configmap.yaml
            kubectl delete namespace ${NAMESPACE}
            echo -e "${GREEN}✅ SETRAF supprimé${NC}"
        else
            echo -e "${YELLOW}Annulé${NC}"
        fi
        ;;
        
    status)
        echo -e "${BLUE}📊 État de SETRAF:${NC}"
        echo ""
        kubectl get all -n ${NAMESPACE}
        echo ""
        echo -e "${BLUE}📦 Volumes:${NC}"
        kubectl get pvc -n ${NAMESPACE}
        echo ""
        echo -e "${BLUE}📋 Logs (dernières 50 lignes):${NC}"
        kubectl logs -n ${NAMESPACE} -l app=setraf --tail=50
        ;;
        
    *)
        echo -e "${RED}Usage: $0 [apply|delete|status]${NC}"
        exit 1
        ;;
esac
