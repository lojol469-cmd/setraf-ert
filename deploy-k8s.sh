#!/bin/bash

###############################################################################
# SETRAF - Kubernetes Deployment Script for Infomaniak
# Déploiement complet de l'application SETRAF sur cluster Kubernetes
###############################################################################

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
K8S_DIR="${SCRIPT_DIR}/k8s"

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     🚀 SETRAF - Kubernetes Deployment (Infomaniak)          ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Vérification des prérequis
check_prerequisites() {
    echo -e "${YELLOW}🔍 Vérification des prérequis...${NC}"
    
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl n'est pas installé${NC}"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ docker n'est pas installé${NC}"
        exit 1
    fi
    
    # Vérifier la connexion au cluster
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ Impossible de se connecter au cluster Kubernetes${NC}"
        echo -e "${YELLOW}Configurez d'abord votre kubeconfig pour Infomaniak${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Prérequis OK${NC}"
    echo ""
}

# Afficher les informations du cluster
show_cluster_info() {
    echo -e "${CYAN}📊 Informations du cluster:${NC}"
    kubectl cluster-info
    echo ""
    kubectl get nodes
    echo ""
}

# Créer le namespace
create_namespace() {
    echo -e "${YELLOW}📦 Création du namespace 'setraf'...${NC}"
    kubectl apply -f "${K8S_DIR}/namespace.yaml"
    echo -e "${GREEN}✓ Namespace créé${NC}"
    echo ""
}

# Créer les secrets depuis .env
create_secrets() {
    echo -e "${YELLOW}🔐 Création des secrets...${NC}"
    
    if [ -f "${SCRIPT_DIR}/.env" ]; then
        echo -e "${CYAN}Lecture du fichier .env...${NC}"
        source "${SCRIPT_DIR}/.env"
        
        # Créer le secret Kubernetes depuis les variables d'environnement
        kubectl create secret generic setraf-secrets \
            --namespace=setraf \
            --from-literal=mongo-uri="${MONGO_URI}" \
            --from-literal=mongo-user="${MONGO_USER}" \
            --from-literal=mongo-password="${MONGO_PASSWORD}" \
            --from-literal=mongo-cluster="${MONGO_CLUSTER}" \
            --from-literal=mongo-db-name="${MONGO_DB_NAME}" \
            --from-literal=jwt-secret="${JWT_SECRET}" \
            --from-literal=jwt-refresh-secret="${JWT_REFRESH_SECRET}" \
            --from-literal=email-user="${EMAIL_USER}" \
            --from-literal=email-pass="${EMAIL_PASS}" \
            --from-literal=public-key="${PUBLIC_KEY:-}" \
            --from-literal=private-key="${PRIVATE_KEY:-}" \
            --from-literal=hf-token="${HF_TOKEN:-}" \
            --from-literal=tavily-api-key="${TAVILY_API_KEY:-}" \
            --dry-run=client -o yaml | kubectl apply -f -
        
        echo -e "${GREEN}✓ Secrets créés depuis .env${NC}"
    else
        echo -e "${YELLOW}⚠️  Fichier .env non trouvé${NC}"
        echo -e "${CYAN}Création depuis le template...${NC}"
        kubectl apply -f "${K8S_DIR}/secret.yaml.template"
        echo -e "${YELLOW}⚠️  N'oubliez pas de modifier les valeurs!${NC}"
    fi
    echo ""
}

# Créer les ConfigMaps
create_configmaps() {
    echo -e "${YELLOW}⚙️  Création des ConfigMaps...${NC}"
    kubectl apply -f "${K8S_DIR}/configmap.yaml"
    echo -e "${GREEN}✓ ConfigMaps créés${NC}"
    echo ""
}

# Créer les PersistentVolumeClaims
create_storage() {
    echo -e "${YELLOW}💾 Création du stockage persistant...${NC}"
    kubectl apply -f "${K8S_DIR}/pvc.yaml"
    echo -e "${GREEN}✓ PVC créés${NC}"
    
    echo -e "${CYAN}Attente de la création des volumes...${NC}"
    sleep 5
    kubectl get pvc -n setraf
    echo ""
}

# Déployer les applications
deploy_applications() {
    echo -e "${YELLOW}🚀 Déploiement des applications...${NC}"
    kubectl apply -f "${K8S_DIR}/deployment.yaml"
    echo -e "${GREEN}✓ Deployments créés${NC}"
    
    echo -e "${CYAN}Attente du démarrage des pods...${NC}"
    kubectl wait --for=condition=ready pod -l app=setraf -n setraf --timeout=300s || true
    echo ""
}

# Créer les services
create_services() {
    echo -e "${YELLOW}🌐 Création des services...${NC}"
    kubectl apply -f "${K8S_DIR}/service.yaml"
    echo -e "${GREEN}✓ Services créés${NC}"
    echo ""
}

# Créer l'Ingress
create_ingress() {
    echo -e "${YELLOW}🔗 Création de l'Ingress...${NC}"
    kubectl apply -f "${K8S_DIR}/ingress.yaml"
    echo -e "${GREEN}✓ Ingress créé${NC}"
    echo ""
}

# Configurer l'autoscaling
setup_autoscaling() {
    echo -e "${YELLOW}📈 Configuration de l'autoscaling...${NC}"
    kubectl apply -f "${K8S_DIR}/hpa.yaml"
    echo -e "${GREEN}✓ HPA configurés${NC}"
    echo ""
}

# Afficher le statut du déploiement
show_status() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                  📊 STATUT DU DÉPLOIEMENT                    ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${YELLOW}Pods:${NC}"
    kubectl get pods -n setraf
    echo ""
    
    echo -e "${YELLOW}Services:${NC}"
    kubectl get svc -n setraf
    echo ""
    
    echo -e "${YELLOW}Ingress:${NC}"
    kubectl get ingress -n setraf
    echo ""
    
    echo -e "${YELLOW}HPA:${NC}"
    kubectl get hpa -n setraf
    echo ""
    
    echo -e "${YELLOW}PVC:${NC}"
    kubectl get pvc -n setraf
    echo ""
}

# Obtenir l'URL d'accès
get_access_url() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    🌐 ACCÈS À L'APPLICATION                  ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Attendre que l'ingress obtienne une IP
    echo -e "${YELLOW}Récupération de l'adresse d'accès...${NC}"
    sleep 5
    
    INGRESS_IP=$(kubectl get ingress setraf-ingress -n setraf -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    INGRESS_HOSTNAME=$(kubectl get ingress setraf-ingress -n setraf -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    
    if [ -n "$INGRESS_IP" ]; then
        echo -e "${GREEN}✓ IP externe: ${INGRESS_IP}${NC}"
    elif [ -n "$INGRESS_HOSTNAME" ]; then
        echo -e "${GREEN}✓ Hostname: ${INGRESS_HOSTNAME}${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}URLs configurées:${NC}"
    echo -e "  ${GREEN}Frontend:${NC} https://setraf.infomaniak.ch"
    echo -e "  ${GREEN}Alternative:${NC} https://ert.infomaniak.ch"
    echo -e "  ${GREEN}Backend API:${NC} https://setraf.infomaniak.ch/api/auth"
    echo -e "  ${GREEN}FastAPI:${NC} https://setraf.infomaniak.ch/api/ert"
    echo ""
    
    echo -e "${YELLOW}⚠️  Configuration DNS requise:${NC}"
    echo -e "Ajoutez les enregistrements DNS suivants chez Infomaniak:"
    if [ -n "$INGRESS_IP" ]; then
        echo -e "  ${CYAN}setraf.infomaniak.ch${NC} → A → ${GREEN}${INGRESS_IP}${NC}"
        echo -e "  ${CYAN}ert.infomaniak.ch${NC} → A → ${GREEN}${INGRESS_IP}${NC}"
    elif [ -n "$INGRESS_HOSTNAME" ]; then
        echo -e "  ${CYAN}setraf.infomaniak.ch${NC} → CNAME → ${GREEN}${INGRESS_HOSTNAME}${NC}"
        echo -e "  ${CYAN}ert.infomaniak.ch${NC} → CNAME → ${GREEN}${INGRESS_HOSTNAME}${NC}"
    else
        echo -e "  ${RED}En attente de l'allocation d'IP par le LoadBalancer...${NC}"
    fi
    echo ""
}

# Commandes utiles
show_useful_commands() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    📚 COMMANDES UTILES                       ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Consulter les logs:${NC}"
    echo -e "  kubectl logs -f deployment/setraf-app -n setraf"
    echo -e "  kubectl logs -f deployment/setraf-auth -n setraf"
    echo -e "  kubectl logs -f deployment/setraf-api -n setraf"
    echo ""
    echo -e "${YELLOW}Redémarrer un déploiement:${NC}"
    echo -e "  kubectl rollout restart deployment/setraf-app -n setraf"
    echo ""
    echo -e "${YELLOW}Scaler manuellement:${NC}"
    echo -e "  kubectl scale deployment/setraf-app --replicas=5 -n setraf"
    echo ""
    echo -e "${YELLOW}Accéder à un pod:${NC}"
    echo -e "  kubectl exec -it <pod-name> -n setraf -- /bin/bash"
    echo ""
    echo -e "${YELLOW}Supprimer tout:${NC}"
    echo -e "  kubectl delete namespace setraf"
    echo ""
}

# Menu interactif
interactive_menu() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              🎯 MENU DE DÉPLOIEMENT SETRAF                   ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Que voulez-vous faire?"
    echo ""
    echo "  1) Déploiement complet (recommandé)"
    echo "  2) Créer uniquement le namespace"
    echo "  3) Déployer les applications"
    echo "  4) Mettre à jour les images"
    echo "  5) Afficher le statut"
    echo "  6) Supprimer tout"
    echo "  7) Quitter"
    echo ""
    read -p "Choix [1-7]: " choice
    
    case $choice in
        1)
            full_deployment
            ;;
        2)
            check_prerequisites
            create_namespace
            ;;
        3)
            check_prerequisites
            deploy_applications
            show_status
            ;;
        4)
            update_images
            ;;
        5)
            show_status
            get_access_url
            show_useful_commands
            ;;
        6)
            delete_all
            ;;
        7)
            echo -e "${GREEN}Au revoir!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Choix invalide${NC}"
            exit 1
            ;;
    esac
}

# Déploiement complet
full_deployment() {
    check_prerequisites
    show_cluster_info
    create_namespace
    create_secrets
    create_configmaps
    create_storage
    deploy_applications
    create_services
    create_ingress
    setup_autoscaling
    show_status
    get_access_url
    show_useful_commands
    
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           ✅ DÉPLOIEMENT TERMINÉ AVEC SUCCÈS!                ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
}

# Mise à jour des images
update_images() {
    echo -e "${YELLOW}🔄 Mise à jour des images Docker...${NC}"
    kubectl set image deployment/setraf-app setraf-frontend=belikanm/kibaertanalyste:1.0.0 -n setraf
    kubectl set image deployment/setraf-auth setraf-backend=belikanm/setraf-auth:latest -n setraf
    kubectl set image deployment/setraf-api setraf-api=belikanm/kibaertanalyste:1.0.0 -n setraf
    
    echo -e "${CYAN}Attente du rollout...${NC}"
    kubectl rollout status deployment/setraf-app -n setraf
    kubectl rollout status deployment/setraf-auth -n setraf
    kubectl rollout status deployment/setraf-api -n setraf
    
    echo -e "${GREEN}✓ Images mises à jour${NC}"
}

# Supprimer tout
delete_all() {
    echo -e "${RED}⚠️  ATTENTION: Cela va supprimer TOUT le déploiement SETRAF!${NC}"
    read -p "Êtes-vous sûr? (oui/non): " confirm
    
    if [ "$confirm" == "oui" ]; then
        echo -e "${YELLOW}Suppression en cours...${NC}"
        kubectl delete namespace setraf
        echo -e "${GREEN}✓ Tout a été supprimé${NC}"
    else
        echo -e "${CYAN}Annulation${NC}"
    fi
}

# Point d'entrée principal
main() {
    if [ $# -eq 0 ]; then
        # Mode interactif
        interactive_menu
    else
        # Mode commande
        case "$1" in
            deploy)
                full_deployment
                ;;
            status)
                show_status
                get_access_url
                show_useful_commands
                ;;
            update)
                update_images
                ;;
            delete)
                delete_all
                ;;
            *)
                echo "Usage: $0 {deploy|status|update|delete}"
                echo "  ou lancez sans argument pour le menu interactif"
                exit 1
                ;;
        esac
    fi
}

# Lancement
main "$@"
