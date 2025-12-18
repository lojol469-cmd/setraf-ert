# 🚀 SETRAF - Guide de Déploiement Docker & Kubernetes

## 📋 Table des matières
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Déploiement Docker Compose](#déploiement-docker-compose)
- [Déploiement Kubernetes](#déploiement-kubernetes)
- [Build et Push Docker Hub](#build-et-push-docker-hub)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture

### Image Docker Optimisée
- **Taille**: ~800 MB (au lieu de 20 GB)
- **Modèles IA**: Téléchargés automatiquement au premier démarrage
- **Cache persistant**: Les modèles ne sont téléchargés qu'une seule fois
- **Temps de build**: 5-8 minutes
- **Temps de premier démarrage**: ~15 minutes (téléchargement modèles)
- **Redémarrages suivants**: <30 secondes

### Modèles téléchargés automatiquement
```
📦 sentence-transformers/all-MiniLM-L6-v2  (88 MB)
   └─ Embeddings pour RAG et base de connaissances

📦 openai/clip-vit-base-patch32  (600 MB)
   └─ Analyse d'images géophysiques

📦 mistralai/Mistral-7B-v0.1  (14 GB) - Optionnel
   └─ Génération de rapports géophysiques
```

---

## 📦 Prérequis

### Pour Docker Compose
```bash
# Vérifier Docker
docker --version  # >= 20.10
docker-compose --version  # >= 1.29

# Vérifier l'espace disque
df -h  # Au moins 20 GB libres
```

### Pour Kubernetes
```bash
# Vérifier kubectl
kubectl version --client

# Vérifier l'accès au cluster
kubectl cluster-info
kubectl get nodes
```

---

## 🐳 Déploiement Docker Compose

### Étape 1: Configuration
```bash
cd /home/belikan/KIbalione8/SETRAF

# Éditer .env si nécessaire
nano .env
```

Contenu minimal de `.env`:
```bash
HF_TOKEN=hf_YOUR_TOKEN_HERE
TAVILY_API_KEY=tvly-YOUR-KEY-HERE
```

### Étape 2: Démarrage
```bash
# Avec docker-compose (mode production)
docker-compose -f docker-compose.production.yml up -d

# Suivre les logs (inclut le téléchargement des modèles)
docker-compose -f docker-compose.production.yml logs -f
```

### Étape 3: Vérification
```bash
# Vérifier le statut
docker-compose -f docker-compose.production.yml ps

# Accéder à l'application
# Ouvrir: http://localhost:8504
```

### Commandes utiles
```bash
# Arrêter
docker-compose -f docker-compose.production.yml down

# Arrêter et supprimer les volumes (⚠️ perte du cache)
docker-compose -f docker-compose.production.yml down -v

# Redémarrer
docker-compose -f docker-compose.production.yml restart

# Voir les logs
docker-compose -f docker-compose.production.yml logs --tail=100 -f

# Entrer dans le container
docker exec -it setraf-production bash
```

---

## ☸️ Déploiement Kubernetes

### Étape 1: Configuration des secrets
```bash
# Encoder vos tokens en base64
echo -n "hf_YOUR_TOKEN" | base64
echo -n "tvly-YOUR-KEY" | base64

# Éditer le fichier secrets
nano kubernetes/configmap.yaml
# Remplacer les valeurs dans la section Secret
```

### Étape 2: Déploiement
```bash
# Méthode 1: Script automatique (RECOMMANDÉ)
cd kubernetes
./deploy-k8s.sh apply

# Méthode 2: Manuel
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/pvc.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

### Étape 3: Vérification
```bash
# État du déploiement
kubectl get all -n setraf

# État des volumes
kubectl get pvc -n setraf

# Logs de l'application
kubectl logs -n setraf -l app=setraf -f

# Détails du pod
kubectl describe pod -n setraf -l app=setraf
```

### Accès à l'application

#### LoadBalancer (AWS, GCP, Azure)
```bash
# Obtenir l'IP externe
kubectl get svc -n setraf setraf-service

# Accéder via:
# http://<EXTERNAL-IP>:8504
```

#### NodePort (Bare-metal)
```bash
# Éditer service.yaml et décommenter nodePort
kubectl apply -f kubernetes/service.yaml

# Accéder via:
# http://<NODE-IP>:30504
```

#### Ingress (avec nom de domaine)
```bash
# 1. Installer Nginx Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# 2. Configurer votre domaine dans service.yaml
nano kubernetes/service.yaml
# Remplacer: setraf.votredomaine.com

# 3. Appliquer
kubectl apply -f kubernetes/service.yaml

# Accéder via: https://setraf.votredomaine.com
```

### Commandes Kubernetes utiles
```bash
# Voir tous les statuts
./deploy-k8s.sh status

# Scaler le déploiement
kubectl scale deployment setraf -n setraf --replicas=2

# Mettre à jour l'image
kubectl set image deployment/setraf setraf=belikanm/setraf:v2.0.1 -n setraf

# Redémarrer le pod
kubectl rollout restart deployment/setraf -n setraf

# Accéder au shell du pod
kubectl exec -it -n setraf $(kubectl get pod -n setraf -l app=setraf -o jsonpath="{.items[0].metadata.name}") -- bash

# Supprimer le déploiement
./deploy-k8s.sh delete
```

---

## 📤 Build et Push Docker Hub

### Méthode automatique (RECOMMANDÉ)
```bash
cd /home/belikan/KIbalione8/SETRAF

# Build et push en une commande
./build_and_push.sh v2.0.0

# Ou pour latest
./build_and_push.sh latest
```

Le script va:
1. ✅ Vérifier les fichiers nécessaires
2. 🔐 Se connecter à Docker Hub (credentials: belikanm)
3. 🔨 Builder l'image (~8 minutes)
4. 🧪 Proposer un test local (optionnel)
5. 📤 Pusher vers Docker Hub
6. ✅ Afficher les commandes de déploiement

### Méthode manuelle
```bash
# 1. Login Docker Hub
echo "YOUR_DOCKER_PAT" | docker login -u belikanm --password-stdin

# 2. Build
docker build -t belikanm/setraf:v2.0.0 -f Dockerfile.optimized .

# 3. Tag latest
docker tag belikanm/setraf:v2.0.0 belikanm/setraf:latest

# 4. Push
docker push belikanm/setraf:v2.0.0
docker push belikanm/setraf:latest
```

### Vérifier l'image sur Docker Hub
```
https://hub.docker.com/r/belikanm/setraf
```

---

## ⚙️ Configuration

### Variables d'environnement importantes

| Variable | Description | Défaut | Optionnel |
|----------|-------------|--------|-----------|
| `HF_TOKEN` | Token HuggingFace | - | ❌ |
| `TAVILY_API_KEY` | Clé API Tavily | - | ✅ |
| `DOWNLOAD_MISTRAL` | Télécharger Mistral au démarrage | `false` | ✅ |
| `STREAMLIT_SERVER_PORT` | Port Streamlit | `8504` | ✅ |
| `TRANSFORMERS_CACHE` | Emplacement cache modèles | `/root/.cache/huggingface` | ✅ |

### Configuration du téléchargement Mistral

Par défaut, Mistral-7B (14 GB) n'est **PAS** téléchargé au démarrage pour accélérer le premier lancement.

Pour le télécharger au démarrage:
```bash
# Docker Compose
DOWNLOAD_MISTRAL=true docker-compose -f docker-compose.production.yml up -d

# Kubernetes
# Éditer kubernetes/configmap.yaml
# Changer: DOWNLOAD_MISTRAL: "true"
kubectl apply -f kubernetes/configmap.yaml
kubectl rollout restart deployment/setraf -n setraf
```

---

## 📊 Monitoring

### Logs Docker Compose
```bash
# Logs en temps réel
docker-compose -f docker-compose.production.yml logs -f

# Logs spécifiques
docker logs setraf-production -f --tail=100
```

### Logs Kubernetes
```bash
# Logs du pod
kubectl logs -n setraf -l app=setraf -f

# Logs depuis le début
kubectl logs -n setraf -l app=setraf --tail=-1

# Logs d'un container spécifique
kubectl logs -n setraf <pod-name> -c setraf
```

### Métriques

#### Taille du cache HuggingFace
```bash
# Docker
docker exec setraf-production du -sh /root/.cache/huggingface

# Kubernetes
kubectl exec -n setraf <pod-name> -- du -sh /root/.cache/huggingface
```

#### État des modèles téléchargés
```bash
# Docker
docker exec setraf-production find /root/.cache/huggingface -name "config.json" | wc -l

# Kubernetes
kubectl exec -n setraf <pod-name> -- find /root/.cache/huggingface -name "config.json" | wc -l
```

---

## 🔧 Troubleshooting

### Problème: Le téléchargement des modèles échoue

**Symptôme**: Erreur lors du démarrage, logs montrent des échecs de téléchargement

**Solutions**:
```bash
# 1. Vérifier la connexion Internet du container
docker exec setraf-production curl -I https://huggingface.co

# 2. Vérifier le token HuggingFace
docker exec setraf-production env | grep HF_TOKEN

# 3. Essayer manuellement
docker exec -it setraf-production python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('OK')
"
```

### Problème: Container redémarre en boucle

**Symptôme**: Pod/Container en `CrashLoopBackOff`

**Solutions**:
```bash
# Voir les logs d'erreur
kubectl logs -n setraf <pod-name> --previous

# Vérifier le healthcheck
kubectl describe pod -n setraf <pod-name>

# Augmenter le start_period
# Éditer deployment.yaml:
# initialDelaySeconds: 300  # 5 minutes au lieu de 3
```

### Problème: Manque d'espace disque

**Symptôme**: Erreur "no space left on device"

**Solutions**:
```bash
# Nettoyer les images Docker inutilisées
docker system prune -a

# Supprimer les volumes orphelins
docker volume prune

# Kubernetes: Augmenter la taille du PVC
kubectl edit pvc setraf-huggingface-cache -n setraf
# Changer: storage: 30Gi
```

### Problème: Application lente/timeout

**Symptômes**: Page ne charge pas, timeout après 30s

**Solutions**:
```bash
# 1. Augmenter les ressources
# Docker Compose: Éditer deploy.resources
# Kubernetes: Éditer deployment.yaml resources.limits

# 2. Vérifier la RAM disponible
docker stats setraf-production

# 3. Désactiver Mistral si pas nécessaire
# DOWNLOAD_MISTRAL=false
```

### Problème: Modèles retéléchargés à chaque redémarrage

**Symptôme**: Premier démarrage à chaque fois

**Solution**:
```bash
# Vérifier que le volume persiste
docker volume ls | grep huggingface

# Kubernetes: Vérifier le PVC
kubectl get pvc -n setraf

# Si le volume n'existe pas:
docker-compose -f docker-compose.production.yml down
docker volume create setraf_huggingface-cache
docker-compose -f docker-compose.production.yml up -d
```

---

## 📚 Ressources

- **Docker Hub**: https://hub.docker.com/r/belikanm/setraf
- **HuggingFace**: https://huggingface.co
- **Kubernetes Docs**: https://kubernetes.io/docs/

---

## 🆘 Support

Pour toute question ou problème:
- **Email**: nyundumathryme@gmail.com
- **GitHub Issues**: (à créer si nécessaire)

---

## 📝 Changelog

### v2.0.0 (2025-01-15)
- ✨ Image Docker optimisée (800 MB vs 20 GB)
- ✨ Téléchargement automatique des modèles IA
- ✨ Configuration Kubernetes complète
- ✨ Script de build et push automatisé
- 🐛 Correction des problèmes de cache persistant
