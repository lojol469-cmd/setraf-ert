# 🚀 Guide de Déploiement SETRAF - Option A (Téléchargement Automatique)

## 📋 Table des Matières
1. [Vue d'ensemble](#vue-densemble)
2. [Déploiement Local](#déploiement-local)
3. [Déploiement Cloud (AWS, Azure, GCP)](#déploiement-cloud)
4. [Déploiement sur Plateformes PaaS](#déploiement-paas)
5. [FAQ & Troubleshooting](#faq--troubleshooting)

---

## 🎯 Vue d'ensemble

### Principe de Fonctionnement

```
┌────────────────────────────────────────────────────────────┐
│ DÉPLOIEMENT                                                │
│ 1. Image Docker légère (800 MB) sans modèles IA          │
│ 2. Au premier démarrage: téléchargement automatique      │
│ 3. Modèles sauvegardés dans volume Docker persistant     │
│ 4. Redémarrages suivants: INSTANTANÉS (cache utilisé)    │
└────────────────────────────────────────────────────────────┘
```

### Avantages

✅ **Image légère**: 800 MB au lieu de 20 GB  
✅ **Build rapide**: 5-8 minutes au lieu de 45 minutes  
✅ **Push/Pull rapide**: 10-15 minutes au lieu de 2 heures  
✅ **Flexible**: Mise à jour des modèles facile  
✅ **Économique**: Moins de stockage et bande passante  

### Inconvénient

⚠️ **Premier démarrage**: +10-15 minutes pour télécharger les modèles  
📶 **Connexion Internet requise** au premier démarrage

---

## 💻 Déploiement Local

### Prérequis

- Docker 20.10+
- Docker Compose 2.0+
- 20 GB d'espace disque libre
- Connexion Internet (pour télécharger les modèles)

### Étape 1: Build de l'image

```bash
cd /home/belikan/KIbalione8/SETRAF

# Build de l'image optimisée
docker build -f Dockerfile.optimized -t setraf:optimized .
```

**Temps attendu**: 5-8 minutes  
**Taille de l'image**: ~800 MB

### Étape 2: Créer le dossier de cache

```bash
# Créer le dossier pour le cache des modèles
sudo mkdir -p /opt/setraf/huggingface-cache
sudo chown -R $USER:$USER /opt/setraf
```

### Étape 3: Lancement avec Docker Compose

```bash
# Démarrer le service
docker-compose -f docker-compose.production.yml up -d

# Voir les logs (téléchargement des modèles)
docker-compose -f docker-compose.production.yml logs -f setraf
```

**Première fois**: Les modèles seront téléchargés (10-15 min)  
**Logs attendus**:
```
🚀 SETRAF - Démarrage de l'application
📦 Vérification et téléchargement des modèles IA...
📦 Modèle: Embeddings (SentenceTransformer)
   ⬇️  Téléchargement en cours...
   ✅ Téléchargé et chargé avec succès!
...
✅ Tous les modèles requis sont prêts!
🎬 Lancement de l'application SETRAF...
```

### Étape 4: Accès à l'application

Ouvrez votre navigateur: **http://localhost:8504**

### Vérification du cache

```bash
# Vérifier que les modèles sont bien en cache
du -sh /opt/setraf/huggingface-cache
# Devrait afficher: ~15 GB

# Redémarrer pour tester le cache
docker-compose -f docker-compose.production.yml restart setraf

# Le redémarrage devrait être RAPIDE (30 secondes)
```

---

## ☁️ Déploiement Cloud

### Option 1: AWS (EC2 + ECR)

#### Étape 1: Créer un repository ECR

```bash
# Créer le repository
aws ecr create-repository --repository-name setraf-optimized

# Obtenir l'URL du registry
ECR_REGISTRY=$(aws ecr describe-repositories --repository-names setraf-optimized --query 'repositories[0].repositoryUri' --output text | cut -d'/' -f1)
```

#### Étape 2: Build et Push

```bash
# Se connecter à ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REGISTRY

# Build
docker build -f Dockerfile.optimized -t setraf:optimized .

# Tag
docker tag setraf:optimized $ECR_REGISTRY/setraf-optimized:latest

# Push (rapide car image légère)
docker push $ECR_REGISTRY/setraf-optimized:latest
```

**Temps de push**: 10-15 minutes (au lieu de 2 heures)

#### Étape 3: Déployer sur EC2

```bash
# Se connecter à l'instance EC2
ssh -i your-key.pem ec2-user@your-instance-ip

# Créer le dossier de cache
sudo mkdir -p /opt/setraf/huggingface-cache
sudo chown ec2-user:ec2-user /opt/setraf

# Pull l'image
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
docker pull $ECR_REGISTRY/setraf-optimized:latest

# Lancer le container
docker run -d \
  --name setraf \
  -p 8504:8504 \
  -v /opt/setraf/huggingface-cache:/root/.cache/huggingface \
  -v /opt/setraf/data:/app/data \
  --restart unless-stopped \
  $ECR_REGISTRY/setraf-optimized:latest
```

#### Étape 4: Accès

```
http://your-instance-ip:8504
```

---

### Option 2: Azure (Container Instances)

```bash
# Créer un groupe de ressources
az group create --name setraf-rg --location eastus

# Créer un Azure Container Registry
az acr create --resource-group setraf-rg --name setrafacr --sku Basic

# Push l'image
az acr login --name setrafacr
docker tag setraf:optimized setrafacr.azurecr.io/setraf:optimized
docker push setrafacr.azurecr.io/setraf:optimized

# Créer un Azure File Share pour le cache
az storage account create --name setrafstorage --resource-group setraf-rg --location eastus
az storage share create --name huggingface-cache --account-name setrafstorage

# Déployer le container avec volume persistant
az container create \
  --resource-group setraf-rg \
  --name setraf-app \
  --image setrafacr.azurecr.io/setraf:optimized \
  --cpu 4 --memory 8 \
  --ports 8504 \
  --azure-file-volume-account-name setrafstorage \
  --azure-file-volume-account-key <storage-key> \
  --azure-file-volume-share-name huggingface-cache \
  --azure-file-volume-mount-path /root/.cache/huggingface \
  --ip-address public
```

---

### Option 3: Google Cloud Platform (Cloud Run)

⚠️ **Limitation**: Cloud Run a un timeout de 60 minutes. Le téléchargement des modèles peut dépasser ce délai au premier démarrage.

**Solution**: Pré-télécharger les modèles dans un volume ou utiliser Compute Engine.

```bash
# Build et push vers GCR
gcloud builds submit --tag gcr.io/your-project/setraf:optimized

# Déployer sur Cloud Run (avec volume persistant)
gcloud run deploy setraf \
  --image gcr.io/your-project/setraf:optimized \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --port 8504 \
  --allow-unauthenticated \
  --execution-environment gen2 \
  --volume-name models \
  --volume-mount-path /root/.cache/huggingface
```

---

## 🌐 Déploiement PaaS

### Heroku

```bash
# Créer l'app
heroku create setraf-app

# Ajouter un volume (Heroku ne supporte pas les volumes persistants nativement)
# Les modèles seront re-téléchargés à chaque démarrage

# Push
heroku container:push web --app setraf-app
heroku container:release web --app setraf-app
```

⚠️ **Attention**: Heroku efface les fichiers à chaque redémarrage. Les modèles seront re-téléchargés à chaque fois.

### Render

1. Créer un nouveau **Web Service**
2. Connecter votre repository GitHub
3. Choisir **Docker** comme environnement
4. Configurer:
   - **Dockerfile Path**: `Dockerfile.optimized`
   - **Port**: 8504
   - **Persistent Disk**: Créer un disk de 20 GB monté sur `/root/.cache/huggingface`

### Railway

1. Créer un nouveau projet
2. Déployer depuis GitHub
3. Ajouter un **Volume** de 20 GB
4. Monter le volume sur `/root/.cache/huggingface`

---

## 📦 Pré-téléchargement des Modèles (Optionnel)

Pour accélérer le premier démarrage, vous pouvez pré-télécharger les modèles:

```bash
# Exécuter le script de pré-téléchargement
chmod +x download_models.sh
./download_models.sh /opt/setraf/huggingface-cache

# Vérifier
du -sh /opt/setraf/huggingface-cache
# ~15 GB
```

Ensuite, montez ce dossier dans Docker:

```yaml
volumes:
  - /opt/setraf/huggingface-cache:/root/.cache/huggingface
```

Le premier démarrage sera alors **INSTANTANÉ** (30 secondes).

---

## ❓ FAQ & Troubleshooting

### Q: Les modèles sont re-téléchargés à chaque redémarrage ?

**R**: Non, si le volume est correctement monté. Vérifiez:

```bash
docker volume inspect setraf_huggingface-cache
```

### Q: Le téléchargement échoue ?

**R**: Vérifiez la connexion Internet:

```bash
docker exec setraf ping -c 3 huggingface.co
```

### Q: Erreur "No space left on device" ?

**R**: Augmentez l'espace disque alloué au volume:

```bash
# Nettoyer les images inutilisées
docker system prune -a
```

### Q: Comment mettre à jour les modèles ?

**R**: Supprimez le cache et redémarrez:

```bash
# Supprimer le cache
docker volume rm setraf_huggingface-cache

# Redémarrer
docker-compose -f docker-compose.production.yml restart setraf

# Les modèles seront re-téléchargés (dernière version)
```

### Q: Comment vérifier que les modèles sont bien en cache ?

**R**:

```bash
# Entrer dans le container
docker exec -it setraf bash

# Lister les modèles
ls -lh /root/.cache/huggingface/hub/
```

### Q: L'application démarre mais crash ?

**R**: Vérifiez les logs:

```bash
docker-compose -f docker-compose.production.yml logs setraf

# Ou en temps réel
docker logs -f setraf
```

---

## 📊 Comparaison des Temps

| Étape | Avec modèles dans l'image | Téléchargement auto |
|-------|---------------------------|---------------------|
| **Build** | 45 minutes | 8 minutes |
| **Push** | 2 heures | 15 minutes |
| **Pull** | 1h30 | 10 minutes |
| **Premier start** | 30 secondes | 15 minutes |
| **Restarts** | 30 secondes | 30 secondes |
| **Taille image** | 20 GB | 800 MB |

---

## ✅ Checklist de Déploiement

- [ ] Docker et Docker Compose installés
- [ ] 20 GB d'espace disque libre
- [ ] Connexion Internet stable
- [ ] Volume persistant créé (`/opt/setraf/huggingface-cache`)
- [ ] Fichiers copiés: `Dockerfile.optimized`, `startup.sh`, `docker-compose.production.yml`
- [ ] `startup.sh` exécutable (`chmod +x startup.sh`)
- [ ] Build de l'image réussi
- [ ] Premier démarrage: modèles téléchargés
- [ ] Application accessible sur `http://localhost:8504`
- [ ] Redémarrage: rapide (cache utilisé)

---

## 🔗 Ressources

- [HuggingFace Hub](https://huggingface.co)
- [Docker Volumes](https://docs.docker.com/storage/volumes/)
- [SentenceTransformers](https://www.sbert.net/)

---

**Auteur**: Belikan M.  
**Date**: Décembre 2025  
**Version**: 1.0.0
