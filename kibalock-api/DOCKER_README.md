# 🔐 KibaLock - Architecture Docker Multi-Conteneurs

## 📋 Vue d'ensemble

KibaLock utilise une **architecture microservices avec Docker** pour isoler les dépendances conflictuelles dans des conteneurs séparés, tout en permettant leur communication via un réseau Docker privé.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Nginx (Port 80)                        │
│              Reverse Proxy & Load Balancer                  │
└────────────┬────────────┬────────────┬──────────────────────┘
             │            │            │
   ┌─────────▼──────┐ ┌──▼─────────┐ ┌▼──────────────┐
   │  Frontend      │ │ LifeModo   │ │ Backend       │
   │  React 3D      │ │ API        │ │ KibaLock      │
   │  (Node 20)     │ │ FastAPI    │ │ Streamlit     │
   │                │ │            │ │               │
   │  Port: 3000    │ │ Port: 8000 │ │ Port: 8505    │
   └────────────────┘ └────┬───────┘ └───┬───────────┘
                           │             │
                      ┌────▼─────────────▼────┐
                      │                       │
                  ┌───▼──────┐         ┌─────▼──────┐
                  │ TTS Svc  │         │  MongoDB   │
                  │ (Coqui)  │         │            │
                  │          │         │  Port:     │
                  │ Port:    │         │  27017     │
                  │ 8001     │         └────────────┘
                  └──────────┘
                      │
              ┌───────┴────────┐
              │  GPU NVIDIA    │
              │  CUDA 13.0     │
              │  (Partagé)     │
              └────────────────┘
```

## 🎯 Services

### 1. **LifeModo API** (Port 8000)
- **Rôle**: Formation et entraînement des modèles IA
- **Stack**: FastAPI + Transformers + LangChain
- **NumPy**: 1.23.5 (compatible transformers)
- **PyTorch**: 2.10.0 CUDA 13.0
- **GPU**: Oui (accélération modèles)

### 2. **Backend KibaLock** (Port 8505)
- **Rôle**: Authentification biométrique + FAISS
- **Stack**: Streamlit + DeepFace + FAISS + OpenCV
- **NumPy**: 2.2.6 (requis par FAISS)
- **PyTorch**: 2.10.0 CUDA 13.0
- **GPU**: Oui (reconnaissance faciale)

### 3. **TTS Service** (Port 8001)
- **Rôle**: Synthèse vocale isolée
- **Stack**: FastAPI + Coqui TTS
- **NumPy**: 1.22.0 (strict - requis par TTS)
- **PyTorch**: 2.10.0 CUDA 13.0
- **GPU**: Oui (génération voix)

### 4. **Frontend** (Port 3000)
- **Rôle**: Interface utilisateur 3D
- **Stack**: React + Three.js + Vite
- **Build**: Multi-stage (Node builder + Nginx prod)

### 5. **MongoDB** (Port 27017)
- **Rôle**: Base de données biométrie + sessions
- **Image**: mongo:7.0
- **Persistence**: Volume Docker

### 6. **Nginx** (Port 80)
- **Rôle**: Reverse proxy + SSL termination
- **Routes**: 
  - `/` → Frontend
  - `/api/lifemodo/` → LifeModo API
  - `/backend/` → KibaLock Backend
  - `/api/tts/` → TTS Service

## 🚀 Déploiement

### Prérequis

```bash
# 1. Docker + Docker Compose
sudo apt install docker.io docker-compose

# 2. NVIDIA Docker (pour GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# 3. Vérification GPU
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

### Installation Complète

```bash
cd /home/belikan/KIbalione8/SETRAF/kibalock-api

# Déploiement automatique (tout-en-un)
./deploy-docker.sh deploy

# OU étape par étape
./deploy-docker.sh start    # Démarrer
./deploy-docker.sh stop     # Arrêter
./deploy-docker.sh restart  # Redémarrer
./deploy-docker.sh rebuild  # Reconstruire images
./deploy-docker.sh logs     # Voir logs en temps réel
./deploy-docker.sh status   # État des services
./deploy-docker.sh clean    # Nettoyage complet
```

### Variables d'environnement (.env)

Le fichier `.env` est généré automatiquement. Vous pouvez le personnaliser :

```bash
# Network
LOCAL_IP=172.20.31.35
API_PORT=8000
BACKEND_PORT=8505
TTS_PORT=8001
FRONTEND_PORT=3000

# MongoDB
MONGO_URI=mongodb://kibalock-mongo:27017/
MONGO_DB=kibalock

# GPU
CUDA_VISIBLE_DEVICES=0

# Security (CHANGEZ EN PRODUCTION!)
JWT_SECRET=your-secret-key-here
MONGO_ROOT_PASSWORD=your-password-here
```

## 📊 Monitoring

### Logs en temps réel

```bash
# Tous les services
docker-compose logs -f

# Service spécifique
docker-compose logs -f lifemodo-api
docker-compose logs -f kibalock-backend
docker-compose logs -f tts-service
```

### État des conteneurs

```bash
docker-compose ps

# Détails complets
docker stats
```

### Health Checks

Chaque service expose un endpoint `/health` :

```bash
curl http://localhost:8000/health  # LifeModo API
curl http://localhost:8001/health  # TTS Service
curl http://localhost:8505/_stcore/health  # Backend
```

## 🔧 Résolution de problèmes

### GPU non détecté

```bash
# Vérifier NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi

# Si échec, réinstaller NVIDIA Container Toolkit
sudo apt purge nvidia-docker2
sudo apt install nvidia-docker2
sudo systemctl restart docker
```

### Conteneur ne démarre pas

```bash
# Voir les logs d'erreur
docker-compose logs [service-name]

# Rebuild sans cache
docker-compose build --no-cache [service-name]
docker-compose up -d [service-name]
```

### Conflits de ports

```bash
# Changer les ports dans .env
nano .env

# Ou utiliser des ports alternatifs
API_PORT=8100
BACKEND_PORT=8505
```

### Espace disque insuffisant

```bash
# Nettoyer images inutilisées
docker system prune -a

# Nettoyer volumes
docker volume prune
```

## 📦 Volumes de données

```
models/
├── huggingface/       # Modèles Transformers
├── transformers/      # Cache transformers
├── tts/              # Modèles TTS
└── faiss/            # Index FAISS

data/
└── faiss_indices/    # Indices biométriques

logs/
├── lifemodo/         # Logs API
├── backend/          # Logs Backend
└── tts/              # Logs TTS
```

## 🔐 Sécurité

### Production

1. **Changer les secrets** dans `.env`
2. **Activer SSL** avec certificats
3. **Limiter les ressources** par conteneur
4. **Activer l'authentification** MongoDB
5. **Utiliser des secrets** Docker

```bash
# Générer certificats SSL
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/kibalock.key -out ssl/kibalock.crt
```

## 🎯 Performance

### Limites de ressources

Modifier `docker-compose.yml` :

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

### Scaling

```bash
# Augmenter le nombre de workers API
docker-compose up -d --scale lifemodo-api=3
```

## 📚 API Endpoints

### LifeModo API (8000)
- `GET /health` - Health check
- `GET /docs` - Documentation interactive
- `POST /train` - Entraîner modèle
- `POST /predict` - Prédiction

### Backend KibaLock (8505)
- Streamlit UI - Interface web complète

### TTS Service (8001)
- `GET /health` - Health check
- `POST /synthesize` - Générer audio

## 🤝 Support

Pour toute question ou problème :
1. Consulter les logs : `docker-compose logs -f`
2. Vérifier le health check : `curl localhost:8000/health`
3. Redémarrer le service : `docker-compose restart [service]`

## 📝 Licence

Voir fichiers LICENSE-* à la racine du projet.

---

**🚀 KibaLock - Authentification Biométrique Nouvelle Génération**
