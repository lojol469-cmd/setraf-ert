# 🚀 KibaLock - Guide de Démarrage Rapide Docker

## ⚡ Installation en 3 minutes

### 1️⃣ Prérequis (une seule fois)

```bash
# Installer Docker + Docker Compose
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Installer NVIDIA Docker pour GPU (optionnel mais recommandé)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

### 2️⃣ Déploiement

```bash
cd /home/belikan/KIbalione8/SETRAF/kibalock-api

# Tout déployer automatiquement
./deploy-docker.sh deploy
```

### 3️⃣ Accès aux services

Après ~2-3 minutes (temps de build des images) :

- **Frontend** : http://localhost:3000
- **LifeModo API** : http://localhost:8000/docs
- **Backend KibaLock** : http://localhost:8505
- **TTS Service** : http://localhost:8001/health

---

## 📝 Commandes essentielles

```bash
# Démarrer tous les services
./deploy-docker.sh start

# Arrêter tous les services
./deploy-docker.sh stop

# Redémarrer un service spécifique
docker-compose restart lifemodo-api

# Voir les logs en temps réel
./deploy-docker.sh logs              # Tous les services
./deploy-docker.sh logs lifemodo-api # Service spécifique

# État des conteneurs
./deploy-docker.sh status

# Reconstruire les images (après modification code)
./deploy-docker.sh rebuild

# Nettoyage complet
./deploy-docker.sh clean
```

---

## 🔍 Vérification rapide

```bash
# Vérifier que tous les services sont OK
docker-compose ps

# Test API
curl http://localhost:8000/health
curl http://localhost:8001/health
```

---

**Documentation complète** : Voir `DOCKER_README.md`
