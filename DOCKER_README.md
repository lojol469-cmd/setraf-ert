# 🐋 SETRAF - Guide Docker

## 📦 Image Docker

**Repository Docker Hub:** `belikanm/kibaertanalyste`

### 🏷️ Tags disponibles

- `latest` - Dernière version stable
- `1.0.0` - Version 1.0.0

## 🚀 Démarrage rapide

### Pull et Run en une commande

```bash
docker run -d \
  --name setraf \
  -p 8504:8504 \
  belikanm/kibaertanalyste:latest
```

Accès: **http://localhost:8504**

### Avec volumes persistants

```bash
docker run -d \
  --name setraf \
  -p 8504:8504 \
  -v ./setraf-logs:/app/logs \
  -v ./setraf-data:/app/data \
  -v ./setraf-uploads:/app/uploads \
  belikanm/kibaertanalyste:latest
```

## 🛠️ Build et Déploiement

### 1. Build l'image localement

```bash
cd /home/belikan/KIbalione8/SETRAF
./docker-build.sh
```

Cette commande:
- Build l'image Docker
- Crée les tags `1.0.0` et `latest`
- Affiche la taille de l'image

### 2. Test en local

```bash
./docker-test.sh
```

Cette commande:
- Lance un container de test
- Vérifie que l'application démarre correctement
- Affiche les logs en temps réel
- Ouvre http://localhost:8504

### 3. Push vers Docker Hub

```bash
./docker-push.sh
```

Cette commande:
- Authentifie sur Docker Hub (si nécessaire)
- Push les versions `1.0.0` et `latest`
- Affiche les instructions d'utilisation

## 📋 Architecture de l'image

### Image de base
- `python:3.10-slim` (Debian)
- Taille optimisée avec multi-stage build

### Dépendances installées
- **Scientifiques:** NumPy, Pandas, SciPy, Scikit-learn
- **Visualisation:** Matplotlib, Plotly, Seaborn
- **Interface:** Streamlit 1.28+
- **PyGIMLi:** Pour analyses ERT avancées
- **FastAPI:** Pour l'API REST

### Ports exposés

| Port | Service | Description |
|------|---------|-------------|
| 8504 | ERTest.py | Application Streamlit principale |
| 8505 | api_setraf.py | API REST FastAPI |
| 8506 | ERT.py | Version Kibali complète |

### Volumes recommandés

| Volume | Description |
|--------|-------------|
| `/app/logs` | Logs de l'application |
| `/app/data` | Données ERT (.dat) |
| `/app/uploads` | Fichiers uploadés |
| `/app/exports` | Rapports PDF générés |

## 🔧 Commandes Docker

### Gestion du container

```bash
# Démarrer
docker start setraf

# Arrêter
docker stop setraf

# Redémarrer
docker restart setraf

# Supprimer
docker rm setraf

# Voir les logs
docker logs setraf

# Logs en temps réel
docker logs -f setraf

# Statistiques
docker stats setraf
```

### Inspection

```bash
# Informations du container
docker inspect setraf

# Processus en cours
docker top setraf

# Entrer dans le container
docker exec -it setraf bash
```

## 🐙 Docker Compose

### Fichier docker-compose.setraf.yml

```yaml
version: '3.8'

services:
  setraf-app:
    image: belikanm/kibaertanalyste:latest
    container_name: setraf-ert-analyzer
    ports:
      - "8504:8504"
      - "8505:8505"
      - "8506:8506"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./uploads:/app/uploads
      - ./exports:/app/exports
    restart: unless-stopped
```

### Démarrage avec Compose

```bash
# Démarrer
docker-compose -f docker-compose.setraf.yml up -d

# Arrêter
docker-compose -f docker-compose.setraf.yml down

# Voir les logs
docker-compose -f docker-compose.setraf.yml logs -f
```

## 🌍 Déploiement en production

### Sur un serveur distant

```bash
# 1. Sur le serveur, pull l'image
docker pull belikanm/kibaertanalyste:latest

# 2. Créer les dossiers
mkdir -p setraf/{logs,data,uploads,exports}

# 3. Lancer le container
docker run -d \
  --name setraf-prod \
  -p 8504:8504 \
  -v $(pwd)/setraf/logs:/app/logs \
  -v $(pwd)/setraf/data:/app/data \
  -v $(pwd)/setraf/uploads:/app/uploads \
  -v $(pwd)/setraf/exports:/app/exports \
  --restart unless-stopped \
  belikanm/kibaertanalyste:latest

# 4. Vérifier
curl http://localhost:8504/_stcore/health
```

### Avec Nginx reverse proxy

```nginx
server {
    listen 80;
    server_name setraf.example.com;

    location / {
        proxy_pass http://localhost:8504;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔐 Variables d'environnement

```bash
docker run -d \
  --name setraf \
  -p 8504:8504 \
  -e STREAMLIT_SERVER_PORT=8504 \
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  -e API_PORT=8505 \
  belikanm/kibaertanalyste:latest
```

## 📊 Monitoring

### Health check

```bash
# Vérifier la santé du container
docker inspect --format='{{.State.Health.Status}}' setraf

# Endpoint de santé
curl http://localhost:8504/_stcore/health
```

### Métriques

```bash
# Utilisation CPU/RAM
docker stats setraf --no-stream

# Logs avec timestamps
docker logs --timestamps setraf
```

## 🐛 Dépannage

### Container ne démarre pas

```bash
# Voir les logs d'erreur
docker logs setraf

# Vérifier la configuration
docker inspect setraf | grep -A 10 Config
```

### Port déjà utilisé

```bash
# Trouver le processus sur le port 8504
lsof -i :8504

# Utiliser un autre port
docker run -d -p 9000:8504 belikanm/kibaertanalyste:latest
```

### Problème de permissions

```bash
# Exécuter avec l'utilisateur courant
docker run -d \
  --user $(id -u):$(id -g) \
  -p 8504:8504 \
  belikanm/kibaertanalyste:latest
```

## 📝 Changelog

### Version 1.0.0 (14 Nov 2025)
- ✅ Image Docker initiale
- ✅ Support Streamlit ERTest.py
- ✅ API FastAPI intégrée
- ✅ Health checks configurés
- ✅ Volumes persistants
- ✅ Multi-port support (8504, 8505, 8506)

## 📞 Support

- **Auteur:** Belikan M.
- **Email:** nyundumathryme@gmail.com
- **Repository:** github.com/BelikanM/KIbalione8
- **Docker Hub:** hub.docker.com/r/belikanm/kibaertanalyste

## 📄 Licence

Copyright © 2025 Belikan M. - Tous droits réservés.
