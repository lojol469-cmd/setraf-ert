# 🐳 SETRAF - Déploiement Docker Complet

## 📦 Images Docker Créées

### 1. **Frontend Python (Streamlit ERTest.py)**
- **Nom:** `belikanm/kibaertanalyste`
- **Tags:** `1.0.0`, `1.0.1` (en cours), `latest`
- **Base:** `python:3.10-slim`
- **Taille:** ~2 GB
- **Ports:** 8504 (Streamlit), 8505 (FastAPI optionnel)
- **Status:** ✅ v1.0.0 pushée, 🔄 v1.0.1 en build (avec PyGIMLi)

**Contenu:**
- ERTest.py (interface Streamlit principale)
- ERT.py (logique d'analyse ERT)
- api_setraf.py (API FastAPI optionnelle)
- auth_module.py (connexion au backend Node.js)
- requirements.txt (avec pygimli>=1.5.0)

**Dépendances spéciales:**
- PyGIMLi (nécessite cmake, libboost, libeigen3, liblapack)
- Streamlit, Plotly, Pandas, NumPy, SciPy, Matplotlib

### 2. **Backend Node.js (Authentication Server)**
- **Nom:** `belikanm/setraf-auth`
- **Tags:** `1.0.0`, `latest`
- **Base:** `node:18-alpine`
- **Taille:** 279 MB
- **Port:** 5000
- **Status:** ✅ Pushée avec succès
- **Digest:** `sha256:ec0386268adc9ef8700e8ce27f92c5c9962a470e7c7a8cca33293b5ff5c7f6ad`

**Contenu:**
- server.js (Express + WebSocket)
- Routes: auth.js, users.js
- Controllers & Middleware
- Mongoose (MongoDB)
- JWT authentication
- Socket.IO (WebSocket temps réel)

**Dépendances:**
- express, mongoose, bcryptjs, jsonwebtoken
- cors, helmet, express-rate-limit
- socket.io, nodemailer

## 🚀 Commandes de Déploiement

### Pull les images depuis Docker Hub

```bash
# Frontend Streamlit
docker pull belikanm/kibaertanalyste:latest

# Backend Node.js
docker pull belikanm/setraf-auth:latest
```

### Lancement rapide

#### 1. Backend d'authentification
```bash
docker run -d \
  --name setraf-backend \
  -p 5000:5000 \
  --env-file .env \
  belikanm/setraf-auth:latest
```

#### 2. Frontend Streamlit
```bash
docker run -d \
  --name setraf-frontend \
  -p 8504:8504 \
  --env-file .env \
  --link setraf-backend:backend \
  belikanm/kibaertanalyste:latest
```

### Avec Docker Compose (Stack complète)

```bash
# Utiliser docker-compose.full.yml
cd /home/belikan/KIbalione8/SETRAF
docker-compose -f docker-compose.full.yml up -d
```

**Services démarrés:**
- `setraf-backend` : Port 5000 (Auth API)
- `setraf-frontend` : Port 8504 (Streamlit)
- `setraf-api` : Port 8505 (FastAPI optionnel)

## 🔑 Variables d'Environnement Requises

Le fichier `.env` doit contenir:

```env
# Backend
PORT=5000
AUTH_PORT=5000

# MongoDB
MONGO_URI=mongodb+srv://...
MONGO_USER=SETRAF
MONGO_PASSWORD=...
MONGO_CLUSTER=...
MONGO_DB_NAME=myDatabase10

# JWT
JWT_SECRET=...
JWT_REFRESH_SECRET=...

# Email (Nodemailer)
EMAIL_USER=...
EMAIL_PASS=...

# API Keys
PUBLIC_KEY=...
PRIVATE_KEY=...
```

## 📊 Architecture du Stack

```
┌──────────────────────────────────────────────┐
│         SETRAF Docker Stack                  │
├──────────────────────────────────────────────┤
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │   Frontend (Port 8504)              │    │
│  │   belikanm/kibaertanalyste:latest   │    │
│  │   - Streamlit ERTest.py             │    │
│  │   - Interface utilisateur           │    │
│  │   - Visualisations ERT              │    │
│  └──────────────┬──────────────────────┘    │
│                 │ HTTP Requests              │
│                 ▼                            │
│  ┌─────────────────────────────────────┐    │
│  │   Backend Auth (Port 5000)          │    │
│  │   belikanm/setraf-auth:latest       │    │
│  │   - Express + Node.js               │    │
│  │   - JWT Authentication              │    │
│  │   - WebSocket (Socket.IO)           │    │
│  └──────────────┬──────────────────────┘    │
│                 │                            │
│                 ▼                            │
│  ┌─────────────────────────────────────┐    │
│  │   MongoDB Atlas (Cloud)             │    │
│  │   - Users collection                │    │
│  │   - Sessions & OTP                  │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  Optional:                                   │
│  ┌─────────────────────────────────────┐    │
│  │   FastAPI (Port 8505)               │    │
│  │   - API REST programmatique         │    │
│  │   - Analyse ERT par API             │    │
│  └─────────────────────────────────────┘    │
│                                              │
└──────────────────────────────────────────────┘
```

## 🔧 Build Local (Développement)

### Backend Node.js

```bash
cd /home/belikan/KIbalione8/SETRAF/node-auth

# Build
./docker-build.sh

# Push vers Docker Hub
./docker-push.sh

# Test local
docker run -p 5000:5000 --env-file ../.env belikanm/setraf-auth:latest
```

### Frontend Python

```bash
cd /home/belikan/KIbalione8/SETRAF

# Build
./docker-build.sh

# Push vers Docker Hub
./docker-push.sh

# Test local
docker run -p 8504:8504 --env-file .env belikanm/kibaertanalyste:latest
```

## 📝 Fichiers Docker Créés

### Backend (node-auth/)
- ✅ `Dockerfile` - Configuration image Node.js Alpine
- ✅ `.dockerignore` - Exclusions build
- ✅ `docker-build.sh` - Script de build automatisé
- ✅ `docker-push.sh` - Script de push Docker Hub

### Frontend (SETRAF/)
- ✅ `Dockerfile` - Configuration image Python 3.10
- ✅ `.dockerignore` - Exclusions build
- ✅ `docker-build.sh` - Script de build automatisé
- ✅ `docker-push.sh` - Script de push Docker Hub
- ✅ `docker-test.sh` - Script de test local

### Orchestration
- ✅ `docker-compose.setraf.yml` - Compose simple (frontend + API)
- ✅ `docker-compose.full.yml` - Stack complet (3 services)

## 🔍 Tests et Validation

### Test du Backend
```bash
# Health check
curl http://localhost:5000/api/health

# Info système
curl http://localhost:5000/
```

### Test du Frontend
```bash
# Health check
curl http://localhost:8504/_stcore/health

# Accès navigateur
xdg-open http://localhost:8504
```

### Logs
```bash
# Backend
docker logs setraf-backend -f

# Frontend
docker logs setraf-frontend -f
```

## 📈 Statut du Déploiement

| Component | Image | Tag | Status | Size | Docker Hub |
|-----------|-------|-----|--------|------|------------|
| Frontend | belikanm/kibaertanalyste | 1.0.0 | ✅ Pushed | 2 GB | ✅ Public |
| Frontend | belikanm/kibaertanalyste | 1.0.1 | 🔄 Building | ~2 GB | ⏳ Pending |
| Backend | belikanm/setraf-auth | 1.0.0 | ✅ Pushed | 279 MB | ✅ Public |
| Backend | belikanm/setraf-auth | latest | ✅ Pushed | 279 MB | ✅ Public |

## 🚨 Problèmes Résolus

### 1. PyGIMLi manquant (v1.0.0 → v1.0.1)
**Problème:** ModuleNotFoundError: No module named 'pygimli'  
**Solution:** 
- Ajout de pygimli>=1.5.0 dans requirements.txt
- Ajout des dépendances système (cmake, libboost, libeigen3, etc.)
- Build v1.0.1 en cours

### 2. npm ci échec (Backend)
**Problème:** package-lock.json absent  
**Solution:** Utilisation de `npm install --omit=dev` au lieu de `npm ci`

### 3. Docker non accessible (WSL2)
**Problème:** Docker not found dans WSL2  
**Solution:** Activation WSL Integration dans Docker Desktop + utilisation du path Windows

## 🔗 Liens Docker Hub

- **Frontend:** https://hub.docker.com/r/belikanm/kibaertanalyste
- **Backend:** https://hub.docker.com/r/belikanm/setraf-auth

## 📚 Documentation Supplémentaire

- `DOCKER_README.md` - Guide général Docker
- `DOCKER_DEPLOYMENT_GUIDE.txt` - Guide de déploiement détaillé
- `DOCKER_SETUP_GUIDE.txt` - Installation et configuration
- `README.md` - Documentation principale SETRAF

## 🎯 Prochaines Étapes

1. ⏳ Attendre completion du build v1.0.1 (avec PyGIMLi)
2. ✅ Tester v1.0.1 localement
3. ✅ Pousser v1.0.1 sur Docker Hub
4. ✅ Valider le stack complet avec docker-compose
5. 📝 Documenter les commandes de déploiement production

## 💡 Notes Importantes

- **v1.0.0 Frontend:** Fonctionnel mais **sans PyGIMLi** (ne peut pas faire d'analyse ERT complète)
- **v1.0.1 Frontend:** Build en cours avec PyGIMLi complet
- **Backend:** Totalement fonctionnel et opérationnel
- **MongoDB:** Utilise MongoDB Atlas (cloud) - pas de container local nécessaire
- **WebSocket:** Port 5000 gère à la fois HTTP et WebSocket

---

**Dernière mise à jour:** 14 novembre 2025  
**Auteur:** BelikanM  
**License:** Apache-2.0
