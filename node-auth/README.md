# 🔒 SETRAF Authentication Backend

Backend d'authentification Node.js pour SETRAF (Subaquifère ERT Analysis Tool).

[![Docker Image](https://img.shields.io/badge/Docker-belikanm%2Fsetraf--auth-blue?logo=docker)](https://hub.docker.com/r/belikanm/setraf-auth)
[![Node.js](https://img.shields.io/badge/Node.js-18.x-green?logo=node.js)](https://nodejs.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb)](https://www.mongodb.com/cloud/atlas)

## 🚀 Fonctionnalités

- ✅ **Authentification JWT** (Access & Refresh tokens)
- ✅ **Système OTP** par email
- ✅ **WebSocket** temps réel (Socket.IO)
- ✅ **MongoDB Atlas** intégration
- ✅ **Rate Limiting** et sécurité (Helmet)
- ✅ **Auto-détection IP** réseau
- ✅ **CORS** configuré pour production
- ✅ **Email** avec Nodemailer

## 📦 Stack Technique

- **Runtime:** Node.js 18 (Alpine Linux)
- **Framework:** Express.js
- **Database:** MongoDB Atlas
- **Auth:** JWT + bcryptjs
- **WebSocket:** Socket.IO
- **Email:** Nodemailer (Gmail)
- **Security:** Helmet, express-rate-limit
- **Validation:** express-validator

## 🐳 Docker

### Image Docker Hub

```bash
docker pull belikanm/setraf-auth:latest
```

### Lancement rapide

```bash
docker run -d \
  --name setraf-backend \
  -p 5000:5000 \
  --env-file .env \
  belikanm/setraf-auth:latest
```

### Build local

```bash
# Build
./docker-build.sh

# Push vers Docker Hub
./docker-push.sh
```

## 🌐 Déploiement sur Render

### Méthode 1: Via Docker Hub (Recommandé)

1. Créer un compte sur [Render](https://dashboard.render.com)
2. **New +** → **Web Service** → **Deploy an existing image**
3. Image URL: `docker.io/belikanm/setraf-auth:latest`
4. Configurer:
   - **Name:** setraf-auth-backend
   - **Region:** Oregon ou Frankfurt
   - **Port:** 5000
   - **Health Check Path:** `/api/health`
5. Ajouter les variables d'environnement (voir `.env.example`)
6. Déployer !

### Méthode 2: Via GitHub

1. Connecter ce repository à Render
2. Render utilisera automatiquement le Dockerfile
3. Configurer les variables d'environnement
4. Déployer

### Aide au déploiement

```bash
./deploy-to-render.sh
```

Ce script génère:
- `render-env-variables.txt` - Variables d'environnement prêtes à copier
- Instructions étape par étape

**Documentation complète:** [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md)

## 🔧 Installation locale

### Prérequis

- Node.js 18+
- MongoDB Atlas account
- Gmail account pour emails (ou autre SMTP)

### Installation

```bash
# Cloner le repository
git clone https://github.com/lojol469-cmd/setraf-auth.git
cd setraf-auth

# Installer les dépendances
npm install

# Configurer .env (voir .env.example)
cp .env.example .env
nano .env

# Démarrer le serveur
npm start

# Ou en mode développement
npm run dev
```

## 🔑 Variables d'Environnement

Créer un fichier `.env`:

```env
# Server
NODE_ENV=production
AUTH_PORT=5000

# MongoDB Atlas
MONGO_URI=mongodb+srv://...
MONGO_USER=...
MONGO_PASSWORD=...
MONGO_CLUSTER=...
MONGO_DB_NAME=...

# JWT
JWT_SECRET=your-secret-here
JWT_REFRESH_SECRET=your-refresh-secret-here

# Email (Nodemailer)
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-app-password

# API Keys
PUBLIC_KEY=...
PRIVATE_KEY=...
```

### Configuration MongoDB Atlas

**Important:** Autoriser l'accès depuis n'importe quelle IP pour Render:

1. MongoDB Atlas → **Network Access**
2. **Add IP Address** → **Allow from anywhere** (`0.0.0.0/0`)
3. Confirmer

## 📡 API Endpoints

### Health Check
```bash
GET /api/health
```

### Authentication
```bash
POST /api/auth/register    # Inscription
POST /api/auth/login       # Connexion
POST /api/auth/refresh     # Refresh token
POST /api/auth/logout      # Déconnexion
POST /api/auth/verify-otp  # Vérifier OTP
```

### Users
```bash
GET  /api/users/profile    # Profil utilisateur (auth required)
PUT  /api/users/profile    # Mettre à jour profil
```

### Documentation
```bash
GET /                      # Info serveur
GET /api/docs              # Documentation API
```

## 🧪 Tests

```bash
# Health check
curl http://localhost:5000/api/health

# Info serveur
curl http://localhost:5000/

# Test WebSocket
wscat -c ws://localhost:5000
```

## 📊 Structure du Projet

```
node-auth/
├── config/
│   ├── database.js          # Configuration MongoDB
│   └── networkUtils.js      # Détection IP
├── controllers/
│   ├── authController.js    # Logique auth
│   └── userController.js    # Logique users
├── middleware/
│   └── auth.js              # Middleware JWT
├── models/
│   └── User.js              # Modèle utilisateur
├── routes/
│   ├── auth.js              # Routes auth
│   └── users.js             # Routes users
├── server.js                # Point d'entrée
├── package.json             # Dépendances
├── Dockerfile               # Image Docker
├── render.yaml              # Config Render
└── RENDER_DEPLOYMENT.md     # Guide déploiement
```

## 🔒 Sécurité

- ✅ Mots de passe hashés avec bcryptjs
- ✅ JWT avec expiration (1h access, 7j refresh)
- ✅ Rate limiting (100 req/15min)
- ✅ Helmet.js pour headers HTTP sécurisés
- ✅ CORS configuré
- ✅ Validation des entrées avec express-validator
- ✅ Secrets en variables d'environnement

## 📈 Performance

- **Image Docker:** 279 MB (Alpine Linux)
- **Cold start:** ~2-3 secondes
- **Mémoire:** ~100-150 MB
- **CPU:** Minimal (Node.js efficace)

## 🔗 Liens Utiles

- **Docker Hub:** https://hub.docker.com/r/belikanm/setraf-auth
- **GitHub:** https://github.com/lojol469-cmd/setraf-auth
- **MongoDB Atlas:** https://cloud.mongodb.com
- **Render:** https://dashboard.render.com

## 🤝 Contribution

Les contributions sont les bienvenues !

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

Apache License 2.0 - voir [LICENSE](LICENSE)

## 👨‍💻 Auteur

**BelikanM**

- GitHub: [@BelikanM](https://github.com/BelikanM)
- Email: nyundumathryme@gmail.com

## 🙏 Remerciements

- SETRAF ERT Analysis Tool
- MongoDB Atlas
- Render.com
- Docker Hub

---

**⭐ Si ce projet vous aide, n'oubliez pas de lui donner une étoile !**
