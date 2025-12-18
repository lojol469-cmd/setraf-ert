# 🚀 Déploiement Backend SETRAF sur Render

## 📦 Image Docker à déployer
**Image:** `belikanm/setraf-auth:latest`  
**Docker Hub:** https://hub.docker.com/r/belikanm/setraf-auth

## 🔧 Configuration Render

### 1. Créer un nouveau Web Service

#### Via Render Dashboard:
1. Aller sur https://render.com
2. Cliquer **"New +"** → **"Web Service"**
3. Choisir **"Deploy an existing image from a registry"**

#### Paramètres:
```
Image URL: belikanm/setraf-auth:latest
Name: setraf-auth-backend
Region: Oregon (US West) ou Frankfurt (Europe)
Branch: main (optionnel si via Docker Hub)
```

### 2. Configuration du Service

#### Instance Type:
- **Starter** (7$/mois) recommandé
- Free tier possible mais avec limitations

#### Port:
```
Port: 5000
```

#### Health Check:
```
Health Check Path: /api/health
```

### 3. Variables d'Environnement

Ajouter dans Render Dashboard → Environment:

```bash
# Node Environment
NODE_ENV=production
AUTH_PORT=5000

# MongoDB Atlas
MONGO_URI=mongodb+srv://SETRAF:Dieu19961991%3F%3F%21%3F%3F%21@cluster0.5tjz9v0.mongodb.net/myDatabase10?retryWrites=true&w=majority&appName=Cluster0
MONGO_USER=SETRAF
MONGO_PASSWORD=Dieu19961991??!??!
MONGO_CLUSTER=cluster0.5tjz9v0.mongodb.net
MONGO_DB_NAME=myDatabase10

# JWT Secrets
JWT_SECRET=Dieu19961991??!??!
JWT_REFRESH_SECRET=Dieu19961991??!??!_refresh

# Email Configuration
EMAIL_USER=nyundumathryme@gmail.com
EMAIL_PASS=zsrrymlixizhiybl

# API Keys
PUBLIC_KEY=qazghazz
PRIVATE_KEY=264419a2-cd4e-471a-81b3-04c522669052
```

### 4. Déploiement

#### Option A: Via Docker Hub (Recommandé)
```bash
# L'image est déjà sur Docker Hub
# Render la pullera automatiquement
Image: docker.io/belikanm/setraf-auth:latest
```

#### Option B: Via GitHub (avec Dockerfile)
1. Push le code sur GitHub:
```bash
cd /home/belikan/KIbalione8/SETRAF/node-auth
git add .
git commit -m "Add Render deployment config"
git push origin main
```

2. Connecter le repo GitHub à Render
3. Render utilisera le Dockerfile automatiquement

### 5. Configuration Réseau

#### Expose:
```
Internal: Non (service public)
Port: 5000
```

#### CORS:
Le backend accepte déjà toutes les origines en production.

### 6. Commandes Render CLI (Optionnel)

Installation:
```bash
# Install Render CLI
curl -s https://render.com/install | bash
```

Déploiement:
```bash
render login
render create web --name setraf-auth-backend \
  --image docker.io/belikanm/setraf-auth:latest \
  --port 5000 \
  --env NODE_ENV=production \
  --health-check-path /api/health
```

## 🔍 Vérification Post-Déploiement

### URLs générées par Render:
```
https://setraf-auth-backend.onrender.com
https://setraf-auth-backend.onrender.com/api/health
```

### Tests:
```bash
# Health check
curl https://setraf-auth-backend.onrender.com/api/health

# Info serveur
curl https://setraf-auth-backend.onrender.com/

# Test WebSocket
wscat -c wss://setraf-auth-backend.onrender.com
```

## 📊 Monitoring

### Logs:
- Dashboard Render → Service → Logs
- Temps réel visible dans l'interface

### Metrics:
- CPU usage
- Memory usage
- Request count
- Response times

## 🔄 Mise à jour

### Automatique (Docker Hub):
```bash
# 1. Build nouvelle version localement
cd /home/belikan/KIbalione8/SETRAF/node-auth
./docker-build.sh

# 2. Push vers Docker Hub
./docker-push.sh

# 3. Redéployer sur Render
# Via Dashboard: Manual Deploy → "Clear build cache & deploy"
# Ou via CLI:
render deploy --service setraf-auth-backend
```

### Manuel (GitHub):
```bash
git push origin main
# Render détectera automatiquement et redéploiera
```

## 🚨 Problèmes Courants

### 1. Connexion MongoDB
- Vérifier que l'IP de Render est autorisée dans MongoDB Atlas
- MongoDB Atlas → Network Access → Add IP Address → "Allow from anywhere" (0.0.0.0/0)

### 2. Variables d'environnement
- Vérifier l'encodage des caractères spéciaux
- MONGO_PASSWORD doit utiliser %3F pour ? et %21 pour !

### 3. Health check fail
- Vérifier que le port 5000 est bien exposé
- Le path /api/health doit retourner 200

### 4. WebSocket
- Render supporte WebSocket nativement
- Pas de configuration supplémentaire nécessaire

## 💰 Coûts

### Free Tier:
- 750 heures/mois
- ⚠️ Service s'arrête après 15min d'inactivité
- Cold start ~30s

### Starter ($7/mois):
- Toujours actif
- Pas de cold start
- 512 MB RAM
- Recommandé pour production

## 🔗 Liens Utiles

- Render Dashboard: https://dashboard.render.com
- Docs Render Docker: https://render.com/docs/deploy-an-image
- MongoDB Atlas: https://cloud.mongodb.com
- Docker Hub Image: https://hub.docker.com/r/belikanm/setraf-auth

## ✅ Checklist de Déploiement

- [ ] Image Docker pushée sur Docker Hub
- [ ] Compte Render créé
- [ ] Service créé sur Render
- [ ] Variables d'environnement configurées
- [ ] MongoDB Atlas IP whitelist configuré (0.0.0.0/0)
- [ ] Health check validé
- [ ] Test API endpoints
- [ ] Test WebSocket connection
- [ ] Frontend mis à jour avec nouvelle URL backend

## 🎯 Prochaine Étape

Une fois le backend déployé, mettre à jour le frontend:

```python
# Dans auth_module.py
BACKEND_URL = "https://setraf-auth-backend.onrender.com/api"
```

Puis déployer le frontend sur Render également.
