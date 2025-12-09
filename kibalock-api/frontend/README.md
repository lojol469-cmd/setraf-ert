# KibaLock Frontend - Guide de lancement

## 🚀 Démarrage rapide

### Option 1: Lanceur universel (RECOMMANDÉ)
Lance tous les services automatiquement (API + Backend + Frontend):
```bash
cd /home/belikan/KIbalione8/SETRAF/kibalock-api
./start_all.sh
```

### Option 2: Lancement manuel du frontend seul
```bash
cd frontend
npm install
npm run dev
```

## 📋 Prérequis

- Node.js >= 18.0.0
- npm >= 9.0.0
- Python 3.10+ (pour les APIs backend)

## 🔧 Configuration automatique

Le système détecte automatiquement:
- ✅ **IP locale** (WSL, Docker, Linux, macOS)
- ✅ **Ports disponibles** (évite les conflits)
- ✅ **URLs des APIs** (LifeModo + Backend KibaLock)

Configuration générée dans `.env`:
```env
VITE_API_URL=http://192.168.1.X:8000
VITE_BACKEND_URL=http://192.168.1.X:8505
VITE_WS_URL=ws://192.168.1.X:8000
```

## 📦 Structure du projet

```
frontend/
├── src/
│   ├── components/      # Composants réutilisables
│   │   ├── Scene3D.jsx          # Scène 3D Three.js
│   │   ├── VoiceRecorder.jsx    # Enregistrement vocal
│   │   ├── WebcamCapture.jsx    # Capture webcam
│   │   ├── ErrorBoundary.jsx    # Gestion erreurs
│   │   └── LoadingScreen.jsx    # Écran chargement
│   ├── pages/          # Pages principales
│   │   ├── Register.jsx         # Inscription biométrique
│   │   ├── Login.jsx            # Connexion biométrique
│   │   ├── Dashboard.jsx        # Tableau de bord
│   │   ├── Chat.jsx             # Chat IA
│   │   └── Training.jsx         # Entraînement temps réel
│   ├── store/          # État global (Zustand)
│   │   └── authStore.js         # Authentification
│   ├── App.jsx         # Composant racine
│   ├── main.jsx        # Point d'entrée
│   └── index.css       # Styles globaux
├── public/             # Assets statiques
├── .vscode/            # Configuration VSCode
├── vite.config.js      # Configuration Vite avec auto-discovery
├── tailwind.config.js  # Configuration Tailwind CSS
├── package.json        # Dépendances
└── index.html          # Template HTML
```

## 🎨 Technologies utilisées

### Core
- **React 18** - Framework UI
- **Vite** - Build tool ultra-rapide
- **React Router** - Navigation

### 3D & Animations
- **Three.js** - Rendu 3D WebGL
- **@react-three/fiber** - React renderer pour Three.js
- **@react-three/drei** - Helpers Three.js
- **Framer Motion** - Animations fluides

### Biométrie
- **RecordRTC** - Enregistrement audio/vidéo
- **face-api.js** - Détection faciale temps réel
- **WaveSurfer.js** - Visualisation audio

### État & API
- **Zustand** - État global léger
- **Axios** - Requêtes HTTP
- **Socket.io** - WebSocket temps réel

### Styling
- **Tailwind CSS** - Utility-first CSS
- **Lucide React** - Icônes

## 🔍 Auto-discovery expliqué

### 1. Détection de l'IP (vite.config.js)
```javascript
import os from 'os'

function getLocalIP() {
  const interfaces = os.networkInterfaces()
  // Recherche de l'interface réseau principale
  // Retourne l'IP locale (ex: 192.168.1.100)
}
```

### 2. Configuration dynamique des proxies
```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://IP_AUTO:8000',
      changeOrigin: true,
    }
  }
}
```

### 3. Fallback intelligent (authStore.js)
```javascript
const getApiUrl = () => {
  // 1. Depuis .env (priorité)
  if (import.meta.env.VITE_API_URL) return import.meta.env.VITE_API_URL
  
  // 2. Auto-détecté par Vite
  if (import.meta.env.VITE_API_URL_AUTO) return import.meta.env.VITE_API_URL_AUTO
  
  // 3. Depuis l'URL courante
  const hostname = window.location.hostname
  if (hostname !== 'localhost') return `http://${hostname}:8000`
  
  // 4. Fallback localhost
  return 'http://localhost:8000'
}
```

## 🛠️ Commandes disponibles

| Commande | Description |
|----------|-------------|
| `npm install` | Installer les dépendances |
| `npm run dev` | Lancer le serveur de développement |
| `npm run build` | Build de production |
| `npm run preview` | Prévisualiser le build |
| `npm run lint` | Vérifier le code avec ESLint |

## 🐛 Résolution des problèmes

### ❌ Erreur "Unknown at rule @tailwind"
**Cause**: VSCode ne reconnaît pas les directives Tailwind  
**Solution**: Configuration `.vscode/settings.json` déjà créée, recharger VSCode

### ❌ Erreur "Cannot find module 'os'"
**Cause**: Import Node.js dans vite.config.js  
**Solution**: Déjà géré, Vite supporte les modules Node.js

### ❌ Erreur "CORS" lors des requêtes API
**Cause**: API backend non accessible  
**Solution**: 
1. Vérifier que l'API est lancée (`./start_all.sh`)
2. Vérifier le port dans `.env`
3. Vérifier le firewall

### ❌ Erreur "EADDRINUSE" (port déjà utilisé)
**Cause**: Port 3000 déjà occupé  
**Solution**: Le script `start_all.sh` trouve automatiquement un port libre

## 🌐 URLs par défaut

| Service | Port | URL |
|---------|------|-----|
| Frontend React | 3000 | http://localhost:3000 |
| Backend KibaLock | 8505 | http://localhost:8505 |
| LifeModo API | 8000 | http://localhost:8000 |
| API Docs (Swagger) | 8000 | http://localhost:8000/docs |

## 📱 Fonctionnalités

### ✅ Implémenté
- ✅ Architecture React 3D avec Three.js
- ✅ Auto-discovery IP/Plateforme
- ✅ Enregistrement vocal multi-échantillons
- ✅ Capture webcam avec détection faciale
- ✅ Routing avec React Router
- ✅ État global avec Zustand
- ✅ Animations Framer Motion
- ✅ Design glassmorphism

### 🚧 En cours
- 🚧 Page Login complète
- 🚧 Chat IA avec Phi-3.5
- 🚧 Entraînement temps réel
- 🚧 Dashboard utilisateur
- 🚧 Intégration WebSocket

## 🔐 Workflow d'authentification

1. **Inscription** (`/register`):
   - Saisie username + email
   - Enregistrement 3 échantillons vocaux
   - Capture 3-5 photos faciales
   - Envoi à LifeModo API pour training
   - Création compte + embeddings FAISS

2. **Connexion** (`/login`):
   - Enregistrement vocal
   - Capture photo faciale
   - Vérification via FAISS (ultra-rapide)
   - Session JWT créée

3. **Dashboard** (`/dashboard`):
   - Accès chat IA
   - Entraînement continu
   - Gestion profil

## 📊 Performance

- **Build time**: ~5s
- **HMR (Hot Module Replacement)**: < 50ms
- **First load**: ~2s
- **Subsequent loads**: < 500ms

## 🎯 Prochaines étapes

1. [ ] Finaliser pages Login/Chat/Training
2. [ ] Implémenter WebSocket pour temps réel
3. [ ] Ajouter clonage vocal avec TTS
4. [ ] Dashboard avec statistiques
5. [ ] Tests E2E avec Playwright

## 📞 Support

Pour toute question ou problème:
- Logs: `logs/frontend_*.log`
- Vérifier l'API: `curl http://localhost:8000/health`
- Recharger VSCode si warnings Tailwind persistent

---

**Note**: Les erreurs `@tailwind` dans VSCode sont des faux positifs - le code compile correctement. La configuration `.vscode/settings.json` les ignore automatiquement.
