# 🌊 SETRAF - Intégration ERTest.py ↔ ERT.py

## ✅ Configuration Actuelle

### 📊 Applications Déployées

1. **ERTest.py** (Port 8504) - Application ERT Principale
   - 🌡️ Calculateur de température Ravensgate
   - 📊 Analyse de fichiers .dat
   - 🌍 Pseudo-sections ERT 2D/3D
   - 🪨 Stratigraphie complète
   - 🔬 Inversion PyGIMLI
   - **🆕 Tab 6 : Kibali Analyst** (importé depuis ERT.py)

2. **ERT.py** (Port 8506) - Kibali Analyst Standalone
   - 🗺️ Calcul de trajets OSM
   - 📸 Analyse d'images (YOLO)
   - 🌐 Recherche web avancée
   - 💬 Chat RAG avec agents IA
   - 📊 Visualisations avancées
   - **🆕 Tab 6 : ERTest** (importé depuis ERTest.py)

### 🔄 Intégration Bidirectionnelle

```
ERTest.py (8504) ←→ ERT.py (8506)
     ↓                    ↓
  Tab 6: Kibali      Tab 6: ERTest
```

## 🚀 Lancement

### Méthode 1 : Kernel Complet (Recommandé)
```bash
bash /home/belikan/KIbalione8/SETRAF/setraf-kernel.sh start
```
**Lance automatiquement :**
- Node.js Auth Server (port 5000)
- ERTest.py (port 8504)
- ERT.py (port 8506)

### Méthode 2 : Individuel

**ERTest seul :**
```bash
conda run -n gestmodo streamlit run ERTest.py --server.port 8504
```

**Kibali seul :**
```bash
bash /home/belikan/KIbalione8/SETRAF/launch-ert-kibali.sh
```

## 🌐 URLs d'Accès

| Service | URL | Description |
|---------|-----|-------------|
| **ERTest** | http://localhost:8504 | App ERT complète + Tab Kibali |
| **Kibali** | http://localhost:8506 | IA avancée + Tab ERTest |
| **Auth** | http://localhost:5000 | Authentification Node.js |

### Réseau Local
| Service | URL Réseau |
|---------|------------|
| **ERTest** | http://172.20.31.35:8504 |
| **Kibali** | http://172.20.31.35:8506 |
| **Auth** | http://172.20.31.35:5000 |

## 🛠️ Gestion des Services

```bash
# Démarrer
bash setraf-kernel.sh start

# Arrêter
bash setraf-kernel.sh stop

# Redémarrer
bash setraf-kernel.sh restart

# Statut
bash setraf-kernel.sh status

# Logs
bash setraf-kernel.sh logs all

# Monitoring temps réel
bash setraf-kernel.sh monitor
```

## 📝 Logs

| Application | Fichier Log |
|-------------|-------------|
| ERTest | `/home/belikan/KIbalione8/SETRAF/logs/streamlit.log` |
| Kibali | `/home/belikan/KIbalione8/SETRAF/logs/ert-kibali.log` |
| Node.js | `/home/belikan/KIbalione8/SETRAF/logs/node-auth.log` |
| Kernel | `/home/belikan/KIbalione8/SETRAF/logs/kernel.log` |

## 🔧 Architecture

```
SETRAF/
├── ERTest.py          # App principale ERT (8504)
│   └── Tab 6: Kibali Analyst (import dynamique ERT.py)
│
├── ERT.py             # Kibali IA Avancée (8506)
│   └── Tab 6: ERTest (import dynamique ERTest.py)
│
├── setraf-kernel.sh   # Gestionnaire de services
├── launch-ert-kibali.sh  # Lanceur Kibali standalone
│
└── node-auth/         # Serveur d'authentification
    └── server.js      # API Node.js (5000)
```

## 🎯 Avantages de l'Intégration

### Mode ERTest (8504)
✅ Accès rapide aux analyses ERT
✅ Kibali disponible sans changer d'onglet
✅ Session unique pour tout
✅ Poids léger en mémoire

### Mode Kibali (8506)
✅ Ressources dédiées pour l'IA
✅ ERTest disponible pour analyses ponctuelles
✅ Session séparée = meilleure stabilité
✅ Idéal pour analyses IA lourdes

## 📦 Environnement

**Environnement Conda :** `gestmodo`
**Python :** 3.10.19
**Streamlit :** 1.51.0
**PyGIMLI :** 1.5.4
**Scikit-learn :** 1.7.2

## 🔐 Authentification

Les deux applications sont connectées au serveur d'authentification Node.js.
- Connexion unique valide pour les deux ports
- Session partagée via MongoDB Atlas

---

**Version :** 1.0.0
**Date :** 10 Novembre 2025
**Auteur :** BelikanM
