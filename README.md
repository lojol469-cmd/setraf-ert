# 💧 SETRAF - Subaquifère ERT Analysis Tool

**Version 1.0.0** - Outil d'analyse géophysique avancé pour tomographie électrique (ERT)

---

## 📋 Description

SETRAF est une application complète pour l'analyse de données ERT (Electrical Resistivity Tomography) avec visualisation 3D interactive, classification automatique des matériaux géologiques et génération de rapports PDF professionnels.

### 🎯 Fonctionnalités principales

- ✅ **Calculateur Température Ts** (Ravensgate Sonic)
- ✅ **Analyse fichiers .dat** avec sections d'eau automatiques
- ✅ **Pseudo-sections ERT 2D/3D** avec interpolation cubique
- ✅ **Stratigraphie Complète** avec 30+ matériaux géologiques
- ✅ **Visualisation 3D interactive** (Plotly) des couches
- ✅ **Précision millimétrique** (3 décimales sur tous les axes)
- ✅ **Classification automatique** en 8 catégories géologiques
- ✅ **Export PDF haute résolution** (150 DPI)
- ✅ **API REST** pour intégration programmatique

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- Conda (Miniconda ou Anaconda)
- Git

### Étapes d'installation

```bash
# 1. Cloner le dépôt (si nécessaire)
cd /home/belikan/KIbalione8/SETRAF

# 2. Créer l'environnement conda
conda create -n gestmodo python=3.10
conda activate gestmodo

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Vérifier l'installation
python -c "import streamlit; import fastapi; print('✅ Installation OK')"
```

---

## 📱 Utilisation

### Lancement de l'application Streamlit

```bash
# Méthode 1: Script de lancement (recommandé)
./launch_setraf.sh

# Méthode 2: Lancement direct
~/miniconda3/envs/gestmodo/bin/python -m streamlit run ERTest.py --server.port 8504

# Méthode 3: Port personnalisé
./launch_setraf.sh 8600
```

L'application sera accessible à : **http://localhost:8504**

### Lancement de l'API

```bash
# Lancer l'API FastAPI
~/miniconda3/envs/gestmodo/bin/python api_setraf.py

# Ou avec uvicorn
~/miniconda3/envs/gestmodo/bin/uvicorn api_setraf:app --host 0.0.0.0 --port 8505 --reload
```

Documentation API : **http://localhost:8505/api/docs**

---

## 📊 Utilisation de l'interface

### Tab 1 : Calculateur Température Ts
- Entrer Tw (température eau) et Tg (température géothermique)
- Calcul automatique de Ts avec table Ravensgate Sonic

### Tab 2 : Analyse fichiers .dat
- Upload fichier .dat (format : survey-point, depth, data, project)
- Détection automatique des sections d'eau (mer, salée, douce, pure)
- Visualisation 2D/3D avec interpolation
- Export CSV/Excel/PDF

### Tab 3 : Pseudo-sections ERT
- Visualisation pseudo-sections 2D avec couleurs résistivité
- Modèle théorique vs données réelles
- Comparaison multicouche

### Tab 4 : Stratigraphie Complète
- Classification automatique en 8 catégories
- Visualisation 3D interactive (rotation 360°)
- 8 plages de résistivité avec coupes détaillées
- Précision millimétrique sur tous les axes
- Export PDF stratigraphique complet

---

## 🔌 Utilisation de l'API

### Exemple Python

```python
import requests
import json

# URL de l'API
API_URL = "http://localhost:8505"

# 1. Vérifier le statut
response = requests.get(f"{API_URL}/api/status")
print(response.json())

# 2. Upload fichier .dat
files = {'file': open('frequ.dat', 'rb')}
response = requests.post(f"{API_URL}/api/upload", files=files)
result = response.json()
print(f"Analysis ID: {result['analysis_id']}")
print(f"Total measurements: {result['analysis']['statistics']['total_measurements']}")

# 3. Analyser données directement
data = {
    "survey_points": [1, 2, 3, 4, 5],
    "depths": [-2, -2, -2, -2, -2],
    "resistivities": [0.36, 0.41, 0.41, 0.37, 0.36],
    "project_id": "20251108"
}
response = requests.post(f"{API_URL}/api/analyze", json=data)
analysis = response.json()
print(json.dumps(analysis, indent=2))
```

### Exemple cURL

```bash
# Statut de l'API
curl http://localhost:8505/api/status

# Upload fichier
curl -X POST http://localhost:8505/api/upload \
  -F "file=@frequ.dat"

# Analyse de données
curl -X POST http://localhost:8505/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "survey_points": [1,2,3],
    "depths": [-2,-2,-2],
    "resistivities": [0.36,0.41,0.37]
  }'
```

---

## 📁 Structure du projet

```
SETRAF/
├── ERTest.py               # Application Streamlit principale (1863 lignes)
├── api_setraf.py           # API FastAPI
├── launch_setraf.sh        # Script de lancement
├── requirements.txt        # Dépendances Python
├── README.md               # Cette documentation
├── logo_belikan.png        # Logo de l'application
└── .env                    # Configuration (à créer)
```

---

## 🎨 Classifications géologiques

### 8 catégories automatiques

| Catégorie | Résistivité (Ω·m) | Couleur | Description |
|-----------|-------------------|---------|-------------|
| 💎 Minéraux métalliques | 0.001-1 | 🟡 Gold | Sulfures, graphite |
| 💧 Eaux salées + Argiles | 1-10 | 🔴 Rouge | Eau de mer, argiles marines |
| 🧱 Argiles compactes | 10-50 | 🟤 Marron | Argiles saturées |
| 💧 Eaux douces + Sols | 50-200 | 🟢 Vert | Nappes phréatiques |
| 🏖️ Sables + Graviers | 200-1000 | 🟠 Sable | Aquifères sableux |
| 🏔️ Roches sédimentaires | 1000-5000 | 🔵 Bleu ciel | Calcaires, grès |
| 🌋 Roches ignées | 5000-100000 | 🔴 Rose | Granites, basaltes |
| 💎 Quartzite | >100000 | ⚪ Gris | Minéraux isolants |

---

## 📊 Format des fichiers .dat

```
survey-point	depth	data	project
1	-2	0.36289272	20251030
2	-2	0.40952906	20251030
3	-2	0.41214067	20251030
...
```

**Colonnes requises :**
- `survey-point` : Position le long du profil (m)
- `depth` : Profondeur (m, négatif = sous la surface)
- `data` : Résistivité mesurée (Ω·m)
- `project` : ID du projet (optionnel)

---

## 🔧 Configuration avancée

### Variables d'environnement (.env)

```env
# Port Streamlit
STREAMLIT_PORT=8504

# Port API
API_PORT=8505

# Mode debug
DEBUG=False

# Chemin des logs
LOG_PATH=./logs

# Clé API (pour authentification future)
API_KEY=votre_cle_secrete_ici
```

---

## 📄 Exports disponibles

- **CSV** : Données brutes tabulaires
- **Excel** : Tableaux formatés avec métadonnées
- **PDF Standard** : Rapport d'analyse DTW (150 DPI)
- **PDF Stratigraphique** : Classification géologique complète (150 DPI)

---

## 🐛 Dépannage

### Problème : Streamlit ne démarre pas
```bash
# Vérifier l'environnement
conda activate gestmodo
python --version  # Doit afficher Python 3.10.x

# Réinstaller streamlit
pip install --upgrade streamlit
```

### Problème : Logo non affiché
```bash
# Vérifier que le logo existe
ls -lh logo_belikan.png

# Si absent, copier depuis le dossier parent
cp ../logo_belikan.png ./
```

### Problème : API ne répond pas
```bash
# Vérifier que le port 8505 est libre
lsof -i :8505

# Tuer le processus si nécessaire
kill -9 <PID>
```

---

## 📞 Support

- **Auteur** : Belikan M.
- **Email** : nyundumathryme@gmail.com
- **Repository** : github.com/BelikanM/KIbalione8

---

## 📜 Licence

Copyright © 2025 Belikan M. - Tous droits réservés.

---

## 🎉 Changelog

### Version 1.0.0 (08 Novembre 2025)
- ✅ Interface Streamlit complète avec 4 tabs
- ✅ Visualisation 3D interactive (Plotly)
- ✅ Précision millimétrique (3 décimales)
- ✅ Classification automatique 8 catégories
- ✅ Export PDF haute résolution
- ✅ API REST FastAPI fonctionnelle
- ✅ Documentation complète
