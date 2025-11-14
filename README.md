---
title: SETRAF - Subaquifère ERT Analysis
emoji: 💧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: agpl-3.0
app_port: 7860
---

# 💧 SETRAF - Subaquifère ERT Analysis Tool

Plateforme d'analyse géophysique avancée pour l'étude des nappes phréatiques et aquifères souterrains par tomographie de résistivité électrique (ERT).

## 🌟 Fonctionnalités

- 📊 **Analyse ERT avancée** avec PyGIMLi
- 🗺️ **Visualisations 2D/3D** des coupes géologiques
- 💧 **Classification des types d'eau** (mer, salée, douce, pure)
- 🔐 **Authentification sécurisée** avec backend Node.js
- 📈 **Clustering K-means** pour segmentation automatique
- 📄 **Export PDF** des résultats d'analyse

## 🚀 Architecture

- **Frontend**: Streamlit (Python 3.10)
- **Backend**: Node.js + MongoDB sur Render
- **Géophysique**: PyGIMLi + Matplotlib
- **Authentification**: JWT + OTP

## 🔧 Configuration

Les variables d'environnement sont gérées via les Secrets de Hugging Face Spaces :

```bash
PRODUCTION_BACKEND_URL=https://setraf-auth.onrender.com
USE_PRODUCTION_BACKEND=true
```

## 📖 Utilisation

1. Connectez-vous avec vos identifiants
2. Chargez votre fichier de données ERT (.dat, .txt, .csv)
3. Analysez les résultats avec les visualisations interactives
4. Exportez vos rapports en PDF

## 🔗 Liens

- Backend API: https://setraf-auth.onrender.com
- Documentation: [GitHub](https://github.com/BelikanM/KIbalione8)

## 👨‍💻 Développé par

**BelikanM** - Analyse géophysique et développement full-stack

---

**Note**: Application développée pour l'analyse scientifique des ressources en eau souterraine.
