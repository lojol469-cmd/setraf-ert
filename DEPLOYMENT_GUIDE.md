# 🚀 Guide de Déploiement SETRAF

## 📋 Vue d'ensemble

Ce guide explique comment déployer l'application SETRAF sur Hugging Face Spaces via GitHub.

## 🏗️ Architecture de Déploiement

```
GitHub Repository (lojol469-cmd/setraf-ert)
    ↓ (auto-sync)
Hugging Face Space (BelikanM/setraf-ert)
    ↓ (Docker build)
Application Live (https://huggingface.co/spaces/BelikanM/setraf-ert)
```

## ⚙️ Prérequis

- [x] Compte GitHub (lojol469-cmd)
- [x] Compte Hugging Face Pro (BelikanM)
- [x] Backend déployé sur Render (https://setraf-auth.onrender.com)
- [x] Git installé localement

## 📦 Étape 1 : Déploiement sur GitHub

### Option A : Script automatique (Recommandé)

```bash
cd /home/belikan/setraf-frontend-hf
./deploy-to-hf.sh
```

### Option B : Manuel

```bash
cd /home/belikan/setraf-frontend-hf

# Initialiser Git
git init
git config user.name "lojol469-cmd"
git config user.email "nyundumathryme@gmail.com"

# Ajouter les fichiers
git add .
git commit -m "🚀 Initial deployment of SETRAF ERT Analysis Tool"

# Pousser vers GitHub
git remote add origin https://github.com/lojol469-cmd/setraf-ert.git
git branch -M main
git push -u origin main
```

## 🌐 Étape 2 : Créer le Space Hugging Face

1. **Créer un nouveau Space**
   - Aller sur : https://huggingface.co/new-space
   - **Owner** : BelikanM
   - **Space name** : setraf-ert
   - **License** : agpl-3.0
   - **Space SDK** : Docker
   - **Visibility** : Public (ou Private selon besoin)

2. **Configurer le Hardware**
   - **CPU Basic (gratuit)** : Convient pour tests légers
   - **CPU Upgraded ($0.03/h)** : Recommandé pour PyGIMLi
   - **GPU T4 ($0.60/h)** : Pour analyses intensives

## 🔗 Étape 3 : Connecter GitHub au Space

1. Aller dans les **Settings** du Space
2. Section **Repository**
3. Cliquer sur **Link to GitHub repository**
4. Autoriser Hugging Face à accéder à GitHub
5. Sélectionner : `lojol469-cmd/setraf-ert`
6. Activer **Auto-sync** pour déploiements automatiques

## 🔐 Étape 4 : Configurer les Variables d'Environnement

Dans **Settings > Variables and secrets**, ajouter :

| Variable | Valeur | Description |
|----------|--------|-------------|
| `USE_PRODUCTION_BACKEND` | `true` | Active le backend Render |
| `PRODUCTION_BACKEND_URL` | `https://setraf-auth.onrender.com` | URL du backend Node.js |

## 🚀 Étape 5 : Premier Déploiement

1. Dans le Space, cliquer sur **Factory reboot**
2. Le build Docker démarrera (5-10 minutes)
3. Surveiller les logs dans l'onglet **Logs**
4. Une fois terminé, l'application sera accessible !

## ✅ Vérification du Déploiement

### URLs à tester :

- **Application** : https://huggingface.co/spaces/BelikanM/setraf-ert
- **Backend API** : https://setraf-auth.onrender.com/api/health
- **GitHub Repo** : https://github.com/lojol469-cmd/setraf-ert

### Tests de fonctionnement :

1. ✅ Page de connexion s'affiche
2. ✅ Authentification fonctionne (connexion OTP ou mot de passe)
3. ✅ Upload de fichier .dat fonctionne
4. ✅ Visualisations s'affichent correctement
5. ✅ Export PDF fonctionne

## 🔄 Mises à Jour Continues

Une fois le Space lié à GitHub, **chaque push sur la branche main** déclenchera :

1. Auto-sync du code vers Hugging Face
2. Rebuild automatique de l'image Docker
3. Redémarrage du Space avec la nouvelle version

```bash
# Workflow de mise à jour
cd /home/belikan/setraf-frontend-hf
git add .
git commit -m "✨ Feature: Nouvelle fonctionnalité"
git push origin main
# → Le Space se met à jour automatiquement !
```

## 🐛 Dépannage

### Build Docker échoue

**Problème** : `ERROR: Unable to install pygimli`

**Solution** :
- Vérifier que le Dockerfile installe toutes les dépendances système
- Augmenter le hardware du Space (CPU Upgraded ou GPU)
- Vérifier les logs de build dans l'onglet **Logs**

### Application ne démarre pas

**Problème** : `Application error`

**Solution** :
1. Vérifier les variables d'environnement (Settings > Variables)
2. Vérifier que le backend Render est actif
3. Consulter les logs du container Docker
4. Tester en local avec Docker :
   ```bash
   cd /home/belikan/setraf-frontend-hf
   docker build -t setraf-test .
   docker run -p 7860:7860 -e USE_PRODUCTION_BACKEND=true setraf-test
   ```

### Authentification ne fonctionne pas

**Problème** : Erreur de connexion au backend

**Solution** :
1. Vérifier que `PRODUCTION_BACKEND_URL` est correct dans les secrets
2. Tester le backend : `curl https://setraf-auth.onrender.com/api/health`
3. Vérifier les CORS dans le backend Node.js

## 📊 Monitoring

### Métriques à surveiller :

- **Uptime** : Disponibilité du Space
- **Build time** : Durée de construction Docker
- **Memory usage** : Utilisation RAM (limite 16 GB en Pro)
- **API calls** : Requêtes vers le backend Render

### Logs utiles :

```bash
# Logs de l'application Streamlit
# Accessibles dans : Space > Logs

# Logs du backend Render
# https://dashboard.render.com/web/setraf-auth > Logs
```

## 💰 Coûts Estimés

### Hugging Face Pro :
- **CPU Basic** : Gratuit (mais limité)
- **CPU Upgraded** : ~$22/mois (usage continu)
- **GPU T4** : ~$432/mois (usage continu)

**Recommandation** : CPU Upgraded pour usage normal, GPU pour démos intensives

### Render :
- **Backend Node.js** : Gratuit (avec limitations)
- **Upgrade si besoin** : $7-25/mois

## 🎯 Optimisations

### Réduire le temps de build :

1. **Utiliser un cache Docker** (activé par défaut sur HF)
2. **Minimiser les dépendances** dans requirements.txt
3. **Pré-construire des images** pour PyGIMLi

### Améliorer les performances :

1. **Activer le GPU** pour les calculs PyGIMLi intensifs
2. **Utiliser st.cache_data** dans Streamlit pour les visualisations
3. **Optimiser les imports** (import lazy)

## 🔒 Sécurité

### Bonnes pratiques :

- ✅ Jamais pousser `.env` sur GitHub (dans .gitignore)
- ✅ Utiliser les Secrets HF pour variables sensibles
- ✅ Activer 2FA sur GitHub et Hugging Face
- ✅ Renouveler les tokens régulièrement
- ✅ Monitorer les logs pour activités suspectes

## 📚 Ressources

- **Hugging Face Spaces** : https://huggingface.co/docs/hub/spaces
- **Docker Documentation** : https://docs.docker.com/
- **Streamlit Docs** : https://docs.streamlit.io/
- **PyGIMLi** : https://www.pygimli.org/

## 🆘 Support

En cas de problème :

1. **Consulter les logs** du Space et du backend
2. **Tester localement** avec Docker
3. **Vérifier la documentation** Hugging Face
4. **Contacter le support** : support@huggingface.co

---

**Développé par** : BelikanM / lojol469-cmd  
**Dernière mise à jour** : Novembre 2025  
**Version** : 1.0.0
