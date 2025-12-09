# 🚀 Déploiement Docker sur NAS - Guide Complet

## 📋 Fichiers créés

### 1. GUIDE_DOCKER_NAS.md
Guide illustré détaillé pour importer une image Docker .tar sur NAS Synology/QNAP via GUI.

**Contenu :**
- Export depuis WSL2
- Transfert vers NAS
- Import via interface graphique
- Lancement du container
- Dépannage

### 2. deploy-to-nas.sh
Script automatisé pour exporter et transférer l'image Docker vers le NAS.

**Utilisation :**
```bash
# Modifier la configuration dans le script
nano deploy-to-nas.sh

# Exécuter
./deploy-to-nas.sh
```

## ⚙️ Configuration requise

### Dans deploy-to-nas.sh :
```bash
IMAGE_NAME="belikanm/kibaertanalyste"  # Votre image
IMAGE_TAG="latest"                     # Tag de l'image
NAS_USER="admin"                       # Utilisateur NAS
NAS_IP="192.168.1.100"                # IP du NAS
NAS_PATH="/volume1/docker/images"      # Chemin sur le NAS
```

## 🎯 Workflow recommandé

### Option 1 : Automatique (Script)
```bash
cd /home/belikan/KIbalione8/SETRAF
./deploy-to-nas.sh
# Puis suivre les étapes GUI du guide
```

### Option 2 : Manuel (Guide)
```bash
# Suivre GUIDE_DOCKER_NAS.md étape par étape
docker save -o my-app.tar belikanm/kibaertanalyste:latest
# Transfert manuel + Import GUI
```

## 📞 Support

- **Guide détaillé :** `GUIDE_DOCKER_NAS.md`
- **Script auto :** `deploy-to-nas.sh`
- **Logs :** Vérifiez les logs Docker sur le NAS

---
**Prêt pour le déploiement sur votre NAS ! 🎉**</content>
<filePath">/home/belikan/KIbalione8/SETRAF/README_DEPLOIEMENT_NAS.md