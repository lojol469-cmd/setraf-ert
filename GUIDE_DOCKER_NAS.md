# 📋 Guide Illustré : Importer une Image Docker .tar sur NAS (Synology & QNAP)

## 🎯 Vue d'ensemble

Ce guide vous explique **pas à pas** comment exporter une image Docker depuis WSL2, la transférer vers votre NAS, et l'importer via l'interface graphique (GUI) de Docker Manager.

**Temps estimé :** 15-30 minutes  
**Prérequis :** Docker installé sur WSL2, accès au NAS

---

## 📤 Étape 1 : Exporter l'image Docker depuis WSL2

### 1.1 Identifier votre image Docker

Dans WSL2, listez vos images :

```bash
docker images
```

Exemple de sortie :
```
REPOSITORY          TAG       IMAGE ID       CREATED         SIZE
my-app              latest    abc123def456   2 hours ago     1.2GB
belikanm/kibaertanalyste latest  def789ghi012   1 day ago      2.1GB
```

### 1.2 Exporter l'image en .tar

Utilisez la commande `docker save` :

```bash
docker save -o my-app.tar my-app:latest
```

**Explication des paramètres :**
- `-o my-app.tar` : nom du fichier de sortie
- `my-app:latest` : nom:tag de votre image

### 1.3 Vérifier l'export

```bash
ls -lh my-app.tar
```

Sortie attendue :
```
-rw-r--r-- 1 user user 1.2G Dec 1 10:30 my-app.tar
```

**✅ L'image est maintenant sauvegardée en .tar**

---

## 📁 Étape 2 : Transférer le .tar vers le NAS

### Option A : Via Explorateur Windows (SMB)

1. **Ouvrez l'Explorateur Windows**
2. **Dans la barre d'adresse**, tapez :
   ```
   \\VOTRE_NAS_IP
   ```
   *(Remplacez VOTRE_NAS_IP par l'adresse IP de votre NAS)*

3. **Connectez-vous** avec vos identifiants NAS

4. **Naviguez** vers un dossier partagé, par exemple :
   ```
   \\NAS_IP\docker\images\
   ```

5. **Copiez** `my-app.tar` depuis WSL2 vers ce dossier

### Option B : Via SCP (ligne de commande)

```bash
scp my-app.tar admin@192.168.1.100:/volume1/docker/images/
```

**Paramètres :**
- `admin` : votre utilisateur NAS
- `192.168.1.100` : IP du NAS
- `/volume1/docker/images/` : chemin sur le NAS

---

## 🖥️ Étape 3 : Importer via GUI - Synology DSM

### 3.1 Accéder à Docker Manager

1. **Connectez-vous** à DSM (interface web du Synology)
2. **Ouvrez** "Docker" depuis le menu principal
3. **Cliquez** sur l'onglet "Image"

### 3.2 Importer l'image

1. **Cliquez** sur "Ajouter" → "Importer depuis un fichier"
2. **Sélectionnez** votre fichier `my-app.tar`
3. **Cliquez** sur "Importer"

**Interface DSM :**
```
Docker → Image → Ajouter → Importer depuis un fichier
```

### 3.3 Vérifier l'import

L'image apparaît dans la liste :
```
REPOSITORY          TAG       IMAGE ID       CREATED         SIZE
my-app              latest    abc123def456   Just now        1.2GB
```

---

## 🖥️ Étape 4 : Importer via GUI - QNAP Qsirch/OS

### 4.1 Accéder à Container Station

1. **Connectez-vous** à Qsirch/OS (interface web du QNAP)
2. **Ouvrez** "Container Station" depuis le menu
3. **Cliquez** sur l'onglet "Images"

### 4.2 Importer l'image

1. **Cliquez** sur "Importer" → "Depuis un fichier local"
2. **Parcourez** et sélectionnez `my-app.tar`
3. **Cliquez** sur "Importer"

**Interface Qsirch :**
```
Container Station → Images → Importer → Depuis un fichier local
```

### 4.3 Vérifier l'import

L'image apparaît dans la liste des images disponibles.

---

## 🚀 Étape 5 : Lancer le container via GUI

### Sur Synology DSM :

1. **Sélectionnez** votre image importée
2. **Cliquez** sur "Lancer"
3. **Configurez :**
   - **Nom du container** : `my-app-container`
   - **Ports** : Ajoutez `8080:80` (NAS:Container)
   - **Volumes** : Ajoutez `/volume1/docker/data:/app/data`
   - **Variables d'environnement** : Si nécessaire
4. **Cliquez** sur "Appliquer"

### Sur QNAP Qsirch/OS :

1. **Sélectionnez** votre image
2. **Cliquez** sur "Créer" → "Container"
3. **Configurez :**
   - **Nom** : `my-app-container`
   - **Réseau** : Bridge ou Host
   - **Ports** : Mappez les ports nécessaires
   - **Volumes** : Montez les dossiers persistants
4. **Cliquez** sur "Créer et exécuter"

---

## 🔍 Étape 6 : Vérifier le fonctionnement

### Vérifier l'état du container

**Synology :** Docker → Container → État  
**QNAP :** Container Station → Containers → État

### Accéder à l'application

Ouvrez votre navigateur :
```
http://VOTRE_NAS_IP:8080
```

### Voir les logs

**Synology :** Sélectionnez le container → "Détails" → "Logs"  
**QNAP :** Sélectionnez le container → "Logs"

---

## 🛠️ Dépannage

### Problème : Import échoue

**Cause :** Fichier corrompu ou permissions  
**Solution :**
```bash
# Vérifier l'intégrité du .tar
docker load < my-app.tar
```

### Problème : Container ne démarre pas

**Cause :** Ports déjà utilisés ou configuration incorrecte  
**Solution :** Vérifiez les logs et ajustez la configuration

### Problème : Accès refusé au NAS

**Cause :** Permissions SMB/SCP  
**Solution :** Vérifiez les droits d'accès utilisateur

---

## 💡 Conseils avancés

### Automatisation avec script

Créez un script `deploy-nas.sh` :

```bash
#!/bin/bash
# Export depuis WSL2
docker save -o my-app.tar my-app:latest

# Transfert vers NAS
scp my-app.tar admin@192.168.1.100:/volume1/docker/images/

# Commande pour importer sur NAS (via SSH)
ssh admin@192.168.1.100 "docker load < /volume1/docker/images/my-app.tar"
```

### Gestion des versions

- **Taggez vos images** : `my-app:v1.0`, `my-app:v1.1`
- **Gardez plusieurs versions** sur le NAS
- **Documentez** les changements

### Sécurité

- **Utilisez HTTPS** pour accéder au NAS
- **Changez les ports par défaut** si nécessaire
- **Limitez l'accès** aux dossiers partagés

---

## 📞 Support

Si vous rencontrez des problèmes :

1. **Vérifiez les logs** du container
2. **Testez l'import** directement sur le NAS via SSH :
   ```bash
   docker load < my-app.tar
   ```
3. **Consultez la documentation** officielle :
   - Synology : https://kb.synology.com/
   - QNAP : https://www.qnap.com/

---

**✅ Guide terminé !** Votre image Docker est maintenant déployée sur votre NAS via l'interface graphique.</content>
<filePath">/home/belikan/KIbalione8/SETRAF/GUIDE_DOCKER_NAS.md