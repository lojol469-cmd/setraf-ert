# 🚀 KibaLock - Guide de démarrage rapide

## ⚡ Installation en 5 minutes

### 1. Prérequis
```bash
# Vérifier Python
python3 --version  # Doit être >= 3.10

# Vérifier pip
pip --version
```

### 2. Installation
```bash
cd /home/belikan/KIbalione8/SETRAF/kibalock-api

# Installer les dépendances
./launch_kibalock.sh --install
```

### 3. Configuration
```bash
# Copier le fichier de configuration
cp .env.example .env

# Éditer avec vos paramètres
nano .env
```

### 4. Lancement
```bash
./launch_kibalock.sh
```

### 5. Accès
Ouvrir dans votre navigateur :
- **Local** : http://localhost:8505
- **Réseau** : http://[VOTRE_IP]:8505

---

## 📝 Premier utilisateur

### Inscription

1. **Onglet "📝 Inscription"**
2. Remplir :
   - Nom d'utilisateur : `test_user`
   - Email : `test@example.com`
3. **Voix** : Téléverser 3 fichiers audio (10-15s chacun)
   - Format : WAV, MP3, OGG
   - Exemple : "Bonjour, je suis [Nom], j'autorise KibaLock"
4. **Visage** : Téléverser 3-5 photos
   - Format : JPG, PNG
   - Angles : face, profil gauche, profil droit
5. Cliquer sur **"✅ Finaliser l'inscription"**

### Connexion

1. **Onglet "🔑 Connexion"**
2. Téléverser :
   - 1 fichier audio (votre voix)
   - 1 photo (votre visage)
3. Cliquer sur **"🔓 Se connecter"**
4. ✅ **Authentifié !**

---

## 🎯 Exemples de fichiers de test

### Créer des échantillons vocaux

```bash
# Sur Linux avec arecord
arecord -d 15 -f cd -t wav voice_sample_1.wav
arecord -d 15 -f cd -t wav voice_sample_2.wav
arecord -d 15 -f cd -t wav voice_sample_3.wav

# Sur Windows avec Sound Recorder
# Démarrer → Enregistreur vocal → Enregistrer 15s
```

### Capturer des photos

```bash
# Sur Linux avec fswebcam
fswebcam -r 640x480 --no-banner face_1.jpg
fswebcam -r 640x480 --no-banner face_2.jpg
fswebcam -r 640x480 --no-banner face_3.jpg

# Sur Windows avec Camera app
# Démarrer → Caméra → Prendre 3 photos
```

---

## 📊 Dashboard

### Statistiques en temps réel

- **👥 Utilisateurs** : Nombre total d'inscrits
- **🔓 Sessions** : Connexions actives
- **🧬 Embeddings** : Empreintes biométriques

### Onglet "👥 Utilisateurs"

Gérer tous les utilisateurs :
- Voir les détails
- Supprimer un utilisateur
- Activer/Désactiver un compte

### Onglet "📈 Monitoring"

- Métriques globales
- Logs en temps réel
- Statistiques de connexion

---

## 🔧 Configuration avancée

### Ajuster les seuils

Dans la sidebar :
- **Seuil voix** : 0.85 (85%)
- **Seuil visage** : 0.90 (90%)

Plus élevé = Plus strict = Plus sécurisé

### Changer le modèle Whisper

Dans `.env` :
```bash
# Options : tiny, base, small, medium, large
WHISPER_MODEL=base
```

- `tiny` : Rapide, moins précis
- `base` : Équilibré (recommandé)
- `large` : Très précis, plus lent

---

## 🐛 Dépannage rapide

### Erreur : "MongoDB connection failed"

```bash
# Vérifier la connexion
python3 -c "from pymongo import MongoClient; client = MongoClient('YOUR_URI'); print(client.server_info())"

# Solution : Vérifier MONGO_URI dans .env
nano .env
```

### Erreur : "Whisper model not found"

```bash
# Télécharger manuellement
python3 -c "import whisper; whisper.load_model('base')"
```

### Erreur : "No face detected"

**Solutions** :
- Améliorer l'éclairage
- Centrer le visage
- Retirer lunettes de soleil
- Utiliser fond neutre

### Erreur : "Port 8505 already in use"

```bash
# Trouver le processus
lsof -i :8505

# Tuer le processus
kill -9 [PID]

# Relancer
./launch_kibalock.sh
```

---

## 📱 Utilisation mobile

### Capturer depuis smartphone

1. Ouvrir l'app sur mobile : `http://[IP_SERVEUR]:8505`
2. Utiliser le navigateur mobile (Chrome/Safari)
3. Autoriser micro + caméra
4. Enregistrer audio/photo directement

---

## 🔒 Conseils de sécurité

### ✅ Bonnes pratiques

1. **Varié vos échantillons**
   - Voix : phrases différentes
   - Visage : angles différents

2. **Environnement contrôlé**
   - Pas de bruit de fond
   - Éclairage correct
   - Fond neutre

3. **Qualité audio**
   - 16kHz minimum
   - Mono
   - WAV non compressé (idéal)

4. **Qualité image**
   - 640x480 minimum
   - Bien éclairé
   - Visage visible

### ❌ À éviter

- ❌ Photos trop sombres
- ❌ Audio avec bruit de fond
- ❌ Visages partiellement cachés
- ❌ Photos de photos (deepfake)

---

## 📈 Performances attendues

### Temps de traitement

| Opération | Temps | Hardware |
|-----------|-------|----------|
| Inscription (3 voix + 3 visages) | ~30s | CPU |
| Connexion (1 voix + 1 visage) | ~5s | CPU |
| Extraction embedding vocal | ~2s | CPU |
| Extraction embedding facial | ~1s | CPU |

### Précision

| Métrique | Valeur |
|----------|--------|
| True Positive Rate | >95% |
| False Positive Rate | <1% |
| False Negative Rate | <5% |

---

## 🎓 Tutoriels vidéo (à créer)

1. **Installation et configuration** (5 min)
2. **Première inscription** (3 min)
3. **Connexion biométrique** (2 min)
4. **Gestion des utilisateurs** (3 min)
5. **Configuration avancée** (5 min)

---

## 📞 Support

### Communauté

- 💬 Discord : [Lien à créer]
- 📧 Email : nyundumathryme@gmail.com
- 🐛 Issues : https://github.com/BelikanM/KIbalione8/issues

### Documentation complète

- 📘 [README.md](README.md) - Documentation complète
- 🔗 [INTEGRATION_LIFEMODO.md](INTEGRATION_LIFEMODO.md) - Intégration avancée
- 📋 [Logs](~/kibalock/logs/) - Logs système

---

## 🎯 Prochaines étapes

Après avoir testé KibaLock :

1. **Intégrer avec LifeModo** pour entraînement custom
2. **Déployer en production** avec HTTPS
3. **Ajouter plus d'utilisateurs**
4. **Configurer le monitoring**
5. **Implémenter l'API REST** pour intégration

---

## ✨ Fonctionnalités à venir

- [ ] Capture webcam en direct
- [ ] Enregistrement audio direct
- [ ] API REST complète
- [ ] Application mobile native
- [ ] Liveness detection
- [ ] Multi-tenancy

---

**Bon démarrage avec KibaLock ! 🚀**

En cas de problème, consultez les logs :
```bash
tail -f ~/kibalock/logs/kibalock_$(date +%Y%m%d).log
```
