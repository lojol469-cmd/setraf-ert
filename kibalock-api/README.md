# 🔐 KibaLock - Authentification Biométrique Multimodale

## 🎯 Vue d'ensemble

**KibaLock** est un système d'authentification biométrique de nouvelle génération utilisant l'intelligence artificielle pour combiner **reconnaissance vocale** et **reconnaissance faciale** dans un système unifié et ultra-sécurisé.

### ⭐ Caractéristiques principales

- 🎤 **Authentification vocale** : Analyse de l'empreinte vocale unique via Whisper AI
- 📸 **Authentification faciale** : Reconnaissance faciale avec FaceNet512
- 🧠 **Fusion multimodale** : Combinaison intelligente des deux modalités (60% voix + 40% visage)
- 🔒 **Sécurité renforcée** : Embeddings vectoriels chiffrés dans MongoDB
- 📊 **Monitoring temps réel** : Dashboard complet avec statistiques et logs
- 🚀 **Interface moderne** : Application Streamlit intuitive et responsive

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    KIBALOCK SYSTEM                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📱 Frontend (Streamlit)                                │
│     ├── Inscription (Voix + Visage)                    │
│     ├── Connexion (Vérification biométrique)           │
│     ├── Gestion des utilisateurs                       │
│     └── Dashboard de monitoring                        │
│                                                         │
│  🧠 AI Core                                             │
│     ├── Whisper (Embeddings vocaux)                    │
│     ├── DeepFace + FaceNet512 (Embeddings faciaux)     │
│     └── Fusion multimodale (Scoring)                   │
│                                                         │
│  💾 Database (MongoDB)                                  │
│     ├── users : Informations utilisateurs              │
│     ├── embeddings : Vecteurs biométriques             │
│     └── sessions : Sessions actives                    │
│                                                         │
│  📊 Monitoring                                          │
│     ├── Logs JSON structurés                           │
│     └── Métriques temps réel                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- MongoDB (local ou Atlas)
- Webcam (pour capture faciale)
- Microphone (pour capture vocale)
- 8 GB RAM minimum
- GPU recommandé (optionnel)

### Installation étape par étape

```bash
# 1. Cloner le projet
cd /home/belikan/KIbalione8/SETRAF/kibalock-api

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer MongoDB
# Créer un fichier .env
cat > .env << EOF
MONGO_URI=mongodb+srv://USERNAME:PASSWORD@cluster.mongodb.net/kibalock
EOF

# 5. Lancer l'application
streamlit run kibalock.py --server.port 8505
```

---

## 📖 Guide d'utilisation

### 1️⃣ Inscription d'un nouvel utilisateur

#### Étape 1 : Informations de base
- Nom d'utilisateur unique
- Adresse email

#### Étape 2 : Capture vocale
- Enregistrez **3 échantillons vocaux** de 10-15 secondes chacun
- Prononcez des phrases naturelles comme :
  > "Bonjour, je suis [Votre Nom], j'autorise KibaLock à reconnaître ma voix."
  
- Formats acceptés : WAV, MP3, OGG
- Qualité recommandée : 16kHz, mono

#### Étape 3 : Capture faciale
- Capturez **3-5 photos** de votre visage
- Angles variés : face, profil gauche/droit, avec/sans sourire
- Conditions :
  - Éclairage correct
  - Visage centré
  - Pas de lunettes de soleil
  - Fond neutre recommandé

#### Étape 4 : Traitement
- Le système extrait automatiquement :
  - **Embedding vocal** : 1280 dimensions (Whisper)
  - **Embedding facial** : 512 dimensions (FaceNet)
  - **Embedding combiné** : 1792 dimensions total

### 2️⃣ Connexion biométrique

#### Étape 1 : Vérification vocale
- Enregistrez une phrase d'identification (10-15 secondes)
- Peut être différente de l'inscription

#### Étape 2 : Vérification faciale
- Capturez une photo de votre visage
- Conditions similaires à l'inscription

#### Étape 3 : Authentification
Le système calcule :
- **Score vocal** : Similarité cosinus (seuil: 85%)
- **Score facial** : Similarité cosinus (seuil: 90%)
- **Score combiné** : (0.6 × voix) + (0.4 × visage)

✅ **Authentification réussie** si les deux seuils sont atteints

---

## 🔬 Fonctionnement technique

### Extraction d'embeddings vocaux

```python
1. Audio (WAV/MP3) → Whisper.load_audio()
2. Padding/Trim → 30 secondes standard
3. Mel Spectrogram → Features audio
4. Encoder → Embedding 1280D
5. Normalisation L2 → Vecteur unitaire
```

### Extraction d'embeddings faciaux

```python
1. Image (JPG/PNG) → OpenCV
2. Détection de visage → Haar Cascade
3. Alignement → Rotation/Crop
4. FaceNet512 → Embedding 512D
5. Normalisation L2 → Vecteur unitaire
```

### Calcul de similarité

```python
def calculate_similarity(emb1, emb2):
    return 1 - cosine(emb1, emb2)

# Score combiné
combined_score = (voice_sim × 0.6) + (face_sim × 0.4)
```

---

## 📊 Base de données MongoDB

### Collection `users`

```json
{
  "_id": ObjectId(),
  "user_id": "sha256_hash",
  "username": "francis_nyundu",
  "email": "francis@example.com",
  "created_at": ISODate("2025-11-08T14:00:00Z"),
  "active": true,
  "login_count": 42,
  "last_login": ISODate("2025-11-08T13:45:00Z")
}
```

### Collection `embeddings`

```json
{
  "_id": ObjectId(),
  "user_id": "sha256_hash",
  "voice_embedding": [0.221, -0.985, 0.332, ...],  // 1280D
  "face_embedding": [0.155, -0.551, 0.883, ...],   // 512D
  "combined_embedding": [...],                      // 1792D
  "voice_samples_count": 3,
  "face_samples_count": 5,
  "transcriptions": ["Phrase 1", "Phrase 2", "Phrase 3"],
  "created_at": ISODate("2025-11-08T14:00:00Z")
}
```

### Collection `sessions`

```json
{
  "_id": ObjectId(),
  "session_id": "sha256_hash",
  "user_id": "sha256_hash",
  "created_at": ISODate("2025-11-08T13:45:00Z"),
  "expires_at": ISODate("2025-11-09T13:45:00Z"),
  "scores": {
    "voice_similarity": 0.92,
    "face_similarity": 0.95,
    "combined_score": 0.934,
    "transcription": "Bonjour je me connecte"
  }
}
```

---

## 🔒 Sécurité

### Mesures de sécurité implémentées

1. **Chiffrement des embeddings**
   - AES-256 pour le stockage
   - Aucune donnée biométrique brute conservée

2. **Authentification multifactorielle**
   - Voix + Visage obligatoires
   - Seuils de similarité élevés

3. **Anti-spoofing**
   - Détection de liveness (visage)
   - Analyse de qualité audio
   - Vérification de cohérence temporelle

4. **Gestion des sessions**
   - Expiration automatique (24h)
   - Invalidation manuelle possible
   - Tracking des connexions

5. **Logs de sécurité**
   - Toutes les tentatives enregistrées
   - Alertes en cas d'échecs répétés
   - Audit trail complet

---

## 📈 Monitoring

### Métriques disponibles

- **Utilisateurs totaux** : Nombre d'utilisateurs enregistrés
- **Utilisateurs actifs** : Comptes non désactivés
- **Sessions actives** : Connexions en cours
- **Connexions totales** : Historique complet
- **Taux de réussite** : % d'authentifications réussies

### Logs structurés

```json
{
  "timestamp": "2025-11-08T13:45:23.123Z",
  "event_type": "SUCCESS",
  "message": "Connexion réussie pour francis_nyundu",
  "user_id": "abc123..."
}
```

Types d'événements :
- `INFO` : Informations générales
- `SUCCESS` : Opérations réussies
- `WARNING` : Alertes non critiques
- `ERROR` : Erreurs critiques

---

## 🎛️ Configuration

### Paramètres ajustables

```python
# Seuils de similarité
VOICE_THRESHOLD = 0.85  # 85% minimum
FACE_THRESHOLD = 0.90   # 90% minimum

# Pondération fusion
VOICE_WEIGHT = 0.6      # 60% voix
FACE_WEIGHT = 0.4       # 40% visage

# Session
SESSION_DURATION = 24   # heures

# Modèles IA
WHISPER_MODEL = "base"  # tiny, base, small, medium, large
FACE_MODEL = "Facenet512"  # VGG-Face, Facenet, OpenFace, DeepFace
```

---

## 🧪 Tests

### Test d'inscription

```bash
# Préparer des fichiers de test
voice1.wav, voice2.wav, voice3.wav
face1.jpg, face2.jpg, face3.jpg

# Lancer l'app et suivre le workflow d'inscription
streamlit run kibalock.py
```

### Test de connexion

```bash
# Préparer des fichiers de vérification
test_voice.wav
test_face.jpg

# Lancer l'app et tester la connexion
```

### Test de performance

```python
# Mesurer le temps de traitement
import time

start = time.time()
embedding = extract_voice_embedding("test.wav")
print(f"Temps vocal: {time.time() - start:.2f}s")

start = time.time()
embedding = extract_face_embedding("test.jpg")
print(f"Temps facial: {time.time() - start:.2f}s")
```

---

## 🚧 Roadmap

### Version 1.0 (Actuelle)
- ✅ Inscription multimodale
- ✅ Connexion biométrique
- ✅ Dashboard de monitoring
- ✅ Gestion des utilisateurs

### Version 1.1 (Prévue)
- ⏳ Capture webcam en temps réel
- ⏳ Enregistrement audio direct
- ⏳ API REST pour intégration
- ⏳ Rate limiting anti-bruteforce

### Version 2.0 (Future)
- 🔮 Liveness detection avancée
- 🔮 Multi-tenancy
- 🔮 Export mobile (iOS/Android)
- 🔮 Blockchain pour audit trail
- 🔮 Authentification comportementale

---

## 🤝 Intégration avec d'autres systèmes

### Exemple : Intégration dans SETRAF

```python
from kibalock import verify_user

def setraf_login():
    voice_path = capture_voice()
    face_path = capture_face()
    
    success, user, scores = verify_user(voice_path, face_path)
    
    if success:
        # Créer session SETRAF
        create_setraf_session(user['user_id'])
        return True
    return False
```

---

## 📝 API (Future)

### Endpoints prévus

```
POST /api/v1/register
POST /api/v1/login
POST /api/v1/verify
GET  /api/v1/users
GET  /api/v1/users/{user_id}
DELETE /api/v1/users/{user_id}
GET  /api/v1/sessions
POST /api/v1/sessions/invalidate
```

---

## 🐛 Dépannage

### Problème : Whisper ne charge pas

```bash
# Vérifier l'installation
pip install openai-whisper --upgrade

# Tester manuellement
python -c "import whisper; model = whisper.load_model('base')"
```

### Problème : DeepFace erreur de détection

```bash
# Installer les backends
pip install opencv-python
pip install tensorflow

# Vérifier l'image
python -c "import cv2; img = cv2.imread('test.jpg'); print(img.shape)"
```

### Problème : MongoDB connection

```bash
# Vérifier la connection string
python -c "from pymongo import MongoClient; client = MongoClient('your_uri'); print(client.server_info())"
```

---

## 📚 Références

- **Whisper** : https://github.com/openai/whisper
- **DeepFace** : https://github.com/serengil/deepface
- **FaceNet** : https://arxiv.org/abs/1503.03832
- **MongoDB** : https://www.mongodb.com/docs/

---

## 👨‍💻 Développement

### Structure du projet

```
kibalock-api/
├── kibalock.py          # Application principale
├── lifemodo.py          # Pipeline d'entraînement
├── requirements.txt     # Dépendances
├── README.md            # Ce fichier
├── .env                 # Configuration (à créer)
└── ~/kibalock/          # Données (créé automatiquement)
    ├── embeddings/      # Cache embeddings
    ├── temp/            # Fichiers temporaires
    └── logs/            # Logs JSON
```

### Contribuer

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing`)
5. Ouvrir une Pull Request

---

## 📄 Licence

**AGPL v3** - Voir [LICENSE](../LICENSE-AGPLv3.txt)

---

## 👏 Crédits

- **Développé par** : Francis Nyundu (BelikanM)
- **Basé sur** : LifeModo Multimodal Pipeline
- **Framework** : Streamlit
- **IA** : OpenAI Whisper, DeepFace, FaceNet
- **Database** : MongoDB

---

## 📞 Support

Pour toute question ou problème :
- 📧 Email : nyundumathryme@gmail.com
- 🐛 Issues : https://github.com/BelikanM/KIbalione8/issues

---

**KibaLock** - Authentification biométrique du futur 🚀
