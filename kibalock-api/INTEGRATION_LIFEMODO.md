# 🔗 Intégration KibaLock ↔ LifeModo

## 📋 Vue d'ensemble

Ce document explique comment **KibaLock** utilise **LifeModo** comme moteur d'apprentissage biométrique et comment les deux systèmes peuvent travailler ensemble.

---

## 🎯 Synergie entre les deux systèmes

| Système | Rôle | Technologies |
|---------|------|--------------|
| **LifeModo** | Pipeline d'entraînement multimodal | YOLO, Whisper, OCR, NLP |
| **KibaLock** | Authentification biométrique en production | Whisper, DeepFace, MongoDB |

### 🔄 Workflow combiné

```
1. LifeModo entraîne les modèles personnalisés
   ↓
2. Export des modèles (.onnx, .tflite, .tfjs)
   ↓
3. KibaLock utilise ces modèles pour l'authentification
   ↓
4. Feedback des performances → Amélioration LifeModo
```

---

## 🧠 Utiliser LifeModo pour améliorer KibaLock

### Scénario 1 : Entraînement personnalisé de reconnaissance vocale

**LifeModo peut entraîner un modèle vocal spécifique à votre environnement :**

1. **Collecte de données** via LifeModo
   ```python
   # Dans lifemodo.py
   uploaded_audios = st.file_uploader("Audios d'entraînement", type=["wav"])
   for audio in uploaded_audios:
       process_audio(audio, text_output_path)
   ```

2. **Entraînement** sur vos données vocales
   ```python
   def train_audio_model(audio_files):
       model = whisper.load_model("base")
       # Fine-tuning sur vos données
       for audio_file in audio_files:
           result = model.transcribe(audio_file)
           # Extraction features + entraînement custom
   ```

3. **Export du modèle** pour KibaLock
   ```python
   model.export(format="onnx")
   # Copier vers ~/kibalock/models/
   ```

4. **Utilisation dans KibaLock**
   ```python
   # Dans kibalock.py
   custom_model = whisper.load_model("~/kibalock/models/custom_voice.onnx")
   ```

### Scénario 2 : Entraînement de détection faciale custom

**LifeModo peut créer un détecteur de visages optimisé :**

1. **Annotations automatiques** via LifeModo
   ```python
   # LifeModo génère automatiquement des annotations YOLO
   generate_annotations_with_ocr(image_path, label_output_path)
   ```

2. **Entraînement YOLO** sur visages
   ```python
   model = YOLO('yolov8n.pt')
   model.train(
       data='faces_dataset.yaml',
       epochs=100,
       imgsz=640
   )
   ```

3. **Export pour KibaLock**
   ```python
   model.export(format="onnx")
   ```

4. **Intégration dans KibaLock**
   ```python
   face_detector = YOLO("~/kibalock/models/custom_face_detector.onnx")
   ```

---

## 🔧 Configuration de l'intégration

### Fichier de configuration partagé

Créer `config.json` pour les deux systèmes :

```json
{
  "shared": {
    "models_dir": "~/kibalock/models",
    "embeddings_dir": "~/kibalock/embeddings",
    "temp_dir": "~/kibalock/temp"
  },
  "lifemodo": {
    "training_enabled": true,
    "export_formats": ["onnx", "tflite"],
    "auto_export_to_kibalock": true
  },
  "kibalock": {
    "use_custom_models": true,
    "fallback_to_pretrained": true,
    "model_update_check": true
  }
}
```

### Script de synchronisation

```bash
#!/bin/bash
# sync_models.sh - Synchronise les modèles LifeModo → KibaLock

LIFEMODO_EXPORT="~/lifemodo/exported_models"
KIBALOCK_MODELS="~/kibalock/models"

# Copier les modèles
cp $LIFEMODO_EXPORT/*.onnx $KIBALOCK_MODELS/
cp $LIFEMODO_EXPORT/*.tflite $KIBALOCK_MODELS/

echo "✅ Modèles synchronisés"
```

---

## 🚀 Pipeline d'entraînement continu

### Étape 1 : Collecte de données authentiques

```python
# Dans KibaLock, après chaque authentification réussie
def save_training_sample(user_id, voice_path, face_path, quality_score):
    if quality_score > 0.95:  # Seulement les meilleures
        training_dir = "~/lifemodo/dataset/real_auth"
        shutil.copy(voice_path, f"{training_dir}/audio/{user_id}_{timestamp}.wav")
        shutil.copy(face_path, f"{training_dir}/images/{user_id}_{timestamp}.jpg")
```

### Étape 2 : Réentraînement périodique avec LifeModo

```python
# Script automatique (cron job)
def retrain_models():
    # Lancer LifeModo en mode batch
    os.system("python lifemodo.py --batch --auto-train --export")
```

### Étape 3 : Mise à jour KibaLock

```python
# KibaLock détecte automatiquement les nouveaux modèles
def check_model_updates():
    latest_model = get_latest_model("~/kibalock/models")
    if latest_model.timestamp > current_model.timestamp:
        load_new_model(latest_model)
```

---

## 📊 Architecture combinée

```
┌───────────────────────────────────────────────────────────┐
│                 SYSTÈME COMPLET                           │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  📚 LifeModo (Training Pipeline)                          │
│     ├── Upload PDFs, Images, Audio                       │
│     ├── OCR + Annotations automatiques                   │
│     ├── Entraînement YOLO (visages, objets)              │
│     ├── Entraînement Whisper (voix custom)               │
│     ├── Export modèles (.onnx, .tflite, .tfjs)           │
│     └── Stockage dans ~/kibalock/models/                 │
│                                                           │
│  🔐 KibaLock (Production Auth)                            │
│     ├── Charge modèles depuis ~/kibalock/models/         │
│     ├── Inscription utilisateurs (voix + visage)         │
│     ├── Authentification temps réel                      │
│     ├── Stockage embeddings MongoDB                      │
│     └── Feedback qualité → LifeModo                      │
│                                                           │
│  💾 Storage MongoDB                                       │
│     ├── users : Infos utilisateurs                       │
│     ├── embeddings : Vecteurs biométriques               │
│     ├── sessions : Sessions actives                      │
│     └── training_samples : Données pour réentraînement   │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

---

## 🎓 Cas d'usage : Formation continue

### Scénario : Amélioration du modèle vocal d'un utilisateur

1. **Utilisateur se connecte 50 fois** avec KibaLock
2. **KibaLock collecte** 50 échantillons vocaux de qualité
3. **Export automatique** vers LifeModo
4. **LifeModo réentraîne** un modèle personnalisé pour cet utilisateur
5. **KibaLock charge** le nouveau modèle → Meilleure précision

### Code d'implémentation

```python
# Dans KibaLock
class PersonalizedAuth:
    def __init__(self, user_id):
        self.user_id = user_id
        self.personal_model_path = f"~/kibalock/models/user_{user_id}_voice.onnx"
        
        if os.path.exists(self.personal_model_path):
            self.voice_model = load_model(self.personal_model_path)
        else:
            self.voice_model = load_default_whisper()
    
    def authenticate(self, voice_path):
        # Utilise le modèle personnalisé si disponible
        embedding = self.voice_model.extract_embedding(voice_path)
        return embedding
```

---

## 🔄 API d'intégration

### Endpoints pour communication LifeModo ↔ KibaLock

```python
# Dans KibaLock (Future API REST)

@app.post("/api/export_training_data")
def export_training_data(user_id: str, limit: int = 100):
    """Exporte les données d'un utilisateur pour LifeModo"""
    samples = get_user_samples(user_id, limit)
    return {"samples": samples, "count": len(samples)}

@app.post("/api/import_model")
def import_model(model_file: UploadFile):
    """Importe un modèle entraîné par LifeModo"""
    save_path = f"~/kibalock/models/{model_file.filename}"
    save_file(model_file, save_path)
    return {"status": "imported", "path": save_path}

@app.get("/api/model_performance")
def get_model_performance(model_name: str):
    """Retourne les stats de performance d'un modèle"""
    stats = calculate_performance(model_name)
    return stats
```

### Dans LifeModo

```python
@app.post("/api/retrain_request")
def retrain_request(user_id: str, data_source: str):
    """Demande de réentraînement depuis KibaLock"""
    training_job = create_training_job(user_id, data_source)
    return {"job_id": training_job.id, "status": "queued"}

@app.get("/api/model_status/{job_id}")
def get_training_status(job_id: str):
    """Statut d'un job d'entraînement"""
    job = get_job(job_id)
    return {"status": job.status, "progress": job.progress}
```

---

## 🧪 Tests d'intégration

### Test 1 : Export de modèle LifeModo → KibaLock

```bash
#!/bin/bash
# test_model_export.sh

echo "🔬 Test export modèle..."

# 1. Entraîner avec LifeModo
python lifemodo.py --train --export --model voice_test

# 2. Vérifier l'export
ls ~/lifemodo/exported_models/voice_test.onnx

# 3. Copier vers KibaLock
cp ~/lifemodo/exported_models/voice_test.onnx ~/kibalock/models/

# 4. Tester dans KibaLock
python -c "
from kibalock import load_custom_model
model = load_custom_model('voice_test.onnx')
print('✅ Modèle chargé avec succès')
"
```

### Test 2 : Collecte de données KibaLock → LifeModo

```python
# test_data_collection.py

def test_data_export():
    # Simuler 10 authentifications
    for i in range(10):
        user_id = "test_user"
        voice_path = f"test_voice_{i}.wav"
        face_path = f"test_face_{i}.jpg"
        
        # Authentifier
        success, user, scores = verify_user(voice_path, face_path)
        
        # Exporter pour LifeModo si qualité élevée
        if success and scores['combined_score'] > 0.95:
            export_to_lifemodo(user_id, voice_path, face_path)
    
    print("✅ Export de données terminé")
```

---

## 📈 Monitoring de l'intégration

### Dashboard combiné

```python
# dashboard_integration.py

import streamlit as st

st.title("🔗 Dashboard LifeModo ↔ KibaLock")

col1, col2 = st.columns(2)

with col1:
    st.header("📚 LifeModo")
    st.metric("Modèles entraînés", get_trained_models_count())
    st.metric("Modèles exportés", get_exported_models_count())
    st.metric("Datasets disponibles", get_datasets_count())

with col2:
    st.header("🔐 KibaLock")
    st.metric("Modèles actifs", get_active_models_count())
    st.metric("Utilisateurs", get_users_count())
    st.metric("Précision moyenne", f"{get_avg_accuracy()*100:.1f}%")

# Graphique de performance
st.line_chart(get_performance_over_time())
```

---

## 🎯 Best Practices

### 1. Séparation des environnements

- **LifeModo** : Environnement d'entraînement (GPU recommandé)
- **KibaLock** : Environnement de production (CPU suffisant)

### 2. Versioning des modèles

```
~/kibalock/models/
├── voice_v1.0.0.onnx
├── voice_v1.1.0.onnx
├── voice_v2.0.0.onnx (actuel)
├── face_v1.0.0.onnx
└── face_v1.1.0.onnx (actuel)
```

### 3. Rollback automatique

```python
def load_model_with_fallback(model_name):
    try:
        return load_model(f"{model_name}_latest.onnx")
    except Exception as e:
        log_error(f"Erreur chargement modèle: {e}")
        return load_model(f"{model_name}_stable.onnx")
```

### 4. Monitoring de qualité

```python
def monitor_model_quality():
    current_accuracy = calculate_accuracy(last_100_auths())
    
    if current_accuracy < 0.90:
        alert("⚠️ Précision du modèle dégradée, réentraînement recommandé")
        trigger_lifemodo_retrain()
```

---

## 🚀 Commandes rapides

```bash
# Lancer LifeModo pour entraînement
cd ~/KIbalione8/SETRAF/kibalock-api
python lifemodo.py --train --export

# Synchroniser les modèles
./sync_models.sh

# Lancer KibaLock
./launch_kibalock.sh

# Vérifier les modèles
ls -lh ~/kibalock/models/

# Tester un modèle
python -c "from kibalock import test_model; test_model('voice_latest.onnx')"
```

---

## 📚 Ressources

- **LifeModo Documentation** : [lifemodo.py](lifemodo.py)
- **KibaLock Documentation** : [README.md](README.md)
- **Whisper Documentation** : https://github.com/openai/whisper
- **YOLO Documentation** : https://docs.ultralytics.com

---

**Auteur** : Francis Nyundu (BelikanM)  
**Date** : Novembre 2025  
**Version** : 1.0
