# === KibaLock Biometric Authentication API ===
# Système d'authentification multimodal (Voix + Visage)
# Basé sur LifeModo - Version Streamlit complète

import os
import time
import json
import hashlib
import numpy as np
import cv2
import streamlit as st
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient
import whisper
import torch
from deepface import DeepFace
from scipy.spatial.distance import cosine
import threading
import queue

# === CONFIGURATION ===
BASE_DIR = os.path.expanduser("~/kibalock")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

for d in [EMBEDDINGS_DIR, TEMP_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://SETRAF:setraf2025@cluster0.5tjz9v0.mongodb.net/myDatabase10")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["kibalock"]
users_collection = db["users"]
embeddings_collection = db["embeddings"]
sessions_collection = db["sessions"]

# Créer les index vectoriels
try:
    embeddings_collection.create_index([("voice_embedding", "2dsphere")])
    embeddings_collection.create_index([("face_embedding", "2dsphere")])
except:
    pass

# === CONFIGURATION DE LA PAGE ===
st.set_page_config(
    page_title="KibaLock - Authentification Biométrique",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === STYLE CSS ===
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}
.biometric-card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 15px 0;
}
.success-box {
    background: #d4edda;
    border-left: 5px solid #28a745;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
}
.warning-box {
    background: #fff3cd;
    border-left: 5px solid #ffc107;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
}
.error-box {
    background: #f8d7da;
    border-left: 5px solid #dc3545;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# === LOGS ===
def log_event(event_type, message, user_id=None):
    """Enregistre les événements du système"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "message": message,
        "user_id": user_id
    }
    
    log_file = os.path.join(LOGS_DIR, f"kibalock_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # Affichage dans Streamlit
    if event_type == "ERROR":
        st.error(f"🔴 [{timestamp}] {message}")
    elif event_type == "WARNING":
        st.warning(f"🟡 [{timestamp}] {message}")
    elif event_type == "SUCCESS":
        st.success(f"🟢 [{timestamp}] {message}")
    else:
        st.info(f"🔵 [{timestamp}] {message}")

# === MODÈLES IA ===
@st.cache_resource
def load_whisper_model():
    """Charge le modèle Whisper pour la reconnaissance vocale"""
    try:
        model = whisper.load_model("base")
        log_event("INFO", "Modèle Whisper chargé avec succès")
        return model
    except Exception as e:
        log_event("ERROR", f"Erreur chargement Whisper: {e}")
        return None

# === EXTRACTION D'EMBEDDINGS VOCAUX ===
def extract_voice_embedding(audio_path):
    """
    Extrait l'embedding vocal à partir d'un fichier audio
    Utilise Whisper pour transcription + extraction de features
    """
    try:
        model = load_whisper_model()
        if model is None:
            return None, "Modèle Whisper non disponible"
        
        # Transcription et extraction
        result = model.transcribe(audio_path)
        
        # Extraction de features audio (mel spectrogram)
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Encoder les features
        with torch.no_grad():
            embedding = model.encode(mel.unsqueeze(0)).cpu().numpy().flatten()
        
        # Normaliser l'embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        log_event("INFO", f"Embedding vocal extrait: {embedding.shape[0]} dimensions")
        return embedding.tolist(), result["text"]
    
    except Exception as e:
        log_event("ERROR", f"Erreur extraction voix: {e}")
        return None, str(e)

# === EXTRACTION D'EMBEDDINGS FACIAUX ===
def extract_face_embedding(image_path):
    """
    Extrait l'embedding facial à partir d'une image
    Utilise DeepFace avec FaceNet
    """
    try:
        # Détecter et extraire l'embedding
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name='Facenet512',
            enforce_detection=True,
            detector_backend='opencv'
        )
        
        if not embedding_objs:
            return None, "Aucun visage détecté"
        
        # Prendre le premier visage détecté
        embedding = np.array(embedding_objs[0]['embedding'])
        
        # Normaliser
        embedding = embedding / np.linalg.norm(embedding)
        
        log_event("INFO", f"Embedding facial extrait: {embedding.shape[0]} dimensions")
        return embedding.tolist(), "Visage détecté et encodé"
    
    except Exception as e:
        log_event("ERROR", f"Erreur extraction visage: {e}")
        return None, str(e)

# === CALCUL DE SIMILARITÉ ===
def calculate_similarity(emb1, emb2):
    """Calcule la similarité cosinus entre deux embeddings"""
    try:
        similarity = 1 - cosine(emb1, emb2)
        return similarity
    except Exception as e:
        log_event("ERROR", f"Erreur calcul similarité: {e}")
        return 0.0

# === INSCRIPTION UTILISATEUR ===
def register_user(username, email, voice_samples, face_images):
    """
    Inscrit un nouvel utilisateur avec ses données biométriques
    
    Args:
        username: Nom d'utilisateur unique
        email: Email de l'utilisateur
        voice_samples: Liste de chemins vers fichiers audio
        face_images: Liste de chemins vers images faciales
    """
    try:
        # Vérifier si l'utilisateur existe déjà
        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            return False, "Nom d'utilisateur déjà pris"
        
        # Extraire les embeddings vocaux (moyenne de plusieurs échantillons)
        voice_embeddings = []
        transcriptions = []
        
        for audio_path in voice_samples:
            emb, text = extract_voice_embedding(audio_path)
            if emb:
                voice_embeddings.append(emb)
                transcriptions.append(text)
        
        if not voice_embeddings:
            return False, "Impossible d'extraire l'empreinte vocale"
        
        # Moyenne des embeddings vocaux
        avg_voice_embedding = np.mean(voice_embeddings, axis=0).tolist()
        
        # Extraire les embeddings faciaux (moyenne de plusieurs angles)
        face_embeddings = []
        
        for image_path in face_images:
            emb, msg = extract_face_embedding(image_path)
            if emb:
                face_embeddings.append(emb)
        
        if not face_embeddings:
            return False, "Impossible d'extraire l'empreinte faciale"
        
        # Moyenne des embeddings faciaux
        avg_face_embedding = np.mean(face_embeddings, axis=0).tolist()
        
        # Créer l'embedding combiné
        combined_embedding = avg_voice_embedding + avg_face_embedding
        
        # Générer un ID unique
        user_id = hashlib.sha256(f"{username}{email}{time.time()}".encode()).hexdigest()
        
        # Enregistrer dans MongoDB
        user_doc = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "created_at": datetime.now(),
            "active": True,
            "login_count": 0,
            "last_login": None
        }
        
        embedding_doc = {
            "user_id": user_id,
            "voice_embedding": avg_voice_embedding,
            "face_embedding": avg_face_embedding,
            "combined_embedding": combined_embedding,
            "voice_samples_count": len(voice_embeddings),
            "face_samples_count": len(face_embeddings),
            "transcriptions": transcriptions,
            "created_at": datetime.now()
        }
        
        users_collection.insert_one(user_doc)
        embeddings_collection.insert_one(embedding_doc)
        
        log_event("SUCCESS", f"Utilisateur {username} enregistré avec succès", user_id)
        return True, f"Inscription réussie ! ID: {user_id}"
    
    except Exception as e:
        log_event("ERROR", f"Erreur inscription: {e}")
        return False, str(e)

# === VÉRIFICATION UTILISATEUR ===
def verify_user(voice_path, face_path, threshold_voice=0.85, threshold_face=0.90):
    """
    Vérifie l'identité d'un utilisateur par voix + visage
    
    Args:
        voice_path: Chemin vers fichier audio
        face_path: Chemin vers image faciale
        threshold_voice: Seuil de similarité vocale (défaut: 0.85)
        threshold_face: Seuil de similarité faciale (défaut: 0.90)
    
    Returns:
        success, user_info, scores
    """
    try:
        # Extraire les embeddings de la tentative de connexion
        voice_emb, transcription = extract_voice_embedding(voice_path)
        if not voice_emb:
            return False, None, {"error": "Impossible d'analyser la voix"}
        
        face_emb, msg = extract_face_embedding(face_path)
        if not face_emb:
            return False, None, {"error": "Impossible d'analyser le visage"}
        
        # Rechercher dans la base de données
        best_match = None
        best_score = 0.0
        best_scores_detail = {}
        
        all_embeddings = embeddings_collection.find()
        
        for user_emb in all_embeddings:
            # Calculer les similarités
            voice_sim = calculate_similarity(voice_emb, user_emb['voice_embedding'])
            face_sim = calculate_similarity(face_emb, user_emb['face_embedding'])
            
            # Score combiné (pondération: 60% voix, 40% visage)
            combined_score = (voice_sim * 0.6) + (face_sim * 0.4)
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = user_emb
                best_scores_detail = {
                    "voice_similarity": voice_sim,
                    "face_similarity": face_sim,
                    "combined_score": combined_score,
                    "transcription": transcription
                }
        
        # Vérifier les seuils
        if best_match and best_scores_detail["voice_similarity"] >= threshold_voice and best_scores_detail["face_similarity"] >= threshold_face:
            # Récupérer les infos utilisateur
            user = users_collection.find_one({"user_id": best_match['user_id']})
            
            if user and user.get('active', True):
                # Mettre à jour les stats de connexion
                users_collection.update_one(
                    {"user_id": user['user_id']},
                    {
                        "$set": {"last_login": datetime.now()},
                        "$inc": {"login_count": 1}
                    }
                )
                
                # Créer une session
                session_id = hashlib.sha256(f"{user['user_id']}{time.time()}".encode()).hexdigest()
                session_doc = {
                    "session_id": session_id,
                    "user_id": user['user_id'],
                    "created_at": datetime.now(),
                    "expires_at": datetime.now() + timedelta(hours=24),
                    "scores": best_scores_detail
                }
                sessions_collection.insert_one(session_doc)
                
                log_event("SUCCESS", f"Connexion réussie pour {user['username']}", user['user_id'])
                
                return True, user, best_scores_detail
            else:
                return False, None, {"error": "Compte inactif"}
        
        log_event("WARNING", f"Tentative de connexion échouée - Scores: {best_scores_detail}")
        return False, None, best_scores_detail
    
    except Exception as e:
        log_event("ERROR", f"Erreur vérification: {e}")
        return False, None, {"error": str(e)}

# === INTERFACE STREAMLIT ===
def main():
    # En-tête
    st.markdown("""
    <div class="main-header">
        <h1>🔐 KibaLock - Authentification Biométrique</h1>
        <p>Système d'authentification multimodal par Voix + Visage</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Statistiques
    with st.sidebar:
        st.markdown("### 📊 Statistiques du système")
        
        total_users = users_collection.count_documents({})
        active_sessions = sessions_collection.count_documents({
            "expires_at": {"$gt": datetime.now()}
        })
        total_embeddings = embeddings_collection.count_documents({})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Utilisateurs", total_users)
        with col2:
            st.metric("🔓 Sessions", active_sessions)
        with col3:
            st.metric("🧬 Embeddings", total_embeddings)
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        voice_threshold = st.slider("Seuil voix", 0.0, 1.0, 0.85, 0.01)
        face_threshold = st.slider("Seuil visage", 0.0, 1.0, 0.90, 0.01)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Inscription",
        "🔑 Connexion",
        "👥 Utilisateurs",
        "📈 Monitoring"
    ])
    
    # === ONGLET INSCRIPTION ===
    with tab1:
        st.header("📝 Inscription d'un nouvel utilisateur")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="biometric-card">', unsafe_allow_html=True)
            st.subheader("📋 Informations")
            username = st.text_input("👤 Nom d'utilisateur", key="reg_username")
            email = st.text_input("📧 Email", key="reg_email")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="biometric-card">', unsafe_allow_html=True)
            st.subheader("🎤 Capture vocale")
            st.info("Enregistrez 3 échantillons vocaux (10-15 secondes chacun)")
            
            # Upload audio
            audio_files = st.file_uploader(
                "🎵 Téléversez vos échantillons vocaux",
                type=["wav", "mp3", "ogg"],
                accept_multiple_files=True,
                key="reg_audio"
            )
            
            if audio_files:
                st.success(f"✅ {len(audio_files)} échantillon(s) vocal(aux) chargé(s)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="biometric-card">', unsafe_allow_html=True)
            st.subheader("📸 Capture faciale")
            st.info("Capturez 3-5 photos de votre visage sous différents angles")
            
            # Upload images
            face_files = st.file_uploader(
                "🖼️ Téléversez vos photos",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="reg_face"
            )
            
            if face_files:
                st.success(f"✅ {len(face_files)} photo(s) chargée(s)")
                # Prévisualisation
                cols = st.columns(min(len(face_files), 3))
                for idx, img_file in enumerate(face_files[:3]):
                    with cols[idx]:
                        st.image(img_file, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Bouton d'inscription
        st.markdown("---")
        if st.button("✅ Finaliser l'inscription", type="primary", use_container_width=True):
            if not username or not email:
                st.error("❌ Veuillez remplir tous les champs")
            elif not audio_files or len(audio_files) < 2:
                st.error("❌ Veuillez fournir au moins 2 échantillons vocaux")
            elif not face_files or len(face_files) < 2:
                st.error("❌ Veuillez fournir au moins 2 photos faciales")
            else:
                with st.spinner("🔄 Traitement des données biométriques..."):
                    # Sauvegarder temporairement
                    audio_paths = []
                    for audio_file in audio_files:
                        audio_path = os.path.join(TEMP_DIR, f"voice_{time.time()}_{audio_file.name}")
                        with open(audio_path, "wb") as f:
                            f.write(audio_file.read())
                        audio_paths.append(audio_path)
                    
                    face_paths = []
                    for face_file in face_files:
                        face_path = os.path.join(TEMP_DIR, f"face_{time.time()}_{face_file.name}")
                        with open(face_path, "wb") as f:
                            f.write(face_file.read())
                        face_paths.append(face_path)
                    
                    # Inscription
                    success, message = register_user(username, email, audio_paths, face_paths)
                    
                    # Nettoyage
                    for path in audio_paths + face_paths:
                        try:
                            os.remove(path)
                        except:
                            pass
                    
                    if success:
                        st.balloons()
                        st.success(f"🎉 {message}")
                    else:
                        st.error(f"❌ {message}")
    
    # === ONGLET CONNEXION ===
    with tab2:
        st.header("🔑 Connexion biométrique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="biometric-card">', unsafe_allow_html=True)
            st.subheader("🎤 Vérification vocale")
            st.info("Enregistrez une phrase d'identification (10-15 secondes)")
            
            login_audio = st.file_uploader(
                "🎵 Téléversez votre échantillon vocal",
                type=["wav", "mp3", "ogg"],
                key="login_audio"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="biometric-card">', unsafe_allow_html=True)
            st.subheader("📸 Vérification faciale")
            st.info("Capturez une photo de votre visage")
            
            login_face = st.file_uploader(
                "🖼️ Téléversez votre photo",
                type=["jpg", "jpeg", "png"],
                key="login_face"
            )
            
            if login_face:
                st.image(login_face, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Bouton de connexion
        st.markdown("---")
        if st.button("🔓 Se connecter", type="primary", use_container_width=True):
            if not login_audio or not login_face:
                st.error("❌ Veuillez fournir votre voix ET votre visage")
            else:
                with st.spinner("🔄 Vérification en cours..."):
                    # Sauvegarder temporairement
                    audio_path = os.path.join(TEMP_DIR, f"login_voice_{time.time()}.wav")
                    face_path = os.path.join(TEMP_DIR, f"login_face_{time.time()}.jpg")
                    
                    with open(audio_path, "wb") as f:
                        f.write(login_audio.read())
                    with open(face_path, "wb") as f:
                        f.write(login_face.read())
                    
                    # Vérification
                    success, user, scores = verify_user(
                        audio_path,
                        face_path,
                        voice_threshold,
                        face_threshold
                    )
                    
                    # Nettoyage
                    try:
                        os.remove(audio_path)
                        os.remove(face_path)
                    except:
                        pass
                    
                    if success:
                        st.balloons()
                        st.success(f"🎉 Bienvenue {user['username']} !")
                        
                        # Afficher les scores
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🎤 Score vocal", f"{scores['voice_similarity']*100:.1f}%")
                        with col2:
                            st.metric("📸 Score facial", f"{scores['face_similarity']*100:.1f}%")
                        with col3:
                            st.metric("🔐 Score combiné", f"{scores['combined_score']*100:.1f}%")
                        
                        # Infos utilisateur
                        st.json({
                            "user_id": user['user_id'],
                            "username": user['username'],
                            "email": user['email'],
                            "login_count": user['login_count'],
                            "transcription": scores.get('transcription', '')
                        })
                    else:
                        st.error("❌ Authentification échouée")
                        if 'voice_similarity' in scores:
                            st.warning(f"🎤 Score vocal: {scores['voice_similarity']*100:.1f}% (seuil: {voice_threshold*100:.1f}%)")
                        if 'face_similarity' in scores:
                            st.warning(f"📸 Score facial: {scores['face_similarity']*100:.1f}% (seuil: {face_threshold*100:.1f}%)")
    
    # === ONGLET UTILISATEURS ===
    with tab3:
        st.header("👥 Gestion des utilisateurs")
        
        users = list(users_collection.find().sort("created_at", -1))
        
        if users:
            for user in users:
                with st.expander(f"👤 {user['username']} ({user['email']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🔑 Connexions", user.get('login_count', 0))
                    with col2:
                        status = "🟢 Actif" if user.get('active', True) else "🔴 Inactif"
                        st.write(f"**Statut:** {status}")
                    with col3:
                        created = user['created_at'].strftime("%Y-%m-%d %H:%M")
                        st.write(f"**Créé:** {created}")
                    
                    # Infos détaillées
                    if user.get('last_login'):
                        last = user['last_login'].strftime("%Y-%m-%d %H:%M")
                        st.info(f"📅 Dernière connexion: {last}")
                    
                    # Actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🗑️ Supprimer", key=f"del_{user['user_id']}"):
                            users_collection.delete_one({"user_id": user['user_id']})
                            embeddings_collection.delete_one({"user_id": user['user_id']})
                            st.success("✅ Utilisateur supprimé")
                            st.rerun()
                    with col2:
                        if st.button(f"🔄 {'Désactiver' if user.get('active', True) else 'Activer'}", key=f"toggle_{user['user_id']}"):
                            new_status = not user.get('active', True)
                            users_collection.update_one(
                                {"user_id": user['user_id']},
                                {"$set": {"active": new_status}}
                            )
                            st.success(f"✅ Statut mis à jour")
                            st.rerun()
        else:
            st.info("Aucun utilisateur enregistré")
    
    # === ONGLET MONITORING ===
    with tab4:
        st.header("📈 Monitoring du système")
        
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("👥 Utilisateurs totaux", total_users)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            active_users = users_collection.count_documents({"active": True})
            st.metric("✅ Utilisateurs actifs", active_users)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🔓 Sessions actives", active_sessions)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            total_logins = sum([u.get('login_count', 0) for u in users_collection.find()])
            st.metric("🔐 Connexions totales", total_logins)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Logs récents
        st.subheader("📋 Logs récents")
        
        log_file = os.path.join(LOGS_DIR, f"kibalock_{datetime.now().strftime('%Y%m%d')}.log")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = f.readlines()[-20:]  # 20 derniers logs
            
            for log in reversed(logs):
                try:
                    log_data = json.loads(log)
                    event_type = log_data['event_type']
                    
                    if event_type == "ERROR":
                        st.error(f"🔴 [{log_data['timestamp']}] {log_data['message']}")
                    elif event_type == "WARNING":
                        st.warning(f"🟡 [{log_data['timestamp']}] {log_data['message']}")
                    elif event_type == "SUCCESS":
                        st.success(f"🟢 [{log_data['timestamp']}] {log_data['message']}")
                    else:
                        st.info(f"🔵 [{log_data['timestamp']}] {log_data['message']}")
                except:
                    pass
        else:
            st.info("Aucun log disponible aujourd'hui")

if __name__ == "__main__":
    main()
