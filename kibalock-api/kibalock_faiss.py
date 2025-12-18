# === KibaLock with FAISS Vector Database ===
# Système d'authentification biométrique haute performance
# Utilise FAISS pour la recherche vectorielle ultra-rapide

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
import faiss
import pickle
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# === CONFIGURATION ===
BASE_DIR = os.path.expanduser(os.getenv("BASE_DIR", "~/kibalock"))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_indexes")

for d in [EMBEDDINGS_DIR, TEMP_DIR, LOGS_DIR, FAISS_DIR]:
    os.makedirs(d, exist_ok=True)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[os.getenv("MONGO_DB_NAME", "kibalock")]
users_collection = db["users"]
embeddings_collection = db["embeddings"]
sessions_collection = db["sessions"]

# FAISS Configuration
FAISS_DIMENSION = int(os.getenv("FAISS_DIMENSION", "1792"))  # 1280 (voice) + 512 (face)
USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"

# Index FAISS pour recherche rapide
voice_index = None
face_index = None
combined_index = None
user_id_mapping = {}  # Mapping index_id -> user_id

# === CONFIGURATION DE LA PAGE ===
st.set_page_config(
    page_title="KibaLock FAISS - Authentification Biométrique",
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
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.faiss-badge {
    background: #4CAF50;
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
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
    
    if event_type == "ERROR":
        st.error(f"🔴 [{timestamp}] {message}")
    elif event_type == "WARNING":
        st.warning(f"🟡 [{timestamp}] {message}")
    elif event_type == "SUCCESS":
        st.success(f"🟢 [{timestamp}] {message}")
    else:
        st.info(f"🔵 [{timestamp}] {message}")

# === INITIALISATION FAISS ===
@st.cache_resource
def initialize_faiss_indexes():
    """Initialise les index FAISS pour la recherche vectorielle"""
    global voice_index, face_index, combined_index, user_id_mapping
    
    try:
        # Chemins des index
        voice_index_path = os.path.join(FAISS_DIR, "voice_index.faiss")
        face_index_path = os.path.join(FAISS_DIR, "face_index.faiss")
        combined_index_path = os.path.join(FAISS_DIR, "combined_index.faiss")
        mapping_path = os.path.join(FAISS_DIR, "user_mapping.pkl")
        
        # Charger ou créer les index
        if os.path.exists(voice_index_path):
            voice_index = faiss.read_index(voice_index_path)
            log_event("INFO", f"Index vocal chargé: {voice_index.ntotal} vecteurs")
        else:
            voice_index = faiss.IndexFlatIP(1280)  # Inner Product (cosine similarity)
            log_event("INFO", "Nouvel index vocal créé")
        
        if os.path.exists(face_index_path):
            face_index = faiss.read_index(face_index_path)
            log_event("INFO", f"Index facial chargé: {face_index.ntotal} vecteurs")
        else:
            face_index = faiss.IndexFlatIP(512)
            log_event("INFO", "Nouvel index facial créé")
        
        if os.path.exists(combined_index_path):
            combined_index = faiss.read_index(combined_index_path)
            log_event("INFO", f"Index combiné chargé: {combined_index.ntotal} vecteurs")
        else:
            combined_index = faiss.IndexFlatIP(FAISS_DIMENSION)
            log_event("INFO", "Nouvel index combiné créé")
        
        # Charger le mapping
        if os.path.exists(mapping_path):
            with open(mapping_path, "rb") as f:
                user_id_mapping = pickle.load(f)
            log_event("INFO", f"Mapping chargé: {len(user_id_mapping)} utilisateurs")
        
        return True
    except Exception as e:
        log_event("ERROR", f"Erreur initialisation FAISS: {e}")
        return False

def save_faiss_indexes():
    """Sauvegarde les index FAISS sur disque"""
    try:
        faiss.write_index(voice_index, os.path.join(FAISS_DIR, "voice_index.faiss"))
        faiss.write_index(face_index, os.path.join(FAISS_DIR, "face_index.faiss"))
        faiss.write_index(combined_index, os.path.join(FAISS_DIR, "combined_index.faiss"))
        
        with open(os.path.join(FAISS_DIR, "user_mapping.pkl"), "wb") as f:
            pickle.dump(user_id_mapping, f)
        
        log_event("INFO", "Index FAISS sauvegardés")
        return True
    except Exception as e:
        log_event("ERROR", f"Erreur sauvegarde FAISS: {e}")
        return False

# === MODÈLES IA ===
@st.cache_resource
def load_whisper_model():
    """Charge le modèle Whisper pour la reconnaissance vocale"""
    try:
        model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
        log_event("INFO", "Modèle Whisper chargé avec succès")
        return model
    except Exception as e:
        log_event("ERROR", f"Erreur chargement Whisper: {e}")
        return None

# === EXTRACTION D'EMBEDDINGS VOCAUX ===
def extract_voice_embedding(audio_path):
    """Extrait l'embedding vocal à partir d'un fichier audio"""
    try:
        model = load_whisper_model()
        if model is None:
            return None, "Modèle Whisper non disponible"
        
        result = model.transcribe(audio_path)
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        with torch.no_grad():
            embedding = model.encode(mel.unsqueeze(0)).cpu().numpy().flatten()
        
        embedding = embedding / np.linalg.norm(embedding)
        
        log_event("INFO", f"Embedding vocal extrait: {embedding.shape[0]} dimensions")
        return embedding, result["text"]
    
    except Exception as e:
        log_event("ERROR", f"Erreur extraction voix: {e}")
        return None, str(e)

# === EXTRACTION D'EMBEDDINGS FACIAUX ===
def extract_face_embedding(image_path):
    """Extrait l'embedding facial à partir d'une image"""
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=os.getenv("FACE_MODEL", "Facenet512"),
            enforce_detection=True,
            detector_backend='opencv'
        )
        
        if not embedding_objs:
            return None, "Aucun visage détecté"
        
        embedding = np.array(embedding_objs[0]['embedding'])
        embedding = embedding / np.linalg.norm(embedding)
        
        log_event("INFO", f"Embedding facial extrait: {embedding.shape[0]} dimensions")
        return embedding, "Visage détecté et encodé"
    
    except Exception as e:
        log_event("ERROR", f"Erreur extraction visage: {e}")
        return None, str(e)

# === INSCRIPTION UTILISATEUR AVEC FAISS ===
def register_user_faiss(username, email, voice_samples, face_images):
    """Inscrit un nouvel utilisateur avec FAISS"""
    try:
        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            return False, "Nom d'utilisateur déjà pris"
        
        # Extraire embeddings vocaux
        voice_embeddings = []
        transcriptions = []
        
        for audio_path in voice_samples:
            emb, text = extract_voice_embedding(audio_path)
            if emb is not None:
                voice_embeddings.append(emb)
                transcriptions.append(text)
        
        if not voice_embeddings:
            return False, "Impossible d'extraire l'empreinte vocale"
        
        avg_voice_embedding = np.mean(voice_embeddings, axis=0)
        
        # Extraire embeddings faciaux
        face_embeddings = []
        
        for image_path in face_images:
            emb, msg = extract_face_embedding(image_path)
            if emb is not None:
                face_embeddings.append(emb)
        
        if not face_embeddings:
            return False, "Impossible d'extraire l'empreinte faciale"
        
        avg_face_embedding = np.mean(face_embeddings, axis=0)
        
        # Créer l'embedding combiné
        combined_embedding = np.concatenate([avg_voice_embedding, avg_face_embedding])
        
        # Générer un ID unique
        user_id = hashlib.sha256(f"{username}{email}{time.time()}".encode()).hexdigest()
        
        # Ajouter aux index FAISS
        index_id = voice_index.ntotal
        
        voice_index.add(np.array([avg_voice_embedding], dtype=np.float32))
        face_index.add(np.array([avg_face_embedding], dtype=np.float32))
        combined_index.add(np.array([combined_embedding], dtype=np.float32))
        
        # Mettre à jour le mapping
        user_id_mapping[index_id] = user_id
        
        # Sauvegarder les index
        save_faiss_indexes()
        
        # Enregistrer dans MongoDB
        user_doc = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "created_at": datetime.now(),
            "active": True,
            "login_count": 0,
            "last_login": None,
            "faiss_index_id": index_id
        }
        
        embedding_doc = {
            "user_id": user_id,
            "voice_embedding": avg_voice_embedding.tolist(),
            "face_embedding": avg_face_embedding.tolist(),
            "combined_embedding": combined_embedding.tolist(),
            "voice_samples_count": len(voice_embeddings),
            "face_samples_count": len(face_embeddings),
            "transcriptions": transcriptions,
            "created_at": datetime.now(),
            "faiss_index_id": index_id
        }
        
        users_collection.insert_one(user_doc)
        embeddings_collection.insert_one(embedding_doc)
        
        log_event("SUCCESS", f"Utilisateur {username} enregistré avec succès (FAISS ID: {index_id})", user_id)
        return True, f"Inscription réussie ! ID: {user_id}"
    
    except Exception as e:
        log_event("ERROR", f"Erreur inscription: {e}")
        return False, str(e)

# === VÉRIFICATION UTILISATEUR AVEC FAISS ===
def verify_user_faiss(voice_path, face_path, threshold_voice=0.85, threshold_face=0.90, k=5):
    """Vérifie l'identité d'un utilisateur avec FAISS (recherche ultra-rapide)"""
    try:
        start_time = time.time()
        
        # Extraire embeddings
        voice_emb, transcription = extract_voice_embedding(voice_path)
        if voice_emb is None:
            return False, None, {"error": "Impossible d'analyser la voix"}
        
        face_emb, msg = extract_face_embedding(face_path)
        if face_emb is None:
            return False, None, {"error": "Impossible d'analyser le visage"}
        
        # Recherche FAISS (ultra-rapide)
        voice_emb_np = np.array([voice_emb], dtype=np.float32)
        face_emb_np = np.array([face_emb], dtype=np.float32)
        
        # Rechercher les k plus proches voisins
        voice_distances, voice_indices = voice_index.search(voice_emb_np, k)
        face_distances, face_indices = face_index.search(face_emb_np, k)
        
        # Trouver les candidats communs
        candidates = {}
        for idx, dist in zip(voice_indices[0], voice_distances[0]):
            if idx in user_id_mapping:
                candidates[idx] = {"voice_sim": float(dist), "face_sim": 0.0}
        
        for idx, dist in zip(face_indices[0], face_distances[0]):
            if idx in user_id_mapping:
                if idx in candidates:
                    candidates[idx]["face_sim"] = float(dist)
                else:
                    candidates[idx] = {"voice_sim": 0.0, "face_sim": float(dist)}
        
        # Calculer les scores combinés
        best_match_idx = None
        best_score = 0.0
        best_scores_detail = {}
        
        for idx, scores in candidates.items():
            combined_score = (scores["voice_sim"] * 0.6) + (scores["face_sim"] * 0.4)
            
            if combined_score > best_score and scores["voice_sim"] >= threshold_voice and scores["face_sim"] >= threshold_face:
                best_score = combined_score
                best_match_idx = idx
                best_scores_detail = {
                    "voice_similarity": scores["voice_sim"],
                    "face_similarity": scores["face_sim"],
                    "combined_score": combined_score,
                    "transcription": transcription,
                    "search_time": time.time() - start_time
                }
        
        if best_match_idx is not None and best_match_idx in user_id_mapping:
            user_id = user_id_mapping[best_match_idx]
            user = users_collection.find_one({"user_id": user_id})
            
            if user and user.get('active', True):
                users_collection.update_one(
                    {"user_id": user['user_id']},
                    {
                        "$set": {"last_login": datetime.now()},
                        "$inc": {"login_count": 1}
                    }
                )
                
                session_id = hashlib.sha256(f"{user['user_id']}{time.time()}".encode()).hexdigest()
                session_doc = {
                    "session_id": session_id,
                    "user_id": user['user_id'],
                    "created_at": datetime.now(),
                    "expires_at": datetime.now() + timedelta(hours=24),
                    "scores": best_scores_detail,
                    "method": "FAISS"
                }
                sessions_collection.insert_one(session_doc)
                
                log_event("SUCCESS", f"Connexion réussie pour {user['username']} (FAISS en {best_scores_detail['search_time']:.3f}s)", user['user_id'])
                
                return True, user, best_scores_detail
            else:
                return False, None, {"error": "Compte inactif"}
        
        log_event("WARNING", f"Tentative de connexion échouée - Scores: {best_scores_detail}")
        return False, None, best_scores_detail if best_scores_detail else {"error": "Aucune correspondance trouvée"}
    
    except Exception as e:
        log_event("ERROR", f"Erreur vérification: {e}")
        return False, None, {"error": str(e)}

# === INTERFACE STREAMLIT ===
def main():
    # Initialiser FAISS
    if not initialize_faiss_indexes():
        st.error("❌ Erreur d'initialisation de FAISS")
        return
    
    # En-tête
    st.markdown("""
    <div class="main-header">
        <h1>🔐 KibaLock FAISS - Authentification Biométrique</h1>
        <p>Système d'authentification multimodal par Voix + Visage</p>
        <span class="faiss-badge">⚡ POWERED BY FAISS</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Statistiques
    with st.sidebar:
        st.markdown("### 📊 Statistiques du système")
        
        total_users = users_collection.count_documents({})
        active_sessions = sessions_collection.count_documents({
            "expires_at": {"$gt": datetime.now()}
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👥 Utilisateurs", total_users)
            st.metric("🧬 FAISS Voice", voice_index.ntotal if voice_index else 0)
        with col2:
            st.metric("🔓 Sessions", active_sessions)
            st.metric("📸 FAISS Face", face_index.ntotal if face_index else 0)
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        voice_threshold = st.slider("Seuil voix", 0.0, 1.0, float(os.getenv("VOICE_THRESHOLD", "0.85")), 0.01)
        face_threshold = st.slider("Seuil visage", 0.0, 1.0, float(os.getenv("FACE_THRESHOLD", "0.90")), 0.01)
        k_neighbors = st.slider("Candidats FAISS", 1, 20, 5, 1)
    
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
            
            face_files = st.file_uploader(
                "🖼️ Téléversez vos photos",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="reg_face"
            )
            
            if face_files:
                st.success(f"✅ {len(face_files)} photo(s) chargée(s)")
                cols = st.columns(min(len(face_files), 3))
                for idx, img_file in enumerate(face_files[:3]):
                    with cols[idx]:
                        st.image(img_file, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Bouton d'inscription
        st.markdown("---")
        if st.button("✅ Finaliser l'inscription (FAISS)", type="primary", use_container_width=True):
            if not username or not email:
                st.error("❌ Veuillez remplir tous les champs")
            elif not audio_files or len(audio_files) < 2:
                st.error("❌ Veuillez fournir au moins 2 échantillons vocaux")
            elif not face_files or len(face_files) < 2:
                st.error("❌ Veuillez fournir au moins 2 photos faciales")
            else:
                with st.spinner("🔄 Traitement des données biométriques..."):
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
                    
                    success, message = register_user_faiss(username, email, audio_paths, face_paths)
                    
                    for path in audio_paths + face_paths:
                        try:
                            os.remove(path)
                        except:
                            pass
                    
                    if success:
                        st.balloons()
                        st.success(f"🎉 {message}")
                        st.info(f"⚡ Index FAISS mis à jour: {voice_index.ntotal} vecteurs")
                    else:
                        st.error(f"❌ {message}")
    
    # === ONGLET CONNEXION ===
    with tab2:
        st.header("🔑 Connexion biométrique FAISS")
        
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
        if st.button("🔓 Se connecter (FAISS)", type="primary", use_container_width=True):
            if not login_audio or not login_face:
                st.error("❌ Veuillez fournir votre voix ET votre visage")
            else:
                with st.spinner("🔄 Vérification en cours..."):
                    audio_path = os.path.join(TEMP_DIR, f"login_voice_{time.time()}.wav")
                    face_path = os.path.join(TEMP_DIR, f"login_face_{time.time()}.jpg")
                    
                    with open(audio_path, "wb") as f:
                        f.write(login_audio.read())
                    with open(face_path, "wb") as f:
                        f.write(login_face.read())
                    
                    success, user, scores = verify_user_faiss(
                        audio_path,
                        face_path,
                        voice_threshold,
                        face_threshold,
                        k_neighbors
                    )
                    
                    try:
                        os.remove(audio_path)
                        os.remove(face_path)
                    except:
                        pass
                    
                    if success:
                        st.balloons()
                        st.success(f"🎉 Bienvenue {user['username']} !")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎤 Score vocal", f"{scores['voice_similarity']*100:.1f}%")
                        with col2:
                            st.metric("📸 Score facial", f"{scores['face_similarity']*100:.1f}%")
                        with col3:
                            st.metric("🔐 Score combiné", f"{scores['combined_score']*100:.1f}%")
                        with col4:
                            st.metric("⚡ Temps FAISS", f"{scores['search_time']:.3f}s")
                        
                        st.json({
                            "user_id": user['user_id'],
                            "username": user['username'],
                            "email": user['email'],
                            "login_count": user['login_count'],
                            "transcription": scores.get('transcription', ''),
                            "method": "FAISS Ultra-Fast Search"
                        })
                    else:
                        st.error("❌ Authentification échouée")
                        if 'voice_similarity' in scores:
                            st.warning(f"🎤 Score vocal: {scores['voice_similarity']*100:.1f}% (seuil: {voice_threshold*100:.1f}%)")
                        if 'face_similarity' in scores:
                            st.warning(f"📸 Score facial: {scores['face_similarity']*100:.1f}% (seuil: {face_threshold*100:.1f}%)")
                        if 'search_time' in scores:
                            st.info(f"⚡ Temps de recherche FAISS: {scores['search_time']:.3f}s")
    
    # === ONGLET UTILISATEURS ===
    with tab3:
        st.header("👥 Gestion des utilisateurs")
        
        users = list(users_collection.find().sort("created_at", -1))
        
        if users:
            for user in users:
                with st.expander(f"👤 {user['username']} ({user['email']})"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🔑 Connexions", user.get('login_count', 0))
                    with col2:
                        status = "🟢 Actif" if user.get('active', True) else "🔴 Inactif"
                        st.write(f"**Statut:** {status}")
                    with col3:
                        faiss_id = user.get('faiss_index_id', 'N/A')
                        st.write(f"**FAISS ID:** {faiss_id}")
                    with col4:
                        created = user['created_at'].strftime("%Y-%m-%d %H:%M")
                        st.write(f"**Créé:** {created}")
                    
                    if user.get('last_login'):
                        last = user['last_login'].strftime("%Y-%m-%d %H:%M")
                        st.info(f"📅 Dernière connexion: {last}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🗑️ Supprimer", key=f"del_{user['user_id']}"):
                            users_collection.delete_one({"user_id": user['user_id']})
                            embeddings_collection.delete_one({"user_id": user['user_id']})
                            st.success("✅ Utilisateur supprimé (Note: Index FAISS à reconstruire)")
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
        
        # Statistiques FAISS
        st.subheader("⚡ Performances FAISS")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎤 Vecteurs vocaux", voice_index.ntotal if voice_index else 0)
        with col2:
            st.metric("📸 Vecteurs faciaux", face_index.ntotal if face_index else 0)
        with col3:
            st.metric("🧬 Vecteurs combinés", combined_index.ntotal if combined_index else 0)
        
        st.markdown("---")
        
        # Logs récents
        st.subheader("📋 Logs récents")
        
        log_file = os.path.join(LOGS_DIR, f"kibalock_{datetime.now().strftime('%Y%m%d')}.log")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = f.readlines()[-20:]
            
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
