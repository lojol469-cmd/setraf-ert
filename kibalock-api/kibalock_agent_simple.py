#!/usr/bin/env python3
"""
KibaLock Agent Simple - Utilise le pattern exact d'ERT.py
Agent direct sans complexité, juste des outils + appels directs
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import logging

# Configuration des caches (pattern ERT.py)
USER_HOME = os.path.expanduser("~")
os.environ['HF_HOME'] = os.path.join(USER_HOME, '.cache', 'huggingface')

# Imports pour biométrie
import faiss
import whisper
from pymongo import MongoClient
from PIL import Image

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

###############################################################################
# Configuration
###############################################################################

WHISPER_MODEL_NAME = "base"
FAISS_INDEX_DIR = os.path.expanduser("~/kibalock/faiss_indexes")
AUDIO_SAMPLES_DIR = os.path.expanduser("~/kibalock/audio_samples")
FACE_IMAGES_DIR = os.path.expanduser("~/kibalock/face_images")

# MongoDB
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://SETRAF:Dieu19961991%3F%3F%21%3F%3F%21@cluster0.5tjz9v0.mongodb.net/kibalock?retryWrites=true&w=majority"
)

###############################################################################
# Chargement des ressources (pattern ERT.py - lazy loading)
###############################################################################

class KibaLockResources:
    """Singleton pour ressources partagées"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KibaLockResources, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("Initialisation des ressources KibaLock...")
            self._initialize()
            self._initialized = True
    
    def _initialize(self):
        """Initialisation paresseuse"""
        try:
            # Whisper
            logger.info(f"Chargement Whisper {WHISPER_MODEL_NAME}...")
            self.whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
            logger.info("✓ Whisper chargé")
            
            # FAISS
            logger.info("Chargement des index FAISS...")
            self.faiss_indexes = self._load_faiss_indexes()
            logger.info(f"✓ {len(self.faiss_indexes)} index FAISS chargés")
            
            # MongoDB
            logger.info("Connexion MongoDB...")
            self.mongo_client = MongoClient(MONGODB_URI)
            self.db = self.mongo_client.kibalock
            logger.info("✓ MongoDB connecté")
            
        except Exception as e:
            logger.error(f"Erreur initialisation: {e}")
            raise
    
    def _load_faiss_indexes(self):
        """Charger les index FAISS"""
        indexes = {}
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        
        index_files = {
            'voice': os.path.join(FAISS_INDEX_DIR, 'voice_index.faiss'),
            'face': os.path.join(FAISS_INDEX_DIR, 'face_index.faiss'),
            'combined': os.path.join(FAISS_INDEX_DIR, 'combined_index.faiss')
        }
        
        for name, path in index_files.items():
            if os.path.exists(path):
                indexes[name] = faiss.read_index(path)
                logger.info(f"  - {name}: {indexes[name].ntotal} vecteurs")
            else:
                dim = 1280 if name == 'voice' else (512 if name == 'face' else 1792)
                indexes[name] = faiss.IndexFlatIP(dim)
                logger.info(f"  - {name}: nouvel index (dim={dim})")
        
        return indexes

###############################################################################
# Outils (pattern ERT.py - fonctions simples)
###############################################################################

def register_user_tool(user_data: str) -> str:
    """Enregistrer un nouvel utilisateur"""
    try:
        data = json.loads(user_data)
        username = data.get('username')
        email = data.get('email')
        
        if not username or not email:
            return "❌ Erreur: username et email requis"
        
        resources = KibaLockResources()
        
        if resources.db.users.find_one({'username': username}):
            return f"❌ Utilisateur '{username}' existe déjà"
        
        user_doc = {
            'username': username,
            'email': email,
            'created_at': datetime.utcnow(),
            'voice_embedding': None,
            'face_embedding': None
        }
        
        result = resources.db.users.insert_one(user_doc)
        return f"✅ Utilisateur '{username}' créé!\nID: {str(result.inserted_id)}"
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"


def verify_user_tool(username: str) -> str:
    """Vérifier un utilisateur"""
    try:
        resources = KibaLockResources()
        user = resources.db.users.find_one({'username': username})
        
        if not user:
            return f"❌ Utilisateur '{username}' non trouvé"
        
        return f"""✅ Utilisateur: {username}
📧 Email: {user.get('email', 'N/A')}
📅 Créé: {user.get('created_at', 'N/A')}
🎤 Voix: {'✓' if user.get('voice_embedding') else '✗'}
📸 Visage: {'✓' if user.get('face_embedding') else '✗'}"""
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"


def list_users_tool(limit: str = "10") -> str:
    """Lister les utilisateurs"""
    try:
        resources = KibaLockResources()
        limit_int = int(limit) if limit.isdigit() else 10
        users = list(resources.db.users.find().limit(limit_int))
        
        if not users:
            return "📋 Aucun utilisateur"
        
        result = f"📋 {len(users)} utilisateur(s):\n\n"
        for i, user in enumerate(users, 1):
            voice = "✓" if user.get('voice_embedding') else "✗"
            face = "✓" if user.get('face_embedding') else "✗"
            result += f"{i}. {user['username']} ({user['email']}) - Voix:{voice} Visage:{face}\n"
        
        return result.strip()
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"


def search_similar_voice_tool(username: str) -> str:
    """Rechercher voix similaires"""
    try:
        resources = KibaLockResources()
        user = resources.db.users.find_one({'username': username})
        
        if not user:
            return f"❌ Utilisateur '{username}' non trouvé"
        
        voice_emb = user.get('voice_embedding')
        if not voice_emb:
            return f"❌ Pas d'embedding vocal pour '{username}'"
        
        query_vector = np.array(voice_emb, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        voice_index = resources.faiss_indexes.get('voice')
        if voice_index.ntotal == 0:
            return "📊 Index vocal vide"
        
        k = min(5, voice_index.ntotal)
        distances, indices = voice_index.search(query_vector, k)
        
        result = f"🔍 Top {k} voix similaires:\n\n"
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            similarity = float(dist) * 100
            result += f"{i}. Index {idx} - Similarité: {similarity:.2f}%\n"
        
        return result.strip()
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"


def get_statistics_tool(dummy: str = "") -> str:
    """Obtenir statistiques KibaLock"""
    try:
        resources = KibaLockResources()
        
        total_users = resources.db.users.count_documents({})
        users_with_voice = resources.db.users.count_documents({'voice_embedding': {'$ne': None}})
        users_with_face = resources.db.users.count_documents({'face_embedding': {'$ne': None}})
        users_complete = resources.db.users.count_documents({
            'voice_embedding': {'$ne': None},
            'face_embedding': {'$ne': None}
        })
        
        voice_vectors = resources.faiss_indexes['voice'].ntotal
        face_vectors = resources.faiss_indexes['face'].ntotal
        combined_vectors = resources.faiss_indexes['combined'].ntotal
        
        audio_files = len(os.listdir(AUDIO_SAMPLES_DIR)) if os.path.exists(AUDIO_SAMPLES_DIR) else 0
        face_files = len(os.listdir(FACE_IMAGES_DIR)) if os.path.exists(FACE_IMAGES_DIR) else 0
        
        return f"""📊 Statistiques KibaLock
{'='*40}

👥 Utilisateurs:
  • Total: {total_users}
  • Avec voix: {users_with_voice}
  • Avec visage: {users_with_face}
  • Profil complet: {users_complete}

🔢 Index FAISS:
  • Vecteurs voix: {voice_vectors}
  • Vecteurs visage: {face_vectors}
  • Vecteurs combinés: {combined_vectors}

💾 Stockage:
  • Audio samples: {audio_files}
  • Images visage: {face_files}"""
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"


def delete_user_tool(username: str) -> str:
    """Supprimer un utilisateur"""
    try:
        resources = KibaLockResources()
        
        user = resources.db.users.find_one({'username': username})
        if not user:
            return f"❌ Utilisateur '{username}' non trouvé"
        
        result = resources.db.users.delete_one({'username': username})
        
        if result.deleted_count > 0:
            return f"✅ Utilisateur '{username}' supprimé"
        else:
            return f"❌ Échec suppression '{username}'"
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"


###############################################################################
# Dictionnaire des outils (pattern ERT.py simple)
###############################################################################

TOOLS = {
    'register_user': {
        'func': register_user_tool,
        'description': 'Enregistrer un nouvel utilisateur. Input: JSON {"username": "...", "email": "..."}'
    },
    'verify_user': {
        'func': verify_user_tool,
        'description': 'Vérifier un utilisateur. Input: username'
    },
    'list_users': {
        'func': list_users_tool,
        'description': 'Lister les utilisateurs. Input: limit (nombre)'
    },
    'search_similar_voice': {
        'func': search_similar_voice_tool,
        'description': 'Chercher voix similaires. Input: username'
    },
    'get_statistics': {
        'func': get_statistics_tool,
        'description': 'Statistiques complètes. No input.'
    },
    'delete_user': {
        'func': delete_user_tool,
        'description': 'Supprimer un utilisateur. Input: username'
    }
}


def call_tool(tool_name: str, tool_input: str = "") -> str:
    """Appeler un outil (pattern ERT.py)"""
    if tool_name not in TOOLS:
        return f"❌ Outil '{tool_name}' inconnu"
    
    try:
        return TOOLS[tool_name]['func'](tool_input)
    except Exception as e:
        logger.error(f"Erreur outil {tool_name}: {e}")
        return f"❌ Erreur: {str(e)}"


###############################################################################
# Interface CLI interactive
###############################################################################

def main():
    """Interface interactive"""
    print("\n" + "="*60)
    print("🤖 KibaLock Agent Simple - Mode CLI")
    print("="*60)
    print("Pattern: ERT.py (outils directs)")
    print("="*60 + "\n")
    
    # Initialiser les ressources
    try:
        resources = KibaLockResources()
        print("✅ Ressources chargées!\n")
    except Exception as e:
        print(f"❌ Erreur chargement: {e}\n")
        return
    
    # Afficher les outils
    print("🔧 Outils disponibles:")
    for i, (name, tool) in enumerate(TOOLS.items(), 1):
        print(f"  {i}. {name:25s} - {tool['description'][:50]}...")
    print()
    
    # Commandes suggérées
    print("📝 Commandes suggérées:")
    print("  1. get_statistics")
    print("  2. list_users 10")
    print("  3. verify_user admin")
    print("  4. register_user {\"username\":\"test\",\"email\":\"test@mail.com\"}")
    print("  5. quit\n")
    
    # Boucle interactive
    while True:
        try:
            user_input = input("💬 Commande: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Au revoir!")
                break
            
            # Parser: tool_name argument
            parts = user_input.split(maxsplit=1)
            tool_name = parts[0]
            tool_arg = parts[1] if len(parts) > 1 else ""
            
            if tool_name == 'help':
                print("\n🔧 Outils disponibles:")
                for name, tool in TOOLS.items():
                    print(f"  • {name}: {tool['description']}")
                print()
                continue
            
            if tool_name not in TOOLS:
                print(f"❌ Outil '{tool_name}' inconnu. Tapez 'help' pour la liste.\n")
                continue
            
            # Exécuter
            print(f"\n🔧 Exécution: {tool_name}({tool_arg[:50]}...)...\n")
            result = call_tool(tool_name, tool_arg)
            
            print("="*60)
            print("📊 Résultat:")
            print("="*60)
            print(result)
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            logger.error(f"Erreur: {e}", exc_info=True)
            print(f"\n❌ Erreur: {e}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nArrêt...")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
        sys.exit(1)
