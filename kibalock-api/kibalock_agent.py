#!/usr/bin/env python3
"""
KibaLock Agent - Agent IA avec LangChain et modèles locaux
Agent intelligent pour l'authentification biométrique avec outils
Utilise le même pattern que ERT.py pour LangChain
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Configuration des caches AVANT imports HuggingFace
USER_HOME = os.path.expanduser("~")
os.environ['HF_HOME'] = os.path.join(USER_HOME, '.cache', 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(USER_HOME, '.cache', 'huggingface', 'transformers')
os.environ['HF_DATASETS_CACHE'] = os.path.join(USER_HOME, '.cache', 'huggingface', 'datasets')
os.environ['TORCH_HOME'] = os.path.join(USER_HOME, '.cache', 'torch')

# LangChain imports (même pattern qu'ERT.py)
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Transformers pour modèle local
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Imports pour biométrie
import faiss
import whisper
from pymongo import MongoClient
from PIL import Image
import io
import base64

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

###############################################################################
# Configuration
###############################################################################

# Chemins et modèles (utilise les modèles déjà téléchargés)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Qwen déjà installé comme ERT.py
PHI2_MODEL = "microsoft/phi-2"  # Phi-2 aussi disponible
WHISPER_MODEL_NAME = "base"
FAISS_INDEX_DIR = os.path.expanduser("~/kibalock/faiss_indexes")
AUDIO_SAMPLES_DIR = os.path.expanduser("~/kibalock/audio_samples")
FACE_IMAGES_DIR = os.path.expanduser("~/kibalock/face_images")

# Token HuggingFace (optionnel, pour API)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# MongoDB
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://SETRAF:Dieu19961991%3F%3F%21%3F%3F%21@cluster0.5tjz9v0.mongodb.net/kibalock?retryWrites=true&w=majority"
)

# Device (CPU pour compatibilité - RTX 5090 pas supporté par PyTorch actuel)
DEVICE = "cpu"  # Force CPU car CUDA sm_120 non supporté
logger.info(f"Device utilisé: {DEVICE} (GPU RTX 5090 non supporté par PyTorch 2.5.1)")

###############################################################################
# Chargement des modèles
###############################################################################

class BiometricModels:
    """Singleton pour gérer les modèles biométriques"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BiometricModels, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("Initialisation des modèles biométriques...")
            self._initialize_models()
            self._initialized = True
    
    def _initialize_models(self):
        """Initialiser tous les modèles (pattern ERT.py)"""
        try:
            # 1. Modèle Qwen2.5-1.5B pour l'agent (comme ERT.py)
            logger.info(f"Chargement de {MODEL_NAME}...")
            
            # Vérifier cache local
            cache_dir = os.path.join(USER_HOME, '.cache', 'huggingface', 'hub')
            model_cache = os.path.join(cache_dir, f"models--{MODEL_NAME.replace('/', '--')}")
            use_local = os.path.exists(model_cache)
            
            if use_local:
                logger.info(f"📦 Modèle {MODEL_NAME} trouvé en cache")
            
            # Charger tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                token=HF_TOKEN if HF_TOKEN else None,
                use_fast=True,
                resume_download=True,
                local_files_only=use_local
            )
            
            # Corriger pad_token comme ERT.py
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Charger modèle selon device
            if DEVICE == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    token=HF_TOKEN if HF_TOKEN else None,
                    low_cpu_mem_usage=True,
                    resume_download=True,
                    local_files_only=use_local
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    token=HF_TOKEN if HF_TOKEN else None,
                    low_cpu_mem_usage=True,
                    resume_download=True,
                    local_files_only=use_local
                ).to(DEVICE)
            
            logger.info(f"✓ {MODEL_NAME} chargé sur {DEVICE}")
            
            # 2. Whisper pour la voix
            logger.info(f"Chargement de Whisper {WHISPER_MODEL_NAME}...")
            self.whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
            logger.info("✓ Whisper chargé")
            
            # 3. FAISS pour recherche vectorielle
            logger.info("Chargement des index FAISS...")
            self.faiss_indexes = self._load_faiss_indexes()
            logger.info(f"✓ FAISS chargé ({len(self.faiss_indexes)} index)")
            
            # 4. MongoDB
            logger.info("Connexion à MongoDB...")
            self.mongo_client = MongoClient(MONGODB_URI)
            self.db = self.mongo_client.kibalock
            logger.info("✓ MongoDB connecté")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    def _load_faiss_indexes(self) -> Dict[str, faiss.Index]:
        """Charger les index FAISS existants"""
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
                # Créer un nouvel index si inexistant
                dim = 1280 if name == 'voice' else (512 if name == 'face' else 1792)
                indexes[name] = faiss.IndexFlatIP(dim)
                logger.info(f"  - {name}: nouvel index créé (dim={dim})")
        
        return indexes
    
    def save_faiss_indexes(self):
        """Sauvegarder les index FAISS"""
        for name, index in self.faiss_indexes.items():
            path = os.path.join(FAISS_INDEX_DIR, f'{name}_index.faiss')
            faiss.write_index(index, path)
            logger.info(f"Index {name} sauvegardé: {index.ntotal} vecteurs")

###############################################################################
# Outils pour l'agent
###############################################################################

def register_user_tool(user_data: str) -> str:
    """
    Enregistrer un nouvel utilisateur avec données biométriques
    
    Args:
        user_data: JSON string avec {username, email, audio_base64, image_base64}
    
    Returns:
        Message de succès ou d'erreur
    """
    try:
        data = json.loads(user_data)
        username = data.get('username')
        email = data.get('email')
        
        if not username or not email:
            return "❌ Erreur: username et email requis"
        
        models = BiometricModels()
        
        # Vérifier si l'utilisateur existe déjà
        if models.db.users.find_one({'username': username}):
            return f"❌ Utilisateur '{username}' existe déjà"
        
        # Créer l'utilisateur
        user_doc = {
            'username': username,
            'email': email,
            'created_at': datetime.utcnow(),
            'voice_embedding': None,
            'face_embedding': None,
            'combined_embedding': None
        }
        
        result = models.db.users.insert_one(user_doc)
        user_id = str(result.inserted_id)
        
        logger.info(f"Utilisateur créé: {username} (ID: {user_id})")
        
        return f"✅ Utilisateur '{username}' enregistré avec succès!\nID: {user_id}"
        
    except json.JSONDecodeError:
        return "❌ Erreur: format JSON invalide"
    except Exception as e:
        logger.error(f"Erreur register_user: {e}")
        return f"❌ Erreur: {str(e)}"


def verify_user_tool(username: str) -> str:
    """
    Vérifier si un utilisateur existe et obtenir ses informations
    
    Args:
        username: Nom d'utilisateur à vérifier
    
    Returns:
        Informations sur l'utilisateur ou message d'erreur
    """
    try:
        models = BiometricModels()
        user = models.db.users.find_one({'username': username})
        
        if not user:
            return f"❌ Utilisateur '{username}' non trouvé"
        
        info = f"""
✅ Utilisateur trouvé: {username}
📧 Email: {user.get('email', 'N/A')}
📅 Créé le: {user.get('created_at', 'N/A')}
🎤 Voix: {'✓' if user.get('voice_embedding') else '✗'}
📸 Visage: {'✓' if user.get('face_embedding') else '✗'}
🔗 Combiné: {'✓' if user.get('combined_embedding') else '✗'}
"""
        return info.strip()
        
    except Exception as e:
        logger.error(f"Erreur verify_user: {e}")
        return f"❌ Erreur: {str(e)}"


def list_users_tool(limit: str = "10") -> str:
    """
    Lister tous les utilisateurs enregistrés
    
    Args:
        limit: Nombre maximum d'utilisateurs à afficher
    
    Returns:
        Liste des utilisateurs
    """
    try:
        models = BiometricModels()
        limit_int = int(limit)
        users = list(models.db.users.find().limit(limit_int))
        
        if not users:
            return "📋 Aucun utilisateur enregistré"
        
        result = f"📋 {len(users)} utilisateur(s) trouvé(s):\n\n"
        for i, user in enumerate(users, 1):
            voice_status = "✓" if user.get('voice_embedding') else "✗"
            face_status = "✓" if user.get('face_embedding') else "✗"
            result += f"{i}. {user['username']} ({user['email']}) - Voix:{voice_status} Visage:{face_status}\n"
        
        return result.strip()
        
    except Exception as e:
        logger.error(f"Erreur list_users: {e}")
        return f"❌ Erreur: {str(e)}"


def search_similar_voice_tool(username: str, top_k: str = "5") -> str:
    """
    Rechercher les voix similaires à un utilisateur
    
    Args:
        username: Nom d'utilisateur de référence
        top_k: Nombre de résultats à retourner
    
    Returns:
        Liste des voix similaires avec scores
    """
    try:
        models = BiometricModels()
        user = models.db.users.find_one({'username': username})
        
        if not user:
            return f"❌ Utilisateur '{username}' non trouvé"
        
        voice_emb = user.get('voice_embedding')
        if not voice_emb:
            return f"❌ Pas d'embedding vocal pour '{username}'"
        
        # Convertir en numpy array
        query_vector = np.array(voice_emb, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Recherche dans FAISS
        voice_index = models.faiss_indexes.get('voice')
        if voice_index.ntotal == 0:
            return "📊 Index vocal vide"
        
        k = min(int(top_k), voice_index.ntotal)
        distances, indices = voice_index.search(query_vector, k)
        
        result = f"🔍 Top {k} voix similaires à '{username}':\n\n"
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            similarity = float(dist) * 100
            result += f"{i}. Index {idx} - Similarité: {similarity:.2f}%\n"
        
        return result.strip()
        
    except Exception as e:
        logger.error(f"Erreur search_similar_voice: {e}")
        return f"❌ Erreur: {str(e)}"


def get_statistics_tool(dummy: str = "") -> str:
    """
    Obtenir des statistiques sur la base de données KibaLock
    
    Returns:
        Statistiques complètes
    """
    try:
        models = BiometricModels()
        
        total_users = models.db.users.count_documents({})
        users_with_voice = models.db.users.count_documents({'voice_embedding': {'$ne': None}})
        users_with_face = models.db.users.count_documents({'face_embedding': {'$ne': None}})
        users_complete = models.db.users.count_documents({
            'voice_embedding': {'$ne': None},
            'face_embedding': {'$ne': None}
        })
        
        voice_vectors = models.faiss_indexes['voice'].ntotal
        face_vectors = models.faiss_indexes['face'].ntotal
        combined_vectors = models.faiss_indexes['combined'].ntotal
        
        result = f"""
📊 Statistiques KibaLock
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
  • Audio samples: {len(os.listdir(AUDIO_SAMPLES_DIR)) if os.path.exists(AUDIO_SAMPLES_DIR) else 0}
  • Images visage: {len(os.listdir(FACE_IMAGES_DIR)) if os.path.exists(FACE_IMAGES_DIR) else 0}
"""
        return result.strip()
        
    except Exception as e:
        logger.error(f"Erreur get_statistics: {e}")
        return f"❌ Erreur: {str(e)}"


def delete_user_tool(username: str) -> str:
    """
    Supprimer un utilisateur et ses données biométriques
    
    Args:
        username: Nom d'utilisateur à supprimer
    
    Returns:
        Message de confirmation ou d'erreur
    """
    try:
        models = BiometricModels()
        
        user = models.db.users.find_one({'username': username})
        if not user:
            return f"❌ Utilisateur '{username}' non trouvé"
        
        # Supprimer de MongoDB
        result = models.db.users.delete_one({'username': username})
        
        if result.deleted_count > 0:
            logger.info(f"Utilisateur supprimé: {username}")
            return f"✅ Utilisateur '{username}' supprimé avec succès"
        else:
            return f"❌ Échec de la suppression de '{username}'"
        
    except Exception as e:
        logger.error(f"Erreur delete_user: {e}")
        return f"❌ Erreur: {str(e)}"


###############################################################################
# Configuration de l'agent LangChain
###############################################################################

def create_kibalock_agent():
    """Créer l'agent KibaLock avec tous les outils (pattern ERT.py)"""
    
    logger.info("Création de l'agent KibaLock...")
    
    # Initialiser les modèles
    models = BiometricModels()
    
    # Définir les outils LangChain (comme ERT.py)
    tools = [
        Tool(
            name="register_user",
            func=register_user_tool,
            description="📝 Enregistrer un nouvel utilisateur dans KibaLock. Input: JSON avec username, email. Retourne confirmation ou erreur."
        ),
        Tool(
            name="verify_user",
            func=verify_user_tool,
            description="🔍 Vérifier si un utilisateur existe et voir ses données biométriques. Input: username. Retourne profil complet avec statut voix/visage."
        ),
        Tool(
            name="list_users",
            func=list_users_tool,
            description="📋 Lister tous les utilisateurs enregistrés dans la base. Input: limit (nombre max, défaut 10). Retourne liste avec statuts."
        ),
        Tool(
            name="search_similar_voice",
            func=search_similar_voice_tool,
            description="🎤 Rechercher les voix similaires à un utilisateur via FAISS. Input: username, top_k. Retourne similarités avec scores."
        ),
        Tool(
            name="get_statistics",
            func=get_statistics_tool,
            description="📊 Obtenir statistiques complètes de KibaLock (utilisateurs, vecteurs FAISS, stockage). No input needed."
        ),
        Tool(
            name="delete_user",
            func=delete_user_tool,
            description="🗑️ Supprimer un utilisateur et toutes ses données biométriques. Input: username. Action irréversible!"
        )
    ]
    
    # Fonction pour appeler les outils
    def call_tool(tool_name: str, tool_input: str) -> str:
        """Appeler un outil par son nom"""
        for tool in tools:
            if tool.name == tool_name:
                try:
                    return tool.func(tool_input)
                except Exception as e:
                    return f"❌ Erreur: {str(e)}"
        return f"❌ Outil '{tool_name}' non trouvé"
    
    # Créer une fonction de génération simple
    def generate_response(query: str) -> str:
        """Générer une réponse avec le modèle local"""
        # Préparer le prompt
        system_msg = """Tu es KibaLock Agent, assistant IA pour l'authentification biométrique.
Tu as accès aux outils suivants:
- register_user: Enregistrer utilisateur
- verify_user: Vérifier utilisateur  
- list_users: Lister utilisateurs
- search_similar_voice: Chercher voix similaires
- get_statistics: Statistiques complètes
- delete_user: Supprimer utilisateur

Réponds directement aux questions sur KibaLock."""

        full_prompt = f"{system_msg}\n\nQuestion: {query}\nRéponse:"
        
        # Générer avec le modèle
        inputs = models.tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
        outputs = models.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=models.tokenizer.eos_token_id
        )
        
        response = models.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extraire seulement la réponse
        if "Réponse:" in response:
            response = response.split("Réponse:")[-1].strip()
        
        return response
    
    logger.info("✓ Agent KibaLock créé avec succès")
    
    return tools, call_tool, generate_response


###############################################################################
# Interface de test
###############################################################################

def test_agent():
    """Tester l'agent en mode interactif"""
    
    print("\n" + "="*60)
    print("🤖 KibaLock Agent - Mode Test")
    print("="*60)
    print("Modèle:", MODEL_NAME)
    print("Device:", DEVICE)
    print("LangChain: Activé")
    print("="*60 + "\n")
    
    # Créer l'agent
    tools, call_tool, generate_response = create_kibalock_agent()
    
    print("\n✅ Agent prêt! Tapez 'quit' pour quitter.\n")
    
    # Afficher les outils disponibles
    print("🔧 Outils disponibles:")
    for i, tool in enumerate(tools, 1):
        print(f"  {i}. {tool.name}: {tool.description[:60]}...")
    print()
    
    # Commandes de test
    test_commands = [
        ("get_statistics", ""),
        ("list_users", "5"),
        ("help", "")
    ]
    
    print("📝 Commandes suggérées:")
    print("  1. get_statistics          # Voir statistiques")
    print("  2. list_users 10           # Lister 10 utilisateurs")
    print("  3. verify_user admin       # Vérifier utilisateur 'admin'")
    print("  4. <question naturelle>    # Poser une question")
    print()
    
    # Boucle interactive
    while True:
        try:
            user_input = input("💬 Vous: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Au revoir!")
                break
            
            # Vérifier si c'est une commande d'outil directe
            parts = user_input.split(maxsplit=1)
            tool_names = [t.name for t in tools]
            
            if parts[0] in tool_names:
                # Commande d'outil directe
                tool_name = parts[0]
                tool_arg = parts[1] if len(parts) > 1 else ""
                print(f"\n🔧 Exécution: {tool_name}({tool_arg})...\n")
                result = call_tool(tool_name, tool_arg)
            else:
                # Question naturelle -> utiliser le modèle
                print("\n🤔 KibaLock Agent analyse votre question...\n")
                result = generate_response(user_input)
            
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


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    try:
        test_agent()
    except KeyboardInterrupt:
        print("\n\nArrêt du programme...")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
        sys.exit(1)
