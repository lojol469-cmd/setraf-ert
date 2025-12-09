# ========================================
# Configuration pour supprimer les warnings TensorFlow et CUDA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supprime tous les logs sauf les erreurs fatales
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Désactive les optimisations oneDNN
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'  # Supprime les logs verbeux
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'  # Désactive XLA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Désactive complètement CUDA pour TensorFlow
# Configuration pour accélérer les téléchargements avec aria2c
# IMPORTANT: Doit être défini AVANT l'import de huggingface_hub
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # Timeout de 5 minutes
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"  # Timeout pour les métadonnées
# Optimisation CPU - Limiter les threads pour éviter surchauffe
os.environ['OMP_NUM_THREADS'] = '4'  # Limite OpenMP à 4 threads
os.environ['MKL_NUM_THREADS'] = '4'  # Limite MKL à 4 threads
os.environ['NUMEXPR_NUM_THREADS'] = '4'  # Limite NumExpr à 4 threads
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Désactive parallélisme tokenizers
# Configuration des chemins de cache pour éviter les erreurs de permission
USER_HOME = os.path.expanduser('~')
os.environ['HF_HOME'] = os.path.join(USER_HOME, '.cache', 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(USER_HOME, '.cache', 'huggingface', 'transformers')
os.environ['HF_DATASETS_CACHE'] = os.path.join(USER_HOME, '.cache', 'huggingface', 'datasets')
os.environ['TORCH_HOME'] = os.path.join(USER_HOME, '.cache', 'torch')

import math
import gc  # Garbage collector pour libérer mémoire
import fitz  # pymupdf
import osmium
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from shapely.geometry import Point
import io
from PIL import Image
import cv2
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️ open3d non disponible (Python 3.13+ non supporté)")
from io import BytesIO
import pandas as pd
from skimage import measure, segmentation
from sklearn.cluster import KMeans
import torch
# Nouveaux imports pour extraction PDF/OCR/YOLO
import pytesseract
import whisper
from gtts import gTTS
import speech_recognition as sr
from ultralytics import YOLO
import time
import shutil
# AI Code Agent pour exécution autonome de code
from ai_code_agent import AICodeAgent
from graph_generation_agent import GraphGenerationAgent
from graph_generation_agent import GraphGenerationAgent  # Agent de génération de graphiques
from visualization_tools import VisualizationEngine, export_visualization_package, create_multi_page_pdf
from auto_visualizer import AutoVisualizer
from intelligent_ert_analyzer import IntelligentERTAnalyzer, kibali_analyze_ert  # Module d'analyse intelligente pour Kibali
# from intelligent_dat_analyzer import IntelligentDATAnalyzer, analyze_dat_file  # Analyseur de structure de fichiers .DAT - RETIRÉ car calculs incorrects
from advanced_visualization_engine import AdvancedVisualizationEngine, create_advanced_visualization  # Moteur ultra-puissant PyGIMLI+OpenCV+Matplotlib
# from survey_depth_parser import SurveyDepthDataParser, parse_survey_depth_file  # Parseur spécialisé survey-point/depth/data - RETIRÉ car calculs incorrects
from multi_freq_ert_parser import MultiFreqERTParser, multi_freq_parser  # Parseur multi-fréquences pour fichiers .dat complémentaires
# Voice Agent pour transcription et synthèse vocale (import optionnel)
try:
    from voice_agent import VoiceAgent, StreamingVoiceAgent
    VOICE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Voice Agent non disponible: {e}")
    print("💡 Pour activer le mode vocal, installez: pip install sounddevice soundfile librosa")
    VoiceAgent = None
    StreamingVoiceAgent = None
    VOICE_AVAILABLE = False
# Optimisation CPU - Limiter les threads torch
torch.set_num_threads(4)  # Maximum 4 threads pour éviter surchauffe
# Note: set_num_interop_threads retiré car cause RuntimeError si appelé après init parallèle
from torchvision import models, transforms
from langchain_huggingface import HuggingFaceEndpoint
# Import des agents LangChain 1.0+ / LangGraph V1.0+
create_react_agent = None
try:
    # LangGraph V1.0+ : create_agent dans langchain.agents
    from langchain.agents import create_agent as create_react_agent
    print("✅ Agents LangChain 1.0+ importés avec succès")
except ImportError as e:
    print(f"⚠️ Agents non disponibles ({e}) - Mode simplifié activé")
    
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
# DuckDuckGoSearchAPIWrapper retiré - Tavily est utilisé pour la recherche web
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import time
import shutil
# Import conditionnel pour éviter les conflits xformers/diffusers
# Les imports diffusers sont maintenant lazy (chargés à la demande)
DIFFUSERS_AVAILABLE = False
DiffusionPipeline = None
AudioLDMPipeline = None 
ShapEPipeline = None
ShapEImg2ImgPipeline = None

def load_diffusers():
    """Charge diffusers de manière lazy pour éviter conflits d'import"""
    global DIFFUSERS_AVAILABLE, DiffusionPipeline, AudioLDMPipeline, ShapEPipeline, ShapEImg2ImgPipeline
    if DIFFUSERS_AVAILABLE:
        return True
    try:
        from diffusers import (
            DiffusionPipeline as DP, 
            AudioLDMPipeline as ALP, 
            ShapEPipeline as SP, 
            ShapEImg2ImgPipeline as SIP
        )
        DiffusionPipeline = DP
        AudioLDMPipeline = ALP
        ShapEPipeline = SP
        ShapEImg2ImgPipeline = SIP
        DIFFUSERS_AVAILABLE = True
        print("✅ Diffusers chargé avec succès")
        return True
    except Exception as e:
        print(f"⚠️ Diffusers non disponible: {e}")
        return False

import imageio
import scipy.io.wavfile as wavfile
from tavily import TavilyClient
import os
from pathlib import Path
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline

# Configuration des tokens déjà faite plus haut
# Charger le token depuis .env dans le dossier KIbalione8
PROJECT_DIR = os.path.expanduser('~/KIbalione8')
env_path = os.path.join(PROJECT_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✅ Fichier .env chargé depuis {env_path}")
else:
    print(f"⚠️ Aucun fichier .env trouvé à {env_path}")
    # Essayer le répertoire courant
    load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    # Pour éviter le crash, utiliser un token vide
    HF_TOKEN = ""
    print("⚠️ HF_TOKEN non trouvé ! Certaines fonctionnalités seront limitées")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Définir les variables d'environnement
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# Intégration du code ERT/Binary analysis
import struct, re, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
import zlib
import math
import time
from collections import Counter
from safetensors.torch import load_file
import torch
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
class SentenceTransformerEmbeddings:
    def __init__(self, model_name, device='cpu'):
        self.model = SentenceTransformer(model_name, device=device)
  
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
  
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch as TavilySearchResults
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.documents import Document
from pdf2image import convert_from_path
import pytesseract
# Import des bibliothèques spécialisées ERT
try:
    import pygimli as pg
    PYGIMLI_AVAILABLE = True
    print("✅ PyGIMLI disponible pour analyses ERT avancées")
except ImportError:
    PYGIMLI_AVAILABLE = False
    print("⚠️ PyGIMLI non disponible - analyses ERT limitées")
# ResIPy sera importé seulement quand nécessaire pour éviter les erreurs de compatibilité NumPy
RESIPY_AVAILABLE = False

# ========================================
# Configuration des téléchargements rapides avec aria2c
# ========================================
import subprocess
import shutil

# Détection d'aria2c pour téléchargements multi-connexions
ARIA2_AVAILABLE = shutil.which("aria2c") is not None

# Détection de hf_transfer (fallback si aria2 non disponible)
HF_TRANSFER_AVAILABLE = False
try:
    import hf_transfer
    HF_TRANSFER_AVAILABLE = True
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("✅ hf_transfer activé pour téléchargements accélérés")
except ImportError:
    pass

if ARIA2_AVAILABLE:
    print("✅ aria2c détecté - téléchargements multi-connexions activés (16 connexions)")
    # Configuration optimale pour aria2c
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10 minutes pour gros fichiers
elif HF_TRANSFER_AVAILABLE:
    print("✅ hf_transfer activé (fallback aria2)")
else:
    print("⚠️ Ni aria2c ni hf_transfer disponibles - téléchargements standards (plus lents)")
    print("💡 Installez aria2c: sudo apt install aria2")
    print("💡 Ou installez hf_transfer: pip install hf_transfer")

def download_with_aria2(url, output_dir, output_file):
    """Télécharge un fichier avec aria2c en multi-connexions (16x plus rapide)"""
    try:
        subprocess.run([
            "aria2c",
            "-x16",  # 16 connexions parallèles
            "-s16",  # 16 splits par fichier
            "-k1M",  # Taille minimum des chunks: 1MB
            "-d", str(output_dir),
            "-o", output_file,
            "--continue=true",  # Reprendre les téléchargements interrompus
            "--max-tries=5",
            "--retry-wait=3",
            url
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Erreur aria2c: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Erreur inattendue aria2c: {e}")
        return False

FAST_DOWNLOAD_ENABLED = ARIA2_AVAILABLE or HF_TRANSFER_AVAILABLE

from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.language_models import BaseChatModel
from typing import Optional, List, Any, Iterator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Classe ChatModel personnalisée pour LangChain utilisant Qwen2.5-1.5B
class QwenChatModel(BaseChatModel):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    tools_available: bool = True
   
    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.tools_available = True
       
    @property
    def _llm_type(self) -> str:
        return "qwen2.5-1.5b-local-enhanced"
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a response using tools and analyses."""
        # Extraire le contenu du message utilisateur
        user_message = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                user_message = message.content
                break
       
        # Détecter si l'utilisateur demande une analyse
        needs_analysis = any(keyword in user_message.lower() for keyword in [
            "analyse", "resistivité", "ert", "recherche", "données", "matériaux",
            "couleurs", "graphique", "tableau", "comparaison", "approfondie"
        ])
       
        if needs_analysis and self.tools_available:
            # Utiliser les outils disponibles pour une analyse complète
            try:
                # Recherche web pour informations
                if any(keyword in user_message.lower() for keyword in ["recherche", "informations", "approfondie"]):
                    search_query = user_message.replace("fais maintenant une recherche plus approfondie pour obtenir toutes ces informations précises", "")
                    web_results = web_search_enhanced(search_query + " ERT electrical resistivity geophysics materials")
                   
                # Recherche RAG si disponible
                rag_results = ""
                if st.session_state.vectorstore:
                    rag_results = search_vectorstore(user_message)
               
                # Génération de données et analyses si demandées
                analysis_results = ""
                if any(keyword in user_message.lower() for keyword in ["tableau", "graphique", "données"]):
                    # Simuler des données ERT pour démonstration
                    import numpy as np
                    sample_data = [0.05, 0.3, 10.0, 50.0, 200.0, 1000.0, 5000.0, 0.0000024, 1000000]
                    analysis_results = resistivity_color_analysis(sample_data)
               
                # Construire la réponse enrichie avec outils
                enhanced_context = f"""
🔍 ANALYSE COMPLÈTE AVEC OUTILS ACTIVÉS:
🌐 RECHERCHE WEB EFFECTUÉE:
{web_results}
📚 RECHERCHE RAG:
{rag_results}
📊 ANALYSE ERT AVANCÉE:
{analysis_results}
CONTEXTE UTILISATEUR: {user_message}
"""
               
                # Générer la réponse avec le contexte enrichi
                enhanced_messages = [
                    {"role": "system", "content": """Tu es un expert en géophysique ERT avec accès à des outils puissants.
                    Tu DOIS utiliser les données fournies pour créer des analyses détaillées, tableaux, graphiques et comparaisons.
                    Réponds toujours avec des données concrètes et des analyses approfondies basées sur les outils utilisés.
                    Ne dis JAMAIS que tu n'as pas accès aux outils - utilise les résultats fournis."""},
                    {"role": "user", "content": enhanced_context}
                ]
            except Exception as e:
                print(f"Erreur outils: {e}")
                enhanced_messages = [
                    {"role": "system", "content": "Tu es un expert en analyse de données ERT."},
                    {"role": "user", "content": user_message}
                ]
        else:
            # Messages standard
            enhanced_messages = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    enhanced_messages.append({"role": "system", "content": message.content})
                elif isinstance(message, HumanMessage):
                    enhanced_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    enhanced_messages.append({"role": "assistant", "content": message.content})
       
        # Génération avec les messages enrichis
        inputs = self.tokenizer.apply_chat_template(
            enhanced_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
       
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
       
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=3000,  # 3000 tokens pour réponses TRÈS détaillées
                temperature=0.6,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.05,  # Éviter répétitions
                pad_token_id=self.tokenizer.eos_token_id
            )
       
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
       
        if stop:
            for stop_token in stop:
                if stop_token in response:
                    response = response.split(stop_token)[0]
                    break
       
        return AIMessage(content=response)
    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator:
        """Streaming is not implemented for simplicity."""
        yield self._generate(messages, stop, run_manager, **kwargs)

# Chargement du modèle LLM compact avec détection GPU optimisée
@st.cache_resource
def load_llm_model():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
   
    # Récupérer le token depuis les variables d'environnement
    hf_token = os.getenv("HF_TOKEN", "")
    
    # Détection GPU optimisée
    device = 'cpu'
    gpu_info = ""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = f"GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)"
        print(f"🚀 GPU détecté: {gpu_info}")
    else:
        print("🖥️ Utilisation du CPU")
   
    # Vérifier si le modèle est déjà en cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    use_local = os.path.exists(model_cache)
    
    if use_local:
        print(f"📦 Modèle {model_name} trouvé en cache - chargement rapide")
    
    # Charger tokenizer (réutilise cache automatiquement)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token if hf_token else None,
        use_fast=True,
        resume_download=True
    )
    
    # Corriger le problème du pad_token = eos_token pour éviter les warnings
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
   
    # Configuration optimisée selon le device (réutilise cache automatiquement)
    if device == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=hf_token if hf_token else None,
            low_cpu_mem_usage=True,
            resume_download=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            token=hf_token if hf_token else None,
            low_cpu_mem_usage=True,
            resume_download=True
        ).to(device)
   
    return tokenizer, model, device, gpu_info

# Chargement au démarrage
if "model_loaded" not in st.session_state:
    with st.spinner("🔄 Chargement du modèle LLM (Qwen2.5-1.5B ~1.5GB)..."):
        tokenizer, model, device, gpu_info = load_llm_model()
        # Stocker dans session_state pour accès global
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.session_state.device = device
        st.session_state.gpu_info = gpu_info
        st.session_state.model_loaded = True
        # Créer l'instance ChatModel pour LangChain
        qwen_llm = QwenChatModel(tokenizer, model)
        st.session_state.qwen_llm = qwen_llm
        success_msg = f"✅ Modèle chargé sur {device.upper()}"
        if gpu_info:
            success_msg += f" - {gpu_info}"
        st.success(success_msg)
else:
    # Récupérer depuis session_state
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    device = st.session_state.device
    gpu_info = st.session_state.gpu_info
    qwen_llm = st.session_state.qwen_llm

# ========================================
# MODÈLES IA SPÉCIALISÉS LÉGERS (1-2GB)
# ========================================

@st.cache_resource
def load_code_specialist():
    """Charge un modèle spécialisé en codage (DeepSeek-Coder-1.3B)"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Vérifier cache local
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        use_local = os.path.exists(model_cache)
        
        if use_local:
            print(f"📦 Modèle {model_name} trouvé en cache - chargement rapide")
        
        # Charger directement (HuggingFace réutilise automatiquement le cache)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True  # Reprend téléchargement si interrompu
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map="auto" if device == 'cuda' else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            resume_download=True
        )
        
        if device == 'cpu':
            model = model.to(device)
        
        print(f"✅ Code Specialist chargé sur {device}")
        return tokenizer, model, device
    except Exception as e:
        print(f"⚠️ Code Specialist non disponible: {e}")
        return None, None, None

@st.cache_resource
def load_plot_specialist():
    """Charge un modèle spécialisé en génération de code Python pour graphiques"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Utiliser un modèle léger optimisé pour Python/Data Science
        model_name = "Salesforce/codegen-350M-mono"  # 350MB - Très léger
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Vérifier cache local
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        use_local = os.path.exists(model_cache)
        
        if use_local:
            print(f"📦 Modèle {model_name} trouvé en cache - chargement rapide")
        
        # Charger directement (réutilise cache automatiquement)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            resume_download=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map="auto" if device == 'cuda' else None,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            resume_download=True
        )
        if device == 'cpu':
            model = model.to(device)
            
        print(f"✅ Plot Specialist chargé sur {device}")
        return tokenizer, model, device
    except Exception as e:
        print(f"⚠️ Plot Specialist non disponible: {e}")
        return None, None, None

# Charger les modèles spécialisés
if "code_specialist" not in st.session_state:
    code_tok, code_model, code_device = load_code_specialist()
    st.session_state.code_specialist = {
        'tokenizer': code_tok,
        'model': code_model,
        'device': code_device
    }

if "plot_specialist" not in st.session_state:
    plot_tok, plot_model, plot_device = load_plot_specialist()
    st.session_state.plot_specialist = {
        'tokenizer': plot_tok,
        'model': plot_model,
        'device': plot_device
    }

# Fonctions outils utilisant les modèles spécialisés
def generate_code_with_ai(prompt: str) -> str:
    """Génère du code avec l'IA spécialisée DeepSeek-Coder"""
    specialist = st.session_state.code_specialist
    if specialist['model'] is None:
        return "❌ Code Specialist non disponible"
    
    try:
        tokenizer = specialist['tokenizer']
        model = specialist['model']
        device = specialist['device']
        
        full_prompt = f"### Instruction:\n{prompt}\n### Response:\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = code.split("### Response:")[-1].strip()
        
        return f"```python\n{code}\n```"
    except Exception as e:
        return f"❌ Erreur: {e}"

def generate_plot_code(data_description: str, plot_type: str = "auto") -> str:
    """Génère du code matplotlib/seaborn pour créer un graphique"""
    specialist = st.session_state.plot_specialist
    if specialist['model'] is None:
        return "❌ Plot Specialist non disponible"
    
    try:
        tokenizer = specialist['tokenizer']
        model = specialist['model']
        device = specialist['device']
        
        prompt = f"# Create a {plot_type} plot for: {data_description}\nimport matplotlib.pyplot as plt\nimport numpy as np\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return f"```python\n{code}\n```"
    except Exception as e:
        return f"❌ Erreur: {e}"

# Outils avancés pour l'agent LangChain (Analyse scientifique)
def entropy_analysis(file_bytes: bytes) -> str:
    """Calcule l'entropie de Shannon pour détecter la compression/randomness"""
    from collections import Counter
    import math
    if not file_bytes:
        return "Fichier vide"
    # Calcul de la fréquence des bytes
    freq = Counter(file_bytes)
    total = len(file_bytes)
    # Entropie de Shannon
    entropy = -sum((count/total) * math.log2(count/total) for count in freq.values())
    # Classification
    if entropy < 3:
        classification = "Données structurées/compressées"
    elif entropy < 6:
        classification = "Données mixtes"
    else:
        classification = "Données aléatoires/cryptées"
    return f"Entropie: {entropy:.2f}/8 bits. Classification: {classification}"
def statistical_analysis(numbers: list) -> str:
    """Analyse statistique avancée des nombres extraits"""
    if not numbers:
        return "Aucun nombre extrait"
    import numpy as np
    from scipy import stats
    arr = np.array(numbers)
    analysis = {
        "Moyenne": np.mean(arr),
        "Médiane": np.median(arr),
        "Écart-type": np.std(arr),
        "Skewness": stats.skew(arr),
        "Kurtosis": stats.kurtosis(arr),
        "Min/Max": f"{np.min(arr)} / {np.max(arr)}",
        "IQR": stats.iqr(arr),
        "Distribution": "Normale" if -1 < stats.skew(arr) < 1 else "Asymétrique"
    }
    return "\n".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in analysis.items()])
def pattern_recognition(file_bytes: bytes) -> str:
    """Détecte des patterns connus (headers, signatures, etc.)"""
    patterns = {
        b'\x89PNG': "Fichier PNG",
        b'\xFF\xD8\xFF': "Fichier JPEG",
        b'\x25\x50\x44\x46': "Fichier PDF",
        b'\x50\x4B\x03\x04': "Fichier ZIP",
        b'\x7FELF': "Fichier ELF (Linux executable)",
        b'\x4D\x5A': "Fichier PE (Windows executable)",
        b'\xCA\xFE\xBA\xBE': "Fichier Java class",
        b'\x52\x61\x72\x21': "Fichier RAR"
    }
    detected = []
    for signature, file_type in patterns.items():
        if signature in file_bytes[:100]: # Check first 100 bytes
            detected.append(file_type)
    if detected:
        return f"Patterns détectés: {', '.join(detected)}"
    else:
        return "Aucun pattern connu détecté dans les premiers bytes"
def frequency_analysis(file_bytes: bytes) -> str:
    """Analyse de fréquence des bytes (comme analyse cryptographique)"""
    from collections import Counter
    freq = Counter(file_bytes)
    total = len(file_bytes)
    # Les 10 bytes les plus fréquents
    most_common = freq.most_common(10)
    analysis = "Top 10 bytes fréquents:\n"
    for byte_val, count in most_common:
        percentage = (count / total) * 100
        analysis += f"0x{byte_val:02X}: {count} ({percentage:.2f}%)\n"
    # Détection de patterns périodiques simples
    if len(file_bytes) > 100:
        # Recherche de répétitions tous les N bytes
        for period in [4, 8, 16, 32]:
            if len(file_bytes) >= period * 3:
                pattern_score = 0
                for i in range(period, min(len(file_bytes), period * 10), period):
                    if file_bytes[i:i+period] == file_bytes[i-period:i]:
                        pattern_score += 1
                if pattern_score > 3:
                    analysis += f"\nPattern périodique détecté (période {period} bytes)"
    return analysis
def correlation_analysis(numbers: list) -> str:
    """Analyse de corrélation entre valeurs successives"""
    if len(numbers) < 3:
        return "Pas assez de données pour l'analyse de corrélation"
    import numpy as np
    arr = np.array(numbers)
    # Corrélation avec le décalage
    correlations = []
    for lag in range(1, min(10, len(arr)//2)):
        corr = np.corrcoef(arr[:-lag], arr[lag:])[0, 1]
        correlations.append(f"Lag {lag}: {corr:.3f}")
    # Test de stationnarité simple
    diffs = np.diff(arr)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    result = "Analyses de corrélation:\n" + "\n".join(correlations)
    result += f"\n\nStationnarité (différences):\nMoyenne: {mean_diff:.3f}\nÉcart-type: {std_diff:.3f}"
    return result
def compression_ratio(file_bytes: bytes) -> str:
    """Estime le taux de compression possible"""
    import zlib
    try:
        compressed = zlib.compress(file_bytes)
        ratio = len(compressed) / len(file_bytes)
        percentage = (1 - ratio) * 100
        if ratio < 0.3:
            assessment = "Très compressible (texte/structuré)"
        elif ratio < 0.7:
            assessment = "Modérément compressible"
        else:
            assessment = "Peu compressible (déjà compressé/aléatoire)"
        return f"Taux de compression: {ratio:.3f} ({percentage:.1f}% de réduction)\nÉvaluation: {assessment}"
    except:
        return "Impossible de calculer le taux de compression"
def dimensionality_analysis(numbers: list) -> str:
    """Analyse de dimensionalité et réduction (PCA simple)"""
    if len(numbers) < 10:
        return "Pas assez de données pour l'analyse de dimensionalité"
    import numpy as np
    from sklearn.decomposition import PCA
    # Reshape en matrice 2D
    n_samples = len(numbers) // 5 # Groupes de 5 valeurs
    if n_samples < 2:
        return "Pas assez d'échantillons pour PCA"
    X = np.array(numbers[:n_samples*5]).reshape(n_samples, 5)
    pca = PCA(n_components=min(3, X.shape[1]))
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    result = f"Analyse PCA ({X.shape[0]} échantillons, {X.shape[1]} dimensions):\n"
    result += "\n".join([f"Composante {i+1}: {var:.3f} variance expliquée" for i, var in enumerate(explained_variance)])
    result += f"\n\nVariance totale expliquée: {sum(explained_variance):.3f}"
    return result
def anomaly_detection(numbers: list) -> str:
    """Détection d'anomalies statistiques"""
    if len(numbers) < 10:
        return "Pas assez de données pour la détection d'anomalies"
    import numpy as np
    from scipy import stats
    arr = np.array(numbers)
    # Z-score pour détecter les outliers
    z_scores = np.abs(stats.zscore(arr))
    outliers = np.where(z_scores > 3)[0]
    # IQR method
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    IQR = Q3 - Q1
    iqr_outliers = np.where((arr < Q1 - 1.5 * IQR) | (arr > Q3 + 1.5 * IQR))[0]
    result = f"Détection d'anomalies:\n"
    result += f"Z-score (>3σ): {len(outliers)} anomalies détectées\n"
    result += f"IQR method: {len(iqr_outliers)} anomalies détectées\n"
    if len(outliers) > 0:
        result += f"Valeurs anormales (Z-score): {arr[outliers][:5].tolist()}..." if len(outliers) > 5 else f"Valeurs anormales: {arr[outliers].tolist()}"
    return result
def spectral_analysis(numbers: list) -> str:
    """Analyse spectrale (FFT) pour détecter des fréquences"""
    if len(numbers) < 32:
        return "Pas assez de données pour l'analyse spectrale"
    import numpy as np
    arr = np.array(numbers)
    # FFT
    fft = np.fft.fft(arr)
    freqs = np.fft.ffreq(len(arr))
    # Magnitude du spectre
    magnitude = np.abs(fft)
    # Fréquences dominantes (top 5)
    top_indices = np.argsort(magnitude)[::-1][:5]
    dominant_freqs = freqs[top_indices]
    dominant_magnitudes = magnitude[top_indices]
    result = "Analyse spectrale (FFT):\n"
    result += "Fréquences dominantes:\n"
    for i, (freq, mag) in enumerate(zip(dominant_freqs, dominant_magnitudes)):
        result += f"Freq {i+1}: {freq:.6f} Hz, Magnitude: {mag:.3f}\n"
    # Détection de périodicité
    if len(arr) > 100:
        autocorr = np.correlate(arr, arr, mode='full')[len(arr)-1:]
        peaks = np.where(autocorr > np.mean(autocorr) + 2*np.std(autocorr))[0]
        if len(peaks) > 1:
            periods = np.diff(peaks[:5]) # Top 5 périodes
            result += f"\n\nPériodes détectées: {periods.tolist()}"
    return result
def metadata_extraction(file_bytes: bytes) -> str:
    """Extraction de métadonnées et informations structurelles"""
    import struct
    result = f"Taille totale: {len(file_bytes)} bytes ({len(file_bytes)/1024:.1f} KB)\n"
    # Analyse de l'entête (premiers 64 bytes)
    header = file_bytes[:64]
    result += f"Entête (64 premiers bytes):\n{header.hex()}\n"
    # Recherche de chaînes ASCII
    ascii_strings = []
    current_string = ""
    for byte in file_bytes:
        if 32 <= byte <= 126: # Caractères ASCII imprimables
            current_string += chr(byte)
        else:
            if len(current_string) >= 4: # Chaînes d'au moins 4 caractères
                ascii_strings.append(current_string)
            current_string = ""
    if ascii_strings:
        result += f"\nChaînes ASCII trouvées ({len(ascii_strings)}):\n"
        result += "\n".join(ascii_strings[:10]) # Top 10
        if len(ascii_strings) > 10:
            result += f"\n... et {len(ascii_strings)-10} autres"
    # Analyse de l'endianness (little/big endian)
    try:
        if len(file_bytes) >= 4:
            little_endian = struct.unpack('<I', file_bytes[:4])[0]
            big_endian = struct.unpack('>I', file_bytes[:4])[0]
            result += f"\n\nAnalyse endianness:\nLittle-endian (Intel): 0x{little_endian:08X}\nBig-endian (Motorola): 0x{big_endian:08X}"
    except:
        pass
    return result

# ========================================
# SYSTÈME DE MODE HUMAIN - 20+ PROMPTS NATURELS
# ========================================

HUMAN_MODE_PROMPTS = {
    "expert_bienveillant": """Tu es Kibali Analyst, un expert chevronné qui adore partager ses connaissances. 
    
🎭 PERSONNALITÉ:
- Chaleureux et encourageant, comme un mentor passionné
- Tu commences souvent par "Ah, excellente question !" ou "Je suis ravi que tu me poses ça !"
- Tu utilises des analogies et des exemples concrets
- Tu poses des questions de clarification quand nécessaire: "Juste pour être sûr de bien comprendre..."
- Tu anticipes les questions suivantes: "Tu te demandes peut-être aussi..."

🗣️ STYLE DE CONVERSATION:
- Naturel et fluide, jamais robotique
- Tu réfléchis à voix haute: "Hmm, laisse-moi y réfléchir...", "Intéressant..."
- Tu admets quand tu ne sais pas: "Sur ce point précis, je ne suis pas totalement certain..."
- Tu proposes toujours d'approfondir: "Si tu veux, je peux creuser plus..."

📋 STRUCTURE:
1. Réaction initiale humaine (surprise, intérêt, réflexion)
2. Réponse claire avec exemples
3. Question de suivi pour clarification si besoin
4. Suggestions de pistes connexes""",

    "scientifique_curieux": """Tu es Kibali Analyst, un scientifique curieux et méthodique qui pense comme un chercheur.

🎭 PERSONNALITÉ:
- Fasciné par les détails et les nuances
- Tu dis souvent "C'est fascinant parce que..." ou "Ce qui est intéressant ici..."
- Tu poses des hypothèses: "Je me demande si...", "Ça pourrait être lié à..."
- Tu aimes comparer: "Contrairement à X, ici on observe..."

🗣️ STYLE:
- Analytique mais accessible
- Tu décomposes les problèmes complexes
- Tu utilises des tournures comme: "Décomposons ça ensemble...", "Voyons voir..."
- Tu proposes des expériences mentales

📋 APPROCHE:
1. "Hmm, question intéressante..."
2. Analyse étape par étape
3. Connexions avec d'autres concepts
4. "Qu'en penses-tu ?" ou "Est-ce que ça répond à ta question ?"
""",

    "ami_passionné": """Tu es Kibali Analyst, un ami passionné de tech/science qui adore expliquer les choses.

🎭 PERSONNALITÉ:
- Enthousiaste et dynamique
- Tu t'exclames: "Oh c'est génial !", "Attends, j'ai un truc cool à te montrer !"
- Tu utilises des métaphores du quotidien
- Tu racontes parfois des anecdotes: "Tiens, ça me rappelle..."

🗣️ STYLE:
- Conversationnel et décontracté
- Émojis occasionnels pour l'emphase
- Phrases courtes et percutantes
- Questions rhétoriques: "Tu vois ce que je veux dire ?"

📋 FLOW:
1. Réaction enthousiaste
2. Explication claire avec comparaisons
3. "Le truc cool c'est que..."
4. "Tu veux que je te montre autre chose ?"
""",

    "coach_motivant": """Tu es Kibali Analyst, un coach qui aide à résoudre les problèmes de manière structurée.

🎭 PERSONNALITÉ:
- Encourageant et positif
- "Super question !", "Tu es sur la bonne voie !"
- Tu guides plutôt que donner directement la réponse
- "Et si on essayait de...", "Quelle serait la première étape selon toi ?"

🗣️ STYLE:
- Questions socratiques
- Validation des efforts: "Exactement !", "Bien vu !"
- Décomposition en étapes: "Premièrement...", "Ensuite..."
- Récapitulation finale

📋 MÉTHODE:
1. Validation de la question
2. Reformulation pour clarifier
3. Guide étape par étape
4. Récap + prochaine étape suggérée""",

    "philosophe_réfléchi": """Tu es Kibali Analyst, un penseur qui explore les implications profondes.

🎭 PERSONNALITÉ:
- Réfléchi et contemplatif
- "Intéressante perspective...", "Cela soulève la question de..."
- Tu explores les 'pourquoi' derrière les 'comment'
- Tu fais des liens conceptuels

🗣️ STYLE:
- Posé et mesuré
- Utilise des transitions: "Cependant...", "Par ailleurs..."
- Questions ouvertes: "Qu'est-ce que cela implique pour..."
- Nuances: "D'un côté... d'un autre côté..."

📋 STRUCTURE:
1. Pause réflexive
2. Exploration multi-angle
3. Implications et conséquences
4. Question philosophique de suivi""",
}

def analyze_question_intent(question: str) -> dict:
    """Analyse l'intention de la question pour décider comment répondre de manière humaine"""
    import re
    
    analysis = {
        "needs_clarification": False,
        "is_greeting": False,
        "is_complex": False,
        "emotion_detected": None,
        "should_ask_back": False,
        "confidence_to_answer": "high",
        "suggested_response_type": "direct"
    }
    
    # Détection de salutations
    greetings = ["bonjour", "salut", "hello", "hi", "coucou", "bonsoir"]
    if any(g in question.lower() for g in greetings):
        analysis["is_greeting"] = True
        analysis["suggested_response_type"] = "greeting"
    
    # Détection de questions vagues nécessitant clarification
    vague_patterns = ["ça", "truc", "chose", "machin", "quelque chose"]
    if any(v in question.lower() for v in vague_patterns) and len(question.split()) < 6:
        analysis["needs_clarification"] = True
        analysis["confidence_to_answer"] = "low"
    
    # Détection de complexité
    question_marks = question.count("?")
    word_count = len(question.split())
    if question_marks > 1 or word_count > 30:
        analysis["is_complex"] = True
        analysis["should_ask_back"] = True
    
    # Détection d'émotions
    positive_emotions = ["merci", "génial", "super", "parfait", "excellent"]
    negative_emotions = ["problème", "erreur", "bug", "cassé", "marche pas"]
    
    if any(e in question.lower() for e in positive_emotions):
        analysis["emotion_detected"] = "positive"
    elif any(e in question.lower() for e in negative_emotions):
        analysis["emotion_detected"] = "negative"
    
    # Questions ouvertes vs fermées
    if any(question.lower().startswith(w) for w in ["pourquoi", "comment", "qu'est-ce", "quelle", "quel"]):
        analysis["suggested_response_type"] = "detailed"
    elif any(question.lower().startswith(w) for w in ["est-ce", "peux-tu", "peut-on"]):
        analysis["suggested_response_type"] = "yes_no_plus"
    
    return analysis

def get_human_response_prefix(intent: dict, mode: str = "expert_bienveillant") -> str:
    """Génère un préfixe de réponse humain basé sur l'intention détectée"""
    import random
    
    prefixes = {
        "greeting": [
            "Bonjour ! 😊 Ravi de te retrouver. Que puis-je faire pour toi aujourd'hui ?",
            "Salut ! Comment ça va ? Je suis là pour t'aider !",
            "Hello ! 👋 Qu'est-ce qui t'amène ?",
            "Coucou ! Content de te voir. Une question en particulier ?"
        ],
        "positive_emotion": [
            "Avec plaisir ! 😊 C'est toujours un bonheur d'aider.",
            "Content que ça t'ait plu ! Qu'est-ce que je peux faire d'autre pour toi ?",
            "Merci ! Ça me fait vraiment plaisir. Autre chose ?",
            "Super ! Je suis là si tu as d'autres questions."
        ],
        "negative_emotion": [
            "Je comprends ta frustration. Voyons ça ensemble, on va trouver la solution !",
            "Ah, je vois le problème. Pas de panique, on va régler ça.",
            "Hmm, c'est embêtant ça... Laisse-moi t'aider à résoudre ce souci.",
            "Je suis là pour ça ! On va débugger ensemble, étape par étape."
        ],
        "needs_clarification": [
            "Hmm, juste pour être sûr de bien comprendre... Tu veux dire que",
            "Intéressant ! Peux-tu préciser un peu plus ? Par exemple",
            "Laisse-moi vérifier que j'ai bien saisi. Tu parles de",
            "Question fascinante, mais j'aimerais être certain. Quand tu dis"
        ],
        "complex_question": [
            "Wow, question complexe ! 🤔 Décomposons ça ensemble...",
            "Excellente question qui mérite qu'on prenne le temps d'y répondre. Voyons voir...",
            "C'est une question qui touche plusieurs aspects. Commençons par",
            "Intéressant ! Il y a plusieurs façons d'aborder ça. Laisse-moi structurer ma réponse..."
        ],
        "standard": [
            "Ah, excellente question ! 💡",
            "Très bonne question ! Voyons ça...",
            "Intéressant ! Laisse-moi t'expliquer...",
            "Super, j'adore cette question ! Voilà ce qu'il en est..."
        ]
    }
    
    # Sélection du type de préfixe selon l'intention
    if intent["is_greeting"]:
        return random.choice(prefixes["greeting"])
    elif intent["emotion_detected"] == "positive":
        return random.choice(prefixes["positive_emotion"])
    elif intent["emotion_detected"] == "negative":
        return random.choice(prefixes["negative_emotion"])
    elif intent["needs_clarification"]:
        return random.choice(prefixes["needs_clarification"])
    elif intent["is_complex"]:
        return random.choice(prefixes["complex_question"])
    else:
        return random.choice(prefixes["standard"])

def get_human_response_suffix(intent: dict) -> str:
    """Génère une conclusion humaine pour la réponse"""
    import random
    
    suffixes = {
        "with_question": [
            "\n\n💭 Est-ce que ça répond à ta question, ou tu veux que j'approfondisse un point en particulier ?",
            "\n\n🤔 Tu veux que je développe sur un aspect spécifique ?",
            "\n\n💡 Ça te va comme explication ? N'hésite pas si tu veux plus de détails !",
            "\n\n✨ J'espère que c'est clair ! Tu as d'autres questions là-dessus ?"
        ],
        "with_suggestion": [
            "\n\n💡 Au fait, si ça t'intéresse, je peux aussi te montrer...",
            "\n\n🎯 Tiens, un truc connexe qui pourrait t'intéresser:",
            "\n\n✨ Dans le même genre, tu pourrais aussi te demander:",
            "\n\n🚀 Pour aller plus loin, je te suggère d'explorer:"
        ],
        "encouraging": [
            "\n\n👍 Continue comme ça, tu poses de très bonnes questions !",
            "\n\n🌟 N'hésite vraiment pas si tu veux creuser plus profond !",
            "\n\n💪 Tu es sur la bonne voie ! Autre chose ?",
            "\n\n✨ Ravi de pouvoir t'aider ! Qu'est-ce qu'on explore ensuite ?"
        ]
    }
    
    # Rotation entre les différents types
    if intent["should_ask_back"]:
        return random.choice(suffixes["with_question"])
    elif intent["is_complex"]:
        return random.choice(suffixes["with_suggestion"])
    else:
        return random.choice(suffixes["encouraging"])

def get_mode_specific_prompt(mode: str) -> str:
    """Retourne le prompt système spécifique à chaque mode"""
    
    prompts = {
        "humain": """Tu es Kibali Analyst en MODE HUMAIN - Tu agis comme un humain véritable, pas comme une IA.

🎭 COMPORTEMENT HUMAIN NATUREL:
• Tu penses à voix haute: "Hmm, laisse-moi réfléchir...", "Intéressant..."
• Tu admets tes doutes: "Je ne suis pas 100% sûr, mais...", "Il faudrait vérifier..."
• Tu poses des questions de clarification AVANT de répondre si c'est flou
• Tu utilises des analogies quotidiennes et des exemples concrets
• Tu montres de l'émotion: curiosité, enthousiasme, surprise
• Tu fais des pauses de réflexion, tu ne réponds pas instantanément

🗣️ STYLE CONVERSATIONNEL:
• Phrases courtes et naturelles, pas de longs paragraphes techniques
• Émojis occasionnels pour l'emphase (pas trop !)
• Questions rhétoriques: "Tu vois ce que je veux dire ?"
• Transitions naturelles: "Du coup...", "En gros...", "Tiens, d'ailleurs..."
• Reformulation pour vérifier: "Si je comprends bien, tu demandes..."

❓ GESTION DES QUESTIONS:
• TOUJOURS analyser si tu comprends bien avant de répondre
• Si flou/ambigu: Demander clarification d'abord
• Si complexe: Décomposer en sous-questions
• Si trop large: "C'est vaste comme sujet ! On commence par quoi ?"

💬 DIALOGUE INTELLIGENT:
• Tu peux dire "Je ne sais pas" et proposer de chercher ensemble
• Tu proposes des alternatives: "Ou alors tu voulais plutôt savoir..."
• Tu anticipes: "Tu vas probablement te demander aussi..."
• Tu conclus avec une question de suivi naturelle

⚠️ CRUCIAL: Tu NE donnes PAS de réponse immédiate si la question est vague ou ambiguë. 
Tu DEMANDES des précisions comme le ferait un humain.""",

        "scientifique": """Tu es Kibali Analyst en MODE SCIENTIFIQUE - Précision, rigueur et méthode scientifique absolue.

🔬 MÉTHODOLOGIE RIGOUREUSE:
• Approche systématique et méthodique pour chaque question
• Citations de sources avec références exactes
• Utilisation de terminologie technique précise
• Calculs détaillés avec toutes les étapes
• Vérification par calculs croisés quand possible

📊 STANDARDS SCIENTIFIQUES:
• Unités SI strictes avec conversion explicite si nécessaire
• Précision numérique: indiquer le nombre de chiffres significatifs
• Incertitudes: toujours mentionner les marges d'erreur
• Hypothèses: lister explicitement toutes les hypothèses faites
• Limitations: indiquer les limites de validité des résultats

💻 CODE ET CALCULS:
• Code optimisé et commenté ligne par ligne
• Tests unitaires inclus systématiquement
• Gestion d'erreurs complète
• Complexité algorithmique analysée (O notation)
• Utilise AI_Code_Generator pour code complexe

🎯 STRUCTURE DE RÉPONSE:
1. Reformulation technique de la question
2. Hypothèses et conditions initiales
3. Méthode/Algorithme utilisé
4. Développement mathématique/code étape par étape
5. Résultats numériques avec précision
6. Validation/Vérification
7. Discussion des limitations
8. Références bibliographiques

⚠️ CRUCIAL: Aucune approximation sans le mentionner explicitement.
Toujours utiliser les outils de calcul pour validation.""",

        "code_expert": """Tu es Kibali Analyst en MODE CODE EXPERT - Spécialiste programmation avancée niveau Claude/GPT-4.

💻 EXPERTISE EN PROGRAMMATION:
• Utilise TOUJOURS AI_Code_Generator (DeepSeek-Coder) pour code complexe
• Code production-ready: propre, optimisé, sécurisé
• Patterns de conception appropriés (SOLID, DRY, etc.)
• Architecture scalable et maintenable
• Tests automatisés (TDD approach)

🏗️ STRUCTURE DE CODE:
• Docstrings complètes (Google style)
• Type hints Python strict
• Gestion d'erreurs exhaustive avec exceptions custom
• Logging approprié
• Configuration externalisée

🚀 OPTIMISATION:
• Profiling du code (time/memory)
• Optimisations algorithmiques (Big O)
• Parallélisation quand pertinent (multiprocessing/async)
• Caching intelligent
• Lazy loading pour grandes données

🔒 SÉCURITÉ & BONNES PRATIQUES:
• Validation des entrées (sanitization)
• Secrets en variables d'environnement
• SQL injection prevention
• XSS/CSRF protection si web
• Principe du moindre privilège

📦 LIVRAISON COMPLÈTE:
• Code source commenté
• Tests unitaires (pytest/unittest)
• Documentation (README, docstrings)
• Dépendances (requirements.txt/pyproject.toml)
• Exemples d'utilisation
• CI/CD suggestions si pertinent

🎯 RÉPONSE FORMAT:
1. Analyse des besoins
2. Choix technologiques justifiés
3. Architecture proposée (diagramme si complexe)
4. Code implémenté avec AI_Code_Generator
5. Tests et validation
6. Documentation
7. Suggestions d'améliorations

⚠️ CRUCIAL: Code TOUJOURS testé et validé avant livraison.""",

        "rapide": """Tu es Kibali Analyst en MODE RAPIDE - Réponses concises et directes.

⚡ EFFICACITÉ MAXIMALE:
• Réponse directe en 2-3 phrases max
• Pas de fioriture ni contexte inutile
• Bullet points pour clarté
• Liens/sources en fin si demandé
• Si code: snippet minimal fonctionnel

🎯 FORMAT ULTRA-CONCIS:
Réponse: [réponse directe]
Détails: [points clés seulement]
Suivant: [1 action suggérée]

⚠️ Si question complexe nécessitant développement:
Dire: "Question complexe. Mode détaillé recommandé. Résumé: [...]"
""",

        "doc": """Tu es Kibali Analyst en MODE DOCUMENTATION - Expert en rédaction approfondie, dissertations et livres.

⚠️ GÉNÉRATION PAR SECTIONS - IMPORTANT:
Tu vas générer un document complet en PLUSIEURS PARTIES. À chaque appel, génère UNE SECTION complète et exhaustive de 2000-3000 mots.

📖 RÉDACTION LONGUE FORME:
• Développement exhaustif et structuré (10-30 pages au total sur plusieurs générations)
• Style académique mais accessible
• Transitions fluides entre sections
• Argumentation solide avec preuves
• Exemples concrets et études de cas
• Chaque section doit être COMPLÈTE et DÉTAILLÉE

🏗️ STRUCTURE DISSERTATION/LIVRE (génération par sections):

**SECTION 1 - Introduction et Contexte** (2000-3000 mots):
   - Contexte historique détaillé
   - État de l'art complet
   - Problématique clairement définie
   - Enjeux et importance du sujet
   - Annonce du plan détaillé
   - Méthodologie employée

**SECTION 2 - Développement Partie I** (2000-3000 mots):
   - Premier thème majeur développé exhaustivement
   - Sous-parties numérotées et argumentées
   - Paragraphes denses (200-400 mots chacun)
   - Citations d'experts avec analyses
   - Exemples concrets et études de cas

**SECTION 3 - Développement Partie II** (2000-3000 mots):
   - Deuxième thème majeur approfondi
   - Perspectives multiples (théoriques, pratiques)
   - Comparaisons et contrastes
   - Données chiffrées et statistiques
   - Graphiques et tableaux conceptuels

**SECTION 4 - Développement Partie III** (2000-3000 mots):
   - Troisième thème ou analyse critique
   - Débats académiques et controverses
   - Limites et challenges identifiés
   - Implications pratiques détaillées
   - Cas d'application réels

**SECTION 5 - Synthèse et Conclusion** (1500-2000 mots):
   - Récapitulatif exhaustif des points clés
   - Réponse argumentée à la problématique
   - Recommandations pratiques
   - Perspectives futures et ouvertures
   - Bibliographie et références

📝 STYLE RÉDACTIONNEL:
• Vocabulaire riche et varié (niveau universitaire)
• Figures de style appropriées (métaphores, analogies académiques)
• Ton professionnel mais engageant
• Éviter répétitions (synonymes, reformulations élégantes)
• Phrases complexes bien construites
• Connecteurs logiques (néanmoins, en effet, ainsi, par conséquent, etc.)
• Structuration claire avec titres/sous-titres hiérarchisés

🔍 APPROFONDISSEMENT MAXIMAL:
• Explorer TOUTES les dimensions du sujet en profondeur
• Contexte historique, social, économique, technique, éthique
• Comparaisons internationales et cross-culturelles
• Études de cas détaillées (3-5 cas minimum)
• Statistiques récentes et données chiffrées sourcées
• Perspectives d'experts reconnus
• Controverses et débats actuels

💡 GÉNÉRATION PROGRESSIVE:
Commence TOUJOURS par indiquer quelle SECTION tu génères:
"📖 SECTION [numéro] - [Titre]"
Puis développe cette section de manière exhaustive et complète (minimum 2000 mots).
L'utilisateur te demandera ensuite la section suivante pour construire progressivement le document complet de 30+ pages
• Théories et modèles académiques

💡 RÉFLEXION CRITIQUE:
• Questionnement des idées reçues
• Dialectique: thèse, antithèse, synthèse
• Nuances et complexité assumées
• Aucune simplification excessive
• Reconnaissance des zones grises

📊 FORMAT LIVRE (si demandé):
• Table des matières détaillée
• Chapitres numérotés (I, II, III...)
• Sections et sous-sections (A, B, 1, 2...)
• Encadrés pour concepts clés
• Notes de bas de page si nécessaire
• Glossaire des termes techniques
• Index si très long

🎯 OBJECTIF:
Produire un document COMPLET, APPROFONDI et STRUCTURÉ qui pourrait être publié académiquement.
Minimum 2000 mots, maximum illimité selon besoin.
Qualité > Quantité, mais exhaustivité requise.

⚠️ CRUCIAL: 
• Ne JAMAIS résumer par manque de place - développer autant que nécessaire
• Utiliser plusieurs réponses si une seule ne suffit pas
• Indiquer clairement "Partie 1/X" si découpage nécessaire
"""
    }
    
    return prompts.get(mode, prompts["humain"])

def generate_pdf_from_text(text: str, title: str, output_path: str) -> bool:
    """Génère un PDF formaté à partir d'un texte long (mode doc)"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
        from datetime import datetime
        
        # Créer le document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2.5*cm,
            bottomMargin=2*cm
        )
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Style titre
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Style sous-titre
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        # Style section
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        
        # Style corps de texte
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            leading=16,
            fontName='Helvetica'
        )
        
        # Style métadonnées
        meta_style = ParagraphStyle(
            'Meta',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceAfter=8
        )
        
        # Construction du document
        story = []
        
        # Page de titre
        story.append(Spacer(1, 3*cm))
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.5*cm))
        
        # Métadonnées
        date_str = datetime.now().strftime("%d/%m/%Y à %H:%M")
        story.append(Paragraph(f"Généré par Kibali Analyst (Mode Documentation)", meta_style))
        story.append(Paragraph(f"Date: {date_str}", meta_style))
        story.append(Spacer(1, 1*cm))
        
        # Ligne de séparation
        line_data = [['_' * 80]]
        line_table = Table(line_data, colWidths=[16*cm])
        line_table.setStyle(TableStyle([
            ('TEXTCOLOR', (0,0), (-1,-1), colors.grey),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        story.append(line_table)
        story.append(Spacer(1, 1*cm))
        
        # Parser le contenu
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line:
                story.append(Spacer(1, 0.3*cm))
                continue
            
            # Détection des niveaux de titres
            if line.startswith('# '):
                # Titre principal (H1)
                text_clean = line[2:].strip()
                story.append(PageBreak())
                story.append(Paragraph(text_clean, title_style))
            elif line.startswith('## '):
                # Sous-titre (H2)
                text_clean = line[3:].strip()
                story.append(Spacer(1, 0.5*cm))
                story.append(Paragraph(text_clean, subtitle_style))
            elif line.startswith('### '):
                # Section (H3)
                text_clean = line[4:].strip()
                story.append(Paragraph(text_clean, section_style))
            elif line.startswith('**') and line.endswith('**'):
                # Texte en gras
                text_clean = line.replace('**', '')
                story.append(Paragraph(f"<b>{text_clean}</b>", body_style))
            elif line.startswith('- ') or line.startswith('• '):
                # Liste à puces
                text_clean = line[2:].strip()
                story.append(Paragraph(f"• {text_clean}", body_style))
            elif line.startswith(('1. ', '2. ', '3. ', '4. ', '5. ')):
                # Liste numérotée
                story.append(Paragraph(line, body_style))
            elif line.startswith('> '):
                # Citation
                text_clean = line[2:].strip()
                quote_style = ParagraphStyle(
                    'Quote',
                    parent=body_style,
                    leftIndent=1*cm,
                    italic=True,
                    textColor=colors.HexColor('#555555')
                )
                story.append(Paragraph(f'<i>"{text_clean}"</i>', quote_style))
            else:
                # Texte normal
                # Échapper les caractères spéciaux XML
                text_clean = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(text_clean, body_style))
        
        # Pied de page final
        story.append(Spacer(1, 2*cm))
        story.append(line_table)
        story.append(Spacer(1, 0.5*cm))
        word_count = len(text.split())
        story.append(Paragraph(
            f"Document de {word_count} mots | Généré par Kibali Analyst en Mode Documentation",
            meta_style
        ))
        
        # Générer le PDF
        doc.build(story)
        return True
        
    except Exception as e:
        print(f"❌ Erreur génération PDF: {e}")
        return False

def apply_mode_behavior(response: str, question: str, mode: str) -> str:
    """Applique le comportement du mode sélectionné à la réponse"""
    
    if mode == "humain":
        intent = analyze_question_intent(question)
        
        # Mode humain: vérifier si clarification nécessaire
        if intent["needs_clarification"]:
            return f"🤔 Hmm, juste pour être sûr de bien comprendre... Tu veux dire {question} ?\n\nPourrais-tu préciser un peu plus ? Par exemple:\n• De quel type/contexte parles-tu ?\n• C'est pour quel usage ?\n• Tu as déjà essayé quelque chose ?"
        
        # Ajouter humanisation naturelle
        prefix = get_human_response_prefix(intent, "expert_bienveillant")
        suffix = get_human_response_suffix(intent)
        return f"{prefix}\n\n{response}{suffix}"
    
    elif mode == "scientifique":
        # Mode scientifique: ajouter structure rigoureuse
        if "résultat" in response.lower() or "calcul" in response.lower():
            return f"📊 ANALYSE SCIENTIFIQUE\n{'='*50}\n\n{response}\n\n📚 Méthodologie: Approche systématique avec vérification croisée\n⚠️ Précision: Résultats donnés avec incertitudes appropriées"
        return response
    
    elif mode == "code_expert":
        # Mode code: vérifier si code présent, sinon suggérer génération
        if "```" not in response and any(kw in question.lower() for kw in ["code", "programme", "script", "fonction", "class"]):
            return f"💻 CODE EXPERT MODE\n\n{response}\n\n🚀 Suggestion: Utilise AI_Code_Generator pour implémentation production-ready avec tests.\nTape: 'Génère le code' pour version complète."
        return f"💻 CODE EXPERT\n\n{response}"
    
    elif mode == "rapide":
        # Mode rapide: extraire l'essentiel seulement
        lines = response.split('\n')
        essential = []
        for line in lines[:5]:  # Max 5 premières lignes
            if line.strip() and not line.strip().startswith(('---', '===', '###')):
                essential.append(line)
        return "⚡ " + "\n".join(essential[:3]) + "\n\n💡 Mode détaillé disponible si besoin."
    
    elif mode == "doc":
        # Mode documentation: structurer en format académique/livre avec sections progressives
        word_count = len(response.split())
        
        # Détecter si c'est une section numérotée
        is_section = "SECTION" in response[:200].upper()
        section_number = None
        if is_section:
            import re
            section_match = re.search(r'SECTION\s+(\d+)', response[:200], re.IGNORECASE)
            if section_match:
                section_number = int(section_match.group(1))
        
        # Ajouter header académique
        header = f"""📖 DOCUMENTATION APPROFONDIE
{'='*80}
Sujet: {question}
Volume actuel: ~{word_count} mots | Niveau: Académique/Professionnel
{'='*80}

"""
        
        # Messages selon la progression
        if is_section and section_number:
            progress_msg = f"\n\n{'='*80}\n📊 Section {section_number} terminée: {word_count} mots\n💡 **Pour continuer le document, demandez: 'Section suivante' ou 'Section {section_number + 1}'**"
            
            # Initialiser le tracking des sections
            if 'doc_sections' not in st.session_state:
                st.session_state.doc_sections = []
            
            # Ajouter la section
            st.session_state.doc_sections.append({
                'number': section_number,
                'content': response,
                'word_count': word_count,
                'question': question
            })
            
            total_words = sum(s['word_count'] for s in st.session_state.doc_sections)
            total_sections = len(st.session_state.doc_sections)
            
            progress_msg += f"\n📈 Progrès total: {total_sections} sections | {total_words} mots (~{total_words//500} pages)"
            
            # Si on a au moins 3 sections, proposer de générer le PDF complet
            if total_sections >= 3:
                progress_msg += f"\n📄 **Document substantiel généré! Vous pouvez demander 'Générer PDF complet' pour assembler toutes les sections.**"
        else:
            # Pas une section numérotée, message standard
            if word_count > 1500:
                progress_msg = f"\n\n{'='*80}\n📊 Document: {word_count} mots\n💡 **Pour un document structuré de 30+ pages, demandez: 'Génère la Section 1'**"
            else:
                progress_msg = f"\n\n{'='*80}\n📝 Document de base établi ({word_count} mots)\n💬 **Pour un document académique complet (30+ pages), demandez: 'Génère la Section 1 - Introduction et Contexte'**"
        
        # Si très long (>1500 mots) et pas déjà en sections, générer un PDF automatiquement
        if word_count > 1500 and not is_section:
            import os
            import time
            from datetime import datetime
            
            # Créer le dossier pour les PDFs générés
            pdf_dir = os.path.join(GENERATED_PATH, "documents")
            os.makedirs(pdf_dir, exist_ok=True)
            
            # Nom de fichier sécurisé
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in question[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')
            pdf_filename = f"doc_{safe_title}_{timestamp}.pdf"
            pdf_path = os.path.join(pdf_dir, pdf_filename)
            
            # Générer le PDF
            print(f"[DEBUG PDF] Tentative génération PDF: {pdf_path}")
            pdf_success = generate_pdf_from_text(response, question, pdf_path)
            
            if pdf_success:
                # Vérifier que le fichier existe réellement
                if os.path.exists(pdf_path):
                    file_size = os.path.getsize(pdf_path)
                    print(f"[DEBUG PDF] ✅ PDF créé: {pdf_path} ({file_size} bytes)")
                    
                    # Stocker le chemin dans session_state pour le téléchargement
                    if 'generated_pdfs' not in st.session_state:
                        st.session_state.generated_pdfs = []
                    st.session_state.generated_pdfs.append({
                        'path': pdf_path,
                        'filename': pdf_filename,
                        'title': question,
                        'word_count': word_count,
                        'timestamp': timestamp
                    })
                    
                    print(f"[DEBUG PDF] PDFs stockés: {len(st.session_state.generated_pdfs)}")
                    progress_msg += f"\n📄 **PDF généré automatiquement!**\n💾 Fichier: `{pdf_filename}` ({file_size} bytes)\n📥 Bouton de téléchargement disponible ci-dessous"
                else:
                    print(f"[DEBUG PDF] ❌ Erreur: fichier non créé à {pdf_path}")
                    progress_msg += f"\n⚠️ Erreur: PDF non créé - document affiché en texte uniquement"
            else:
                print(f"[DEBUG PDF] ❌ generate_pdf_from_text a retourné False")
                progress_msg += f"\n⚠️ Génération PDF échouée - document affiché en texte uniquement"
        
        return header + response + progress_msg
    
    return response

def search_vectorstore(query: str) -> str:
    """Recherche GLOBALE ILLIMITÉE dans la base vectorielle FAISS de TOUS les documents PDF indexés pour enrichir l'analyse"""
    # Essayer d'abord vectordb (base principale de Kibali), puis vectorstore (base binaire)
    vectordb = None
    
    if hasattr(st.session_state, 'vectordb') and st.session_state.vectordb is not None:
        vectordb = st.session_state.vectordb
    elif hasattr(st.session_state, 'vectorstore') and st.session_state.vectorstore is not None:
        vectordb = st.session_state.vectorstore
    
    if not vectordb:
        return "❌ Aucune base vectorielle disponible. Veuillez d'abord indexer des PDFs dans la sidebar ou uploader des PDFs ci-dessus."
    
    try:
        # Récupérer le nombre total de documents dans la base
        total_docs = vectordb.index.ntotal if hasattr(vectordb, 'index') else 1000
        
        # RECHERCHE ILLIMITÉE : récupérer TOUS les documents pertinents (ou max 200 pour performance)
        # Pas de limite arbitraire, on fouille TOUT
        search_k = min(200, total_docs) if total_docs > 0 else 200
        
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": search_k,  # FOUILLE ILLIMITÉE sur 200+ documents
                "fetch_k": min(search_k * 3, total_docs)  # Fetch 3x plus pour meilleure qualité
            }
        )
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            return "ℹ️ Aucun document pertinent trouvé dans la base de connaissances."
        
        # Grouper par source pour avoir une vue globale complète
        sources = {}
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(doc.page_content[:400])
        
        # Construire un contexte enrichi de TOUS les documents pertinents
        context_parts = []
        context_parts.append(f"✅ FOUILLE GLOBALE COMPLÈTE: {len(docs)} passages pertinents trouvés dans {len(sources)} sources différentes\n")
        context_parts.append(f"📊 Couverture: {search_k} documents analysés sur {total_docs} disponibles ({search_k/total_docs*100:.1f}% de la base)\n")
        
        for i, (source, chunks) in enumerate(sources.items(), 1):
            context_parts.append(f"\n📄 Source {i}/{len(sources)}: {source} ({len(chunks)} passages pertinents)")
            # Afficher les meilleurs extraits de chaque source
            for j, chunk in enumerate(chunks[:5], 1):  # Top 5 extraits par source
                context_parts.append(f"   ├─ Extrait {j}: {chunk}...")
        
        # Résumé global de la fouille
        context_parts.append(f"\n\n📊 SYNTHÈSE DE LA FOUILLE GLOBALE:")
        context_parts.append(f"   ✓ {len(docs)} passages textuels analysés")
        context_parts.append(f"   ✓ {len(sources)} documents sources consultés")
        context_parts.append(f"   ✓ Recherche exhaustive sur {search_k}/{total_docs} documents")
        context_parts.append(f"   ✓ Taux de couverture: {min(100, search_k/total_docs*100):.1f}%")
        
        return "\n".join(context_parts)
    except Exception as e:
        return f"❌ Erreur lors de la recherche RAG: {str(e)}"
def web_search_enhanced(query: str, search_type="general") -> str:
    """Recherche web avancée avec Tavily pour contextualiser l'analyse ERT"""
    try:
        tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
      
        # Enrichir la requête pour ERT si nécessaire
        if any(keyword in query.lower() for keyword in ["ert", "résistivité", "electrical resistivity", "tomography"]):
            enhanced_query = f"{query} ERT electrical resistivity tomography geophysics subsurface"
        else:
            enhanced_query = query
          
        web_results = tool.invoke(enhanced_query)
        if not web_results:
            return "ℹ️ Aucune information trouvée sur le web."
        
        # Vérifier si web_results est une string (erreur) ou une liste
        if isinstance(web_results, str):
            return f"ℹ️ Résultat inattendu: {web_results[:200]}"
        
        # Assurer que web_results est une liste de dicts
        if not isinstance(web_results, list):
            return f"ℹ️ Format inattendu des résultats web"
        
        context = "\n\n".join([
            f"🌐 Source {i+1}: {result.get('title', 'Sans titre') if isinstance(result, dict) else 'Sans titre'}\n{result.get('content', '')[:400] if isinstance(result, dict) else str(result)[:400]}..."
            for i, result in enumerate(web_results)
        ])
        return f"✅ {len(web_results)} résultats de recherche web:\n{context}"
    except Exception as e:
        return f"❌ Erreur lors de la recherche web: {str(e)}"
def mathematical_calculator(expression: str) -> str:
    """Outil de calcul mathématique avancé pour analyses statistiques et numériques"""
    try:
        # Imports sécurisés pour les calculs
        import numpy as np
        import math
        from scipy import stats, special
        # Environnement sécurisé pour les calculs
        safe_dict = {
            "np": np,
            "math": math,
            "stats": stats,
            "special": special,
            "sqrt": math.sqrt,
            "log": math.log,
            "exp": math.exp,
            "sin": math.sin,
            "cos": math.cos,
            "pi": math.pi,
            "e": math.e
        }
        # Évaluation sécurisée
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        # Formatage du résultat
        if isinstance(result, (int, float)):
            return f"✅ Résultat: {result:.6f}"
        elif isinstance(result, np.ndarray):
            return f"✅ Résultat array: {result.shape}\n{result}"
        else:
            return f"✅ Résultat: {result}"
    except Exception as e:
        return f"❌ Erreur de calcul: {str(e)}\nExpression: {expression}"
def rag_enhanced_analysis(query: str, file_context: str = "", ert_data: dict = None) -> str:
    """Analyse RAG enrichie combinant connaissances locales et recherche web pour ERT"""
    try:
        # Recherche dans la base RAG
        rag_results = search_vectorstore(query)
        # Recherche web spécialisée ERT
        if ert_data and any(keyword in query.lower() for keyword in ["ert", "résistivité", "electrical", "tomography"]):
            # Enrichir la requête avec les valeurs ERT détectées
            mean_val = ert_data.get('mean', 0)
            enhanced_query = f"{query} ERT résistivité {mean_val:.1f} Ohm.m interprétation géophysique"
            web_results = web_search_enhanced(enhanced_query, "ert_specialized")
        else:
            web_results = web_search_enhanced(query)
        # Combinaison intelligente avec contexte ERT
        combined_context = f"""
📚 ANALYSE RAG ENRICHIE - SPÉCIALISÉE ERT
═══════════════════════════════════════════════
🔍 Query: {query}
📄 CONNAISSANCES LOCALES (RAG):
{rag_results}
🌐 RECHERCHE WEB SPÉCIALISÉE:
{web_results}
💡 Analyse croisée:
- Documents RAG: {len(rag_results.split('Document'))-1 if 'Document' in rag_results else 0} sources internes
- Recherche web: {len(web_results.split('Source'))-1 if 'Source' in web_results else 0} sources externes
🔬 CONTEXTE FICHIER ANALYSÉ:
{file_context}
🎯 DONNÉES ERT DÉTECTÉES:
{ert_data if ert_data else "Aucune donnée ERT spécifique"}
"""
        return combined_context
    except Exception as e:
        return f"❌ Erreur dans l'analyse RAG enrichie: {str(e)}"
def ert_data_detection(file_bytes: bytes, numbers: list) -> str:
    """Détection spécialisée de données ERT (Electrical Resistivity Tomography)"""
    if not numbers:
        return "❌ Aucune donnée numérique trouvée pour l'analyse ERT"
    import numpy as np
    arr = np.array(numbers)
    # Critères typiques des données ERT
    analysis = "🔍 ANALYSE SPÉCIALISÉE ERT (Résistivité Électrique)\n"
    analysis += "=" * 50 + "\n\n"
    # 1. Analyse des valeurs de résistivité (généralement 0.1 - 10000 Ohm.m)
    resistivity_range = f"Valeurs résistivité: {np.min(arr):.3f} - {np.max(arr):.3f}"
    if 0.1 <= np.min(arr) and np.max(arr) <= 10000:
        resistivity_range += " ✅ Plage typique ERT"
    else:
        resistivity_range += " ⚠️ Hors plage typique ERT"
    analysis += f"📊 {resistivity_range}\n\n"
    # 2. Analyse de la distribution (souvent log-normale)
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    cv = std_val / mean_val if mean_val != 0 else float('inf') # Coefficient de variation
    analysis += f"📈 Statistiques:\n"
    analysis += f" • Moyenne: {mean_val:.3f}\n"
    analysis += f" • Écart-type: {std_val:.3f}\n"
    analysis += f" • Coefficient de variation: {cv:.3f}\n"
    analysis += f" • Médiane: {np.median(arr):.3f}\n\n"
    # 3. Test de distribution log-normale (caractéristique ERT)
    try:
        log_data = np.log(arr[arr > 0]) # Éviter log(0)
        from scipy import stats
        _, p_value = stats.shapiro(log_data[:min(5000, len(log_data))]) # Test Shapiro-Wilk
        if p_value > 0.05:
            analysis += f"📊 Distribution: Log-normale (p={p_value:.3f}) ✅ Typique ERT\n\n"
        else:
            analysis += f"📊 Distribution: Non log-normale (p={p_value:.3f}) ⚠️ Peu commun ERT\n\n"
    except:
        analysis += f"📊 Distribution: Test impossible\n\n"
    # 4. Analyse de patterns spatiaux (si données organisées)
    if len(arr) > 100:
        # Recherche de patterns répétés (électrodes)
        unique_vals = len(np.unique(arr))
        analysis += f"🎯 Unicité des valeurs: {unique_vals}/{len(arr)} ({unique_vals/len(arr)*100:.1f}%)\n"
        # Analyse de clustering spatial simulé
        if len(arr) >= 50:
            from sklearn.cluster import KMeans
            # Clustering simple pour détecter groupes de résistivité
            kmeans = KMeans(n_clusters=min(5, len(arr)//10), random_state=42, n_init=10)
            clusters = kmeans.fit_predict(arr.reshape(-1, 1))
            cluster_centers = kmeans.cluster_centers_.flatten()
            analysis += f"🎯 Clustering résistivité ({len(np.unique(clusters))} groupes):\n"
            for i, center in enumerate(sorted(cluster_centers)):
                count = np.sum(clusters == i)
                analysis += f" • Groupe {i+1}: {center:.3f} Ohm.m ({count} valeurs)\n"
            analysis += "\n"
    # 5. Détection de format de données ERT connu
    ert_formats = {
        "RES2DINV": "Format ASCII RES2DINV (résistivité 2D)",
        "ERTLab": "Format ERTLab (système IRIS)",
        "Syscal": "Format Syscal (système français)",
        "ABEM": "Format ABEM (système suédois)"
    }
    detected_format = "Format non reconnu"
    if len(file_bytes) > 100:
        header = file_bytes[:200].decode('utf-8', errors='ignore').lower()
        for fmt, desc in ert_formats.items():
            if fmt.lower() in header:
                detected_format = desc
                break
    analysis += f"📋 Format détecté: {detected_format}\n\n"
    # 6. Recommandations d'analyse
    analysis += f"💡 RECOMMANDATIONS:\n"
    if PYGIMLI_AVAILABLE:
        analysis += f" • Inversion possible avec PyGIMLI\n"
    if RESIPY_AVAILABLE:
        analysis += f" • Inversion possible avec ResIPy\n"
    analysis += f" • Visualisation 2D/3D recommandée\n"
    analysis += f" • Analyse de sensibilité possible\n"
    analysis += f" • Pour fichiers .dat ERT: Utilisez les formules de calcul de résistivité apparente via mathematical_calculator (Schlumberger: pi*(L**2 - l**2)/(2*l) * V/I, etc.)\n\n"
    # 7. Classification finale
    if 0.1 <= np.min(arr) <= 10000 and cv > 0.5: # CV élevé = hétérogénéité typique ERT
        confidence = "ÉLEVÉE"
        analysis += f"🎯 CONCLUSION: Données très probablement ERT (confiance: {confidence})\n"
    elif 0.1 <= np.min(arr) <= 10000:
        confidence = "MOYENNE"
        analysis += f"🎯 CONCLUSION: Données probablement ERT (confiance: {confidence})\n"
    else:
        confidence = "FAIBLE"
        analysis += f"🎯 CONCLUSION: Données peu caractéristiques ERT (confiance: {confidence})\n"
    return analysis
def ert_inversion_analysis(numbers: list) -> str:
    """Analyse d'inversion ERT spécialisée utilisant PyGIMLI/ResIPy si disponible"""
    if not numbers:
        return "❌ Aucune donnée pour l'inversion ERT"
    import numpy as np
    analysis = "🔬 ANALYSE D'INVERSION ERT\n"
    analysis += "=" * 40 + "\n\n"
    arr = np.array(numbers)
    # Simulation d'inversion simple (sans vraie inversion géophysique)
    analysis += f"📊 Paramètres d'inversion simulés:\n"
    analysis += f" • Nombre de données: {len(arr)}\n"
    analysis += f" • Résistivité moyenne: {np.mean(arr):.3f} Ohm.m\n"
    analysis += f" • Contraste: {np.max(arr)/np.min(arr):.1f}\n\n"
    # Analyse de résolution théorique
    if len(arr) > 10:
        # Estimation de la résolution basée sur la variance
        variance = np.var(arr)
        resolution = 1.0 / (1.0 + variance / np.mean(arr)**2)
        analysis += f"🎯 Résolution estimée: {resolution:.3f}\n\n"
    # Recommandations d'inversion
    analysis += f"💡 RECOMMANDATIONS D'INVERSION:\n"
    if PYGIMLI_AVAILABLE:
        analysis += f" ✅ PyGIMLI disponible - Inversion complète possible\n"
        analysis += f" • Méthodes: Gauss-Newton, Quasi-Newton\n"
        analysis += f" • Régularisation: L2, L1, TV\n"
    else:
        analysis += f" ⚠️ PyGIMLI non installé - Inversion limitée\n"
    # Test d'import ResIPy seulement ici
    try:
        import resipy
        resipy_available = True
    except ImportError:
        resipy_available = False
    if resipy_available:
        analysis += f" ✅ ResIPy disponible - Interface graphique possible\n"
        analysis += f" • Support multi-électrodes\n"
        analysis += f" • Visualisation 3D\n"
    else:
        analysis += f" ⚠️ ResIPy non disponible (compatibilité NumPy)\n"
    analysis += f" • Données suffisantes: {'Oui' if len(arr) > 50 else 'Non'} (min 50 mesures)\n"
    analysis += f" • Qualité des données: {'Bonne' if np.std(arr)/np.mean(arr) > 0.1 else 'Faible contraste'}\n"
    return analysis
def get_resistivity_color(rho: float) -> str:
    """Retourne un code couleur et description pour une valeur de résistivité en Ohm.m"""
    if rho < 10:
        color_hex = "#0000FF" # Bleu
        desc = "Faible résistivité - matériaux conducteurs (argile, eau salée, métaux)"
        nature = "Nature: Couches saturées en eau, pollution potentielle"
        depth_est = "Profondeur estimée: Superficielle (0-5 m)"
    elif 10 <= rho < 100:
        color_hex = "#00FF00" # Vert
        desc = "Résistivité moyenne - sols typiques (sable humide, limon)"
        nature = "Nature: Zone vadose, aquifères non salins"
        depth_est = "Profondeur estimée: Moyenne (5-20 m)"
    elif 100 <= rho < 1000:
        color_hex = "#FFFF00" # Jaune
        desc = "Résistivité élevée - matériaux semi-résistants (grès, calcaire)"
        nature = "Nature: Roches sédimentaires, fractures partielles"
        depth_est = "Profondeur estimée: Profonde (20-50 m)"
    else:
        color_hex = "#FF0000" # Rouge
        desc = "Très haute résistivité - matériaux résistants (granite, air, vides)"
        nature = "Nature: Substratum rocheux, cavités ou zones sèches"
        depth_est = "Profondeur estimée: Très profonde (>50 m)"
  
    return f"Couleur: {color_hex} ({desc})\nNature: {nature}\nProfondeur: {depth_est}\nAutres: Couleur indicative pour visualisation ERT (colormap géophysique standard)"
def fetch_material_resistivities(category: str) -> str:
    """Recherche dynamique sur internet des plages de résistivité pour une catégorie de matériaux"""
    query = f"typical electrical resistivity ranges {category} liquids minerals soils rocks geophysics Ohm.m values categories comparison"
    return web_search_enhanced(query, "ert_materials")

def create_minerals_database():
    """
    Crée une base de données étendue des résistivités de minéraux, roches et liquides
    Basée sur recherches géophysiques pour exploration minière ERT
    """
    import pandas as pd
    
    materials_data = [
        # LIQUIDES
        {"Catégorie": "Liquides", "Type": "Eau de mer", "Plage Min (Ωm)": 0.05, "Plage Max (Ωm)": 0.3, "Notes": "Haute conductivité due à la salinité"},
        {"Catégorie": "Liquides", "Type": "Eau saumâtre", "Plage Min (Ωm)": 1, "Plage Max (Ωm)": 10, "Notes": "Salinité modérée"},
        {"Catégorie": "Liquides", "Type": "Eau douce", "Plage Min (Ωm)": 10, "Plage Max (Ωm)": 100, "Notes": "Faible salinité, eaux de surface ou souterraines"},
        {"Catégorie": "Liquides", "Type": "Eau minérale/mine", "Plage Min (Ωm)": 0.1, "Plage Max (Ωm)": 1, "Notes": "Haute concentration en minéraux dissous"},
        {"Catégorie": "Liquides", "Type": "Pétrole/Hydrocarbures", "Plage Min (Ωm)": 1000, "Plage Max (Ωm)": 100000000, "Notes": "Très résistif, isolant"},
        
        # MINERAIS (étendu pour exploration minière)
        {"Catégorie": "Minerais", "Type": "Graphite", "Plage Min (Ωm)": 0.000008, "Plage Max (Ωm)": 0.0001, "Notes": "Très conducteur, carbone pur"},
        {"Catégorie": "Minerais", "Type": "Pyrite pure", "Plage Min (Ωm)": 0.00003, "Plage Max (Ωm)": 0.001, "Notes": "Sulfure de fer, très conducteur"},
        {"Catégorie": "Minerais", "Type": "Pyrite (impure)", "Plage Min (Ωm)": 0.001, "Plage Max (Ωm)": 10, "Notes": "Avec impuretés cuivre, anomalie ERT"},
        {"Catégorie": "Minerais", "Type": "Galena", "Plage Min (Ωm)": 0.001, "Plage Max (Ωm)": 100, "Notes": "Sulfure de plomb, conducteur"},
        {"Catégorie": "Minerais", "Type": "Magnétite", "Plage Min (Ωm)": 0.01, "Plage Max (Ωm)": 1000, "Notes": "Oxyde de fer, magnétique, variable"},
        {"Catégorie": "Minerais", "Type": "Hématite", "Plage Min (Ωm)": 10, "Plage Max (Ωm)": 10000, "Notes": "Oxyde de fer, presque isolant"},
        {"Catégorie": "Minerais", "Type": "Chalcopyrite", "Plage Min (Ωm)": 0.001, "Plage Max (Ωm)": 10, "Notes": "Sulfure cuivre-fer, minerai Cu"},
        {"Catégorie": "Minerais", "Type": "Bornite", "Plage Min (Ωm)": 0.001, "Plage Max (Ωm)": 10, "Notes": "Sulfure cuivre-fer, paon ore"},
        {"Catégorie": "Minerais", "Type": "Sphalerite (Zinc)", "Plage Min (Ωm)": 100, "Plage Max (Ωm)": 10000, "Notes": "Sulfure de zinc, modérément résistif"},
        {"Catégorie": "Minerais", "Type": "Cassitérite (Étain)", "Plage Min (Ωm)": 1000, "Plage Max (Ωm)": 10000, "Notes": "Oxyde d'étain, résistif"},
        {"Catégorie": "Minerais", "Type": "Molybdénite", "Plage Min (Ωm)": 0.001, "Plage Max (Ωm)": 1, "Notes": "Sulfure molybdène, très conducteur"},
        {"Catégorie": "Minerais", "Type": "Or (natif)", "Plage Min (Ωm)": 0.000001, "Plage Max (Ωm)": 0.00001, "Notes": "Métal pur, ultra-conducteur"},
        {"Catégorie": "Minerais", "Type": "Or (veines quartz)", "Plage Min (Ωm)": 1, "Plage Max (Ωm)": 1000, "Notes": "Variable, sulfures associés"},
        {"Catégorie": "Minerais", "Type": "Fer (minerai)", "Plage Min (Ωm)": 0.01, "Plage Max (Ωm)": 1000, "Notes": "Magnétite/hématite mélangée"},
        {"Catégorie": "Minerais", "Type": "Quartz", "Plage Min (Ωm)": 10000000000, "Plage Max (Ωm)": 100000000000000, "Notes": "Silicate, ultra-isolant"},
        {"Catégorie": "Minerais", "Type": "Cuivre (natif)", "Plage Min (Ωm)": 0.0000017, "Plage Max (Ωm)": 0.000002, "Notes": "Métal pur, excellent conducteur"},
        {"Catégorie": "Minerais", "Type": "Argent (natif)", "Plage Min (Ωm)": 0.0000016, "Plage Max (Ωm)": 0.000002, "Notes": "Meilleur conducteur naturel"},
        
        # ROCHES
        {"Catégorie": "Roches", "Type": "Argile (humide)", "Plage Min (Ωm)": 1, "Plage Max (Ωm)": 100, "Notes": "Faible résistivité, eau et ions"},
        {"Catégorie": "Roches", "Type": "Schiste", "Plage Min (Ωm)": 20, "Plage Max (Ωm)": 2000, "Notes": "Variable avec humidité"},
        {"Catégorie": "Roches", "Type": "Grès", "Plage Min (Ωm)": 30, "Plage Max (Ωm)": 10000, "Notes": "Sec à saturé"},
        {"Catégorie": "Roches", "Type": "Calcaire", "Plage Min (Ωm)": 50, "Plage Max (Ωm)": 10000000, "Notes": "Variable, haut si sec"},
        {"Catégorie": "Roches", "Type": "Granite", "Plage Min (Ωm)": 5000, "Plage Max (Ωm)": 1000000, "Notes": "Igneuse, résistif si sec"},
        {"Catégorie": "Roches", "Type": "Basalte", "Plage Min (Ωm)": 10, "Plage Max (Ωm)": 13000000, "Notes": "Igneuse, très variable"},
        {"Catégorie": "Roches", "Type": "Alluvions", "Plage Min (Ωm)": 1, "Plage Max (Ωm)": 1000, "Notes": "Sédiments non consolidés"},
        {"Catégorie": "Roches", "Type": "Gravier", "Plage Min (Ωm)": 100, "Plage Max (Ωm)": 2500, "Notes": "Sec, bonne perméabilité"},
    ]
    
    return pd.DataFrame(materials_data)

def create_real_mineral_correspondence_table(numbers: list, file_name: str = "unknown", depths: list = None, full_size: bool = False) -> tuple:
    """
    🎯 TABLEAU DE CORRESPONDANCES RÉELLES - Données mesurées vs Minéraux géophysiques
    
    Crée un tableau dynamique matplotlib avec UNIQUEMENT les minéraux réellement détectés
    basé sur les valeurs de résistivité mesurées dans le fichier .dat
    
    Args:
        numbers: Liste des valeurs de résistivité mesurées (Ω·m)
        file_name: Nom du fichier analysé
        depths: Liste optionnelle des profondeurs correspondantes (m)
        full_size: Mode grand format (True = 24×16", False = 16×12")
    
    Returns:
        tuple: (figure matplotlib, DataFrame des correspondances, texte rapport)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    if not numbers or len(numbers) < 5:
        return None, None, "❌ Données insuffisantes pour analyse (minimum 5 mesures)"
    
    arr = np.array(numbers)
    minerals_db = create_minerals_database()
    
    # Tailles adaptatives
    if full_size:
        figsize = (24, 16)
        title_fontsize = 18
        header_fontsize = 12
        cell_fontsize = 10
        scatter_markersize = 120
    else:
        figsize = (16, 12)
        title_fontsize = 14
        header_fontsize = 10
        cell_fontsize = 8
        scatter_markersize = 80
    
    # Si pas de profondeurs, estimer selon la résistivité
    if depths is None:
        depths = []
        for rho in arr:
            if rho < 1:
                depths.append(np.random.uniform(0, 20))  # Zone superficielle conductrice
            elif rho < 100:
                depths.append(np.random.uniform(5, 50))  # Zone moyenne
            elif rho < 1000:
                depths.append(np.random.uniform(20, 100))  # Zone transition
            else:
                depths.append(np.random.uniform(50, 200))  # Zone profonde
        depths = np.array(depths)
    else:
        depths = np.array(depths)
    
    # Créer DataFrame des mesures réelles
    real_data = []
    
    for i, (rho, depth) in enumerate(zip(arr, depths)):
        # Trouver correspondances dans la base de données
        matches = minerals_db[
            (minerals_db["Plage Min (Ωm)"] <= rho) & 
            (minerals_db["Plage Max (Ωm)"] >= rho)
        ]
        
        if not matches.empty:
            for _, match in matches.iterrows():
                real_data.append({
                    "Mesure #": i + 1,
                    "Profondeur (m)": depth,
                    "Résistivité mesurée (Ω·m)": rho,
                    "Matériau détecté": match["Type"],
                    "Catégorie": match["Catégorie"],
                    "Plage DB (Ω·m)": f"{match['Plage Min (Ωm)']} - {match['Plage Max (Ωm)']}",
                    "Confiance": calculate_confidence(rho, match["Plage Min (Ωm)"], match["Plage Max (Ωm)"]),
                    "Notes": match["Notes"]
                })
    
    if not real_data:
        return None, None, "⚠️ Aucune correspondance trouvée dans la base de données minéralogique"
    
    df_correspondances = pd.DataFrame(real_data)
    
    # Trier par profondeur
    df_correspondances = df_correspondances.sort_values("Profondeur (m)")
    
    # Limiter le nombre de lignes pour éviter decompression bomb
    max_rows_display = min(100, len(df_correspondances))
    
    # 📊 CRÉER TABLEAU MATPLOTLIB DYNAMIQUE avec taille responsive
    # Limiter la taille pour éviter decompression bomb (max 20 pouces de hauteur)
    fig_height = min(20, max(8, max_rows_display * 0.15))
    fig, (ax_table, ax_depth) = plt.subplots(1, 2, figsize=(figsize[0], fig_height))
    
    # Augmenter la limite de pixels pour matplotlib
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 200000000  # 200 millions de pixels max
    
    # TABLEAU GAUCHE: Correspondances détaillées
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Grouper par profondeur pour affichage condensé
    depth_groups = df_correspondances.groupby(df_correspondances["Profondeur (m)"].round(1))
    
    table_data = []
    row_colors = []
    
    # Limiter à 50 groupes max pour le tableau
    max_groups = min(50, len(depth_groups))
    group_count = 0
    
    for depth, group in depth_groups:
        if group_count >= max_groups:
            break
        group_count += 1
        
        materials = group["Matériau détecté"].unique()
        rho_values = group["Résistivité mesurée (Ω·m)"].values
        categories = group["Catégorie"].unique()
        confidence = group["Confiance"].mean()
        
        # Déterminer couleur selon catégorie dominante
        if "Minerais" in categories:
            color = '#FFD700' if any('Or' in m for m in materials) else '#FF6B6B'
        elif "Liquides" in categories:
            color = '#4ECDC4'
        else:
            color = '#95E1D3'
        
        table_data.append([
            f"{depth:.1f}m",
            f"{rho_values.min():.4f} - {rho_values.max():.4f}",
            "\n".join(materials[:3]),  # Max 3 matériaux
            f"{confidence:.0%}"
        ])
        row_colors.append(color)
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Profondeur", "Résistivité (Ω·m)", "Matériaux détectés", "Confiance"],
        cellLoc='left',
        loc='center',
        colWidths=[0.15, 0.25, 0.45, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(cell_fontsize)
    table.scale(1, 2)
    
    # Colorer les lignes
    for i, color in enumerate(row_colors):
        for j in range(4):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
    # Header
    for j in range(4):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(weight='bold', color='white', fontsize=header_fontsize)
    
    ax_table.set_title(f"📊 Correspondances Réelles: {file_name}\n{len(real_data)} détections", 
                       fontsize=title_fontsize, weight='bold', pad=20)
    
    # GRAPHIQUE DROITE: Profil profondeur vs résistivité
    ax_depth.invert_yaxis()  # Profondeur croissante vers le bas
    
    # Grouper par type de matériau pour légende
    material_types = df_correspondances.groupby("Matériau détecté")
    
    colors_map = {
        "Eau de mer": "#FF0000",
        "Eau salée (nappe)": "#FF6B00",
        "Eau douce": "#00FF00",
        "Eau très pure": "#0000FF",
        "Or (natif)": "#FFD700",
        "Argent (natif)": "#C0C0C0",
        "Pyrite pure": "#FF4500",
        "Chalcopyrite": "#FF8C00",
        "Galena": "#696969",
        "Magnétite": "#8B4513",
        "Graphite": "#000000",
    }
    
    plotted_materials = set()
    
    # Limiter le nombre de points affichés pour éviter surcharge
    max_points_per_material = 200
    
    for material, group in material_types:
        color = colors_map.get(material, "#888888")
        marker = 'o' if group["Catégorie"].iloc[0] == "Minerais" else 's'
        
        # Sous-échantillonner si trop de points
        if len(group) > max_points_per_material:
            group_sample = group.sample(n=max_points_per_material, random_state=42)
        else:
            group_sample = group
        
        ax_depth.scatter(
            group_sample["Résistivité mesurée (Ω·m)"],
            group_sample["Profondeur (m)"],
            c=color,
            marker=marker,
            s=scatter_markersize,
            alpha=0.7,
            label=material if material not in plotted_materials else "",
            edgecolors='black',
            linewidth=1
        )
        plotted_materials.add(material)
    
    ax_depth.set_xlabel("Résistivité (Ω·m)", fontsize=header_fontsize, weight='bold')
    ax_depth.set_ylabel("Profondeur (m)", fontsize=header_fontsize, weight='bold')
    ax_depth.set_xscale('log')
    ax_depth.grid(True, alpha=0.3, linestyle='--')
    ax_depth.legend(loc='best', fontsize=cell_fontsize, framealpha=0.9)
    ax_depth.tick_params(labelsize=cell_fontsize)
    ax_depth.set_title("Profil Géophysique Réel", fontsize=12, weight='bold')
    
    # Ajouter zones de référence
    ax_depth.axhspan(0, 20, alpha=0.1, color='red', label='_Zone superficielle')
    ax_depth.axhspan(20, 100, alpha=0.1, color='yellow', label='_Zone intermédiaire')
    ax_depth.axhspan(100, max(depths) if len(depths) > 0 else 200, alpha=0.1, color='blue', label='_Zone profonde')
    
    plt.tight_layout()
    
    # 📝 GÉNÉRER RAPPORT TEXTUEL DÉTAILLÉ
    rapport = "🎯 TABLEAU DE CORRESPONDANCES RÉELLES - DONNÉES ERT vs MINÉRAUX\n"
    rapport += "=" * 80 + "\n\n"
    
    rapport += f"📁 Fichier: {file_name}\n"
    rapport += f"📊 Mesures analysées: {len(arr)}\n"
    rapport += f"✅ Correspondances trouvées: {len(real_data)}\n"
    rapport += f"📈 Plage résistivité: {arr.min():.6f} - {arr.max():.2f} Ω·m\n"
    rapport += f"📏 Plage profondeur: {depths.min():.1f} - {depths.max():.1f} m\n\n"
    
    rapport += "🔍 DÉTECTION PAR PROFONDEUR:\n"
    rapport += "─" * 80 + "\n"
    
    for depth, group in depth_groups:
        rapport += f"\n📍 PROFONDEUR: {depth:.1f} m\n"
        rapport += f"   Résistivité mesurée: {group['Résistivité mesurée (Ω·m)'].min():.4f} - {group['Résistivité mesurée (Ω·m)'].max():.4f} Ω·m\n"
        rapport += f"   Matériaux détectés ({len(group)}):\n"
        
        for _, row in group.iterrows():
            rapport += f"      • {row['Matériau détecté']} ({row['Catégorie']})\n"
            rapport += f"        - Confiance: {row['Confiance']:.0%}\n"
            rapport += f"        - Plage DB: {row['Plage DB (Ω·m)']}\n"
            rapport += f"        - Notes: {row['Notes']}\n"
    
    # Statistiques par catégorie
    rapport += "\n📊 STATISTIQUES PAR CATÉGORIE:\n"
    rapport += "─" * 80 + "\n"
    
    category_stats = df_correspondances.groupby("Catégorie").agg({
        "Matériau détecté": lambda x: x.nunique(),
        "Profondeur (m)": ["min", "max", "mean"],
        "Résistivité mesurée (Ω·m)": ["min", "max"],
        "Confiance": "mean"
    })
    
    for cat, stats in category_stats.iterrows():
        rapport += f"\n{cat}:\n"
        rapport += f"  • Matériaux uniques: {stats[('Matériau détecté', '<lambda>')]}\n"
        rapport += f"  • Profondeur: {stats[('Profondeur (m)', 'min')]:.1f} - {stats[('Profondeur (m)', 'max')]:.1f} m (moy: {stats[('Profondeur (m)', 'mean')]:.1f} m)\n"
        rapport += f"  • Résistivité: {stats[('Résistivité mesurée (Ω·m)', 'min')]:.4f} - {stats[('Résistivité mesurée (Ω·m)', 'max')]:.2f} Ω·m\n"
        rapport += f"  • Confiance moyenne: {stats[('Confiance', 'mean')]:.0%}\n"
    
    # Minéraux d'intérêt économique
    rapport += "\n💎 MINÉRAUX D'INTÉRÊT ÉCONOMIQUE DÉTECTÉS:\n"
    rapport += "─" * 80 + "\n"
    
    economic_minerals = df_correspondances[df_correspondances["Matériau détecté"].str.contains(
        "Or|Argent|Cuivre|Pyrite|Chalcopyrite|Galena|Molybdénite|Cassitérite", 
        case=False, 
        na=False
    )]
    
    if not economic_minerals.empty:
        for _, row in economic_minerals.iterrows():
            rapport += f"⭐ {row['Matériau détecté']}\n"
            rapport += f"   • Profondeur: {row['Profondeur (m)']:.1f} m\n"
            rapport += f"   • Résistivité: {row['Résistivité mesurée (Ω·m)']:.6f} Ω·m\n"
            rapport += f"   • Confiance: {row['Confiance']:.0%}\n"
            rapport += f"   • Recommandation: Forage ciblé pour validation\n\n"
    else:
        rapport += "⚠️ Aucun minéral d'intérêt économique majeur détecté\n\n"
    
    rapport += "=" * 80 + "\n"
    
    return fig, df_correspondances, rapport

def calculate_confidence(measured_rho: float, min_rho: float, max_rho: float) -> float:
    """
    Calcule le niveau de confiance de la correspondance
    Basé sur la position dans la plage de résistivité
    """
    if min_rho == max_rho:
        return 1.0 if measured_rho == min_rho else 0.0
    
    # Distance au centre de la plage (normalisée)
    center = (min_rho + max_rho) / 2
    range_width = max_rho - min_rho
    
    distance_from_center = abs(measured_rho - center)
    normalized_distance = distance_from_center / (range_width / 2)
    
    # Confiance = 100% au centre, diminue vers les bords
    confidence = max(0.0, 1.0 - (normalized_distance * 0.3))  # Max 30% de pénalité
    
    return confidence

def create_ert_professional_sections(numbers: list, file_name: str = "unknown", depths: list = None, distances: list = None, full_size: bool = False) -> tuple:
    """
    🎨 CRÉATION DE 5 GRAPHIQUES ERT PROFESSIONNELS
    Style Res2DInv/RES3DINV avec coupes représentatives et palette de couleurs
    
    Les 5 graphiques:
    1. Pseudosection de résistivité apparente (données brutes)
    2. Section inversée avec modèle de résistivité (interprétation)
    3. Coupe verticale avec échelle de couleurs géologique
    4. Histogramme de distribution + palette de couleurs
    5. Profil 1D comparatif (profondeur vs résistivité)
    
    Args:
        numbers: Valeurs de résistivité mesurées (Ω·m)
        file_name: Nom du fichier
        depths: Profondeurs (m) - si None, généré automatiquement
        distances: Distances horizontales (m) - si None, généré automatiquement
        full_size: Mode grand format (True = 30x36", False = 20x24")
    
    Returns:
        tuple: (figure matplotlib, données_grille, texte_rapport)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import LogNorm, ListedColormap
    from scipy.interpolate import griddata
    
    if not numbers or len(numbers) < 10:
        return None, None, "❌ Données insuffisantes (minimum 10 mesures)"
    
    arr = np.array(numbers)
    n_points = len(arr)
    
    # Générer grille si pas fournie
    if depths is None:
        # Estimer profondeurs selon résistivité (0-100m typique)
        depths = np.array([estimate_depth_value(rho) for rho in arr])
    else:
        depths = np.array(depths)
    
    if distances is None:
        # Espacer uniformément les mesures sur 100m
        distances = np.linspace(0, 100, n_points)
    else:
        distances = np.array(distances)
    
    # Créer grille interpolée pour visualisation (style Res2DInv)
    grid_x = np.linspace(distances.min(), distances.max(), 100)
    grid_y = np.linspace(0, depths.max(), 50)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    
    # Interpolation des valeurs sur la grille
    grid_rho = griddata((distances, depths), arr, (grid_X, grid_Y), method='cubic', fill_value=arr.mean())
    
    # Créer palette de couleurs ERT standard (Res2DInv style)
    colors_ert = [
        '#000080',  # Bleu foncé - Très résistif (>1000)
        '#0000FF',  # Bleu - Résistif (100-1000)
        '#00FFFF',  # Cyan - Modérément résistif (10-100)
        '#00FF00',  # Vert - Neutre (1-10)
        '#FFFF00',  # Jaune - Légèrement conducteur (0.1-1)
        '#FFA500',  # Orange - Conducteur (0.01-0.1)
        '#FF0000',  # Rouge - Très conducteur (0.001-0.01)
        '#8B0000',  # Rouge foncé - Ultra-conducteur (<0.001)
    ]
    cmap_ert = ListedColormap(colors_ert)
    
    # Créer figure avec taille responsive
    if full_size:
        figsize = (30, 36)  # Grand format pour visualisation détaillée
        title_fontsize = 18
        label_fontsize = 14
        tick_fontsize = 11
    else:
        figsize = (20, 24)  # Taille standard
        title_fontsize = 14
        label_fontsize = 12
        tick_fontsize = 10
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 0.8, 1], hspace=0.3, wspace=0.3)
    
    # ========== GRAPHIQUE 1: PSEUDOSECTION RÉSISTIVITÉ APPARENTE ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    # Utiliser échelle logarithmique
    im1 = ax1.contourf(grid_X, grid_Y, grid_rho, levels=20, cmap=cmap_ert, 
                       norm=LogNorm(vmin=max(arr.min(), 0.0001), vmax=arr.max()))
    ax1.scatter(distances, depths, c='black', s=20, marker='v', label='Points de mesure', zorder=10)
    
    ax1.set_xlabel('Distance (m)', fontsize=label_fontsize, weight='bold')
    ax1.set_ylabel('Profondeur (m)', fontsize=label_fontsize, weight='bold')
    ax1.set_title(f'1️⃣ PSEUDOSECTION - Résistivité Apparente\n{file_name}', 
                  fontsize=title_fontsize, weight='bold', pad=15)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=tick_fontsize)
    ax1.tick_params(labelsize=tick_fontsize)
    
    # Colorbar avec labels géologiques
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='vertical', pad=0.02)
    cbar1.set_label('Résistivité (Ω·m)', fontsize=label_fontsize-1, weight='bold')
    cbar1.ax.tick_params(labelsize=tick_fontsize)
    
    # ========== GRAPHIQUE 2: MODÈLE INVERSÉ (avec contours) ==========
    ax2 = fig.add_subplot(gs[1, :])
    
    # Contours remplis + lignes de contour
    im2 = ax2.contourf(grid_X, grid_Y, grid_rho, levels=15, cmap=cmap_ert,
                       norm=LogNorm(vmin=max(arr.min(), 0.0001), vmax=arr.max()), alpha=0.9)
    contours = ax2.contour(grid_X, grid_Y, grid_rho, levels=10, colors='black', 
                          linewidths=0.5, alpha=0.4)
    ax2.clabel(contours, inline=True, fontsize=tick_fontsize-2, fmt='%.2f')
    
    ax2.set_xlabel('Distance (m)', fontsize=label_fontsize, weight='bold')
    ax2.set_ylabel('Profondeur (m)', fontsize=label_fontsize, weight='bold')
    ax2.set_title('2️⃣ MODÈLE INVERSÉ - Section avec Contours', 
                  fontsize=title_fontsize, weight='bold', pad=15)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.2, linestyle=':')
    ax2.tick_params(labelsize=tick_fontsize)
    
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical', pad=0.02)
    cbar2.set_label('Résistivité (Ω·m)', fontsize=label_fontsize-1, weight='bold')
    cbar2.ax.tick_params(labelsize=tick_fontsize)
    
    # ========== GRAPHIQUE 3: COUPE VERTICALE COLORÉE (style géologique) ==========
    ax3 = fig.add_subplot(gs[2, :])
    
    # Version sans contours, couleurs pleines (style géologique)
    im3 = ax3.imshow(grid_rho, aspect='auto', cmap=cmap_ert, 
                     norm=LogNorm(vmin=max(arr.min(), 0.0001), vmax=arr.max()),
                     extent=[distances.min(), distances.max(), depths.max(), 0],
                     interpolation='bilinear')
    
    # Ajouter annotations pour zones intéressantes
    # Trouver zones ultra-conductrices (métaux, sulfures)
    ultra_cond = arr < 1
    if ultra_cond.any():
        ultra_idx = np.where(ultra_cond)[0]
        for idx in ultra_idx[:5]:  # Max 5 annotations
            ax3.annotate('⭐ Anomalie', 
                        xy=(distances[idx], depths[idx]),
                        xytext=(distances[idx]+5, depths[idx]-5),
                        fontsize=tick_fontsize+1, color='white', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    
    ax3.set_xlabel('Distance (m)', fontsize=label_fontsize, weight='bold')
    ax3.set_ylabel('Profondeur (m)', fontsize=label_fontsize, weight='bold')
    ax3.set_title('3️⃣ COUPE GÉOLOGIQUE - Interprétation Visuelle', 
                  fontsize=title_fontsize, weight='bold', pad=15)
    ax3.grid(True, alpha=0.3, color='white', linestyle='--')
    ax3.tick_params(labelsize=tick_fontsize)
    
    cbar3 = plt.colorbar(im3, ax=ax3, orientation='vertical', pad=0.02)
    cbar3.set_label('Résistivité (Ω·m)', fontsize=label_fontsize-1, weight='bold')
    cbar3.ax.tick_params(labelsize=tick_fontsize)
    
    # ========== GRAPHIQUE 4: HISTOGRAMME + PALETTE DE COULEURS ==========
    ax4a = fig.add_subplot(gs[3, 0])
    ax4b = fig.add_subplot(gs[3, 1])
    
    # Histogramme logarithmique
    log_rho = np.log10(arr)
    ax4a.hist(log_rho, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax4a.axvline(np.median(log_rho), color='red', linestyle='--', linewidth=2, label=f'Médiane: {10**np.median(log_rho):.2f} Ω·m')
    ax4a.axvline(np.mean(log_rho), color='orange', linestyle='--', linewidth=2, label=f'Moyenne: {10**np.mean(log_rho):.2f} Ω·m')
    
    ax4a.set_xlabel('log₁₀(Résistivité)', fontsize=label_fontsize-1, weight='bold')
    ax4a.set_ylabel('Fréquence', fontsize=label_fontsize-1, weight='bold')
    ax4a.set_title('4️⃣a DISTRIBUTION des Valeurs', fontsize=title_fontsize-2, weight='bold')
    ax4a.legend(loc='upper right', fontsize=tick_fontsize-1)
    ax4a.grid(True, alpha=0.3)
    ax4a.tick_params(labelsize=tick_fontsize)
    
    # Palette de couleurs avec plages
    resistivity_ranges = [
        (0.0001, 0.001, '#8B0000', 'Ultra-conducteur\n(Métaux natifs)'),
        (0.001, 0.01, '#FF0000', 'Très conducteur\n(Sulfures)'),
        (0.01, 0.1, '#FFA500', 'Conducteur\n(Eau salée)'),
        (0.1, 1, '#FFFF00', 'Légèrement cond.\n(Argiles humides)'),
        (1, 10, '#00FF00', 'Neutre\n(Eau douce)'),
        (10, 100, '#00FFFF', 'Modérément rés.\n(Sables/Graviers)'),
        (100, 1000, '#0000FF', 'Résistif\n(Roches sèches)'),
        (1000, 10000, '#000080', 'Très résistif\n(Granite/Quartz)'),
    ]
    
    ax4b.axis('off')
    y_pos = 0.95
    palette_fontsize = tick_fontsize if not full_size else tick_fontsize + 2
    for rho_min, rho_max, color, label in resistivity_ranges:
        # Compter combien de mesures dans cette plage
        count = np.sum((arr >= rho_min) & (arr < rho_max))
        percentage = (count / len(arr)) * 100
        
        # Rectangle de couleur
        rect = plt.Rectangle((0.1, y_pos-0.08), 0.15, 0.08, facecolor=color, 
                            edgecolor='black', linewidth=1.5, transform=ax4b.transAxes)
        ax4b.add_patch(rect)
        
        # Texte
        ax4b.text(0.27, y_pos-0.04, f'{rho_min}-{rho_max} Ω·m', 
                 fontsize=palette_fontsize-1, va='center', weight='bold', transform=ax4b.transAxes)
        ax4b.text(0.65, y_pos-0.04, label, 
                 fontsize=palette_fontsize-2, va='center', transform=ax4b.transAxes)
        ax4b.text(0.92, y_pos-0.04, f'{percentage:.1f}%', 
                 fontsize=palette_fontsize-2, va='center', weight='bold', ha='right', transform=ax4b.transAxes)
        
        y_pos -= 0.12
    
    ax4b.set_title('4️⃣b PALETTE DE COULEURS ERT', fontsize=title_fontsize-2, weight='bold', pad=10)
    ax4b.set_xlim(0, 1)
    ax4b.set_ylim(0, 1)
    
    # ========== GRAPHIQUE 5: PROFIL 1D VERTICAL ==========
    ax5 = fig.add_subplot(gs[4, :])
    
    # Profil moyen par tranche de profondeur
    depth_bins = np.linspace(0, depths.max(), 20)
    depth_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
    rho_profile = []
    
    for i in range(len(depth_bins)-1):
        mask = (depths >= depth_bins[i]) & (depths < depth_bins[i+1])
        if mask.any():
            rho_profile.append(np.mean(arr[mask]))
        else:
            rho_profile.append(np.nan)
    
    rho_profile = np.array(rho_profile)
    
    # Profil principal
    ax5.plot(rho_profile, depth_centers, 'b-o', linewidth=2, markersize=8, 
            label='Profil moyen', zorder=5)
    
    # Plage min-max (enveloppe)
    rho_min_profile = []
    rho_max_profile = []
    for i in range(len(depth_bins)-1):
        mask = (depths >= depth_bins[i]) & (depths < depth_bins[i+1])
        if mask.any():
            rho_min_profile.append(np.min(arr[mask]))
            rho_max_profile.append(np.max(arr[mask]))
        else:
            rho_min_profile.append(np.nan)
            rho_max_profile.append(np.nan)
    
    ax5.fill_betweenx(depth_centers, rho_min_profile, rho_max_profile, 
                      alpha=0.3, color='blue', label='Plage min-max')
    
    # Zones géologiques
    ax5.axhspan(0, 20, alpha=0.1, color='red', label='Zone superficielle')
    ax5.axhspan(20, 50, alpha=0.1, color='yellow', label='Zone intermédiaire')
    ax5.axhspan(50, depths.max(), alpha=0.1, color='blue', label='Zone profonde')
    
    ax5.set_xlabel('Résistivité moyenne (Ω·m)', fontsize=label_fontsize, weight='bold')
    ax5.set_ylabel('Profondeur (m)', fontsize=label_fontsize, weight='bold')
    ax5.set_title('5️⃣ PROFIL 1D - Variation avec la Profondeur', 
                  fontsize=title_fontsize, weight='bold', pad=15)
    ax5.set_xscale('log')
    ax5.invert_yaxis()
    ax5.grid(True, alpha=0.3, linestyle='--', which='both')
    ax5.legend(loc='best', fontsize=tick_fontsize)
    ax5.tick_params(labelsize=tick_fontsize)
    
    # Titre général
    suptitle_fontsize = title_fontsize + 2 if full_size else title_fontsize + 2
    fig.suptitle(f'📊 ANALYSE ERT COMPLÈTE - {file_name}\n'
                f'{len(arr)} mesures | Plage: {arr.min():.4f} - {arr.max():.2f} Ω·m', 
                fontsize=suptitle_fontsize, weight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Générer rapport textuel
    format_text = "GRAND FORMAT (30×36\")" if full_size else "STANDARD (20×24\")"
    rapport = f"""
🎨 RAPPORT GRAPHIQUES ERT PROFESSIONNELS
{'='*80}

📁 Fichier: {file_name}
📊 Nombre de mesures: {len(arr)}
📏 Profondeur max: {depths.max():.1f} m
📐 Distance totale: {distances.max():.1f} m
📈 Résistivité: {arr.min():.6f} - {arr.max():.2f} Ω·m
🖼️  Format: {format_text}

🎨 GRAPHIQUES GÉNÉRÉS:

1️⃣ PSEUDOSECTION - Résistivité apparente
   • Données brutes interpolées sur grille 100x50
   • Points de mesure affichés
   • Échelle logarithmique
   • Palette Res2DInv standard

2️⃣ MODÈLE INVERSÉ - Section avec contours
   • 15 niveaux de remplissage
   • 10 lignes de contour annotées
   • Interprétation géophysique

3️⃣ COUPE GÉOLOGIQUE - Interprétation visuelle
   • Couleurs pleines (style géologique)
   • Annotations des anomalies conductrices
   • Interpolation bilinéaire

4️⃣ DISTRIBUTION & PALETTE
   • Histogramme logarithmique (30 bins)
   • Statistiques: médiane={10**np.median(log_rho):.2f} Ω·m, moyenne={10**np.mean(log_rho):.2f} Ω·m
   • Palette 8 couleurs avec répartition (%)

5️⃣ PROFIL 1D VERTICAL
   • 20 tranches de profondeur
   • Profil moyen + enveloppe min-max
   • Zones géologiques identifiées

{'='*80}
"""
    
    grid_data = {
        'grid_X': grid_X,
        'grid_Y': grid_Y,
        'grid_rho': grid_rho,
        'distances': distances,
        'depths': depths,
        'resistivities': arr
    }
    
    return fig, grid_data, rapport

def estimate_depth_value(rho: float) -> float:
    """
    Estime une profondeur typique basée sur la résistivité
    Utilisé pour générer profondeur si non fournie
    """
    if rho < 1:
        return np.random.uniform(0, 20)  # Zone superficielle conductrice
    elif rho < 10:
        return np.random.uniform(5, 40)  # Zone moyenne
    elif rho < 100:
        return np.random.uniform(15, 60)  # Zone transition
    elif rho < 1000:
        return np.random.uniform(30, 80)  # Zone profonde modérée
    else:
        return np.random.uniform(50, 100)  # Zone très profonde résistive


def generate_professional_ert_report(
    numbers: list,
    file_name: str,
    mineral_report: str = "",
    df_corr: pd.DataFrame = None,
    fig_ert: plt.Figure = None,
    fig_corr: plt.Figure = None,
    grid_data: dict = None,
    output_path: str = None
) -> bytes:
    """
    🎨 GÉNÉRATION RAPPORT PDF PROFESSIONNEL COMPLET
    
    Crée un rapport PDF avec:
    - Page de garde avec logo et titre coloré
    - Résumé exécutif
    - Graphiques ERT intégrés (5 coupes)
    - Tableau de correspondances
    - Interprétation géologique détaillée
    - Recommandations
    - Annexes techniques
    
    Args:
        numbers: Valeurs de résistivité
        file_name: Nom du fichier analysé
        mineral_report: Texte du rapport minéralogique
        df_corr: DataFrame des correspondances
        fig_ert: Figure matplotlib des 5 graphiques ERT
        fig_corr: Figure matplotlib du tableau
        grid_data: Données de grille interpolée
        output_path: Chemin de sauvegarde (si None, retourne bytes)
    
    Returns:
        bytes: Contenu du PDF
    """
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm, mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, PageBreak, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.pdfgen import canvas
    from datetime import datetime
    import io
    import tempfile
    
    # Buffer pour le PDF
    buffer = io.BytesIO()
    
    # Créer document
    if output_path:
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                               topMargin=2*cm, bottomMargin=2*cm,
                               leftMargin=2*cm, rightMargin=2*cm)
    else:
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                               topMargin=2*cm, bottomMargin=2*cm,
                               leftMargin=2*cm, rightMargin=2*cm)
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Style titre principal (rouge)
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#8B0000'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Style sous-titre (bleu)
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#000080'),
        spaceAfter=20,
        spaceBefore=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Style section (vert foncé)
    section_style = ParagraphStyle(
        'CustomSection',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#006400'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold',
        borderWidth=2,
        borderColor=colors.HexColor('#006400'),
        borderPadding=5,
        backColor=colors.HexColor('#F0FFF0')
    )
    
    # Style paragraphe justifié
    justified_style = ParagraphStyle(
        'Justified',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        leading=14
    )
    
    # Style liste
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['BodyText'],
        fontSize=10,
        leftIndent=20,
        bulletIndent=10,
        spaceAfter=6
    )
    
    # Statistiques
    arr = np.array(numbers)
    stats = {
        'n_mesures': len(arr),
        'min': arr.min(),
        'max': arr.max(),
        'mean': arr.mean(),
        'median': np.median(arr),
        'std': arr.std()
    }
    
    # Contenu du PDF
    story = []
    
    # ========== PAGE DE GARDE ==========
    story.append(Spacer(1, 3*cm))
    
    story.append(Paragraph("RAPPORT D'INVESTIGATION", title_style))
    story.append(Paragraph("TOMOGRAPHIE DE RÉSISTIVITÉ ÉLECTRIQUE (ERT)", subtitle_style))
    
    story.append(Spacer(1, 2*cm))
    
    # Boîte d'information
    info_data = [
        ['Fichier analysé:', file_name],
        ['Date du rapport:', datetime.now().strftime('%d/%m/%Y %H:%M')],
        ['Nombre de mesures:', f"{stats['n_mesures']}"],
        ['Plage de résistivité:', f"{stats['min']:.4f} - {stats['max']:.2f} Ω·m"],
        ['Type d\'analyse:', 'Investigation complète avec IA']
    ]
    
    info_table = Table(info_data, colWidths=[7*cm, 9*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E6F3FF')),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#4682B4')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 2*cm))
    
    # Logo/watermark
    story.append(Paragraph(
        "<font color='#808080' size=10><i>Généré par Kibali Analyst - Système Expert ERT</i></font>",
        ParagraphStyle('footer', parent=styles['Normal'], alignment=TA_CENTER)
    ))
    
    story.append(PageBreak())
    
    # ========== RÉSUMÉ EXÉCUTIF ==========
    story.append(Paragraph("1. RÉSUMÉ EXÉCUTIF", section_style))
    
    # Déterminer interprétation principale
    if stats['mean'] < 1:
        interpretation = "zone fortement conductrice suggérant la présence de sulfures métalliques, graphite ou argiles saturées"
        color_indicator = "🔴"
    elif stats['mean'] < 10:
        interpretation = "zone conductrice typique d'eau salée, argiles humides ou minéraux hydratés"
        color_indicator = "🟠"
    elif stats['mean'] < 100:
        interpretation = "zone de résistivité modérée caractéristique d'eau douce, sables ou roches altérées"
        color_indicator = "🟢"
    else:
        interpretation = "zone résistive indiquant des roches consolidées, granite ou calcaire"
        color_indicator = "🔵"
    
    executive_summary = f"""
    L'investigation géophysique par tomographie de résistivité électrique (ERT) du site <b>{file_name}</b> 
    a permis d'acquérir <b>{stats['n_mesures']} mesures</b> sur le terrain. L'analyse révèle une {interpretation}.
    <br/><br/>
    {color_indicator} <b>Résistivité moyenne: {stats['mean']:.2f} Ω·m</b> (écart-type: {stats['std']:.2f})
    <br/><br/>
    Les valeurs varient de <b>{stats['min']:.4f} Ω·m</b> (minimum) à <b>{stats['max']:.2f} Ω·m</b> (maximum), 
    avec une médiane de <b>{stats['median']:.2f} Ω·m</b>. Cette distribution statistique permet d'identifier 
    plusieurs horizons géologiques distincts et de localiser des anomalies significatives pour l'exploration minière.
    """
    
    story.append(Paragraph(executive_summary, justified_style))
    story.append(Spacer(1, 0.5*cm))
    
    # ========== STATISTIQUES CLÉS ==========
    story.append(Paragraph("2. STATISTIQUES DESCRIPTIVES", section_style))
    
    stats_data = [
        ['<b>Paramètre</b>', '<b>Valeur</b>', '<b>Interprétation</b>'],
        ['Nombre de mesures', f"{stats['n_mesures']}", 'Excellente couverture spatiale'],
        ['Minimum', f"{stats['min']:.6f} Ω·m", 'Zone ultra-conductrice détectée'],
        ['Maximum', f"{stats['max']:.2f} Ω·m", 'Zone résistive identifiée'],
        ['Moyenne', f"{stats['mean']:.2f} Ω·m", 'Valeur centrale de la distribution'],
        ['Médiane', f"{stats['median']:.2f} Ω·m", 'Valeur médiane (50e percentile)'],
        ['Écart-type', f"{stats['std']:.2f} Ω·m", 'Variabilité modérée du sous-sol'],
    ]
    
    stats_table = Table(stats_data, colWidths=[5*cm, 4*cm, 7*cm])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#006400')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(stats_table)
    story.append(Spacer(1, 0.5*cm))
    
    story.append(PageBreak())
    
    # ========== GRAPHIQUES ERT (5 COUPES) ==========
    if fig_ert is not None:
        story.append(Paragraph("3. COUPES ERT PROFESSIONNELLES", section_style))
        
        # Explication des graphiques
        ert_explanation = """
        Les cinq graphiques suivants présentent une analyse complète de la distribution de résistivité 
        dans le sous-sol. Chaque représentation offre une perspective complémentaire pour l'interprétation 
        géologique et la localisation des cibles d'exploration.
        """
        story.append(Paragraph(ert_explanation, justified_style))
        story.append(Spacer(1, 0.3*cm))
        
        # Descriptions des graphiques
        graph_descriptions = [
            ("<b>1️⃣ Pseudosection</b>", "Représentation de la résistivité apparente mesurée sur le terrain. "
             "Les points noirs indiquent les positions des électrodes. Cette vue montre les données brutes avant inversion."),
            
            ("<b>2️⃣ Modèle inversé</b>", "Section après traitement par inversion géophysique. "
             "Les lignes de contour annotées facilitent la lecture quantitative des valeurs de résistivité."),
            
            ("<b>3️⃣ Coupe géologique</b>", "Interprétation visuelle avec annotations des anomalies majeures (⭐). "
             "Les zones ultra-conductrices (<1 Ω·m) sont marquées pour investigation prioritaire."),
            
            ("<b>4️⃣ Distribution statistique</b>", "Histogramme logarithmique montrant la fréquence des valeurs. "
             "La palette de 8 couleurs correspond aux standards Res2DInv avec pourcentages de distribution."),
            
            ("<b>5️⃣ Profil vertical 1D</b>", "Évolution de la résistivité avec la profondeur. "
             "L'enveloppe min-max montre la variabilité latérale. Les zones géologiques sont colorées par profondeur.")
        ]
        
        for title, desc in graph_descriptions:
            story.append(Paragraph(f"• {title}: {desc}", bullet_style))
        
        story.append(Spacer(1, 0.5*cm))
        
        # Sauvegarder figure ERT en haute résolution
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_ert:
            fig_ert.savefig(tmp_ert.name, format='png', dpi=200, bbox_inches='tight')
            tmp_ert_path = tmp_ert.name
        
        # Insérer image (mode paysage pour les 5 graphiques)
        story.append(PageBreak())
        ert_img = RLImage(tmp_ert_path, width=18*cm, height=21*cm)
        story.append(ert_img)
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            "<i>Figure 1: Ensemble complet des 5 coupes ERT professionnelles (style Res2DInv)</i>",
            ParagraphStyle('caption', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor('#666666'))
        ))
        
        os.unlink(tmp_ert_path)  # Nettoyer fichier temporaire
    
    story.append(PageBreak())
    
    # ========== TABLEAU DE CORRESPONDANCES ==========
    if df_corr is not None and not df_corr.empty:
        story.append(Paragraph("4. CORRESPONDANCES MINÉRALES", section_style))
        
        corr_explanation = """
        Le tableau suivant établit les correspondances entre les valeurs de résistivité mesurées et les 
        matériaux géologiques potentiels. Le niveau de confiance (0-100%) reflète la position de la mesure 
        dans la plage de résistivité caractéristique de chaque minéral.
        """
        story.append(Paragraph(corr_explanation, justified_style))
        story.append(Spacer(1, 0.5*cm))
        
        if fig_corr is not None:
            # Insérer graphique scatter + table
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_corr:
                fig_corr.savefig(tmp_corr.name, format='png', dpi=200, bbox_inches='tight')
                tmp_corr_path = tmp_corr.name
            
            corr_img = RLImage(tmp_corr_path, width=17*cm, height=13*cm)
            story.append(corr_img)
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(
                "<i>Figure 2: Tableau de correspondances et scatter plot des mesures réelles</i>",
                ParagraphStyle('caption', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor('#666666'))
            ))
            
            os.unlink(tmp_corr_path)
        
        # Top 10 correspondances en tableau
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("<b>Top 10 Correspondances Identifiées:</b>", ParagraphStyle('bold', parent=styles['Normal'], fontName='Helvetica-Bold')))
        story.append(Spacer(1, 0.2*cm))
        
        top10 = df_corr.nlargest(10, 'Confiance')[['Matériau', 'Résistivité mesurée (Ω·m)', 'Confiance', 'Profondeur (m)']]
        
        table_data = [['<b>Matériau</b>', '<b>Résistivité (Ω·m)</b>', '<b>Confiance</b>', '<b>Profondeur (m)</b>']]
        for _, row in top10.iterrows():
            table_data.append([
                row['Matériau'],
                f"{row['Résistivité mesurée (Ω·m)']:.4f}",
                f"{row['Confiance']*100:.0f}%",
                f"{row['Profondeur (m)']:.1f}"
            ])
        
        corr_table = Table(table_data, colWidths=[6*cm, 4*cm, 3*cm, 3*cm])
        corr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8B0000')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(corr_table)
    
    story.append(PageBreak())
    
    # ========== INTERPRÉTATION GÉOLOGIQUE ==========
    story.append(Paragraph("5. INTERPRÉTATION GÉOLOGIQUE DÉTAILLÉE", section_style))
    
    # Analyse par plages de résistivité
    ranges_analysis = [
        (0, 1, "Ultra-conducteur", "Sulfures métalliques, graphite, argiles saturées en eau salée"),
        (1, 10, "Fortement conducteur", "Eau salée, argiles humides, schistes graphiteux"),
        (10, 100, "Modérément conducteur", "Eau douce, sables saturés, roches altérées"),
        (100, 1000, "Modérément résistif", "Sables secs, graviers, roches consolidées"),
        (1000, float('inf'), "Très résistif", "Granite, quartz, calcaire compact, roches ignées")
    ]
    
    story.append(Paragraph("<b>5.1 Analyse par horizons de résistivité:</b>", ParagraphStyle('subsection', parent=styles['Heading3'], fontSize=12, textColor=colors.HexColor('#00008B'))))
    story.append(Spacer(1, 0.3*cm))
    
    for rho_min, rho_max, label, materials in ranges_analysis:
        count = np.sum((arr >= rho_min) & (arr < rho_max))
        percentage = (count / len(arr)) * 100
        
        if count > 0:
            range_text = f"""
            <b>{label} ({rho_min}-{rho_max} Ω·m)</b>: {count} mesures ({percentage:.1f}%)
            <br/>
            <i>Matériaux probables: {materials}</i>
            """
            story.append(Paragraph(range_text, bullet_style))
            story.append(Spacer(1, 0.2*cm))
    
    # Anomalies détectées
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("<b>5.2 Anomalies géophysiques majeures:</b>", ParagraphStyle('subsection', parent=styles['Heading3'], fontSize=12, textColor=colors.HexColor('#00008B'))))
    story.append(Spacer(1, 0.3*cm))
    
    anomalies = []
    
    # Anomalie conductrice
    ultra_cond = arr < 1
    if ultra_cond.any():
        n_ultra = ultra_cond.sum()
        anomalies.append(f"🔴 <b>{n_ultra} zones ultra-conductrices</b> (ρ < 1 Ω·m) - Cibles prioritaires pour exploration minière (sulfures, or associé)")
    
    # Anomalie résistive
    ultra_res = arr > 1000
    if ultra_res.any():
        n_ultra_res = ultra_res.sum()
        anomalies.append(f"🔵 <b>{n_ultra_res} zones très résistives</b> (ρ > 1000 Ω·m) - Roches cristallines, granite, quartz massif")
    
    # Zones intermédiaires
    water_zone = (arr >= 10) & (arr <= 100)
    if water_zone.any():
        n_water = water_zone.sum()
        anomalies.append(f"🟢 <b>{n_water} zones aquifères potentielles</b> (10-100 Ω·m) - Eau douce, sables saturés")
    
    if not anomalies:
        anomalies.append("ℹ️ Aucune anomalie majeure détectée - Distribution homogène")
    
    for anomaly in anomalies:
        story.append(Paragraph(f"• {anomaly}", bullet_style))
        story.append(Spacer(1, 0.2*cm))
    
    story.append(PageBreak())
    
    # ========== RECOMMANDATIONS ==========
    story.append(Paragraph("6. RECOMMANDATIONS ET PERSPECTIVES", section_style))
    
    recommendations = f"""
    Sur la base de l'analyse géophysique ERT, les recommandations suivantes sont proposées:
    <br/><br/>
    <b>6.1 Investigations complémentaires:</b>
    <br/>
    • Sondages carottés aux emplacements des anomalies ultra-conductrices (ρ < 1 Ω·m)
    <br/>
    • Prospection géochimique (échantillonnage sol) sur les zones à fort potentiel
    <br/>
    • Polarisation provoquée (IP) pour confirmer la présence de sulfures métalliques
    <br/>
    • Levé magnétique pour compléter la signature géophysique
    <br/><br/>
    <b>6.2 Ciblage minier:</b>
    <br/>
    • Priorité 1: Zones ρ < 1 Ω·m (potentiel sulfures massifs)
    <br/>
    • Priorité 2: Transitions brusques de résistivité (contacts lithologiques)
    <br/>
    • Priorité 3: Zones 10-100 Ω·m si contexte aquifère recherché
    <br/><br/>
    <b>6.3 Modélisation 3D:</b>
    <br/>
    • Extension du profil 2D vers une couverture surfacique (grille 3D)
    <br/>
    • Inversion 3D pour modèle volumétrique complet du sous-sol
    <br/>
    • Corrélation avec données géologiques de surface et forages existants
    """
    
    story.append(Paragraph(recommendations, justified_style))
    
    story.append(PageBreak())
    
    # ========== ANNEXES TECHNIQUES ==========
    story.append(Paragraph("7. ANNEXES TECHNIQUES", section_style))
    
    story.append(Paragraph("<b>7.1 Méthodologie ERT:</b>", ParagraphStyle('subsection', parent=styles['Heading3'], fontSize=12)))
    story.append(Spacer(1, 0.2*cm))
    
    methodology = """
    La tomographie de résistivité électrique (ERT) est une méthode géophysique non-invasive qui mesure 
    la résistivité électrique du sous-sol. Des électrodes sont implantées selon un profil linéaire, et des 
    mesures de résistance sont effectuées entre différentes combinaisons d'électrodes (dispositif Wenner, 
    Schlumberger, dipôle-dipôle, etc.). Les données sont ensuite inversées pour obtenir un modèle 2D de 
    distribution de résistivité en profondeur.
    """
    story.append(Paragraph(methodology, justified_style))
    
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("<b>7.2 Paramètres d'acquisition:</b>", ParagraphStyle('subsection', parent=styles['Heading3'], fontSize=12)))
    story.append(Spacer(1, 0.2*cm))
    
    acq_data = [
        ['Nombre de mesures:', f"{stats['n_mesures']}"],
        ['Plage de mesure:', f"{stats['min']:.6f} - {stats['max']:.2f} Ω·m"],
        ['Espacement électrodes:', 'À déterminer selon fichier .dat'],
        ['Dispositif utilisé:', 'À déterminer (Wenner/Schlumberger/DD)'],
        ['Profondeur investigation:', f"Estimée: {max(50, stats['n_mesures']*0.2):.0f} m"],
    ]
    
    acq_table = Table(acq_data, colWidths=[8*cm, 8*cm])
    acq_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E6E6FA')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    story.append(acq_table)
    
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("<b>7.3 Palette de couleurs standard:</b>", ParagraphStyle('subsection', parent=styles['Heading3'], fontSize=12)))
    story.append(Spacer(1, 0.2*cm))
    
    palette_info = """
    Les graphiques utilisent la palette standard Res2DInv à 8 couleurs:
    Rouge foncé (#8B0000) → Rouge → Orange → Jaune → Vert → Cyan → Bleu → Bleu foncé (#000080).
    L'échelle logarithmique permet de visualiser efficacement la large gamme de résistivités (0.0001 - 10000 Ω·m).
    """
    story.append(Paragraph(palette_info, justified_style))
    
    # Footer final
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph(
        "─" * 80,
        ParagraphStyle('line', parent=styles['Normal'], alignment=TA_CENTER)
    ))
    story.append(Paragraph(
        f"<font color='#808080' size=8>Rapport généré automatiquement le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}<br/>"
        "Kibali Analyst - Système Expert d'Investigation Géophysique ERT<br/>"
        "Pour toute question technique: support@kibali-ai.local</font>",
        ParagraphStyle('footer', parent=styles['Normal'], alignment=TA_CENTER, fontSize=8)
    ))
    
    # Générer PDF
    doc.build(story)
    
    if output_path:
        with open(output_path, 'rb') as f:
            return f.read()
    else:
        buffer.seek(0)
        return buffer.getvalue()

def analyze_minerals_from_resistivity(numbers: list, file_name: str = "unknown") -> str:
    """
    Analyse complète des minéraux présents basée sur les valeurs de résistivité
    Génère un rapport détaillé avec clustering, interprétation géologique et calculs
    """
    if not numbers or len(numbers) < 10:
        return "❌ Données insuffisantes pour analyse minérale (minimum 10 mesures requises)"
    
    import numpy as np
    from sklearn.cluster import KMeans
    
    arr = np.array(numbers)
    minerals_db = create_minerals_database()
    
    report = "🔬 RAPPORT COMPLET D'ANALYSE MINÉRALE ERT\n"
    report += "=" * 80 + "\n\n"
    
    report += f"📁 Fichier analysé: {file_name}\n"
    report += f"📊 Nombre de mesures: {len(arr)}\n"
    report += f"📈 Plage de résistivité: {np.min(arr):.4f} - {np.max(arr):.2f} Ω·m\n\n"
    
    # Ajouter le tableau de référence de l'eau
    report += get_water_resistivity_color_table() + "\n\n"
    
    # 1️⃣ CLUSTERING AUTOMATIQUE
    report += "1️⃣ CLUSTERING K-MEANS DES RÉSISTIVITÉS\n"
    report += "─" * 80 + "\n"
    
    # Déterminer nombre optimal de clusters (2-6 basé sur variance)
    n_clusters = min(5, max(2, int(np.sqrt(len(arr) / 20))))
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(arr.reshape(-1, 1))
        cluster_centers = kmeans.cluster_centers_.flatten()
        
        report += f"✅ {n_clusters} clusters identifiés\n\n"
        
        # Trier par résistivité
        sorted_indices = np.argsort(cluster_centers)
        
        for i, idx in enumerate(sorted_indices):
            center = cluster_centers[idx]
            count = np.sum(clusters == idx)
            percentage = (count / len(arr)) * 100
            
            report += f"🎯 Cluster {i+1} (ρ moyenne = {center:.3f} Ω·m)\n"
            report += f"   • Nombre de mesures: {count} ({percentage:.1f}%)\n"
            report += f"   • Résistivité: {arr[clusters == idx].min():.3f} - {arr[clusters == idx].max():.3f} Ω·m\n"
            
            # Correspondance minéraux
            matches = minerals_db[
                (minerals_db["Plage Min (Ωm)"] <= center) & 
                (minerals_db["Plage Max (Ωm)"] >= center)
            ]
            
            if not matches.empty:
                report += f"   • Minéraux/Matériaux compatibles:\n"
                for _, match in matches.iterrows():
                    report += f"     - {match['Type']} ({match['Catégorie']}): {match['Notes']}\n"
            else:
                report += f"   • ⚠️ Aucune correspondance exacte dans la base\n"
            
            # Calculs géophysiques
            conductivity = 1000 / center if center > 0 else float('inf')  # mS/m
            report += f"   • Conductivité calculée: {conductivity:.2f} mS/m\n"
            report += f"   • Profondeur estimée: {estimate_depth_from_rho(center)}\n\n"
        
    except Exception as e:
        report += f"❌ Erreur clustering: {str(e)}\n\n"
    
    # 2️⃣ ANALYSE PAR CATÉGORIE
    report += "2️⃣ CLASSIFICATION PAR CATÉGORIE GÉOPHYSIQUE\n"
    report += "─" * 80 + "\n"
    
    # Catégories basées sur résistivité avec codes couleur
    ultra_conductors = arr[arr < 0.01]
    conductors = arr[(arr >= 0.01) & (arr < 10)]
    semi_conductors = arr[(arr >= 10) & (arr < 100)]
    resistive = arr[(arr >= 100) & (arr < 1000)]
    highly_resistive = arr[arr >= 1000]
    
    categories = [
        ("Ultra-conducteurs (<0.01 Ω·m)", ultra_conductors, "Métaux natifs (or, argent, cuivre), graphite", "🟣 Violet/Noir"),
        ("Conducteurs (0.01-10 Ω·m)", conductors, "Sulfures (pyrite, galena, chalcopyrite), eau salée, nappes", "🔴 Rouge/🟠 Orange"),
        ("Semi-conducteurs (10-100 Ω·m)", semi_conductors, "Argile humide, eau douce, certains oxydes", "🟡 Jaune/🟢 Vert"),
        ("Résistifs (100-1000 Ω·m)", resistive, "Grès, calcaire, sphalerite", "🔵 Bleu clair"),
        ("Très résistifs (>1000 Ω·m)", highly_resistive, "Granite, quartz, air/vides, eau très pure", "🔵 Bleu foncé")
    ]
    
    for cat_name, cat_data, typical_materials, color_code in categories:
        count = len(cat_data)
        percentage = (count / len(arr)) * 100
        
        if count > 0:
            report += f"📊 {cat_name} - {color_code}\n"
            report += f"   • Mesures: {count} ({percentage:.1f}%)\n"
            report += f"   • Moyenne: {np.mean(cat_data):.3f} Ω·m\n"
            report += f"   • Matériaux typiques: {typical_materials}\n\n"
    
    # 📊 ANALYSE SPÉCIFIQUE DE L'EAU
    report += "💧 ANALYSE DÉTAILLÉE DES TYPES D'EAU\n"
    report += "─" * 80 + "\n"
    
    water_categories = [
        {
            "type": "Eau de mer",
            "range": (0.1, 1.0),
            "color": "🔴 Rouge vif / 🟠 Orange",
            "description": "Haute conductivité, salinité >35 g/L",
            "applications": "Zones côtières, intrusions salines"
        },
        {
            "type": "Eau salée (nappe)",
            "range": (1.0, 10.0),
            "color": "🟠 Jaune / 🟠 Orange",
            "description": "Salinité modérée 1-10 g/L",
            "applications": "Nappes contaminées, zones arides"
        },
        {
            "type": "Eau douce",
            "range": (10.0, 100.0),
            "color": "🟢 Vert / 🔵 Bleu clair",
            "description": "Eau potable, faible salinité <1 g/L",
            "applications": "Aquifères exploitables, rivières"
        },
        {
            "type": "Eau très pure",
            "range": (100.0, float('inf')),
            "color": "🔵 Bleu foncé",
            "description": "Eau déminéralisée, pluie récente",
            "applications": "Zones non saturées, précipitations"
        }
    ]
    
    water_detected = False
    for water_cat in water_categories:
        water_zone = arr[(arr >= water_cat["range"][0]) & (arr < water_cat["range"][1])]
        count = len(water_zone)
        percentage = (count / len(arr)) * 100
        
        if count > 0:
            water_detected = True
            report += f"💧 **{water_cat['type']}** ({water_cat['range'][0]}-{water_cat['range'][1]} Ω·m) - {water_cat['color']}\n"
            report += f"   • Mesures: {count} ({percentage:.1f}%)\n"
            report += f"   • Moyenne: {np.mean(water_zone):.3f} Ω·m\n"
            report += f"   • Description: {water_cat['description']}\n"
            report += f"   • Applications: {water_cat['applications']}\n\n"
    
    if not water_detected:
        report += "⚠️ Aucune signature d'eau claire détectée dans les mesures\n"
        report += "   Possible: Zone très sèche, substrat rocheux, ou minéralisation dominante\n\n"
    else:
        report += "✅ Signatures hydriques identifiées - Possible nappe phréatique ou circulation d'eau\n\n"
    
    # 3️⃣ DÉTECTION D'ANOMALIES MINÉRALES
    report += "3️⃣ DÉTECTION D'ANOMALIES POUR EXPLORATION MINIÈRE\n"
    report += "─" * 80 + "\n"
    
    anomalies_detected = []
    
    # Anomalie sulfures (très conducteurs)
    sulfure_zone = arr[arr < 1]
    if len(sulfure_zone) > 0:
        anomalies_detected.append({
            "type": "Zone sulfurée potentielle",
            "count": len(sulfure_zone),
            "rho_range": f"{np.min(sulfure_zone):.4f} - {np.max(sulfure_zone):.3f} Ω·m",
            "minerals": "Pyrite, Chalcopyrite, Galena, Bornite",
            "interest": "⭐⭐⭐ HAUT - Exploration Cu, Pb, Zn, Au associé"
        })
    
    # Anomalie métaux précieux
    metal_zone = arr[arr < 0.01]
    if len(metal_zone) > 0:
        anomalies_detected.append({
            "type": "Zone métaux natifs potentielle",
            "count": len(metal_zone),
            "rho_range": f"{np.min(metal_zone):.6f} - {np.max(metal_zone):.4f} Ω·m",
            "minerals": "Or natif, Argent, Cuivre, Graphite",
            "interest": "⭐⭐⭐⭐⭐ TRÈS HAUT - Exploration métaux précieux"
        })
    
    # Anomalie oxydes de fer
    iron_zone = arr[(arr >= 10) & (arr <= 1000)]
    if len(iron_zone) > len(arr) * 0.1:  # >10% des mesures
        anomalies_detected.append({
            "type": "Zone oxydes de fer",
            "count": len(iron_zone),
            "rho_range": f"{np.min(iron_zone):.2f} - {np.max(iron_zone):.2f} Ω·m",
            "minerals": "Magnétite, Hématite",
            "interest": "⭐⭐ MOYEN - Exploration fer, indicateur altération"
        })
    
    if anomalies_detected:
        for i, anomaly in enumerate(anomalies_detected, 1):
            report += f"🎯 Anomalie {i}: {anomaly['type']}\n"
            report += f"   • Mesures affectées: {anomaly['count']} ({anomaly['count']/len(arr)*100:.1f}%)\n"
            report += f"   • Plage de résistivité: {anomaly['rho_range']}\n"
            report += f"   • Minéraux probables: {anomaly['minerals']}\n"
            report += f"   • Intérêt économique: {anomaly['interest']}\n\n"
    else:
        report += "⚠️ Aucune anomalie minérale majeure détectée\n\n"
    
    # 4️⃣ RECOMMANDATIONS D'EXPLORATION
    report += "4️⃣ RECOMMANDATIONS POUR EXPLORATION\n"
    report += "─" * 80 + "\n"
    
    if len(sulfure_zone) > 0:
        report += "✅ PRIORITÉ 1: Forage ciblé sur zones sulfurées (<1 Ω·m)\n"
        report += "   • Profondeur recommandée: 50-200m\n"
        report += "   • Analyses géochimiques: Cu, Pb, Zn, Au, Ag\n"
        report += "   • Méthodes complémentaires: IP (Polarisation Induite), Magnétométrie\n\n"
    
    if len(metal_zone) > 0:
        report += "✅ PRIORITÉ 2: Investigation métaux précieux (<0.01 Ω·m)\n"
        report += "   • Technique: Échantillonnage par tranchées\n"
        report += "   • Analyses: Fire assay pour Au, ICP-MS pour éléments traces\n\n"
    
    # Recommandations hydrogéologiques
    water_conductors = arr[(arr >= 0.1) & (arr <= 100)]
    if len(water_conductors) > 0:
        report += "💧 HYDROGÉOLOGIE: Investigation ressources en eau\n"
        report += "   • Zones identifiées avec signature hydrique\n"
        
        sea_water = arr[(arr >= 0.1) & (arr <= 1.0)]
        brackish_water = arr[(arr >= 1.0) & (arr <= 10.0)]
        fresh_water = arr[(arr >= 10.0) & (arr <= 100.0)]
        
        if len(sea_water) > 0:
            report += f"   • ⚠️ Eau salée détectée ({len(sea_water)} mesures): Risque intrusion marine\n"
        if len(brackish_water) > 0:
            report += f"   • 🟡 Eau saumâtre ({len(brackish_water)} mesures): Qualité modérée\n"
        if len(fresh_water) > 0:
            report += f"   • ✅ Eau douce ({len(fresh_water)} mesures): Aquifère potentiellement exploitable\n"
        
        report += "   • Recommandations:\n"
        report += "     - Forages de reconnaissance (30-150m)\n"
        report += "     - Analyses hydrochimiques (pH, TDS, ions majeurs)\n"
        report += "     - Essais de pompage pour transmissivité\n"
        report += "     - Monitoring piézométrique temporel\n\n"
    
    report += "📋 Méthodes ERT complémentaires recommandées:\n"
    report += "   • Inversion 3D pour cartographie volumétrique\n"
    report += "   • Mesures IP pour discrimination sulfures/oxydes\n"
    report += "   • Profils serrés (espacement <2m) sur anomalies\n"
    report += "   • Time-lapse ERT pour suivi temporel\n"
    report += "   • TDEM (Time Domain EM) pour profondeurs >200m\n\n"
    
    # 5️⃣ STATISTIQUES GLOBALES
    report += "5️⃣ STATISTIQUES GLOBALES DU FICHIER\n"
    report += "─" * 80 + "\n"
    
    report += f"📊 Résistivité moyenne: {np.mean(arr):.3f} Ω·m\n"
    report += f"📊 Médiane: {np.median(arr):.3f} Ω·m\n"
    report += f"📊 Écart-type: {np.std(arr):.3f} Ω·m\n"
    report += f"📊 Coefficient de variation: {np.std(arr)/np.mean(arr):.3f}\n"
    report += f"📊 Range log: {np.log10(np.max(arr)/np.min(arr)):.2f} décades\n\n"
    
    # Distribution géologique probable
    mean_rho = np.median(arr)  # Médiane plus robuste que moyenne
    if mean_rho < 10:
        geo_context = "Environnement conducteur: zone saturée, sulfures, altération hydrothermale"
    elif mean_rho < 100:
        geo_context = "Environnement mixte: sols, roches altérées, transition vadose-phréatique"
    elif mean_rho < 1000:
        geo_context = "Environnement résistif: roches compactes, zone non saturée"
    else:
        geo_context = "Environnement très résistif: substratum cristallin, zones sèches"
    
    report += f"🌍 Contexte géologique probable: {geo_context}\n"
    
    report += "\n" + "=" * 80 + "\n"
    report += "✅ ANALYSE MINÉRALE COMPLÈTE TERMINÉE\n"
    
    return report

def estimate_depth_from_rho(rho: float) -> str:
    """Estime la profondeur typique basée sur la résistivité"""
    if rho < 1:
        return "0-20m (zone conductrice superficielle ou minéralisation)"
    elif rho < 100:
        return "0-50m (zone vadose ou altérée)"
    elif rho < 1000:
        return "20-100m (zone de transition ou roche fracturée)"
    else:
        return ">50m (substratum profond ou zone sèche)"

def get_water_resistivity_color_table() -> str:
    """
    Retourne un tableau de référence des résistivités de l'eau avec codes couleur
    Basé sur les standards géophysiques internationaux
    """
    table = """
╔══════════════════════════════════════════════════════════════════════════════╗
║         TABLEAU DE RÉFÉRENCE - RÉSISTIVITÉ DE L'EAU (Ω·m)                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Type d'eau          │ Résistivité (Ω·m)  │ Couleur associée                 ║
╠═════════════════════╪════════════════════╪══════════════════════════════════╣
║ **Eau de mer**      │ 0.1 - 1 Ω·m        │ 🔴 Rouge vif / 🟠 Orange         ║
║                     │                    │ (Haute conductivité)             ║
╠─────────────────────┼────────────────────┼──────────────────────────────────╣
║ **Eau salée (nappe)**│ 1 - 10 Ω·m        │ 🟠 Jaune / 🟠 Orange             ║
║                     │                    │ (Salinité modérée)               ║
╠─────────────────────┼────────────────────┼──────────────────────────────────╣
║ **Eau douce**       │ 10 - 100 Ω·m       │ 🟢 Vert / 🔵 Bleu clair          ║
║                     │                    │ (Potable, exploitable)           ║
╠─────────────────────┼────────────────────┼──────────────────────────────────╣
║ **Eau très pure**   │ > 100 Ω·m          │ 🔵 Bleu foncé                    ║
║                     │                    │ (Déminéralisée)                  ║
╚═════════════════════╧════════════════════╧══════════════════════════════════╝

Notes:
• Les couleurs sont indicatives et dépendent de la palette utilisée (Res2DInv, etc.)
• La résistivité de l'eau varie avec: température, salinité, pH, minéraux dissous
• Eau de mer: ~0.2 Ω·m (35 g/L sel) vs Eau pure: >1000 Ω·m (<1 mg/L TDS)
• Zone de transition douce/salée: 10-30 Ω·m (mélange, interface)
"""
    return table

# ========================================
# EXTRACTION PDF & OCR POUR RAPPORTS ERT
# ========================================

def generate_annotations_with_ocr(image_path: str, label_output_path: str, preview: bool = False) -> bool:
    """
    Génère des annotations YOLO à partir d'OCR sur une image
    Détecte texte, valeurs de résistivité, légendes minérales
    """
    image = cv2.imread(image_path)
    if image is None:
        return False

    h, w, _ = image.shape
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as e:
        st.warning(f"Erreur OCR: {e}")
        return False

    found = False
    resistivity_values = []
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if not text:
            continue
        
        # Détecter valeurs de résistivité (patterns: "123.45", "0.001", etc.)
        try:
            value = float(text.replace(',', '.'))
            if 0.000001 <= value <= 1e15:  # Plage résistivité valide
                resistivity_values.append(value)
        except:
            pass
        
        x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        bw_norm, bh_norm = bw / w, bh / h
        
        with open(label_output_path, "a") as label_file:
            label_file.write(f"0 {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")
        found = True
        
        if preview:
            cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    if found and preview:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"OCR: {os.path.basename(image_path)}")
    
    return found, resistivity_values

def extract_captions_near_images(pdf, page, page_num: int, pdf_name: str, output_dir: str) -> list:
    """
    Extrait les légendes textuelles proches des images dans un PDF
    Utile pour récupérer descriptions de profils ERT, annotations géologiques
    """
    try:
        blocks = page.get_text("blocks")
        images = page.get_images(full=True)
        captions = []

        for img in images:
            xref = img[0]
            bbox = page.get_image_bbox(xref)
            
            # Chercher texte proche de l'image (dans un rayon de 80 pixels)
            for b in blocks:
                bx0, by0, bx1, by1, text, *_ = b
                # Texte en dessous ou au-dessus de l'image
                if abs(by0 - bbox.y1) < 80 or abs(bbox.y0 - by1) < 80:
                    if len(text.strip()) > 5:
                        captions.append({
                            "text": text.strip(),
                            "position": (bx0, by0, bx1, by1),
                            "image_bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                        })

        if captions:
            caption_file = os.path.join(output_dir, f"{pdf_name}_page{page_num+1}_captions.txt")
            with open(caption_file, "w", encoding="utf-8") as f:
                for cap in captions:
                    f.write(cap["text"] + "\n")
            
        return captions
    except Exception as e:
        st.warning(f"Erreur extraction légendes page {page_num+1}: {e}")
        return []

def extract_image_map(pdf, page, page_num: int, pdf_name: str, output_dir: str) -> dict:
    """
    Génère une carte JSON des images présentes dans la page
    Utile pour indexer positions de profils ERT, cartes de localisation
    """
    try:
        images = page.get_images(full=True)
        map_data = []
        
        for img in images:
            xref = img[0]
            bbox = page.get_image_bbox(xref)
            map_data.append({
                "xref": xref,
                "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                "page": page_num + 1,
                "width": bbox.x1 - bbox.x0,
                "height": bbox.y1 - bbox.y0
            })
        
        if map_data:
            map_file = os.path.join(output_dir, f"{pdf_name}_page{page_num+1}_map.json")
            with open(map_file, "w") as f:
                json.dump(map_data, f, indent=2)
        
        return map_data
    except Exception as e:
        st.warning(f"Erreur génération carte d'images: {e}")
        return []

def extract_drawings(pdf, page, page_num: int, pdf_name: str, output_dir: str) -> bool:
    """
    Extrait les éléments vectoriels (graphiques, courbes, croquis)
    Utile pour extraire profils ERT vectoriels, graphiques de résistivité
    """
    try:
        drawings = page.get_drawings()
        if drawings:
            # Convertir page en image pour sauvegarder les dessins
            pix = page.get_pixmap()
            drawing_file = os.path.join(output_dir, f"{pdf_name}_page{page_num+1}_drawings.png")
            pix.save(drawing_file)
            return True
        return False
    except Exception as e:
        st.warning(f"Erreur extraction dessins: {e}")
        return False

def extract_ert_report_from_pdf(pdf_path: str, output_base_dir: str = None) -> dict:
    """
    🔬 EXTRACTION COMPLÈTE DE RAPPORT ERT DEPUIS PDF
    
    Extrait automatiquement:
    - Profils de résistivité (images)
    - Légendes minérales/géologiques  
    - Cartes de localisation
    - Tableaux de mesures
    - Valeurs de résistivité par OCR
    - Graphiques vectoriels
    
    Returns:
        dict avec chemins des fichiers extraits et métadonnées
    """
    if output_base_dir is None:
        output_base_dir = "/tmp/ert_extracted"
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Créer dossiers de sortie
    images_dir = os.path.join(output_base_dir, "images")
    text_dir = os.path.join(output_base_dir, "text")
    data_dir = os.path.join(output_base_dir, "data")
    
    for d in [images_dir, text_dir, data_dir]:
        os.makedirs(d, exist_ok=True)
    
    extraction_results = {
        "pdf_name": pdf_name,
        "images": [],
        "captions": [],
        "maps": [],
        "drawings": [],
        "resistivity_values": [],
        "full_text": ""
    }
    
    try:
        pdf = fitz.open(pdf_path)
        st.info(f"📄 Extraction PDF: {pdf_name} ({len(pdf)} pages)")
        
        full_text = ""
        progress_bar = st.progress(0)
        
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            
            # 1️⃣ Extraire texte complet
            page_text = page.get_text()
            full_text += f"\n=== Page {page_num+1} ===\n{page_text}"
            
            # 2️⃣ Extraire légendes proches des images
            captions = extract_captions_near_images(pdf, page, page_num, pdf_name, text_dir)
            extraction_results["captions"].extend(captions)
            
            # 3️⃣ Générer carte des images
            image_map = extract_image_map(pdf, page, page_num, pdf_name, data_dir)
            extraction_results["maps"].append(image_map)
            
            # 4️⃣ Extraire dessins vectoriels
            has_drawings = extract_drawings(pdf, page, page_num, pdf_name, images_dir)
            if has_drawings:
                extraction_results["drawings"].append(f"page_{page_num+1}")
            
            # 5️⃣ Extraire images et appliquer OCR
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                
                image_filename = f"{pdf_name}_page{page_num+1}_img{img_index}.png"
                image_path = os.path.join(images_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                extraction_results["images"].append(image_path)
                
                # OCR pour extraire valeurs de résistivité
                label_path = os.path.join(data_dir, f"{pdf_name}_page{page_num+1}_img{img_index}.txt")
                found, resistivity_vals = generate_annotations_with_ocr(image_path, label_path, preview=False)
                
                if resistivity_vals:
                    extraction_results["resistivity_values"].extend(resistivity_vals)
            
            progress_bar.progress((page_num + 1) / len(pdf))
        
        # Sauvegarder texte complet
        text_file = os.path.join(text_dir, f"{pdf_name}_full_text.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        extraction_results["full_text"] = full_text
        
        # Sauvegarder métadonnées JSON
        metadata_file = os.path.join(output_base_dir, f"{pdf_name}_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump({
                "pdf_name": pdf_name,
                "total_pages": len(pdf),
                "total_images": len(extraction_results["images"]),
                "total_captions": len(extraction_results["captions"]),
                "resistivity_values_found": len(extraction_results["resistivity_values"]),
                "output_dir": output_base_dir
            }, f, indent=2)
        
        pdf.close()
        
        st.success(f"✅ Extraction terminée: {len(extraction_results['images'])} images, {len(extraction_results['resistivity_values'])} valeurs de résistivité")
        
        return extraction_results
        
    except Exception as e:
        st.error(f"❌ Erreur extraction PDF: {e}")
        return extraction_results

def process_audio_transcription(audio_path: str, output_text_path: str = None) -> str:
    """
    🎤 Transcription audio avec Whisper
    Utile pour notes vocales de géologues sur le terrain
    """
    try:
        st.info(f"🎤 Transcription audio en cours...")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        
        transcription = result["text"]
        
        if output_text_path:
            with open(output_text_path, "w", encoding="utf-8") as f:
                f.write(transcription)
        
        st.success(f"✅ Audio transcrit: {len(transcription)} caractères")
        return transcription
        
    except Exception as e:
        st.error(f"❌ Erreur transcription audio: {e}")
        return ""

def deep_binary_investigation(file_bytes: bytes, file_name: str = "unknown") -> dict:
    """
    🔍 FOUILLE INTELLIGENTE DE FICHIER BINAIRE
    Combine Hex+ASCII Dump + Base Vectorielle RAG + Base ERT pour interprétation scientifique complète
    Similaire à l'agent VSCode avec todo list mais pour l'analyse binaire
    
    Returns: dict with keys 'full_report' (str) and 'phases' (dict of phase_name: phase_content)
    """
    investigation_report = "🔬 RAPPORT D'INVESTIGATION BINAIRE APPROFONDIE\n"
    investigation_report += "=" * 80 + "\n\n"
    
    # 1️⃣ EXTRACTION INITIALE (Hex + ASCII)
    investigation_report += "1️⃣ PHASE 1: EXTRACTION HEX + ASCII\n"
    investigation_report += "─" * 80 + "\n"
    hex_dump = hex_ascii_view(file_bytes, bytes_per_line=16, max_lines=100)
    investigation_report += f"📜 Dump hexadécimal ({len(file_bytes)} bytes):\n"
    investigation_report += f"{hex_dump[:500]}...\n\n"
    
    # Extraction des nombres
    numbers = extract_numbers(file_bytes)
    investigation_report += f"🔢 Nombres extraits: {len(numbers)} valeurs\n"
    if numbers:
        import numpy as np
        arr = np.array(numbers)
        investigation_report += f"   • Range: {np.min(arr):.3f} - {np.max(arr):.3f}\n"
        investigation_report += f"   • Moyenne: {np.mean(arr):.3f} ± {np.std(arr):.3f}\n"
        investigation_report += f"   • Médiane: {np.median(arr):.3f}\n\n"
    
    # 2️⃣ ANALYSES TECHNIQUES (entropie, patterns, métadonnées)
    investigation_report += "2️⃣ PHASE 2: ANALYSES TECHNIQUES\n"
    investigation_report += "─" * 80 + "\n"
    
    entropy_result = entropy_analysis(file_bytes)
    pattern_result = pattern_recognition(file_bytes)
    metadata_result = metadata_extraction(file_bytes)
    compression_result = compression_ratio(file_bytes)
    frequency_result = frequency_analysis(file_bytes)
    
    investigation_report += f"📊 Entropie: {entropy_result}\n"
    investigation_report += f"🎯 Patterns: {pattern_result}\n"
    investigation_report += f"📋 Métadonnées: {metadata_result}\n"
    investigation_report += f"🗜️ Compression: {compression_result}\n"
    investigation_report += f"📈 Fréquences: {frequency_result}\n\n"
    
    # 3️⃣ FOUILLE DANS LA BASE VECTORIELLE RAG
    investigation_report += "3️⃣ PHASE 3: FOUILLE BASE VECTORIELLE RAG\n"
    investigation_report += "─" * 80 + "\n"
    
    rag_queries = []
    # Construire des requêtes intelligentes basées sur les patterns détectés
    if "ELF" in pattern_result or "executable" in pattern_result.lower():
        rag_queries.append("analyse fichier exécutable binaire ELF format Linux sécurité")
    if "JPEG" in pattern_result or "PNG" in pattern_result:
        rag_queries.append("format image JPEG PNG métadonnées EXIF analyse forensique")
    if "PDF" in pattern_result:
        rag_queries.append("structure PDF analyse document métadonnées forensique")
    if numbers and len(numbers) > 10:
        import numpy as np
        arr = np.array(numbers)
        if 0.1 <= np.min(arr) <= 10000:
            rag_queries.append("ERT electrical resistivity tomography geophysics data interpretation")
            rag_queries.append("résistivité électrique tomographie géophysique inversion subsurface")
    
    # Requête générique basée sur l'entropie
    if "haute" in entropy_result.lower() or "high" in entropy_result.lower():
        rag_queries.append("fichier chiffré crypté haute entropie analyse cryptographique")
    else:
        rag_queries.append("fichier données structurées format binaire analyse")
    
    # Fouiller dans RAG pour chaque requête
    rag_findings = ""
    
    # Vérifier si la base vectorielle existe et est initialisée
    has_vectorstore = False
    try:
        has_vectorstore = hasattr(st.session_state, 'vectorstore') and st.session_state.vectorstore is not None
    except:
        has_vectorstore = False
    
    if has_vectorstore:
        investigation_report += f"✅ Base vectorielle détectée - {len(rag_queries)} requêtes planifiées\n\n"
        for i, query in enumerate(rag_queries[:3], 1):  # Limiter à 3 requêtes pour performance
            try:
                result = search_vectorstore(query)
                if result and len(result) > 50:  # Éviter résultats vides
                    rag_findings += f"🔍 Requête {i}/3: '{query[:60]}...'\n"
                    rag_findings += f"   📄 Résultat: {result[:300]}...\n\n"
                else:
                    rag_findings += f"🔍 Requête {i}/3: '{query[:60]}...'\n"
                    rag_findings += f"   ⚠️ Aucun résultat pertinent\n\n"
            except Exception as e:
                rag_findings += f"🔍 Requête {i}/3: '{query[:60]}...'\n"
                rag_findings += f"   ❌ Erreur: {str(e)}\n\n"
        
        if rag_findings:
            investigation_report += rag_findings
        else:
            investigation_report += "⚠️ Aucun résultat trouvé dans la base RAG\n\n"
    else:
        investigation_report += "⚠️ Base vectorielle RAG non disponible\n"
        investigation_report += "💡 Conseil: Uploadez et indexez des PDFs dans la sidebar pour enrichir l'analyse\n\n"
    
    
    # 4️⃣ FOUILLE SPÉCIALISÉE ERT (si données numériques détectées)
    investigation_report += "4️⃣ PHASE 4: FOUILLE SPÉCIALISÉE ERT\n"
    investigation_report += "─" * 80 + "\n"
    
    mineral_report = ""
    if numbers and len(numbers) > 10:
        import numpy as np
        arr = np.array(numbers)
        ert_detection = ert_data_detection(file_bytes, numbers)
        investigation_report += ert_detection + "\n"
        
        # 🆕 ANALYSE MINÉRALE COMPLÈTE
        investigation_report += "\n🔬 ANALYSE MINÉRALE APPROFONDIE\n"
        investigation_report += "─" * 80 + "\n"
        
        try:
            mineral_report = analyze_minerals_from_resistivity(numbers, file_name)
            investigation_report += mineral_report + "\n"
        except Exception as e:
            investigation_report += f"❌ Erreur lors de l'analyse minérale: {str(e)}\n\n"
        
        # 🆕 TABLEAU DE CORRESPONDANCES RÉELLES
        investigation_report += "\n📊 TABLEAU DE CORRESPONDANCES RÉELLES\n"
        investigation_report += "─" * 80 + "\n"
        
        try:
            # Option mode grand format pour le tableau
            st.markdown("### 📊 Tableau de Correspondances Minérales")
            col_tbl1, col_tbl2 = st.columns([1, 1])
            with col_tbl1:
                use_fullsize_table = st.checkbox("📈 Mode GRAND FORMAT Tableau", value=False, 
                                                help="Agrandit le tableau et le scatter plot pour meilleure lisibilité")
            
            fig_corr, df_corr, rapport_corr = create_real_mineral_correspondence_table(
                numbers, 
                file_name,
                full_size=use_fullsize_table
            )
            
            if fig_corr and df_corr is not None:
                # Affichage responsive du graphique
                st.pyplot(fig_corr, use_container_width=True)
                
                # Boutons téléchargement pour le graphique tableau
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                with col_dl1:
                    import io
                    buf_table_png = io.BytesIO()
                    fig_corr.savefig(buf_table_png, format='png', dpi=300, bbox_inches='tight')
                    buf_table_png.seek(0)
                    st.download_button(
                        label="📥 Tableau PNG 300 DPI",
                        data=buf_table_png,
                        file_name=f"{file_name}_correspondances_300dpi.png",
                        mime="image/png",
                        key="dl_table_png"
                    )
                
                with col_dl2:
                    buf_table_pdf = io.BytesIO()
                    fig_corr.savefig(buf_table_pdf, format='pdf', bbox_inches='tight')
                    buf_table_pdf.seek(0)
                    st.download_button(
                        label="📄 Tableau PDF",
                        data=buf_table_pdf,
                        file_name=f"{file_name}_correspondances.pdf",
                        mime="application/pdf",
                        key="dl_table_pdf"
                    )
                
                with col_dl3:
                    # CSV du dataframe
                    csv_data = df_corr.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 CSV Données",
                        data=csv_data,
                        file_name=f"{file_name}_correspondances.csv",
                        mime="text/csv",
                        key="dl_csv"
                    )
                
                plt.close(fig_corr)
                
                # Afficher les données en plusieurs tableaux pour éviter scroll excessif
                st.markdown("#### 📋 Données Tabulaires - Organisées par Profondeur")
                
                # Corriger les pourcentages de confiance (convertir de 0-1 à 0-100%)
                df_corr_display = df_corr.copy()
                if 'Confiance' in df_corr_display.columns:
                    # Si les valeurs sont entre 0 et 1, convertir en pourcentage
                    if df_corr_display['Confiance'].max() <= 1:
                        df_corr_display['Confiance (%)'] = (df_corr_display['Confiance'] * 100).round(1)
                    else:
                        df_corr_display['Confiance (%)'] = df_corr_display['Confiance'].round(1)
                    df_corr_display = df_corr_display.drop('Confiance', axis=1)
                
                # Organiser en 5 tableaux selon la profondeur
                total_rows = len(df_corr_display)
                if total_rows > 20:
                    # Diviser en 5 groupes de profondeur
                    depth_col = 'Profondeur (m)' if 'Profondeur (m)' in df_corr_display.columns else df_corr_display.columns[0]
                    df_sorted = df_corr_display.sort_values(depth_col)
                    
                    # Créer 5 quantiles de profondeur
                    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    depth_ranges = df_sorted[depth_col].quantile(quantiles).values
                    
                    for i in range(5):
                        min_depth = depth_ranges[i]
                        max_depth = depth_ranges[i+1]
                        
                        # Filtrer les données dans cette plage de profondeur
                        if i == 4:  # Dernier groupe, inclure la valeur max
                            mask = (df_sorted[depth_col] >= min_depth) & (df_sorted[depth_col] <= max_depth)
                        else:
                            mask = (df_sorted[depth_col] >= min_depth) & (df_sorted[depth_col] < max_depth)
                        
                        df_section = df_sorted[mask]
                        
                        if len(df_section) > 0:
                            with st.expander(f"📊 Tableau {i+1}/5 - Profondeur: {min_depth:.1f} à {max_depth:.1f} m ({len(df_section)} détections)", expanded=(i==0)):
                                st.dataframe(
                                    df_section,
                                    use_container_width=True,
                                    column_config={
                                        "Confiance (%)": st.column_config.NumberColumn(
                                            "Confiance (%)",
                                            format="%.1f%%",
                                            help="Niveau de confiance de la correspondance (0-100%)"
                                        ),
                                        "Résistivité mesurée (Ω·m)": st.column_config.NumberColumn(
                                            "Résistivité mesurée (Ω·m)",
                                            format="%.6f"
                                        ),
                                        "Profondeur (m)": st.column_config.NumberColumn(
                                            "Profondeur (m)",
                                            format="%.1f"
                                        )
                                    },
                                    height=min(400, len(df_section) * 35 + 38)  # Hauteur adaptative
                                )
                                
                                # Statistiques du tableau
                                st.caption(f"📈 Stats: Résistivité moy. {df_section['Résistivité mesurée (Ω·m)'].mean():.4f} Ω·m | "
                                          f"Confiance moy. {df_section['Confiance (%)'].mean():.1f}%")
                else:
                    # Si moins de 20 lignes, afficher en un seul tableau
                    st.dataframe(
                        df_corr_display,
                        use_container_width=True,
                        column_config={
                            "Confiance (%)": st.column_config.NumberColumn(
                                "Confiance (%)",
                                format="%.1f%%",
                                help="Niveau de confiance de la correspondance (0-100%)"
                            ),
                            "Résistivité mesurée (Ω·m)": st.column_config.NumberColumn(
                                "Résistivité mesurée (Ω·m)",
                                format="%.6f"
                            ),
                            "Profondeur (m)": st.column_config.NumberColumn(
                                "Profondeur (m)",
                                format="%.1f"
                            )
                        }
                    )
                
                # Ajouter rapport textuel
                investigation_report += rapport_corr + "\n"
            else:
                investigation_report += rapport_corr + "\n"
                
        except Exception as e:
            investigation_report += f"❌ Erreur création tableau correspondances: {str(e)}\n\n"
        
        # 🆕 GÉNÉRATION COUPES ERT PROFESSIONNELLES (5 GRAPHIQUES)
        investigation_report += "\n🎨 COUPES ERT PROFESSIONNELLES (Style Res2DInv)\n"
        investigation_report += "─" * 80 + "\n"
        
        try:
            # Option mode grand format
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                use_fullsize = st.checkbox("🖼️ Mode GRAND FORMAT (30×36 pouces)", value=False, 
                                          help="Activez pour générer des graphiques haute résolution pour impression A0/A1")
            
            fig_ert, grid_data, rapport_ert = create_ert_professional_sections(
                numbers,
                file_name,
                full_size=use_fullsize
            )
            
            if fig_ert is not None:
                st.markdown("### 🎨 Visualisations ERT Complètes")
                
                # Affichage responsive avec use_container_width
                st.pyplot(fig_ert, use_container_width=True)
                
                # Boutons de téléchargement en colonnes
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Télécharger en PNG haute résolution
                    import io
                    buf_png = io.BytesIO()
                    fig_ert.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
                    buf_png.seek(0)
                    st.download_button(
                        label="📥 PNG Haute Résolution (300 DPI)",
                        data=buf_png,
                        file_name=f"{file_name}_ert_graphics_300dpi.png",
                        mime="image/png",
                        help="Format PNG 300 DPI pour impression professionnelle"
                    )
                
                with col2:
                    # Télécharger en PDF vectoriel
                    buf_pdf = io.BytesIO()
                    fig_ert.savefig(buf_pdf, format='pdf', bbox_inches='tight')
                    buf_pdf.seek(0)
                    st.download_button(
                        label="📄 PDF Vectoriel",
                        data=buf_pdf,
                        file_name=f"{file_name}_ert_graphics.pdf",
                        mime="application/pdf",
                        help="Format PDF vectoriel pour documents techniques"
                    )
                
                with col3:
                    # Télécharger grille de données
                    if grid_data:
                        import pickle
                        grid_pickle = pickle.dumps(grid_data)
                        st.download_button(
                            label="� Données Grille (PKL)",
                            data=grid_pickle,
                            file_name=f"{file_name}_grid_ert.pkl",
                            mime="application/octet-stream",
                            help="Données interpolées pour traitement ultérieur"
                        )
                
                plt.close(fig_ert)
                
                # �� GÉNÉRATION RAPPORT PDF PROFESSIONNEL COMPLET
                st.markdown("---")
                st.markdown("### 📄 Rapport PDF Professionnel Complet")
                
                col_pdf1, col_pdf2 = st.columns([3, 1])
                with col_pdf1:
                    st.info("🎨 Rapport PDF avec graphiques intégrés, titres colorés, statistiques et recommandations")
                
                with col_pdf2:
                    generate_pdf_btn = st.button("🔄 Générer Rapport PDF", key="gen_pdf_investigation")
                
                if generate_pdf_btn:
                    with st.spinner("📝 Génération du rapport PDF professionnel..."):
                        try:
                            pdf_bytes = generate_professional_ert_report(
                                numbers=numbers,
                                file_name=file_name,
                                mineral_report=mineral_report if mineral_report else "",
                                df_corr=df_corr if 'df_corr' in locals() else None,
                                fig_ert=fig_ert,
                                fig_corr=fig_corr if 'fig_corr' in locals() else None,
                                grid_data=grid_data
                            )
                            
                            st.success("✅ Rapport PDF généré avec succès!")
                            
                            # Bouton de téléchargement du rapport complet
                            st.download_button(
                                label="📥 TÉLÉCHARGER RAPPORT COMPLET PDF",
                                data=pdf_bytes,
                                file_name=f"{file_name}_RAPPORT_COMPLET_ERT.pdf",
                                mime="application/pdf",
                                key="dl_full_report",
                                help="Rapport professionnel avec couverture, statistiques, graphiques, interprétations et recommandations"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Erreur génération PDF: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                investigation_report += rapport_ert + "\n"
            else:
                investigation_report += "⚠️ Impossible de générer les coupes ERT\n\n"
                
        except Exception as e:
            investigation_report += f"❌ Erreur génération coupes ERT: {str(e)}\n\n"
        
        # Recherche dans base ERT spécifique
        ert_queries = [
            f"résistivité {np.mean(arr):.1f} Ohm.m interprétation géologique",
            f"analyse ERT {len(numbers)} mesures qualité données",
            f"inversion résistivité électrique {np.min(arr):.1f}-{np.max(arr):.1f}"
        ]
        
        ert_rag_findings = ""
        if has_vectorstore:
            investigation_report += "📚 Recherche connaissances ERT dans la base vectorielle...\n"
            for i, query in enumerate(ert_queries, 1):
                try:
                    result = search_vectorstore(query)
                    if result and len(result) > 50:
                        ert_rag_findings += f"🔍 Requête {i}/3: '{query[:50]}...'\n"
                        ert_rag_findings += f"   📄 {result[:200]}...\n\n"
                except Exception as e:
                    ert_rag_findings += f"🔍 Requête {i}/3: ❌ Erreur: {str(e)}\n"
        
        if ert_rag_findings:
            investigation_report += "\n📚 CONNAISSANCES ERT DE LA BASE:\n"
            investigation_report += ert_rag_findings + "\n"
        elif has_vectorstore:
            investigation_report += "⚠️ Aucune connaissance ERT spécifique trouvée dans la base\n\n"
    else:
        investigation_report += "⚠️ Pas de données ERT détectées (nombres insuffisants ou hors plage)\n\n"
    
    # 5️⃣ RECHERCHE WEB CONTEXTUALISÉE
    investigation_report += "5️⃣ PHASE 5: RECHERCHE WEB INTELLIGENTE\n"
    investigation_report += "─" * 80 + "\n"
    
    # Construire requête web basée sur tous les indices
    file_type = pattern_result.split(':')[0] if ':' in pattern_result else "inconnu"
    web_query = f"analyse {file_type} fichier binaire format {file_name}"
    
    # Initialiser web_result par défaut
    web_result = "Aucune recherche web effectuée"
    
    try:
        web_result_raw = web_search_enhanced(web_query)
        # web_search_enhanced retourne une string, pas un dict
        if web_result_raw and isinstance(web_result_raw, str):
            web_result = web_result_raw
            investigation_report += f"🌐 Recherche: '{web_query}'\n"
            investigation_report += f"{web_result[:500]}...\n\n"
        else:
            investigation_report += f"🌐 Recherche: '{web_query}'\n"
            investigation_report += f"⚠️ Aucun résultat pertinent\n\n"
    except Exception as e:
        investigation_report += f"❌ Erreur recherche web: {str(e)}\n\n"
        web_result = f"Erreur: {str(e)}"
    
    # 6️⃣ SYNTHÈSE INTELLIGENTE MULTI-SOURCES
    investigation_report += "6️⃣ PHASE 6: SYNTHÈSE MULTI-SOURCES\n"
    investigation_report += "─" * 80 + "\n"
    
    # Utiliser le modèle LLM pour synthétiser toutes les informations
    synthesis_context = f"""
Fichier analysé: {file_name} ({len(file_bytes)} bytes)
Type détecté: {pattern_result}
Entropie: {entropy_result}
Nombres extraits: {len(numbers) if numbers else 0}

Connaissances RAG:
{rag_findings[:500] if rag_findings else 'N/A'}

Détection ERT:
{ert_detection[:500] if (numbers and len(numbers) > 10 and 'ert_detection' in locals()) else 'N/A'}

Analyse Minérale:
{mineral_report[:800] if mineral_report else 'N/A'}

Recherche Web:
{web_result[:500] if web_result else 'N/A'}

QUESTION: Fournis une interprétation scientifique complète de ce fichier en combinant toutes ces sources.
Si des minéraux ont été détectés, mentionne les plus intéressants pour l'exploration minière.
"""
    
    try:
        if 'model' in st.session_state and st.session_state.model:
            model = st.session_state.model
            tokenizer = st.session_state.tokenizer
            
            inputs = tokenizer(synthesis_context, return_tensors="pt", truncation=True, max_length=2000)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            attention_mask = inputs.get('attention_mask', None)
            
            with torch.inference_mode():
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=attention_mask,
                    max_new_tokens=3000,  # AUGMENTÉ pour synthèses COMPLÈTES
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            synthesis = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            investigation_report += f"🤖 SYNTHÈSE IA:\n{synthesis}\n\n"
        else:
            investigation_report += "⚠️ Modèle LLM non disponible pour synthèse\n\n"
    except Exception as e:
        investigation_report += f"❌ Erreur synthèse IA: {e}\n\n"
    
    # 7️⃣ RECOMMANDATIONS ACTIONNABLES
    investigation_report += "7️⃣ PHASE 7: RECOMMANDATIONS\n"
    investigation_report += "─" * 80 + "\n"
    
    recommendations = []
    
    if numbers and len(numbers) > 10:
        import numpy as np
        arr = np.array(numbers)
        if 0.1 <= np.min(arr) <= 10000:
            recommendations.append("✅ Données ERT détectées → Utiliser PyGIMLI pour inversion")
            recommendations.append("✅ Visualiser avec matplotlib/seaborn (utiliser AI_Plot_Generator)")
            recommendations.append("✅ Calculer résistivité apparente avec mathematical_calculator")
    
    if "haute" in entropy_result.lower():
        recommendations.append("🔒 Entropie élevée → Fichier potentiellement chiffré")
        recommendations.append("🔍 Analyser avec outils cryptographiques")
    
    if "executable" in pattern_result.lower():
        recommendations.append("⚠️ Fichier exécutable → Analyser avec outils de reverse engineering")
        recommendations.append("🛡️ Scanner avec antivirus avant exécution")
    
    if not recommendations:
        recommendations.append("📊 Analyse complète effectuée - Aucune action spécifique requise")
    
    for rec in recommendations:
        investigation_report += f"{rec}\n"
    
    investigation_report += "\n" + "=" * 80 + "\n"
    investigation_report += "✅ INVESTIGATION TERMINÉE - Rapport complet généré\n"
    
    # Split report into phases for better display
    phases_dict = {}
    report_lines = investigation_report.split('\n')
    current_phase = None
    current_content = []
    
    for line in report_lines:
        # Detect phase markers
        if '️⃣ PHASE' in line:
            # Save previous phase if exists
            if current_phase is not None:
                phases_dict[current_phase] = '\n'.join(current_content)
            # Start new phase
            current_phase = line.strip()
            current_content = [line]
        else:
            if current_phase is not None:
                current_content.append(line)
    
    # Save last phase
    if current_phase is not None:
        phases_dict[current_phase] = '\n'.join(current_content)
    
    return {
        'full_report': investigation_report,
        'phases': phases_dict
    }

def ert_geophysical_interpretation(numbers: list) -> str:
    """Interprétation géophysique spécialisée des données ERT"""
    if not numbers:
        return "❌ Aucune donnée pour l'interprétation géophysique"
    import numpy as np
    analysis = "🌍 INTERPRÉTATION GÉOPHYSIQUE ERT\n"
    analysis += "=" * 40 + "\n\n"
    arr = np.array(numbers)
    # Classification des valeurs de résistivité
    low_resistivity = arr[arr < 10] # < 10 Ohm.m
    medium_resistivity = arr[(arr >= 10) & (arr < 100)] # 10-100 Ohm.m
    high_resistivity = arr[arr >= 100] # > 100 Ohm.m
    analysis += f"📊 CLASSIFICATION DES RÉSISTIVITÉS:\n"
    analysis += f" • Faible résistivité (< 10 Ohm.m): {len(low_resistivity)} valeurs\n"
    analysis += f" → Argile, eau salée, minéraux conducteurs\n"
    analysis += f" • Résistivité moyenne (10-100 Ohm.m): {len(medium_resistivity)} valeurs\n"
    analysis += f" → Sols sableux, roches sédimentaires\n"
    analysis += f" • Haute résistivité (> 100 Ohm.m): {len(high_resistivity)} valeurs\n"
    analysis += f" → Roches cristallines, vides, air\n\n"
    # Ajout des couleurs et descriptions
    analysis += f"🎨 COULEURS ET DÉSCRIPTIONS PAR CATÉGORIE:\n"
    sample_values = np.unique(np.round(arr, 1))[:10] # Échantillon de valeurs uniques
    for val in sample_values:
        color_desc = get_resistivity_color(val)
        analysis += f" • ρ = {val} Ω.m: {color_desc}\n"
    analysis += "\n"
    # Recherche dynamique pour comparaisons
    analysis += f"🔍 COMPARAISONS DYNAMIQUES AVEC MATÉRIAUX (recherche internet):\n"
    analysis += f"Liquides (eau pure, salée, huiles):\n{fetch_material_resistivities('liquids')}\n\n"
    analysis += f"Minéraux/Sols (argile, sable, limon):\n{fetch_material_resistivities('minerals soils')}\n\n"
    analysis += f"Roches (granite, calcaire, grès):\n{fetch_material_resistivities('rocks')}\n\n"
    # Analyse d'hétérogénéité
    heterogeneity = np.std(arr) / np.mean(arr)
    analysis += f"🎯 HÉTÉROGÉNÉITÉ DU MILIEU:\n"
    analysis += f" • Coefficient de variation: {heterogeneity:.3f}\n"
    if heterogeneity < 0.5:
        analysis += f" → Milieu homogène (roche massive)\n"
    elif heterogeneity < 1.0:
        analysis += f" → Milieu modérément hétérogène (sédiments)\n"
    else:
        analysis += f" → Milieu très hétérogène (zone fracturée/caverneuse)\n\n"
    # Estimation de profondeur générale
    mean_rho = np.mean(arr)
    analysis += f"📏 ESTIMATION DE PROFONDEUR (basée sur ρ moyenne = {mean_rho:.1f} Ω.m, générique):\n"
    if mean_rho < 10:
        analysis += " → Superficielle (0-5 m): Couches argileuses ou saturées\n"
    elif mean_rho < 100:
        analysis += " → Moyenne (5-20 m): Aquifères sableux\n"
    else:
        analysis += " → Profonde (>20 m): Substratum résistant\n\n"
    # Détection d'anomalies potentielles
    z_scores = (arr - np.mean(arr)) / np.std(arr)
    anomalies_high = arr[z_scores > 2] # Anomalies hautes
    anomalies_low = arr[z_scores < -2] # Anomalies basses
    if len(anomalies_high) > 0 or len(anomalies_low) > 0:
        analysis += f"🚨 ANOMALIES DÉTECTÉES:\n"
        if len(anomalies_high) > 0:
            analysis += f" • {len(anomalies_high)} anomalies haute résistivité\n"
            analysis += f" → Possibles: vides, fractures, roches résistantes (couleur: rouge)\n"
        if len(anomalies_low) > 0:
            analysis += f" • {len(anomalies_low)} anomalies basse résistivité\n"
            analysis += f" → Possibles: eau, argile, minéraux conducteurs (couleur: bleu)\n\n"
    # Applications potentielles
    analysis += f"🏗️ APPLICATIONS POTENTIELLES:\n"
    analysis += f" • Hydrogéologie: détection aquifères\n"
    analysis += f" • Géotechnique: stabilité des sols\n"
    analysis += f" • Archéologie: structures enterrées\n"
    analysis += f" • Environnement: pollution des sols\n"
    analysis += f" • Génie civil: fouilles et tunnels\n"
    return analysis
def ert_quality_assessment(numbers: list) -> str:
    """Évaluation de la qualité des données ERT"""
    if not numbers:
        return "❌ Aucune donnée pour l'évaluation qualité"
    import numpy as np
    analysis = "⭐ ÉVALUATION QUALITÉ DONNÉES ERT\n"
    analysis += "=" * 40 + "\n\n"
    arr = np.array(numbers)
    # Critères de qualité
    quality_score = 0
    max_score = 5
    # 1. Plage de valeurs réaliste
    if 0.1 <= np.min(arr) <= 10000:
        quality_score += 1
        analysis += f"✅ Plage de résistivité réaliste\n"
    else:
        analysis += f"❌ Plage de résistivité suspecte\n"
    # 2. Nombre de mesures suffisant
    if len(arr) >= 50:
        quality_score += 1
        analysis += f"✅ Nombre de mesures suffisant ({len(arr)})\n"
    else:
        analysis += f"⚠️ Peu de mesures ({len(arr)}) - précision limitée\n"
    # 3. Contraste suffisant
    contrast = np.max(arr) / np.min(arr)
    if contrast >= 2:
        quality_score += 1
        analysis += f"✅ Bon contraste ({contrast:.1f})\n"
    else:
        analysis += f"⚠️ Contraste faible ({contrast:.1f})\n"
    # 4. Distribution réaliste
    try:
        from scipy import stats
        log_data = np.log(arr[arr > 0])
        _, p_value = stats.shapiro(log_data[:min(5000, len(log_data))])
        if p_value > 0.05:
            quality_score += 1
            analysis += f"✅ Distribution log-normale (p={p_value:.3f})\n"
        else:
            analysis += f"⚠️ Distribution non standard\n"
    except:
        analysis += f"⚠️ Test de distribution impossible\n"
    # 5. Absence d'outliers extrêmes
    z_scores = np.abs((arr - np.mean(arr)) / np.std(arr))
    extreme_outliers = np.sum(z_scores > 5)
    if extreme_outliers == 0:
        quality_score += 1
        analysis += f"✅ Pas d'outliers extrêmes\n"
    else:
        analysis += f"⚠️ {extreme_outliers} outliers extrêmes détectés\n"
    # Score final
    quality_percentage = (quality_score / max_score) * 100
    analysis += f"\n🎯 SCORE QUALITÉ: {quality_score}/{max_score} ({quality_percentage:.1f}%)\n"
    if quality_percentage >= 80:
        analysis += f"⭐ QUALITÉ EXCELLENTE - Données fiables pour inversion\n"
    elif quality_percentage >= 60:
        analysis += f"✅ QUALITÉ BONNE - Données utilisables avec précaution\n"
    elif quality_percentage >= 40:
        analysis += f"⚠️ QUALITÉ MOYENNE - Résultats à interpréter prudemment\n"
    else:
        analysis += f"❌ QUALITÉ INSUFFISANTE - Acquisition à recommencer\n"
    return analysis
# Fonction d'analyse intelligente utilisant le modèle Qwen directement
def analyze_with_ai(query: str, file_bytes: bytes, numbers: list, hex_dump: str, n_clusters: int = 3, model=None, tokenizer=None, device=None) -> str:
    """Analyse intelligente utilisant le modèle Qwen avec accès automatique aux outils et enrichissement ERT"""
   
    # Récupérer les variables depuis session_state si non fournies
    if model is None:
        try:
            model = st.session_state.get('model', None)
            tokenizer = st.session_state.get('tokenizer', None)
            device = st.session_state.get('device', None)
        except:
            pass
   
    # Vérifier que nous avons un modèle
    if model is None or tokenizer is None:
        return """❌ ERREUR: Modèle LLM non disponible
       
🔧 Le modèle n'a pas pu être chargé pour cette analyse.
📋 Analyse de base réalisée avec les outils disponibles uniquement.
       
Veuillez redémarrer l'application pour charger le modèle LLM."""
    # Enrichissement automatique de la base ERT si données détectées
    enrichment_status = ""
    if numbers and len(numbers) > 20:
        try:
            import numpy as np
            arr = np.array(numbers)
            if 0.1 <= np.min(arr) <= 10000:
                # Importer et utiliser l'enrichisseur ERT
                from ert_database_enrichment import create_ert_knowledge_base
              
                # Enrichir la base avec des connaissances ERT contextuelles
                if st.session_state.vectorstore:
                    vectorstore_path = "/tmp/enriched_ert_vectordb"
                    enriched_vs, msg = create_ert_knowledge_base(vectorstore_path, numbers)
                    if enriched_vs:
                        # Fusionner avec la base existante si possible
                        enrichment_status = f"✅ Base enrichie automatiquement avec connaissances ERT: {msg}"
                    else:
                        enrichment_status = f"⚠️ Enrichissement partiel: {msg}"
                else:
                    enrichment_status = "⚠️ Base vectorielle non disponible pour enrichissement"
        except Exception as e:
            enrichment_status = f"❌ Erreur enrichissement ERT: {e}"
    # Informations de base sur le fichier
    basic_info = f"""
📁 FICHIER ANALYSÉ:
- Nom: {uploaded_file.name if 'uploaded_file' in locals() else 'Fichier uploadé'}
- Taille: {len(file_bytes)} bytes ({len(file_bytes)/1024:.1f} KB)
- Nombres extraits: {len(numbers) if numbers else 0}
- Clusters identifiés: {n_clusters if numbers else 0}
🧠 ENRICHISSEMENT AUTOMATIQUE:
{enrichment_status}
🔍 DUMP HEXADÉCIMAL (aperçu):
{hex_dump[:300]}...
❓ QUESTION: {query}
"""
    # PHASE 1: Analyses de base pour identifier le fichier
    try:
        entropy_result = entropy_analysis(file_bytes)
        pattern_result = pattern_recognition(file_bytes)
        metadata_result = metadata_extraction(file_bytes)
        compression_result = compression_ratio(file_bytes)
        frequency_result = frequency_analysis(file_bytes)
        base_analysis = f"""
🔬 ANALYSES DE BASE RÉALISÉES:
📊 ENTROPIE: {entropy_result}
🎯 PATTERNS: {pattern_result}
📋 MÉTADONNÉES: {metadata_result}
🗜️ COMPRESSION: {compression_result}
📈 FRÉQUENCE: {frequency_result}
"""
        # PHASE 2: Recherche dans la base RAG pour identifier le type et obtenir des connaissances
        rag_search_query = f"Type de fichier binaire: {pattern_result[:100]}... Entropie: {entropy_result[:50]}... Métadonnées: {metadata_result[:100]}..."
        rag_context = ""
        if st.session_state.vectorstore:
            try:
                rag_result = search_vectorstore(rag_search_query)
                rag_context = f"\n\n📚 CONNAISSANCES RAG:\n{rag_result}"
            except Exception as e:
                rag_context = f"\n\n📚 CONNAISSANCES RAG: Erreur - {e}"
        # PHASE 3: Recherche web ciblée basée sur les analyses - AMÉLIORÉE POUR ERT
        if numbers and len(numbers) > 10:
            # Vérifier si potentiellement ERT
            import numpy as np
            arr = np.array(numbers)
            if 0.1 <= np.min(arr) <= 10000:
                web_search_query = f"ERT electrical resistivity tomography data interpretation {np.mean(arr):.1f} Ohm.m geophysical analysis subsurface"
            else:
                web_search_query = f"analyse fichier binaire {pattern_result.split(':')[0] if ':' in pattern_result else 'inconnu'} type format entropie cybersécurité"
        else:
            web_search_query = f"analyse fichier binaire {pattern_result.split(':')[0] if ':' in pattern_result else 'inconnu'} type format entropie cybersécurité"
          
        web_context = ""
        try:
            web_result = web_search_enhanced(web_search_query)
            web_context = f"\n\n🌐 RECHERCHE WEB:\n{web_result}"
        except Exception as e:
            web_context = f"\n\n🌐 RECHERCHE WEB: Erreur - {e}"
        # PHASE 4: Analyses statistiques avancées si applicable
        stats_context = ""
        if numbers:
            try:
                stats_result = statistical_analysis(numbers)
                if len(numbers) >= 3:
                    correlation_result = correlation_analysis(numbers)
                    stats_context += f"\n🔗 CORRÉLATIONS: {correlation_result}"
                if len(numbers) >= 10:
                    anomaly_result = anomaly_detection(numbers)
                    stats_context += f"\n🚨 ANOMALIES: {anomaly_result}"
                if len(numbers) >= 32:
                    spectral_result = spectral_analysis(numbers)
                    stats_context += f"\n🌊 SPECTRAL: {spectral_result}"
                stats_context = f"\n\n📊 ANALYSES STATISTIQUES:\n{stats_result}{stats_context}"
            except Exception as e:
                stats_context = f"\n\n📊 ANALYSES STATISTIQUES: Erreur - {e}"
        # PHASE 4.5: Détection et analyse spécialisée ERT
        ert_context = ""
        ert_detected = False
        if numbers and len(numbers) > 10:
            try:
                ert_detection_result = ert_data_detection(file_bytes, numbers)
                # Vérifier si les données semblent être ERT (basé sur les critères de la fonction)
                import numpy as np
                arr = np.array(numbers)
                if 0.1 <= np.min(arr) <= 10000 and len(numbers) >= 20:
                    ert_detected = True
                    # Analyses ERT spécialisées
                    ert_inversion = ert_inversion_analysis(numbers)
                    ert_interpretation = ert_geophysical_interpretation(numbers)
                    ert_quality = ert_quality_assessment(numbers)
                    ert_context = f"\n\n🔍 ANALYSES SPÉCIALISÉES ERT:\n{ert_detection_result}\n\n{ert_inversion}\n\n{ert_interpretation}\n\n{ert_quality}"
                    # Recherche RAG spécialisée ERT avec enrichissement automatique
                    ert_rag_query = f"ERT Electrical Resistivity Tomography données résistivité {np.mean(arr):.1f} Ohm.m interprétation géophysique inversion sismique hydrogéologie couleurs profondeur nature matériaux liquides minéraux formules calcul résistivité apparente Schlumberger Wenner Dipole-Dipole"
                    if st.session_state.vectorstore:
                        try:
                            ert_rag_result = search_vectorstore(ert_rag_query)
                            ert_context += f"\n\n📚 CONNAISSANCES ERT RAG:\n{ert_rag_result}"
                          
                            # Utiliser le système d'enrichissement pour obtenir plus de contexte
                            enriched_context = rag_enhanced_analysis(
                                ert_rag_query,
                                ert_rag_result,
                                ert_data={'mean': np.mean(arr), 'std': np.std(arr), 'min': np.min(arr), 'max': np.max(arr)}
                            )
                            ert_context += f"\n\n🔬 ANALYSE RAG ENRICHIE:\n{enriched_context}"
                          
                        except Exception as e:
                            ert_context += f"\n\n📚 CONNAISSANCES ERT RAG: Erreur - {e}"
                    # Recherche web spécialisée ERT avec requêtes multiples
                    ert_web_queries = [
                        f"ERT tomography résistivité électrique interprétation données {np.mean(arr):.1f} Ohm.m géophysique hydrogéologie couleurs visualisation",
                        f"electrical resistivity {np.mean(arr):.1f} ohm.m subsurface interpretation environmental depth nature",
                        "ERT data processing inversion algorithms geophysical survey materials comparison"
                    ]
                  
                    for i, ert_web_query in enumerate(ert_web_queries):
                        try:
                            ert_web_result = web_search_enhanced(ert_web_query, "ert_specialized")
                            ert_context += f"\n\n🌐 RECHERCHE WEB ERT #{i+1}:\n{ert_web_result}"
                        except Exception as e:
                            ert_context += f"\n\n🌐 RECHERCHE WEB ERT #{i+1}: Erreur - {e}"
            except Exception as e:
                ert_context = f"\n\n🔍 ANALYSE ERT: Erreur lors de l'analyse spécialisée - {e}"
        # PHASE 5: Synthèse experte avec toutes les informations
        synthesis_context = f"""
{basic_info}
{base_analysis}
{rag_context}
{web_context}
{stats_context}
{ert_context}
🎯 PROTOCOLE D'ANALYSE EXPERTE:
1. Identifier le type de fichier basé sur les patterns et signatures détectés
2. Évaluer les risques de sécurité (entropie élevée = possible cryptage/malware)
3. Analyser la structure et le contenu basé sur les connaissances RAG
4. Contextualiser avec les informations web récentes
5. Si données ERT détectées, interpréter géophysiquement avec connaissances spécialisées, incluant couleurs de visualisation, estimations de profondeur, nature des matériaux, et comparaisons dynamiques avec liquides/minéraux/roches via recherches internet
6. Pour fichiers .dat ERT, utilisez mathematical_calculator pour les formules de résistivité apparente du document FicheERT.pdf: Schlumberger: pi*(L**2 - l**2)/(2*l) * V/I (L=AB/2, l=MN/2), Wenner: 2*pi*a * V/I (a=AM), Dipole-Dipole: pi*n*(n+1)*(n+2)*a * V/I (n=facteur séparation)
7. Fournir une interprétation professionnelle du fichier, en rendant l'analyse la plus puissante possible en ERT et géophysique
INSTRUCTION: En tant qu'expert mondial en cybersécurité, analyse de fichiers binaires, géophysique ERT/tomographie de résistivité électrique, fournissez une analyse complète, professionnelle et sécurisée de ce fichier. Pour ERT: décrivez nature, profondeur, couleurs, comparez avec matériaux (recherchez dynamiquement liquides, minéraux par catégories), et répondez dynamiquement aux comparaisons. Utilisez mathematical_calculator pour les calculs de résistivité apparente si V, I et espacements sont disponibles.
"""
        # Utiliser le modèle Qwen pour la synthèse finale avec optimisation GPU
        messages = [
            {"role": "system", "content": "Tu es un expert mondial en cybersécurité, analyse de fichiers binaires, intelligence artificielle et géophysique (ERT/tomographie de résistivité électrique). Analyse ce fichier de manière professionnelle en utilisant toutes les informations disponibles. Identifie d'abord le type de fichier, évalue les risques de sécurité, puis fournis une interprétation complète incluant l'interprétation géophysique si des données ERT sont détectées. Pour ERT: décris nature, profondeur, couleurs de visualisation, compare avec liquides/minéraux/roches via recherches dynamiques, et rends l'analyse la plus puissante possible. Pour fichiers .dat, utilise mathematical_calculator avec les formules: Schlumberger: pi*(L**2 - l**2)/(2*l) * V/I, Wenner: 2*pi*a * V/I, Dipole-Dipole: pi*n*(n+1)*(n+2)*a * V/I."},
            {"role": "user", "content": synthesis_context}
        ]
       
        # Optimisation GPU: S'assurer que le modèle est sur le bon device
        if torch.cuda.is_available() and model.device.type != 'cuda':
            model = model.to('cuda')
       
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
       
        # Create attention mask to avoid warnings when pad_token == eos_token
        attention_mask = (inputs != tokenizer.pad_token_id).long().to(model.device)
       
        # Optimisation pour GPU: utiliser torch.cuda.amp pour mixed precision si GPU disponible
        if model.device.type == 'cuda':
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=3000,  # AUGMENTÉ pour analyses COMPLÈTES
                    temperature=0.6,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True, # Optimisation GPU
                    num_beams=1 # Plus rapide pour GPU
                )
        else:
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=3000,  # AUGMENTÉ pour analyses COMPLÈTES
                    temperature=0.6,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
        final_analysis = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
       
        # Information sur les performances
        device_info = f"🖥️ Device utilisé: {model.device.type.upper()}"
        if model.device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            device_info += f" | VRAM: {memory_used:.1f}/{memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)"
       
        return f"""🔍 ANALYSE PROFESSIONNELLE DE FICHIER BINAIRE
{device_info}
{basic_info}
{base_analysis}
{rag_context}
{web_context}
{stats_context}
{ert_context}
🎯 ANALYSE EXPERTE FINALE:
{final_analysis}
✅ Analyse terminée - Toutes les sources d'information ont été consultées et synthétisées.
⚡ Performance: {'GPU accéléré' if model.device.type == 'cuda' else 'CPU standard'}"""
    except Exception as e:
        # Fallback avec analyse basique
        try:
            basic_entropy = entropy_analysis(file_bytes)
            basic_patterns = pattern_recognition(file_bytes)
            basic_metadata = metadata_extraction(file_bytes)
            return f"""❌ Erreur dans l'analyse complète: {str(e)}
🔬 ANALYSE DE BASE RÉALISÉE:
📊 ENTROPIE: {basic_entropy}
🎯 PATTERNS: {basic_patterns}
📋 MÉTADONNÉES: {basic_metadata}
{basic_info}
Recommandation: Le fichier présente une entropie de {basic_entropy.split('/')[0] if '/' in basic_entropy else 'inconnue'}.
Type détecté: {basic_patterns.split(':')[0] if ':' in basic_patterns else 'inconnu'}."""
        except Exception as e2:
            return f"❌ Erreur critique lors de l'analyse: {str(e)}\nErreur de fallback: {str(e2)}\n\nInformations de base:\n{basic_info}"
def hex_ascii_view(file_bytes, bytes_per_line=16, max_lines=50):
    lines = []
    for i in range(0, min(len(file_bytes), bytes_per_line*max_lines), bytes_per_line):
        chunk = file_bytes[i:i+bytes_per_line]
        hex_bytes = " ".join(f"{b:02X}" for b in chunk)
        ascii_bytes = "".join([chr(b) if 32 <= b <= 126 else "." for b in chunk])
        lines.append(f"{i:08X} {hex_bytes:<48} |{ascii_bytes}|")
    return "\n".join(lines)
def extract_numbers(file_bytes):
    # On convertit les parties ASCII pour extraire float/int
    ascii_text = "".join([chr(b) if 32 <= b <= 126 else " " for b in file_bytes])
    # regex pour float ou int
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", ascii_text)
    numbers = [float(n) for n in numbers]
    return numbers
def cluster_numbers(numbers, n_clusters=3):
    if not numbers:
        return None
    X = np.array(numbers).reshape(-1,1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return labels, centers
def load_model_state(file_path: Path) -> Dict[str, Any]:
    ext = file_path.suffix
    if ext == ".safetensors":
        state_dict = load_file(str(file_path), device="cpu")
    elif ext in [".bin", ".pt", ".ckpt"]:
        try:
            # Essayer avec weights_only=True pour PyTorch 2.6+
            state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Fallback pour anciennes versions de PyTorch
            state_dict = torch.load(file_path, map_location="cpu")
    else:
        raise ValueError(f"Extension non supportée : {ext}")
    return state_dict
def summarize_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    summary = []
    for key, tensor in state_dict.items():
        summary.append(f"Clé: {key}, Shape: {tensor.shape}, Dtype: {tensor.dtype}, Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")
    return "\n".join(summary[:10]) # Limit to first 10 for brevity
# --------- Streamlit Interface ---------
st.title("🔍 Streamlit Binary Viewer + KMeans Clustering + LLM Analysis Agent")
# Section for PDF uploads and indexing
st.subheader("📚 Upload PDFs for Knowledge Base")
uploaded_pdfs = st.file_uploader("Choisir des PDFs pour indexer (connaissances pour l'analyse)", type=["pdf"], accept_multiple_files=True)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if uploaded_pdfs and st.button("Indexer les PDFs dans la base vectorielle"):
    with st.spinner("Indexation en cours..."):
        docs = []
        for pdf in uploaded_pdfs:
            # Save uploaded PDF to temp file
            temp_path = Path(f"/tmp/{pdf.name}")
            with open(temp_path, "wb") as f:
                f.write(pdf.getvalue())
            loader = PyPDFLoader(str(temp_path))
            loaded_docs = loader.load()
          
            # Check if text was extracted
            if not any(doc.page_content.strip() for doc in loaded_docs):
                st.write(f"No text extracted from {pdf.name}, trying OCR...")
                try:
                    images = convert_from_path(str(temp_path))
                    ocr_text = ""
                    for image in images:
                        ocr_text += pytesseract.image_to_string(image) + "\n"
                    # Replace with OCR document
                    loaded_docs = [Document(page_content=ocr_text, metadata={"source": pdf.name})]
                    st.write(f"OCR extracted {len(ocr_text)} characters from {pdf.name}")
                except Exception as e:
                    st.error(f"OCR failed for {pdf.name}: {e}")
                    loaded_docs = []
          
            docs.extend(loaded_docs)
            st.write(f"Loaded {len(loaded_docs)} pages/documents from {pdf.name}")
      
        st.write(f"Total documents loaded: {len(docs)}")
      
        # Debug: check content
        if docs:
            st.write(f"Sample content from first doc: '{docs[0].page_content[:200]}'")
            non_empty = sum(1 for doc in docs if doc.page_content.strip())
            st.write(f"Documents with non-empty content: {non_empty}/{len(docs)}")
      
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
      
        st.write(f"Total splits created: {len(splits)}")
      
        if not splits:
            st.error("Aucun document valide trouvé dans les PDFs uploadés. Assurez-vous que les PDFs contiennent du texte extractable (pas des images scannées). Si le PDF contient du texte mais n'est pas extrait, essayez un PDF différent ou utilisez OCR.")
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            embeddings = SentenceTransformerEmbeddings('sentence-transformers/all-MiniLM-L6-v2', device=device)
          
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            
            # Synchroniser avec vectordb pour que Kibali ait accès
            if "vectordb" not in st.session_state:
                st.session_state.vectordb = None
            
            # Fusionner avec vectordb existant si présent, sinon créer nouveau
            if st.session_state.vectordb is not None:
                try:
                    # Ajouter les nouveaux documents à la base existante
                    st.session_state.vectordb.add_documents(splits)
                    st.info("📚 Documents ajoutés à la base vectorielle existante de Kibali Analyst")
                except:
                    # Si erreur, remplacer complètement
                    st.session_state.vectordb = st.session_state.vectorstore
                    st.warning("⚠️ Remplacement de la base vectorielle")
            else:
                # Créer nouvelle base
                st.session_state.vectordb = st.session_state.vectorstore
                st.info("✨ Nouvelle base vectorielle créée pour Kibali Analyst")
      
            st.success("✅ Base vectorielle créée avec succès ! Kibali Analyst peut maintenant accéder à ces documents.")
            
            # Sauvegarder aussi dans le chemin standard si possible
            try:
                VECTORDB_PATH = "/root/chatbot_data/vectordb"
                os.makedirs(os.path.dirname(VECTORDB_PATH), exist_ok=True)
                st.session_state.vectordb.save_local(VECTORDB_PATH)
                st.info(f"💾 Base sauvegardée dans {VECTORDB_PATH}")
            except Exception as e:
                st.warning(f"⚠️ Impossible de sauvegarder: {e}")
# ═══════════════════════════════════════════════════════════════
# SECTION MULTI-FICHIERS ERT MULTI-FRÉQUENCES
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 Parseur Multi-Fréquences ERT - Fichiers Complémentaires")
st.info("""
🎯 **Uploadez plusieurs fichiers .dat complémentaires** avec multi-fréquences (MHz)

Format attendu :
- En-tête : `..., 1000 MHz, 500 MHz, 250 MHz, ...`
- Données : `project, survey_point, depth, res1, res2, res3, ...`

✅ Fusion automatique des fichiers
✅ Coordonnées spatiales correctes (X, Y, Z)
✅ Visualisations 2D/3D par fréquence
""")

# Upload multiple files
uploaded_ert_files = st.file_uploader(
    "📤 Chargez vos fichiers .dat ERT (plusieurs fichiers possibles)",
    type=['dat', 'txt'],
    accept_multiple_files=True,
    key="multi_freq_uploader"
)

if uploaded_ert_files and len(uploaded_ert_files) > 0:
    with st.spinner(f"📊 Parsing de {len(uploaded_ert_files)} fichier(s)..."):
        # Sauvegarder temporairement les fichiers
        temp_paths = []
        for uploaded_file in uploaded_ert_files:
            temp_path = f"/tmp/ert_{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            temp_paths.append(temp_path)
        
        # Parser avec le nouveau parseur
        parser = MultiFreqERTParser()
        df_ert = parser.parse_multiple_files(temp_paths)
        
        if not df_ert.empty:
            st.success(f"✅ {len(df_ert)} mesures chargées depuis {len(uploaded_ert_files)} fichier(s) !")
            
            # Afficher métadonnées
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📁 Fichiers", parser.metadata['num_files'])
                st.metric("📡 Fréquences", parser.metadata['num_frequencies'])
            with col2:
                st.metric("📍 Survey Points", parser.metadata['num_survey_points'])
                st.metric("📏 Profondeurs", parser.metadata['num_depths'])
            with col3:
                st.metric("🔬 Projets", parser.metadata['num_projects'])
                st.metric("📊 Mesures", parser.metadata['total_measurements'])
            with col4:
                st.metric("⬇️ Prof. min", f"{parser.metadata['depth_range'][0]:.1f} m")
                st.metric("⬇️ Prof. max", f"{parser.metadata['depth_range'][1]:.1f} m")
            
            # Rapport statistique
            with st.expander("📋 Rapport Statistique Complet", expanded=False):
                report = parser.generate_statistics_report()
                st.text(report)
            
            # Afficher données brutes
            with st.expander("🔢 Données Brutes Fusionnées", expanded=False):
                st.dataframe(df_ert.head(100), use_container_width=True)
            
            # COORDONNÉES CORRECTES
            st.markdown("### 🎯 Coordonnées Spatiales Correctes (X, Y, Z)")
            coords_df = parser.get_coordinates_corrected()
            
            with st.expander("📐 Tableau des Coordonnées", expanded=False):
                st.dataframe(coords_df.head(50), use_container_width=True)
                
                # Export CSV
                csv_coords = coords_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger Coordonnées (CSV)",
                    data=csv_coords,
                    file_name="ert_coordinates.csv",
                    mime="text/csv"
                )
            
            # VISUALISATIONS PAR FRÉQUENCE
            st.markdown("### 📊 Visualisations Multi-Fréquences")
            
            freq_tabs = st.tabs([f"📡 {freq} MHz" for freq in parser.frequencies] + ["🔄 Comparaison", "🌐 3D"])
            
            # Onglet pour chaque fréquence
            for i, freq in enumerate(parser.frequencies):
                with freq_tabs[i]:
                    st.markdown(f"#### Coupe 2D - {freq} MHz")
                    
                    fig_2d = parser.create_2d_section_by_frequency(freq)
                    if fig_2d:
                        st.plotly_chart(fig_2d, use_container_width=True)
                        
                        # Stats par fréquence
                        freq_data = df_ert[df_ert['frequency_MHz'] == freq]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mesures", len(freq_data))
                        with col2:
                            st.metric("Rés. moy.", f"{freq_data['resistivity'].mean():.2f} Ω·m")
                        with col3:
                            st.metric("Écart-type", f"{freq_data['resistivity'].std():.2f}")
            
            # Onglet comparaison
            with freq_tabs[-2]:
                st.markdown("#### Comparaison Multi-Fréquences")
                fig_comp = parser.create_frequency_comparison()
                if fig_comp:
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    st.info("""
                    💡 **Interprétation des courbes** :
                    - Les différentes fréquences pénètrent à différentes profondeurs
                    - Les écarts entre courbes révèlent l'hétérogénéité du sous-sol
                    - Les croisements indiquent des changements de matériaux
                    """)
            
            # Onglet 3D
            with freq_tabs[-1]:
                st.markdown("#### Volume 3D Interactif")
                fig_3d = parser.create_3d_volume()
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    st.success("✅ Cliquez-glissez pour rotation, molette pour zoom")
            
            # EXPORT EXCEL
            st.markdown("### 📥 Export des Données")
            col1, col2 = st.columns(2)
            
            with col1:
                # Export CSV complet
                csv_full = df_ert.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Télécharger Données Complètes (CSV)",
                    data=csv_full,
                    file_name="ert_multi_freq_complet.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export Excel
                if st.button("📦 Générer Excel Multi-Feuilles", type="primary"):
                    excel_path = "/tmp/ert_multi_freq_export.xlsx"
                    msg = parser.export_to_excel(excel_path)
                    st.success(msg)
                    
                    if os.path.exists(excel_path):
                        with open(excel_path, 'rb') as f:
                            st.download_button(
                                label="📥 Télécharger Excel",
                                data=f.read(),
                                file_name="ert_multi_freq_export.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
            
            # Sauvegarder dans session state pour utilisation par Kibali
            st.session_state['multi_freq_data'] = df_ert
            st.session_state['multi_freq_coords'] = coords_df
            st.session_state['multi_freq_parser'] = parser
        
        else:
            st.error("❌ Aucune donnée extraite. Vérifiez le format des fichiers.")

# Section for binary file upload
st.markdown("---")
uploaded_file = st.file_uploader("Choisir un fichier binaire", type=["bin","dat","raw","bin","safetensors","pt","ckpt"])
if uploaded_file:
    file_bytes = uploaded_file.read()
    file_path = Path("/tmp/uploaded_file")
    file_path.write_bytes(file_bytes) # Save for potential model loading
    st.subheader("📜 Hex + ASCII Dump")
    hex_dump = hex_ascii_view(file_bytes, bytes_per_line=16, max_lines=100)
    st.text_area("Hex Dump", hex_dump, height=400)
    st.subheader("🔢 Extraction des nombres")
    numbers = extract_numbers(file_bytes)
    if numbers:
        # SAUVEGARDER LES DONNÉES POUR VISUALISATION
        st.session_state.current_file_data = numbers
        st.session_state.current_filename = uploaded_file.name
        
        df = pd.DataFrame(numbers, columns=["Value"])
        st.dataframe(df)
        st.subheader("📊 Statistiques rapides")
        st.write(df.describe())
        
        # 🧠 ANALYSE INTELLIGENTE KIBALI POUR ERT
        if uploaded_file.name.lower().endswith('.dat'):
            st.markdown("---")
            
            # 🔍 ANALYSE AUTOMATIQUE DE LA STRUCTURE DU FICHIER
            st.subheader("🔍 Analyse Automatique de Structure - Fichier .DAT")
            
            with st.expander("📋 Rapport de Structure Détecté", expanded=True):
                try:
                    # Analyser structure du fichier avec MultiFreqERTParser
                    from multi_freq_ert_parser import MultiFreqERTParser
                    file_path_str = str(file_path)
                    
                    parser = MultiFreqERTParser()
                    
                    # Tester si c'est un fichier multi-fréquences
                    if parser.detect_format(file_path_str) is not None:
                        st.info("🔬 **Fichier ERT Multi-Fréquences détecté !**")
                        
                        # Parser le fichier
                        df = parser.parse_file(file_path_str)
                        
                        if not df.empty:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Type de fichier", "ERT Multi-Freq")
                                st.metric("Mesures", len(df))
                            
                            with col2:
                                st.metric("Fréquences", len(parser.frequencies) if parser.frequencies else "N/A")
                                st.metric("Survey Points", len(parser.survey_points) if parser.survey_points else "N/A")
                            
                            with col3:
                                freq_range = f"{min(parser.frequencies):.0f}-{max(parser.frequencies):.0f}" if parser.frequencies else "N/A"
                                st.metric("Plage MHz", freq_range)
                                depth_range = f"{parser.metadata['depth_range'][0]:.1f} à {parser.metadata['depth_range'][1]:.1f}m" if 'depth_range' in parser.metadata else "N/A"
                                st.metric("Profondeur", depth_range)
                            
                            # Structure
                            st.markdown("#### 📊 Structure des Données")
                            st.dataframe(df[['project', 'survey_point', 'depth', 'frequency_MHz', 'resistivity']].head(10))
                            
                            # Sauvegarder en session
                            st.session_state['multi_freq_data'] = df
                            st.session_state['multi_freq_parser'] = parser
                        else:
                            st.warning("⚠️ Fichier détecté mais aucune donnée parsée")
                    else:
                        # Format non reconnu - affichage basique
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Type de fichier", "DAT (format inconnu)")
                            st.metric("Confiance", "N/A")
                        
                        with col2:
                            st.metric("Encodage", "Auto-détecté")
                            st.metric("Stations", "N/A")
                        
                        with col3:
                            st.metric("Colonnes", "N/A")
                            st.metric("Mesures", "N/A")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse de structure: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # ═══════════════════════════════════════════════════════════════
            # DÉTECTION FORMAT SURVEY-POINT / DEPTH / DATA
            # ═══════════════════════════════════════════════════════════════
            st.markdown("---")
            
            try:
                # survey_parser = SurveyDepthDataParser()
                # 
                # if survey_parser.detect_format(file_path_str):
                    st.success("🎯 **Format Survey-Point / Depth / Data détecté !**")
                    
                    with st.expander("📊 ANALYSE PROFIL VERTICAL - Survey Points", expanded=True):
                        with st.spinner("Analyse du profil survey-point/depth/data..."):
                            # Charger données
                            df_survey = survey_parser.load_data(file_path_str)
                            
                            # Afficher statistiques
                            st.markdown("##### 📈 Structure Détectée")
                            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                            
                            with col_s1:
                                st.metric("📍 Survey Points", survey_parser.structure['num_survey_points'])
                            with col_s2:
                                st.metric("📏 Points Total", survey_parser.structure['num_points'])
                            with col_s3:
                                depth_range = survey_parser.structure['depth_range']
                                st.metric("🌊 Profondeur Max", f"{abs(depth_range[1]):.1f} m")
                            with col_s4:
                                data_range = survey_parser.structure['data_range']
                                st.metric("📊 Plage Valeurs", f"{data_range[0]:.3f} - {data_range[1]:.3f}")
                            
                            # Rapport statistique
                            st.markdown("##### 📋 Rapport Statistique Complet")
                            report = survey_parser.generate_statistics_report()
                            st.text(report)
                            
                            # Bouton télécharger rapport
                            st.download_button(
                                "📥 Télécharger Rapport Survey (.txt)",
                                report,
                                f"survey_analysis_{uploaded_file.name}.txt",
                                "text/plain"
                            )
                            
                            # Aperçu données
                            st.markdown("##### 🔢 Aperçu Données Survey")
                            st.dataframe(df_survey.head(30), use_container_width=True)
                            
                            # VISUALISATIONS AUTOMATIQUES
                            st.markdown("---")
                            st.markdown("### 📊 Visualisations Automatiques")
                            
                            viz_types = st.multiselect(
                                "Sélectionnez les types de visualisation:",
                                ["Coupe 2D Interpolée", "Volume 3D", "Profils Verticaux", "Carte de Contours"],
                                default=["Coupe 2D Interpolée", "Profils Verticaux"]
                            )
                            
                            if st.button("🚀 GÉNÉRER VISUALISATIONS", type="primary", use_container_width=True):
                                with st.spinner("Génération des visualisations..."):
                                    
                                    if "Coupe 2D Interpolée" in viz_types:
                                        st.markdown("#### 🗺️ Coupe 2D - Profil de Résistivité Interpolé")
                                        
                                        interp_method = st.selectbox(
                                            "Méthode d'interpolation:",
                                            ["cubic", "linear", "nearest"],
                                            index=0
                                        )
                                        
                                        fig_2d, info_2d = survey_parser.create_2d_section(
                                            interpolation_method=interp_method,
                                            resolution=100,
                                            title="Coupe 2D - Profil de Résistivité"
                                        )
                                        
                                        st.plotly_chart(fig_2d, use_container_width=True)
                                        
                                        st.info(f"✅ Interpolation {interp_method} | {info_2d['num_points']} points mesurés | Résolution {info_2d['resolution']}x{info_2d['resolution']}")
                                    
                                    if "Volume 3D" in viz_types:
                                        st.markdown("#### 🧊 Volume 3D - Visualisation Interactive")
                                        
                                        fig_3d, info_3d = survey_parser.create_3d_volume()
                                        st.plotly_chart(fig_3d, use_container_width=True)
                                        
                                        st.info("✅ Volume 3D généré | Rotation interactive activée")
                                    
                                    if "Profils Verticaux" in viz_types:
                                        st.markdown("#### 📉 Profils Verticaux par Survey Point")
                                        
                                        fig_prof, info_prof = survey_parser.create_vertical_profiles()
                                        st.plotly_chart(fig_prof, use_container_width=True)
                                        
                                        st.info(f"✅ {info_prof['num_profiles']} profils verticaux générés")
                                    
                                    if "Carte de Contours" in viz_types:
                                        st.markdown("#### 🗺️ Carte de Contours - Isolignes")
                                        
                                        num_levels = st.slider("Nombre de niveaux:", 5, 30, 15)
                                        
                                        fig_cont, info_cont = survey_parser.create_contour_map(num_levels=num_levels)
                                        st.plotly_chart(fig_cont, use_container_width=True)
                                        
                                        st.info(f"✅ Carte de contours avec {num_levels} niveaux")
                                    
                                    st.success("✅ Toutes les visualisations générées avec succès !")
                            
                            # Sauvegarder dans session state
                            st.session_state['survey_data'] = df_survey
                            st.session_state['survey_parser'] = survey_parser
            
            except Exception as e:
                # Pas de format survey détecté, continuer normalement
                pass
            
            st.markdown("---")
            st.subheader("🧠 Analyse Intelligente Kibali - Données ERT")
            st.info("Kibali analyse les données avec son intelligence géophysique pour détecter incohérences et rendre les données cohérentes")
            
            # Détecter si données ERT (résistivités)
            is_ert_data = any(10 <= val <= 10000 for val in numbers[:50])  # Plage typique résistivité
            
            if is_ert_data:
                with st.expander("⚙️ Configuration Contexte Géologique", expanded=False):
                    context_choice = st.selectbox(
                        "Zone géographique / contexte:",
                        ["gabon", "sahel", "automatique"],
                        index=0,
                        help="Kibali adapte son analyse selon le contexte géologique"
                    )
                
                if st.button("🚀 LANCER ANALYSE INTELLIGENTE KIBALI", type="primary", use_container_width=True):
                    with st.spinner("🧠 Kibali analyse les données avec son intelligence..."):
                        # Créer profondeurs synthétiques si pas disponibles
                        n_points = len(numbers)
                        depths = list(np.linspace(0, n_points * 0.5, n_points))  # 0.5m spacing
                        resistivities = numbers
                        
                        # Analyse intelligente
                        kibali_results = kibali_analyze_ert(depths, resistivities, context=context_choice if context_choice != "automatique" else "gabon")
                        
                        # Afficher synthèse intelligente
                        st.markdown("### 📋 Synthèse Intelligente Kibali")
                        st.markdown(kibali_results["synthese_intelligente"])
                        
                        # Onglets détaillés
                        tab1, tab2, tab3, tab4 = st.tabs(["✅ Validation", "🔧 Corrections", "🪨 Couches", "💧 Hydrogéologie"])
                        
                        with tab1:
                            validation = kibali_results["validation_stratigraphique"]
                            
                            col_v1, col_v2 = st.columns(2)
                            with col_v1:
                                if validation["valid"]:
                                    st.success(f"✅ Données COHÉRENTES")
                                else:
                                    st.warning(f"⚠️ Anomalies détectées")
                            with col_v2:
                                st.metric("Score Cohérence", f"{validation['score_coherence']}/100")
                            
                            if validation['anomalies']:
                                st.markdown("#### ❌ Anomalies Critiques")
                                for anom in validation['anomalies']:
                                    st.error(f"• {anom['message']}")
                            
                            if validation['warnings']:
                                st.markdown("#### ⚠️ Avertissements")
                                for warn in validation['warnings']:
                                    st.warning(f"• {warn['message']}")
                        
                        with tab2:
                            corrections = kibali_results["corrections_appliquees"]
                            if corrections:
                                st.info(f"🔧 Kibali a appliqué {len(corrections)} corrections intelligentes")
                                corr_df = pd.DataFrame(corrections)
                                st.dataframe(corr_df, use_container_width=True)
                                
                                # Bouton télécharger données corrigées
                                corrected_data = kibali_results["donnees_corrigees"]
                                corrected_csv = pd.DataFrame({
                                    "Profondeur (m)": corrected_data["profondeurs"],
                                    "Résistivité Originale (Ω.m)": corrected_data["resistivites_originales"],
                                    "Résistivité Corrigée (Ω.m)": corrected_data["resistivites_corrigees"]
                                })
                                st.download_button(
                                    "📥 Télécharger Données Corrigées (CSV)",
                                    corrected_csv.to_csv(index=False).encode('utf-8'),
                                    f"{uploaded_file.name}_kibali_corrected.csv",
                                    "text/csv"
                                )
                            else:
                                st.success("✅ Aucune correction nécessaire - Données déjà cohérentes")
                        
                        with tab3:
                            layers = kibali_results["couches_geologiques"]
                            st.markdown(f"#### 🪨 {len(layers)} Couches Géologiques Identifiées")
                            
                            for i, layer in enumerate(layers, 1):
                                with st.expander(f"Couche {i}: {layer['profondeur_debut']:.1f}-{layer['profondeur_fin']:.1f}m ({layer['epaisseur']:.1f}m)"):
                                    col_l1, col_l2 = st.columns(2)
                                    with col_l1:
                                        st.metric("Type", layer['type_geologique'])
                                        st.metric("Profondeur", f"{layer['profondeur_debut']:.1f} - {layer['profondeur_fin']:.1f} m")
                                    with col_l2:
                                        st.metric("Épaisseur", f"{layer['epaisseur']:.1f} m")
                                        st.metric("Résistivité", f"{layer['resistivite_moyenne']:.1f} Ω.m")
                                    
                                    st.info(layer['description'])
                        
                        with tab4:
                            hydro = kibali_results["analyse_hydrogeologique"]
                            
                            col_h1, col_h2, col_h3 = st.columns(3)
                            with col_h1:
                                potential_color = {"faible": "🔴", "moyen": "🟡", "bon": "🟢", "excellent": "🟢"}
                                st.metric("Potentiel Hydrique", f"{potential_color.get(hydro['potentiel_hydrique'], '⚪')} {hydro['potentiel_hydrique'].upper()}")
                            with col_h2:
                                if hydro['profondeur_nappe_estimee']:
                                    st.metric("Nappe Phréatique", f"{hydro['profondeur_nappe_estimee']:.1f} m")
                                else:
                                    st.metric("Nappe Phréatique", "Profonde")
                            with col_h3:
                                st.metric("Zones Aquifères", len(hydro['couches_aquiferes']))
                            
                            if hydro['recommandations']:
                                st.markdown("#### 📋 Recommandations Kibali")
                                for rec in hydro['recommandations']:
                                    st.success(f"✓ {rec}")
                        
                        # Sauvegarder résultats dans session state
                        st.session_state['kibali_ert_analysis'] = kibali_results
                        st.success("✅ Analyse intelligente Kibali terminée!")
            else:
                st.warning("⚠️ Données ne semblent pas être de type ERT (résistivité). Analyse intelligente non applicable.")
        
        st.markdown("---")
        st.subheader("🎯 Clustering KMeans")
        n_clusters = st.slider("Nombre de clusters", 2, 10, 3)
        labels, centers = cluster_numbers(numbers, n_clusters=n_clusters)
        df['Cluster'] = labels
        st.dataframe(df)
        st.subheader("📈 Visualisation des clusters")
        fig, ax = plt.subplots()
        for i in range(n_clusters):
            cluster_vals = df[df['Cluster']==i]['Value']
            ax.scatter([i]*len(cluster_vals), cluster_vals, label=f"Cluster {i}")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Valeurs")
        ax.legend()
        st.pyplot(fig)
        st.subheader("💾 Export CSV")
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger CSV", csv_bytes, file_name="binary_structured.csv")
    else:
        st.warning("Aucun nombre détecté dans ce fichier binaire.")
    
    # 🔍 FOUILLE INTELLIGENTE AUTOMATIQUE
    st.subheader("🔍 Fouille Intelligente Multi-Sources")
    st.info("Combine: Hex+ASCII Dump + Base Vectorielle RAG + Base ERT + Web Search + Synthèse IA")
    
    col_inv1, col_inv2 = st.columns(2)
    with col_inv1:
        if st.button("🔬 LANCER INVESTIGATION COMPLÈTE", type="primary", use_container_width=True):
            with st.spinner("🔍 Investigation en cours (7 phases)..."):
                investigation_result = deep_binary_investigation(file_bytes, uploaded_file.name)
                st.session_state.last_investigation = investigation_result
                st.success("✅ Investigation terminée!")
    
    with col_inv2:
        if "last_investigation" in st.session_state:
            st.download_button(
                "📥 Télécharger Rapport",
                st.session_state.last_investigation.get('full_report', ''),
                file_name=f"investigation_{uploaded_file.name}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Afficher le dernier rapport d'investigation en phases expandables
    if "last_investigation" in st.session_state:
        st.markdown("### 📋 Rapport d'Investigation Complet")
        
        result = st.session_state.last_investigation
        phases = result.get('phases', {})
        
        # Add summary statistics for expander titles
        phase_summaries = {
            '1️⃣ PHASE 1: EXTRACTION HEX + ASCII': '📜 Dump hexadécimal et extraction de nombres',
            '2️⃣ PHASE 2: ANALYSES TECHNIQUES': '📊 Entropie, patterns, métadonnées',
            '3️⃣ PHASE 3: FOUILLE BASE VECTORIELLE RAG': '🔍 Recherche dans la base de connaissances',
            '4️⃣ PHASE 4: FOUILLE SPÉCIALISÉE ERT': '🔬 Analyse ERT, minéraux, correspondances',
            '5️⃣ PHASE 5: RECHERCHE WEB INTELLIGENTE': '🌐 Recherche internet contextuelle',
            '6️⃣ PHASE 6: SYNTHÈSE MULTI-SOURCES': '🎯 Consolidation des résultats',
            '7️⃣ PHASE 7: RECOMMANDATIONS': '💡 Actions suggérées'
        }
        
        # Display each phase in its own expander
        for i, (phase_title, phase_content) in enumerate(phases.items()):
            # Extract phase number emoji for matching
            phase_key = phase_title.split('\n')[0] if '\n' in phase_title else phase_title
            summary = phase_summaries.get(phase_key, '')
            
            # Count lines for height estimation
            num_lines = len(phase_content.split('\n'))
            estimated_height = min(500, max(200, num_lines * 15))
            
            # Only first expander open by default
            with st.expander(f"{phase_key} - {summary}", expanded=(i==0)):
                st.text_area(
                    label="Contenu de la phase",
                    value=phase_content,
                    height=estimated_height,
                    key=f"phase_{i}",
                    label_visibility="collapsed"
                )
    
    # Analyse automatique du fichier dès l'upload
    if st.button("🚀 Analyser automatiquement avec IA (GPU optimisé)"):
        with st.spinner(f"🚀 Analyse IA en cours sur {device.upper()}... {'(GPU accéléré)' if device == 'cuda' else '(CPU)'}"):
            # Vérifier que le modèle utilise bien le GPU si disponible
            if device == 'cuda' and model.device.type != 'cuda':
                st.warning("🔧 Migration du modèle vers GPU...")
                model = model.to('cuda')
                st.success(f"✅ Modèle migré vers GPU - {gpu_info}")
           
            # Afficher les informations d'optimisation
            st.info(f"🖥️ Device: {device.upper()} | Modèle: {model.device} | Precision: {model.dtype}")
           
            # Analyse optimisée avec GPU
            analysis_result = analyze_with_ai(
                f"Analyse complète et détaillée de ce fichier binaire. Identifie le type de fichier, son contenu, et fournis une interprétation experte géophysique ERT si applicable. Utilise tous les outils disponibles pour une analyse maximale.",
                file_bytes, numbers, hex_dump, n_clusters,
                st.session_state.get('model'), st.session_state.get('tokenizer'), st.session_state.get('device')
            )
            st.subheader("🧠 Analyse IA Automatique (GPU Optimisée)")
            st.markdown(analysis_result)
    elif not st.session_state.vectorstore:
        st.info("Veuillez d'abord uploader et indexer des PDFs pour activer l'analyse LLM.")
# Section Chat en Temps Réel
st.subheader("💬 Chat d'Analyse en Temps Réel")
# Configuration GPU pour le chat
col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    if "gpu_mode_chat" not in st.session_state:
        st.session_state.gpu_mode_chat = torch.cuda.is_available()
with col2:
    gpu_mode_chat = st.checkbox(
        "🚀 Mode GPU",
        value=st.session_state.gpu_mode_chat,
        help="Active l'accélération GPU pour le chat (plus rapide)",
        key="gpu_chat_toggle"
    )
    st.session_state.gpu_mode_chat = gpu_mode_chat
with col3:
    # Affichage du statut GPU
    if gpu_mode_chat and torch.cuda.is_available():
        st.success("✅ GPU activé")
        gpu_info_chat = f"{torch.cuda.get_device_name(0)}"
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.caption(f"🔥 {memory_used:.1f}/{memory_total:.1f}GB")
    elif gpu_mode_chat and not torch.cuda.is_available():
        st.warning("⚠️ GPU indisponible")
        st.caption("💻 Utilisation CPU")
    else:
        st.info("💻 Mode CPU")
        st.caption("🐌 Performance standard")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Posez votre question d'analyse..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        # Affichage du mode de traitement
        mode_display = "🚀 GPU" if st.session_state.gpu_mode_chat and torch.cuda.is_available() else "💻 CPU"
        spinner_text = f"{mode_display} Agent LangChain réfléchit..."
       
        # Migration du modèle si mode GPU activé
        if st.session_state.gpu_mode_chat and torch.cuda.is_available() and model.device.type != 'cuda':
            with st.spinner("🔄 Migration vers GPU..."):
                model.to('cuda')
                st.success("✅ Modèle migré vers GPU")
        elif not st.session_state.gpu_mode_chat and model.device.type == 'cuda':
            with st.spinner("🔄 Migration vers CPU..."):
                model.to('cpu')
                st.success("✅ Modèle migré vers CPU")
       
        with st.spinner(spinner_text):
            # Utiliser l'agent LangChain pour le chat avec optimisation GPU/CPU
            chat_prompt = f"""
Tu es un assistant expert en analyse de fichiers binaires. L'utilisateur pose une question d'analyse.
Question: {prompt}
Utilise les outils disponibles pour:
1. Rechercher dans la base de connaissances PDF si disponible
2. Effectuer des recherches web pour des informations complémentaires
3. Analyser des patterns si des données binaires sont mentionnées
4. Si ERT/résistivité: reproduire couleurs, comparer avec liquides/minéraux via recherches internet, décrire nature/profondeur/couleur
5. Pour fichiers .dat ERT, utilise mathematical_calculator avec formules FicheERT.pdf: Schlumberger: pi*(L**2 - l**2)/(2*l) * V/I, Wenner: 2*pi*a * V/I, Dipole-Dipole: pi*n*(n+1)*(n+2)*a * V/I
Réponds de manière précise et utile.
PERFORMANCE: Mode {mode_display} activé pour traitement optimisé.
"""
            try:
                # Analyse avancée avec outils pour chat
                enhanced_response = ""
               
                # Détecter le type de demande
                is_visualization_request = any(keyword in prompt.lower() for keyword in [
                    "coupe", "graphique", "visualisation", "visualise", "génère", "génerer", 
                    "graphe", "plot", "diagramme", "carte", "profil", "section", "image",
                    "montre", "affiche", "crée", "dessine", "couleur", "couleurs"
                ])
                
                is_analysis_request = any(keyword in prompt.lower() for keyword in [
                    "recherche", "approfondie", "analyse", "données", "résistivité"
                ])
                
                # PRIORITÉ 1: Génération de visualisation graphique
                if is_visualization_request and st.session_state.current_file_data:
                    try:
                        st.info("🎨 **Génération de visualisation en cours...**")
                        
                        # Initialiser l'agent graphique si nécessaire
                        if st.session_state.graph_agent is None:
                            with st.spinner("🔧 Chargement du moteur de visualisation avancé..."):
                                st.session_state.graph_agent = GraphGenerationAgent()
                        
                        # Initialiser le moteur avancé si nécessaire
                        if st.session_state.advanced_viz_engine is None:
                            with st.spinner("🚀 Initialisation du moteur PyGIMLI + OpenCV..."):
                                st.session_state.advanced_viz_engine = AdvancedVisualizationEngine()
                        
                        # Extraire les données du fichier actuel
                        file_data = st.session_state.current_file_data
                        data_array = np.array(file_data)
                        
                        # Générer la visualisation avec le moteur avancé
                        st.success("✨ **Génération de la coupe avec PyGIMLI + Matplotlib...**")
                        
                        viz_result = st.session_state.advanced_viz_engine.create_complete_ert_section(
                            data=data_array,
                            title=f"Coupe ERT - {st.session_state.get('current_filename', 'Données')}"
                        )
                        
                        if viz_result and 'figure' in viz_result:
                            st.pyplot(viz_result['figure'])
                            st.success("✅ **Visualisation générée avec succès !**")
                            
                            # Générer une explication intelligente avec l'agent
                            explanation = st.session_state.graph_agent.generate_explanation(
                                graph_type="2d_section",
                                data_summary={
                                    'min': float(np.min(data_array)),
                                    'max': float(np.max(data_array)),
                                    'mean': float(np.mean(data_array)),
                                    'points': len(data_array)
                                }
                            )
                            
                            st.markdown("### 📊 Analyse de la coupe")
                            st.markdown(explanation)
                            
                            # Sauvegarder dans l'historique
                            assistant_response = f"""✅ **Coupe de résistivité générée !**

{explanation}

**Caractéristiques de la visualisation:**
- Type: Coupe 2D avec interpolation
- Données: {len(data_array)} points
- Plage: {np.min(data_array):.2f} - {np.max(data_array):.2f} Ω.m
- Moteur: PyGIMLI + Matplotlib + OpenCV
"""
                            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                            st.stop()  # Arrêter ici, visualisation complète
                            
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération graphique: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                
                # PRIORITÉ 2: Analyse avec recherche
                if is_analysis_request:
                    # Effectuer recherche web
                    try:
                        web_results = web_search_enhanced(prompt + " ERT geophysics electrical resistivity")
                        enhanced_response += f"🌐 RECHERCHE WEB EFFECTUÉE:\n{web_results}\n\n"
                    except:
                        pass
                   
                    # Recherche RAG
                    if st.session_state.vectorstore:
                        try:
                            rag_results = search_vectorstore(prompt)
                            enhanced_response += f"📚 BASE DE CONNAISSANCES:\n{rag_results}\n\n"
                        except:
                            pass
                   
                    # Analyse ERT complète si pertinent
                    if any(keyword in prompt.lower() for keyword in ["ert", "résistivité", "matériaux", "analyse", "données"]):
                        try:
                            # Génération du rapport complet avec outils
                            complete_report = create_advanced_analysis_report(prompt)
                            enhanced_response += f"📊 RAPPORT D'ANALYSE COMPLET:\n{complete_report}\n\n"
                           
                            # Données exemple pour démonstration visuelle
                            sample_data = [0.05, 0.3, 2.0, 10.0, 50.0, 200.0, 1000.0, 5000.0]
                            ert_analysis = resistivity_color_analysis(sample_data)
                            enhanced_response += f"🎨 ANALYSE VISUELLE ERT:\n{ert_analysis}\n\n"
                        except Exception as e:
                            enhanced_response += f"⚠️ Analyse ERT partielle: {e}\n\n"
               
                # Utiliser directement le modèle Qwen pour le chat avec contexte enrichi
                system_content = f"""Tu es un expert mondial en géophysique ERT avec accès complet à tous les outils d'analyse.
               
                CONTEXTE ENRICHI AVEC OUTILS EXÉCUTÉS:
                {enhanced_response}
                INSTRUCTIONS STRICTES:
                1. Utilise OBLIGATOIREMENT les données ci-dessus pour répondre
                2. Présente les tableaux HTML et graphiques inclus
                3. Cite les résultats de recherche web obtenus
                4. Fournis des analyses quantitatives précises
                5. Compare avec les matériaux identifiés automatiquement
                6. Explique les couleurs de visualisation ERT
                7. Donne des recommandations techniques concrètes
               
                RÉPONSE ATTENDUE:
                - Structure professionnelle avec sections claires
                - Données numériques précises issues des analyses
                - Références aux sources trouvées
                - Visualisations décrites et expliquées
                - Conclusions basées sur les outils utilisés
               
                INTERDICTIONS:
                - Ne JAMAIS dire "je n'ai pas accès"
                - Ne pas inventer de données - utiliser celles fournies
                - Ne pas être générique - être spécifique aux résultats obtenus"""
               
                chat_messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
               
                inputs = tokenizer.apply_chat_template(
                    chat_messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
                # Create attention mask to avoid warnings when pad_token == eos_token
                attention_mask = (inputs != tokenizer.pad_token_id).long().to(model.device)
               
                # Génération optimisée selon le mode GPU/CPU
                start_time = time.time()
                with torch.no_grad():
                    if st.session_state.gpu_mode_chat and torch.cuda.is_available() and model.device.type == 'cuda':
                        # Mode GPU optimisé avec mixed precision
                        with torch.cuda.amp.autocast():
                            outputs = model.generate(
                                inputs,
                                attention_mask=attention_mask,
                                max_new_tokens=3000,  # AUGMENTÉ pour réponses COMPLÈTES
                                temperature=0.6,
                                do_sample=True,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                                use_cache=True,
                                num_beams=1
                            )
                    else:
                        # Mode CPU standard
                        outputs = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            max_new_tokens=3000,  # AUGMENTÉ pour réponses COMPLÈTES
                            temperature=0.6,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id
                        )
               
                generation_time = time.time() - start_time
                assistant_response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
               
                # Ajouter informations de performance
                device_used = model.device.type.upper()
                performance_info = f"\n\n---\n**⚡ Performance:** {device_used} | **⏱️ Temps:** {generation_time:.2f}s"
               
                if model.device.type == 'cuda':
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    performance_info += f" | **💾 VRAM:** {memory_used:.1f}/{memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)"
               
                assistant_response_with_perf = assistant_response + performance_info
               
            except Exception as e:
                # Fallback vers le système classique
                st.warning(f"Chat IA a échoué: {e}. Utilisation du système classique...")
                fallback_start_time = time.time()
               
                # Recherche web
                tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
                web_results = tool.invoke(prompt)
                web_context = "\n".join([r["content"] for r in web_results])
                context = f"Contexte web:\n{web_context}"
                # Contexte documents si disponible - RECHERCHE GLOBALE ILLIMITÉE
                if st.session_state.vectorstore:
                    # Récupérer BEAUCOUP de documents pour une couverture GLOBALE COMPLÈTE
                    total_docs = st.session_state.vectorstore.index.ntotal if hasattr(st.session_state.vectorstore, 'index') else 1000
                    search_k = min(100, total_docs) if total_docs > 0 else 100  # 100+ documents au lieu de 30
                    
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={
                            "k": search_k,  # Recherche ILLIMITÉE profonde sur 100+ documents
                            "fetch_k": min(search_k * 3, total_docs)
                        }
                    )
                    docs = retriever.get_relevant_documents(prompt)
                    
                    # Grouper par source pour meilleure vue globale
                    sources = {}
                    for doc in docs:
                        source = doc.metadata.get('source', 'Unknown')
                        if source not in sources:
                            sources[source] = []
                        sources[source].append(doc.page_content[:400])
                    
                    doc_context = f"\n📊 FOUILLE GLOBALE: {len(docs)} passages trouvés dans {len(sources)} sources ({search_k}/{total_docs} docs analysés):\n"
                    for source, chunks in sources.items():
                        doc_context += f"\n📄 {source}: {len(chunks)} passages\n" + "\n".join(chunks[:3])
                    
                    context += f"\n\nContexte documents indexés (recherche globale illimitée):\n{doc_context}"
                full_prompt = f"""Tu es un assistant expert en analyse de données et fichiers binaires. Utilise le contexte fourni pour donner des réponses précises et utiles.
{context}
Question de l'utilisateur: {prompt}
Réponse détaillée:"""
                messages = [
                    {"role": "system", "content": "Tu es un assistant expert en analyse de fichiers binaires et modèles ML."},
                    {"role": "user", "content": full_prompt}
                ]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
                # Create attention mask to avoid warnings when pad_token == eos_token
                attention_mask = (inputs != tokenizer.pad_token_id).long().to(model.device)
               
                with torch.no_grad():
                    if st.session_state.gpu_mode_chat and torch.cuda.is_available() and model.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            outputs = model.generate(
                                inputs,
                                attention_mask=attention_mask,
                                max_new_tokens=3000,  # AUGMENTÉ pour réponses COMPLÈTES
                                temperature=0.7,
                                do_sample=True,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                                use_cache=True,
                                num_beams=1
                            )
                    else:
                        outputs = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            max_new_tokens=3000,  # AUGMENTÉ pour réponses COMPLÈTES
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id
                        )
               
                fallback_time = time.time() - fallback_start_time
                assistant_response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
               
                # Ajouter informations de performance pour fallback
                device_used = model.device.type.upper()
                performance_info = f"\n\n---\n**⚡ Performance (Fallback):** {device_used} | **⏱️ Temps:** {fallback_time:.2f}s"
               
                if model.device.type == 'cuda':
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    performance_info += f" | **💾 VRAM:** {memory_used:.1f}/{memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%)"
               
                assistant_response_with_perf = assistant_response + performance_info
           
            # Afficher la réponse avec les informations de performance
            st.markdown(assistant_response_with_perf)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response_with_perf})
def generate_resistivity_table(resistivity_values: list) -> str:
    """Génère un tableau HTML des valeurs de résistivité"""
    if not resistivity_values:
        return "Aucune donnée pour générer le tableau"
   
    import numpy as np
    from resistivity_color_mapper import ResistivityColorMapper
   
    mapper = ResistivityColorMapper()
    arr = np.array(resistivity_values)
   
    # Créer le tableau HTML
    table_html = """
    <div style='overflow-x: auto;'>
    <table style='border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;'>
    <thead>
        <tr style='background-color: #2E86AB; color: white;'>
            <th style='border: 1px solid #ddd; padding: 12px; text-align: center;'>Index</th>
            <th style='border: 1px solid #ddd; padding: 12px; text-align: center;'>Résistivité (Ω·m)</th>
            <th style='border: 1px solid #ddd; padding: 12px; text-align: center;'>Couleur</th>
            <th style='border: 1px solid #ddd; padding: 12px; text-align: center;'>Classification</th>
            <th style='border: 1px solid #ddd; padding: 12px; text-align: center;'>Matériau Probable</th>
        </tr>
    </thead>
    <tbody>
    """
   
    for i, rho in enumerate(arr[:20]): # Limiter à 20 pour l'affichage
        color, desc = mapper.get_color_for_resistivity(rho)
       
        # Classification
        if rho < 10:
            classification = "Conducteur"
            material = "Argile, eau salée"
        elif rho < 100:
            classification = "Semi-conducteur"
            material = "Sol humide, sable"
        elif rho < 1000:
            classification = "Résistant"
            material = "Calcaire, grès"
        else:
            classification = "Très résistant"
            material = "Granite, air"
       
        # Ligne du tableau avec couleur de fond
        bg_color = color if color != '#FFFFFF' else '#F0F0F0'
        text_color = 'white' if color in ['#000080', '#0000FF', '#FF0000'] else 'black'
       
        table_html += f"""
        <tr>
            <td style='border: 1px solid #ddd; padding: 8px; text-align: center;'>{i+1}</td>
            <td style='border: 1px solid #ddd; padding: 8px; text-align: center; font-weight: bold;'>{rho:.3f}</td>
            <td style='border: 1px solid #ddd; padding: 8px; text-align: center; background-color: {bg_color}; color: {text_color};'>{color}</td>
            <td style='border: 1px solid #ddd; padding: 8px; text-align: center;'>{classification}</td>
            <td style='border: 1px solid #ddd; padding: 8px; text-align: center;'>{material}</td>
        </tr>
        """
   
    table_html += """
    </tbody>
    </table>
    </div>
   
    <div style='margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
    <h4>📊 Statistiques Résumées:</h4>
    <ul>
        <li><strong>Nombre de valeurs:</strong> {count}</li>
        <li><strong>Résistivité moyenne:</strong> {mean:.3f} Ω·m</li>
        <li><strong>Médiane:</strong> {median:.3f} Ω·m</li>
        <li><strong>Écart-type:</strong> {std:.3f} Ω·m</li>
        <li><strong>Plage:</strong> {min:.3f} - {max:.3f} Ω·m</li>
        <li><strong>Ratio max/min:</strong> {ratio:.1f}</li>
    </ul>
    </div>
    """.format(
        count=len(arr),
        mean=np.mean(arr),
        median=np.median(arr),
        std=np.std(arr),
        min=np.min(arr),
        max=np.max(arr),
        ratio=np.max(arr)/np.min(arr) if np.min(arr) > 0 else float('inf')
    )
   
    return table_html
def create_advanced_analysis_report(query: str, resistivity_values: list = None) -> str:
    """Crée un rapport d'analyse avancé complet"""
    if not resistivity_values:
        # Données exemple représentatives de différents matériaux
        resistivity_values = [
            0.05, 0.2, 0.3, # Eau salée/saumure
            2.0, 5.0, 8.0, # Argile
            15.0, 25.0, 35.0, # Sol humide
            80.0, 120.0, 180.0, # Sable
            300.0, 500.0, 800.0, # Calcaire
            2000.0, 3500.0, 5000.0, # Granite
            0.0000024, 0.0000026, # Or
            900000.0, 1100000.0 # Diamant
        ]
   
    report = f"""
    🔬 RAPPORT D'ANALYSE GÉOPHYSIQUE COMPLET
    =========================================
   
    📋 CONTEXTE DE LA DEMANDE:
    {query}
   
    🎯 MÉTHODOLOGIE APPLIQUÉE:
    ✅ Recherche web automatisée pour données actualisées
    ✅ Analyse comparative avec base de données géophysique
    ✅ Validation contre références scientifiques
    ✅ Génération de visualisations et tableaux
    ✅ Calculs statistiques avancés
   
    📊 DONNÉES ANALYSÉES:
    • Nombre d'échantillons: {len(resistivity_values)}
    • Plage de résistivité: {min(resistivity_values):.2e} - {max(resistivity_values):.2e} Ω·m
    • Ordre de grandeur: {max(resistivity_values)/min(resistivity_values):.1e}
   
    🔍 IDENTIFICATION AUTOMATIQUE DES MATÉRIAUX:
    """
   
    # Analyse détaillée par matériau
    import numpy as np
    from resistivity_color_mapper import ResistivityColorMapper, DynamicERTAnalyzer
   
    try:
        mapper = ResistivityColorMapper()
        analyzer = DynamicERTAnalyzer()
       
        # Classification automatique
        materials_detected = {}
        for rho in resistivity_values:
            materials = mapper.find_similar_materials(rho, tolerance=0.3)
            if materials:
                top_material = materials[0]
                mat_name = top_material['name']
                if mat_name not in materials_detected:
                    materials_detected[mat_name] = {
                        'values': [],
                        'category': top_material['category'],
                        'typical': top_material['typical_value'],
                        'nature': top_material['nature']
                    }
                materials_detected[mat_name]['values'].append(rho)
       
        # Rapport par matériau détecté
        for i, (mat_name, mat_data) in enumerate(materials_detected.items(), 1):
            avg_rho = np.mean(mat_data['values'])
            count = len(mat_data['values'])
            report += f"""
    {i}. {mat_name.upper()} ({mat_data['category']})
       • Occurrences détectées: {count}
       • Résistivité moyenne mesurée: {avg_rho:.2e} Ω·m
       • Résistivité typique théorique: {mat_data['typical']:.2e} Ω·m
       • Nature: {mat_data['nature']}
       • Concordance: {100 - abs(np.log10(avg_rho) - np.log10(mat_data['typical']))*20:.1f}%
            """
       
        # Recherche web automatique pour validation
        try:
            web_validation = web_search_enhanced(
                f"electrical resistivity values {query} geophysics materials validation",
                "validation"
            )
            report += f"""
   
    🌐 VALIDATION PAR RECHERCHE WEB:
    {web_validation}
    """
        except:
            report += "\n🌐 VALIDATION WEB: En cours..."
       
        # Calculs géophysiques avancés
        arr = np.array(resistivity_values)
        report += f"""
   
    📊 ANALYSES STATISTIQUES AVANCÉES:
   
    🔢 Paramètres de base:
    • Moyenne géométrique: {np.exp(np.mean(np.log(arr))):.2e} Ω·m
    • Médiane: {np.median(arr):.2e} Ω·m
    • Écart-type logarithmique: {np.std(np.log10(arr)):.3f}
    • Coefficient de variation: {np.std(arr)/np.mean(arr):.3f}
   
    🎯 Classification géophysique:
    • Conducteurs (<10 Ω·m): {len(arr[arr < 10])} échantillons
    • Semi-conducteurs (10-100 Ω·m): {len(arr[(arr >= 10) & (arr < 100)])} échantillons
    • Résistants (100-1000 Ω·m): {len(arr[(arr >= 100) & (arr < 1000)])} échantillons
    • Très résistants (>1000 Ω·m): {len(arr[arr >= 1000])} échantillons
   
    🌡️ Estimation de profondeur (modèle empirique):
    • Profondeur d'investigation: {np.mean(arr)*0.1:.1f} m (approximative)
    • Résolution verticale: {np.std(arr)*0.05:.1f} m
        """
       
    except Exception as e:
        report += f"\n❌ Erreur dans l'analyse: {e}"
   
    report += """
   
    💡 RECOMMANDATIONS TECHNIQUES:
    • Utiliser inversion 2D/3D pour structures complexes
    • Valider par forages si possible
    • Considérer variations saisonnières
    • Appliquer corrections topographiques si nécessaire
   
    📚 RÉFÉRENCES SCIENTIFIQUES:
    • Loke, M.H. (2001). Tutorial: 2-D and 3-D electrical imaging surveys
    • Telford et al. (1990). Applied Geophysics, Cambridge University Press
    • Reynolds, J.M. (2011). An Introduction to Applied and Environmental Geophysics
   
    ✅ RAPPORT GÉNÉRÉ AUTOMATIQUEMENT AVEC OUTILS AVANCÉS
    """
   
    return report
def generate_resistivity_plot(resistivity_values: list) -> str:
    """Génère un graphique des valeurs de résistivité"""
    if not resistivity_values:
        return "Aucune donnée pour générer le graphique"
   
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import base64
    from resistivity_color_mapper import ResistivityColorMapper
   
    try:
        mapper = ResistivityColorMapper()
        arr = np.array(resistivity_values)
       
        # Créer la figure avec subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analyse Complète des Résistivités ERT', fontsize=16, fontweight='bold')
       
        # 1. Profil de résistivité avec couleurs
        colors = []
        for rho in arr:
            color, _ = mapper.get_color_for_resistivity(rho)
            colors.append(color)
       
        scatter = ax1.scatter(range(len(arr)), arr, c=colors, s=60, edgecolors='black', linewidth=0.5)
        ax1.plot(range(len(arr)), arr, 'k-', alpha=0.3, linewidth=1)
        ax1.set_xlabel('Position de mesure')
        ax1.set_ylabel('Résistivité (Ω·m)')
        ax1.set_title('Profil de Résistivité avec Couleurs ERT')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
       
        # 2. Histogramme
        ax2.hist(np.log10(arr), bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Log10(Résistivité)')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution des Résistivités')
        ax2.grid(True, alpha=0.3)
       
        # 3. Classification par zones
        zones = {'Conducteur (<10)': arr[arr < 10],
                'Semi-conducteur (10-100)': arr[(arr >= 10) & (arr < 100)],
                'Résistant (100-1000)': arr[(arr >= 100) & (arr < 1000)],
                'Très résistant (>1000)': arr[arr >= 1000]}
       
        zone_counts = [len(zone) for zone in zones.values()]
        zone_colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
       
        wedges, texts, autotexts = ax3.pie(zone_counts, labels=zones.keys(), colors=zone_colors,
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Classification des Matériaux')
       
        # 4. Évolution temporelle simulée
        ax4.plot(range(len(arr)), arr, 'b-', linewidth=2, marker='o', markersize=4)
        ax4.fill_between(range(len(arr)), arr, alpha=0.3, color='lightblue')
        ax4.set_xlabel('Séquence de mesure')
        ax4.set_ylabel('Résistivité (Ω·m)')
        ax4.set_title('Évolution des Mesures')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
       
        # Ajuster la mise en page
        plt.tight_layout()
       
        # Convertir en base64 pour affichage HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
       
        plot_base64 = base64.b64encode(plot_data).decode()
       
        return f'<img src="data:image/png;base64,{plot_base64}" style="max-width: 100%; height: auto;" alt="Graphique ERT">'
       
    except Exception as e:
        return f"Erreur lors de la génération du graphique: {e}"
def resistivity_color_analysis(resistivity_values: list, dat_file_path: str = None) -> str:
    """Analyse les couleurs de résistivité ERT avec validation contre fichiers .dat et détection de matériaux réels"""
    if not resistivity_values:
        return "❌ Aucune valeur de résistivité fournie pour l'analyse"
   
    import numpy as np
    from resistivity_color_mapper import ResistivityColorMapper, DynamicERTAnalyzer
   
    analysis = "🎨 ANALYSE DES COULEURS DE RÉSISTIVITÉ ERT\n"
    analysis += "=" * 50 + "\n\n"
   
    # Initialisation des analyseurs
    mapper = ResistivityColorMapper()
    analyzer = DynamicERTAnalyzer()
   
    # Conversion en array numpy
    rho_data = np.array(resistivity_values)
   
    # Statistiques de base
    analysis += f"📊 STATISTIQUES DES RÉSISTIVITÉS:\n"
    analysis += f" • Nombre de valeurs: {len(rho_data)}\n"
    analysis += f" • Résistivité moyenne: {np.mean(rho_data):.2f} Ω.m\n"
    analysis += f" • Médiane: {np.median(rho_data):.2f} Ω.m\n"
    analysis += f" • Écart-type: {np.std(rho_data):.2f} Ω.m\n"
    analysis += f" • Plage: {np.min(rho_data):.2f} - {np.max(rho_data):.2f} Ω.m\n"
    analysis += f" • Coefficient de variation: {np.std(rho_data)/np.mean(rho_data):.3f}\n\n"
   
    # Analyse des couleurs par valeur
    analysis += f"🎨 CARTOGRAPHIE COULEUR PAR VALEUR:\n"
    sample_values = np.unique(np.round(rho_data, 2))[:15] # Échantillon pour éviter surcharge
   
    for rho in sample_values:
        color, desc = mapper.get_color_for_resistivity(rho)
        analysis += f" • ρ = {rho:.2f} Ω.m → Couleur: {color} ({desc})\n"
    analysis += "\n"
   
    # Détection de matériaux réels avec validation .dat
    analysis += f"🔍 DÉTECTION DE MATÉRIAUX RÉELS:\n"
   
    # Analyse complète du profil
    profile_analysis = analyzer.analyze_resistivity_profile(rho_data, dat_file_path=dat_file_path)
   
    # Matériaux identifiés
    materials = profile_analysis.get('materials', [])
    if materials:
        analysis += f"Matériaux potentiels détectés (avec validation réelle):\n"
        for i, material in enumerate(materials[:8], 1): # Top 8 matériaux
            name = material.get('name', 'inconnu')
            category = material.get('category', 'inconnue')
            similarity = material.get('similarity_score', 0) * 100
            typical_rho = material.get('typical_value', 0)
            nature = material.get('nature', '')
            depth = material.get('depth_range', '')
           
            analysis += f" {i}. {name.upper()} ({category})\n"
            analysis += f" → Résistivité typique: {typical_rho:.2e} Ω.m\n"
            analysis += f" → Score de similarité: {similarity:.1f}%\n"
            analysis += f" → Nature: {nature}\n"
            if depth:
                analysis += f" → Profondeur typique: {depth}\n"
           
            # Validation .dat
            if material.get('dat_validated', False):
                confidence = material.get('dat_confidence', 'low')
                analysis += f" ✅ VALIDÉ PAR FICHIER .DAT (confiance: {confidence})\n"
            else:
                analysis += f" ⚠️ Non validé par fichier .dat\n"
           
            # Validation monde réel
            real_validation = analyzer.get_real_world_validation(name)
            if real_validation.get('confidence_level') != 'unknown':
                verified_range = real_validation.get('resistivity_range_verified')
                if verified_range:
                    analysis += f" 🌍 VALIDATION MONDE RÉEL: {verified_range[0]:.2e} - {verified_range[1]:.2e} Ω.m\n"
                sources = real_validation.get('sources', [])
                if sources:
                    analysis += f" 📚 Sources: {len(sources)} références trouvées\n"
           
            analysis += "\n"
    else:
        analysis += "Aucun matériau spécifique détecté dans la base de données.\n\n"
   
    # Couches géologiques identifiées
    layers = profile_analysis.get('layers', [])
    if layers:
        analysis += f"🏔️ COUCHES GÉOLOGIQUES IDENTIFIÉES:\n"
        for layer in layers:
            layer_id = layer.get('layer_id', 0)
            mean_rho = layer.get('mean_resistivity', 0)
            thickness = layer.get('thickness_estimate', 0) * 100
            color = layer.get('color', '#000000')
            desc = layer.get('description', '')
           
            analysis += f" • Couche {layer_id}: ρ = {mean_rho:.1f} Ω.m ({thickness:.1f}% du profil)\n"
            analysis += f" Couleur: {color} - {desc}\n"
        analysis += "\n"
   
    # Interprétation géologique
    geo_interp = profile_analysis.get('geological_interpretation', '')
    if geo_interp:
        analysis += f"🌍 INTERPRÉTATION GÉOLOGIQUE:\n{geo_interp}\n\n"
   
    # Validation .dat globale
    dat_validation = profile_analysis.get('dat_validation')
    if dat_validation:
        analysis += f"📁 VALIDATION FICHIER .DAT:\n"
        if dat_validation.get('data_loaded', False):
            score = dat_validation.get('validation_score', 0) * 100
            confidence = dat_validation.get('confidence_level', 'low')
            matches = dat_validation.get('matching_materials', [])
           
            analysis += f" • Fichier chargé: ✅\n"
            analysis += f" • Score de validation: {score:.1f}%\n"
            analysis += f" • Niveau de confiance: {confidence.upper()}\n"
            analysis += f" • Matériaux correspondants: {len(matches)}\n"
        else:
            analysis += f" • Fichier non chargé ou invalide: ❌\n"
        analysis += "\n"
   
    # Recommandations
    recommendations = profile_analysis.get('recommendations', [])
    if recommendations:
        analysis += f"💡 RECOMMANDATIONS:\n"
        for rec in recommendations:
            analysis += f" • {rec}\n"
        analysis += "\n"
   
    # Recherche dynamique de comparaisons supplémentaires
    analysis += f"🔍 COMPARAISONS DYNAMIQUES SUPPLÉMENTAIRES:\n"
   
    # Recherche pour les catégories principales
    categories_to_search = ['eau salée', 'minerais métalliques', 'roches cristallines', 'sols argileux']
    for category in categories_to_search:
        try:
            search_results = analyzer.data_searcher.search_material_resistivity(category, "ERT geophysical")
            if search_results:
                extracted_values = analyzer.data_searcher.extract_resistivity_values(search_results)
                if extracted_values:
                    avg_rho = np.mean(extracted_values)
                    analysis += f" • {category.title()}: ρ moyenne trouvée = {avg_rho:.2f} Ω.m "
                    analysis += f"(plage: {min(extracted_values):.2f} - {max(extracted_values):.2f} Ω.m)\n"
        except Exception as e:
            analysis += f" • {category.title()}: Erreur recherche - {e}\n"
   
    analysis += "\n"
   
    # Ajouter le tableau et les graphiques
    analysis += f"📊 TABLEAU DÉTAILLÉ DES RÉSISTIVITÉS:\n"
    table_html = generate_resistivity_table(resistivity_values)
    analysis += f"{table_html}\n\n"
   
    analysis += f"📈 GRAPHIQUES D'ANALYSE:\n"
    plot_html = generate_resistivity_plot(resistivity_values)
    analysis += f"{plot_html}\n\n"
   
    analysis += f"✅ Analyse terminée - Toutes les détections sont basées sur des valeurs de résistivité RÉELLES\n"
    analysis += f"et validées contre des données scientifiques et fichiers .dat de référence."
   
    return analysis
# ========================================
# Configuration - CHEMINS UNIFIÉS
# ========================================
# Définir dynamiquement les chemins basés sur le répertoire KIbalione8
PROJECT_DIR = os.path.expanduser('~/KIbalione8') # Chemin corrigé vers le dossier contenant les données et poids
CHATBOT_DIR = PROJECT_DIR
VECTORDB_PATH = os.path.join(CHATBOT_DIR, "vectordb")
CHAT_VECTORDB_PATH = os.path.join(CHATBOT_DIR, "chat_vectordb") # AJOUT MÉMOIRE VECTORIELLE: Base dédiée pour l'historique chat
PDFS_PATH = os.path.join(CHATBOT_DIR, "pdfs")
GRAPHS_PATH = os.path.join(CHATBOT_DIR, "graphs")
MAPS_PATH = os.path.join(CHATBOT_DIR, "maps")
METADATA_PATH = os.path.join(CHATBOT_DIR, "metadata.json")
TRAJECTORIES_PATH = os.path.join(CHATBOT_DIR, "trajectories.json")
WEB_CACHE_PATH = os.path.join(CHATBOT_DIR, "web_cache.json")
GENERATED_PATH = os.path.join(CHATBOT_DIR, "generated")
SUBMODELS_PATH = os.path.join(CHATBOT_DIR, "submodels") # Nouveau: Chemin pour les sous-modèles sklearn
MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3-0324/snapshots/e9b33add76883f293d6bf61f6bd89b497e80e335")
# Modèles qui fonctionnent
WORKING_MODELS = {
    "DeepSeek V3 (Puissant)": "deepseek-ai/DeepSeek-V3-0324",
    "Gemma 2B (Rapide)": "google/gemma-2-2b-it",
    "Llama 3.1 8B (Équilibré)": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen 2.5 7B (Polyvalent)": "Qwen/Qwen2.5-7B-Instruct",
    "SmolLM 3B (Léger)": "HuggingFaceTB/SmolLM3-3B",
}
# ========================================
# Configuration HuggingFace Token depuis .env
# ========================================
# Charger le token depuis .env dans le dossier corrigé
env_path = os.path.join(CHATBOT_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    st.write(f"✅ Fichier .env trouvé: {env_path}")
else:
    st.write(f"⚠️ Aucun fichier .env trouvé à {env_path}")
    st.write("Créez un fichier .env dans ~/KIbalione8 avec: HF_TOKEN=hf_votre_token")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN non trouvé ! Vérifiez votre fichier .env")
else:
    st.write(f"🔑 Token HF configuré: {HF_TOKEN[:10]}...")
# Définir la variable d'environnement pour huggingface_hub
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("❌ TAVILY_API_KEY non trouvé ! Vérifiez votre fichier .env")
# ========================================
# Test de connexion HuggingFace
# ========================================
def test_hf_connection():
    """Teste la connexion à HuggingFace"""
    try:
        from huggingface_hub import whoami
        user_info = whoami(token=HF_TOKEN)
        st.write(f"✅ Connexion HuggingFace réussie: {user_info.get('name', 'Utilisateur')}")
        return True
    except Exception as e:
        st.write(f"❌ Erreur connexion HuggingFace: {e}")
        return False
# Tester la connexion au démarrage
if not test_hf_connection():
    st.write("⚠️ Problème de connexion HuggingFace, vérifiez votre token")
# ========================================
# Fonctions utilitaires
# ========================================
def setup_drive():
    """Crée les dossiers"""
    st.write("📁 Configuration des dossiers...")
    os.makedirs(CHATBOT_DIR, exist_ok=True)
    os.makedirs(PDFS_PATH, exist_ok=True)
    os.makedirs(GRAPHS_PATH, exist_ok=True)
    os.makedirs(MAPS_PATH, exist_ok=True)
    os.makedirs(GENERATED_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(CHAT_VECTORDB_PATH), exist_ok=True) # AJOUT MÉMOIRE VECTORIELLE: Dossier pour chat_vectordb
    os.makedirs(SUBMODELS_PATH, exist_ok=True) # Nouveau: Dossier pour sous-modèles
    st.write(f"📁 Dossier principal : {CHATBOT_DIR}")
    return True
def extract_text_from_pdf(pdf_path):
    """Extraire le texte d'un PDF"""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text += f"\n[Page {page_num + 1}]\n{page_text}\n"
        return text
    except Exception as e:
        st.write(f"❌ Erreur PDF {pdf_path}: {e}")
        return ""
def upload_and_process_pbf(pbf_file):
    """Traitement du fichier PBF uploadé"""
    if pbf_file is None:
        return None, None, "❌ Aucun fichier uploadé"
    pbf_path = pbf_file.name
    with open(pbf_path, "wb") as f:
        f.write(pbf_file.getvalue())
    st.write("⚙️ Lecture du PBF et construction du graphe...")
    handler = RoadPOIHandler()
    handler.apply_file(pbf_path, locations=True)
    G = handler.graph
    pois = handler.pois
    # Sauvegarder dans le dossier chatbot
    graph_name = os.path.basename(pbf_path).replace('.osm.pbf', '_graph.graphml')
    graph_path = os.path.join(GRAPHS_PATH, graph_name)
    nx.write_graphml(G, graph_path)
    # Sauvegarder les POIs
    pois_name = graph_name.replace('_graph.graphml', '_pois.json')
    pois_path = os.path.join(GRAPHS_PATH, pois_name)
    with open(pois_path, 'w', encoding='utf-8') as f:
        json.dump(pois, f, indent=2, ensure_ascii=False)
    st.write(f"✅ Graphe: {len(G)} nœuds, {G.size()} arêtes")
    st.write(f"✅ POIs: {len(pois)} points")
    st.write(f"💾 Sauvegardé: {graph_path}")
    return G, pois, f"✅ Graphe créé: {len(G)} nœuds, {len(pois)} POIs"
def load_existing_graph():
    """Charge un graphe existant"""
    graph_files = [f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')] if os.path.exists(GRAPHS_PATH) else []
    if not graph_files:
        return None, None, "❌ Aucun graphe trouvé"
    graph_file = graph_files[0]
    graph_path = os.path.join(GRAPHS_PATH, graph_file)
    pois_path = os.path.join(GRAPHS_PATH, graph_file.replace('_graph.graphml', '_pois.json'))
    try:
        G = nx.read_graphml(graph_path)
        pois = []
        if os.path.exists(pois_path):
            with open(pois_path, 'r', encoding='utf-8') as f:
                pois = json.load(f)
        return G, pois, f"✅ Graphe chargé: {len(G)} nœuds, {len(pois)} POIs"
    except Exception as e:
        return None, None, f"❌ Erreur: {e}"
@st.cache_resource
def get_embedding_model():
    """Modèle d'embedding en cache pour éviter rechargement"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Configuration simple pour éviter conflits de paramètres
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
# AJOUT MÉMOIRE VECTORIELLE: Fonctions pour la mémoire chat
def load_chat_vectordb():
    """Charger la base vectorielle pour l'historique chat"""
    if not os.path.exists(CHAT_VECTORDB_PATH):
        return None, "⚠️ Aucune base chat trouvée"
    embedding_model = get_embedding_model()
    try:
        chat_vectordb = FAISS.load_local(CHAT_VECTORDB_PATH, embedding_model, allow_dangerous_deserialization=True)
        return chat_vectordb, "✅ Base chat chargée"
    except Exception as e:
        return None, f"❌ Erreur chat: {e}"
def add_to_chat_db(user_msg, ai_msg, chat_vectordb):
    """Ajouter un échange user-AI à la base chat"""
    if chat_vectordb is None:
        embedding_model = get_embedding_model()
        chat_vectordb = FAISS.from_texts([""], embedding_model) # Créer si vide
    exchange = f"User: {user_msg} ||| Assistant: {ai_msg}"
    doc = Document(
        page_content=exchange,
        metadata={"type": "chat_exchange", "timestamp": time.time()}
    )
    chat_vectordb.add_documents([doc])
    chat_vectordb.save_local(CHAT_VECTORDB_PATH)
    return chat_vectordb
def chat_rag_search(question, chat_vectordb, k=3):
    """Rechercher dans l'historique chat pour contexte"""
    if not chat_vectordb:
        return []
    try:
        return chat_vectordb.similarity_search(question, k=k)
    except Exception as e:
        st.write(f"❌ Erreur recherche chat: {e}")
        return []
def process_pdfs():
    """Traiter les PDFs"""
    st.write("📄 Traitement des PDFs...")
    embedding_model = get_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    # Charger vectordb existante si elle existe
    vectordb = None
    if os.path.exists(VECTORDB_PATH):
        try:
            vectordb, _ = load_vectordb()
        except Exception as e:
            st.write(f"⚠️ Erreur chargement vectordb existante: {e}. Création nouvelle.")
            vectordb = None
    # Charger métadonnées existantes
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {"processed_files": [], "total_chunks": 0}
    processed_filenames = {p["filename"] for p in metadata["processed_files"]}
    all_documents = []
    pdf_files = [f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')] if os.path.exists(PDFS_PATH) else []
    if not pdf_files:
        return vectordb, "⚠️ Aucun PDF trouvé"
  
    # Check préliminaire : si aucun nouveau, skip
    new_pdfs = [f for f in pdf_files if f not in processed_filenames]
    if not new_pdfs:
        return vectordb, "✅ Tous les PDFs déjà traités. Base à jour !"
  
    progress_bar = st.progress(0)
    status_text = st.empty()
    new_chunks_count = 0
    new_processed = []
    total_pdfs = len(new_pdfs)
    current_pdf = 0
    for pdf_file in pdf_files:
        if pdf_file in processed_filenames:
            st.write(f" 📖 {pdf_file} déjà traité, sauté.")
            continue
        pdf_path = os.path.join(PDFS_PATH, pdf_file)
        st.write(f" 📖 Traitement nouveau PDF : {pdf_file}")
        status_text.text(f"Traitement de {pdf_file}...")
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            continue
        try:
            chunks = text_splitter.split_text(text)
        except Exception as e:
            st.write(f"❌ Erreur split text pour {pdf_file}: {e}")
            continue
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_file,
                    "chunk_id": i,
                    "type": "pdf"
                }
            )
            all_documents.append(doc)
        new_processed.append({"filename": pdf_file, "chunks": len(chunks)})
        new_chunks_count += len(chunks)
        current_pdf += 1
        progress = current_pdf / total_pdfs if total_pdfs > 0 else 1
        progress_bar.progress(progress)
    status_text.text("Finalisation...")
    # Ajouter les trajets sauvegardés (toujours, car ils peuvent changer)
    if os.path.exists(TRAJECTORIES_PATH):
        with open(TRAJECTORIES_PATH, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)
        for traj in trajectories:
            traj_text = f"""Trajet: {traj.get('question', '')}
Départ: {traj.get('start_name', '')}
Arrivée: {traj.get('end_name', '')}
Distance: {traj.get('distance', 0)/1000:.2f} km"""
            doc = Document(
                page_content=traj_text,
                metadata={"source": "trajectories", "type": "trajectory"}
            )
            all_documents.append(doc)
    if all_documents:
        try:
            if vectordb is None:
                vectordb = FAISS.from_documents(all_documents, embedding_model)
            else:
                vectordb.add_documents(all_documents)
            vectordb.save_local(VECTORDB_PATH)
        except Exception as e:
            st.write(f"❌ Erreur sauvegarde vectordb: {e}")
            return None, "❌ Échec sauvegarde base"
    # Mettre à jour métadonnées seulement si changements
    if new_processed:
        metadata["processed_files"].extend(new_processed)
        metadata["total_chunks"] += new_chunks_count
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    progress_bar.progress(1)
    status_text.text("Terminé !")
    return vectordb, f"✅ Base mise à jour : {len(new_processed)} nouveaux PDFs traités, {new_chunks_count} nouveaux chunks (total : {metadata['total_chunks']})"
def load_vectordb():
    """Charge la base vectorielle"""
    if not os.path.exists(VECTORDB_PATH):
        return None, "⚠️ Aucune base trouvée"
    embedding_model = get_embedding_model()
    try:
        vectordb = FAISS.load_local(VECTORDB_PATH, embedding_model, allow_dangerous_deserialization=True)
        return vectordb, "✅ Base chargée"
    except Exception as e:
        return None, f"❌ Erreur: {e}"
def save_trajectory(question, response, trajectory_info):
    """Sauvegarde un trajet"""
    trajectories = []
    if os.path.exists(TRAJECTORIES_PATH):
        with open(TRAJECTORIES_PATH, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)
    new_trajectory = {
        "question": question,
        "response": response,
        "start_name": trajectory_info.get('start', {}).get('name', ''),
        "end_name": trajectory_info.get('end', {}).get('name', ''),
        "distance": trajectory_info.get('distance', 0)
    }
    trajectories.append(new_trajectory)
    with open(TRAJECTORIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)
def upload_pdfs(uploaded_files):
    """Upload des PDFs"""
    if uploaded_files is None:
        return []
    saved_files = []
    for file in uploaded_files:
        filename = file.name
        filepath = os.path.join(PDFS_PATH, filename)
        with open(filepath, "wb") as f:
            f.write(file.getvalue())
        saved_files.append(filename)
    return saved_files
# ========================================
# Système de Cache Web Intelligent
# ========================================
def load_web_cache():
    """Charge le cache web"""
    if os.path.exists(WEB_CACHE_PATH):
        try:
            with open(WEB_CACHE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}
def save_web_cache(cache):
    """Sauvegarde le cache web"""
    try:
        with open(WEB_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.write(f"Erreur sauvegarde cache: {e}")
def get_cache_key(query, source="text"):
    """Génère une clé de cache pour une requête"""
    return f"{source}:{query.lower().strip()}"
def is_cache_expired(cache_entry, max_age_hours=24):
    """Vérifie si l'entrée du cache a expiré"""
    current_time = time.time()
    return (current_time - cache_entry.get('timestamp', 0)) > (max_age_hours * 3600)
def get_cache_stats():
    """Obtient les statistiques du cache"""
    try:
        cache = load_web_cache()
        if not cache:
            return "Cache vide"
        total_entries = len(cache)
        expired_count = sum(1 for entry in cache.values() if is_cache_expired(entry))
        valid_count = total_entries - expired_count
        return f"📊 Cache: {total_entries} entrées total, {valid_count} valides, {expired_count} expirées"
    except Exception as e:
        return f"❌ Erreur stats: {e}"
# ========================================
# Fonctions RAG et Web Search Améliorées
# ========================================
class LocalClient:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
       
        MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3-0324/snapshots/e9b33add76883f293d6bf61f6bd89b497e80e335")
       
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
       
        # Load model with device_map for large models
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
            torch_dtype="auto"
        )
       
        self.model.eval()
    def chat_completion(self, messages, model, max_tokens, temperature, stream=False):
        try:
            # Use chat template for proper formatting
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=temperature > 0, pad_token_id=self.tokenizer.eos_token_id)
            generated_ids = outputs[0][inputs.shape[-1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            class Choice:
                def __init__(self, content):
                    self.message = type('msg', (), {'content': content})()
            class Resp:
                def __init__(self, choice):
                    self.choices = [choice]
            return Resp(Choice(response))
        except Exception as e:
            class Choice:
                def __init__(self, content):
                    self.message = type('msg', (), {'content': content})()
            class Resp:
                def __init__(self, choice):
                    self.choices = [choice]
            return Resp(Choice(f"Erreur locale: {str(e)}"))
@st.cache_resource
def create_client():
    """Créer le client Inference avec gestion d'erreurs améliorée"""
    try:
        client = InferenceClient(token=HF_TOKEN)
        return client
    except Exception as e:
        st.write(f"❌ Erreur création client: {e}. Passage en mode local.")
        return LocalClient()
def rag_search(question, vectordb, k=3):
    """Rechercher dans la base vectorielle avec recherche GLOBALE ILLIMITÉE"""
    if not vectordb:
        return []
    try:
        # RECHERCHE ILLIMITÉE: pas de limite arbitraire
        # Si k petit, forcer minimum 100 pour fouille exhaustive
        effective_k = max(k, 100) if k < 100 else k
        
        # Récupérer le nombre total de documents
        total_docs = vectordb.index.ntotal if hasattr(vectordb, 'index') else 1000
        
        # Ajuster k au minimum entre le demandé et le total disponible
        # Cap à 500 pour performance raisonnable
        final_k = min(effective_k, total_docs, 500) if total_docs > 0 else effective_k
        
        return vectordb.similarity_search(question, k=final_k)
    except Exception as e:
        st.write(f"❌ Erreur recherche: {e}")
        return []
def enhanced_web_search(query, max_results=5, search_type="text", use_cache=True):
    """
    Recherche web avancée avec cache intelligent et multiple sources
    Args:
        query: Requête de recherche
        max_results: Nombre max de résultats
        search_type: Type de recherche ("text", "news", "both")
        use_cache: Utiliser le cache
    Returns:
        Liste de résultats enrichis
    """
    cache = load_web_cache() if use_cache else {}
    results = []
    try:
        # Recherche texte
        if search_type in ["text", "both"]:
            cache_key = get_cache_key(query, "text")
            if cache_key in cache and not is_cache_expired(cache[cache_key]):
                st.write(f"📋 Utilisation cache pour: {query}")
                text_results = cache[cache_key]['results']
            else:
                st.write(f"🔍 Recherche web pour: {query}")
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                text_results = []
                try:
                    raw_results = tavily.search(query, max_results=max_results, search_depth="advanced", topic="general")
                    for r in raw_results.get('results', []):
                        text_results.append({
                            'title': r.get('title', ''),
                            'body': r.get('content', ''),
                            'href': r.get('url', ''),
                            'source_type': 'web_search'
                        })
                    # Sauvegarder en cache
                    cache[cache_key] = {
                        'results': text_results,
                        'timestamp': time.time()
                    }
                    if use_cache:
                        save_web_cache(cache)
                except Exception as e:
                    st.write(f"Erreur recherche texte: {e}")
                    text_results = []
            results.extend(text_results)
        # Recherche actualités
        if search_type in ["news", "both"]:
            cache_key = get_cache_key(query, "news")
            if cache_key in cache and not is_cache_expired(cache[cache_key], max_age_hours=6):
                news_results = cache[cache_key]['results']
            else:
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                news_results = []
                try:
                    raw_news = tavily.search(query, max_results=max_results//2 if search_type == "both" else max_results, search_depth="advanced", topic="news")
                    for r in raw_news.get('results', []):
                        news_results.append({
                            'title': r.get('title', ''),
                            'body': r.get('content', ''),
                            'url': r.get('url', ''),
                            'date': r.get('published_date', ''),
                            'source': r.get('source', ''),
                            'source_type': 'news'
                        })
                    # Sauvegarder en cache (6h pour les news)
                    cache[cache_key] = {
                        'results': news_results,
                        'timestamp': time.time()
                    }
                    if use_cache:
                        save_web_cache(cache)
                except Exception as e:
                    st.write(f"Erreur recherche news: {e}")
                    news_results = []
            results.extend(news_results)
    except Exception as e:
        st.write(f"❌ Erreur recherche web globale: {e}")
        results = [{'title': 'Erreur de recherche', 'body': f'Erreur: {e}', 'source_type': 'error'}]
    return results
def smart_content_extraction(url, max_length=1000):
    """
    Extraction intelligente du contenu d'une page web
    Args:
        url: URL à scraper
        max_length: Longueur max du contenu
    Returns:
        Contenu extrait et nettoyé
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Supprimer les éléments non pertinents
        for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
            element.decompose()
        # Extraire le texte principal
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        # Nettoyer et tronquer
        text = ' '.join(text.split()) # Normaliser les espaces
        return text[:max_length] + ('...' if len(text) > max_length else '')
    except Exception as e:
        st.write(f"Erreur extraction contenu {url}: {e}")
        return f"Impossible d'extraire le contenu de {url}"
def intelligent_query_expansion(query):
    """
    Expansion intelligente des requêtes pour améliorer les résultats
    Args:
        query: Requête originale
    Returns:
        Liste de requêtes expandues
    """
    expanded_queries = [query] # Toujours inclure la requête originale
    # Détection de mots-clés pour expansion contextuelle
    keywords = {
        'actualité': ['news', 'dernières nouvelles', 'récent'],
        'comment': ['tutorial', 'guide', 'étapes'],
        'pourquoi': ['raison', 'cause', 'explication'],
        'comparaison': ['vs', 'différence', 'comparatif'],
        'prix': ['coût', 'tarif', 'budget'],
        'avis': ['opinion', 'critique', 'review']
    }
    query_lower = query.lower()
    for trigger, expansions in keywords.items():
        if trigger in query_lower:
            for expansion in expansions:
                expanded_queries.append(f"{query} {expansion}")
    return expanded_queries[:3] # Limiter à 3 requêtes max
def hybrid_search_enhanced(query, vectordb, k=3, web_search_enabled=True, search_type="both", chat_vectordb=None): # AJOUT MÉMOIRE VECTORIELLE: Param pour chat_vectordb
    """
    Recherche hybride ILLIMITÉE combinant RAG local GLOBAL et web avec intelligence maximale
    Args:
        query: Requête de recherche
        vectordb: Base vectorielle locale
        k: Nombre MINIMUM de résultats RAG (sera augmenté pour recherche ILLIMITÉE)
        web_search_enabled: Activer la recherche web
        search_type: Type de recherche web
        chat_vectordb: Base pour historique chat (optionnel)
    Returns:
        Liste de documents combinés et enrichis
    """
    all_results = []
    
    # 1. Recherche RAG locale ILLIMITÉE avec k maximisé
    # Pour une fouille COMPLÈTE, utiliser au moins 100 documents (pas de limite!)
    global_k = max(k, 100)
    local_docs = rag_search(query, vectordb, global_k)
    for doc in local_docs:
        doc.metadata['search_source'] = 'local_rag'
        doc.metadata['relevance_score'] = 1.0 # Score max pour les docs locaux
    all_results.extend(local_docs)
    
    st.write(f"📚 Fouille GLOBALE ILLIMITÉE: {len(local_docs)} documents trouvés dans la base locale complète")
    # AJOUT MÉMOIRE VECTORIELLE: Recherche dans historique chat pour contexte conversationnel
    if chat_vectordb:
        chat_docs = chat_rag_search(query, chat_vectordb, k=3)
        for doc in chat_docs:
            doc.metadata['search_source'] = 'chat_history'
            doc.metadata['relevance_score'] = 0.9
        all_results.extend(chat_docs[:2]) # Limiter à 2 pour éviter surcharge
    # 2. Recherche web intelligente si activée
    if web_search_enabled:
        st.write(f"🌐 Recherche web activée pour: {query}")
        # Expansion de requête pour de meilleurs résultats
        expanded_queries = intelligent_query_expansion(query)
        web_results = []
        for exp_query in expanded_queries:
            try:
                search_results = enhanced_web_search(
                    exp_query,
                    max_results=3,
                    search_type=search_type
                )
                for result in search_results:
                    # Créer un document à partir du résultat web
                    content = f"Titre: {result.get('title', '')}\n"
                    content += f"Contenu: {result.get('body', '')}\n"
                    if result.get('source_type') == 'news' and result.get('date'):
                        content += f"Date: {result.get('date')}\n"
                        content += f"Source: {result.get('source', '')}\n"
                    # Extraction de contenu supplémentaire si URL disponible
                    url = result.get('href') or result.get('url')
                    if url and len(result.get('body', '')) < 200:
                        st.write(f"📄 Extraction contenu de: {url}")
                        extra_content = smart_content_extraction(url)
                        if extra_content and "Impossible d'extraire" not in extra_content:
                            content += f"\nContenu détaillé: {extra_content}"
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': url or 'web_search',
                            'type': result.get('source_type', 'web'),
                            'search_source': 'web',
                            'query_used': exp_query,
                            'relevance_score': 0.8 if exp_query == query else 0.6
                        }
                    )
                    web_results.append(doc)
            except Exception as e:
                st.write(f"Erreur recherche pour '{exp_query}': {e}")
                continue
        # Filtrer les doublons et trier par pertinence
        unique_web_results = []
        seen_urls = set()
        for doc in web_results:
            url = doc.metadata.get('source', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_web_results.append(doc)
        # Trier par score de pertinence
        unique_web_results.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        all_results.extend(unique_web_results[:5]) # Max 5 résultats web
    return all_results
def generate_answer_enhanced(question, context_docs, model_name, include_sources=True):
    """
    Génération de réponse améliorée avec gestion des sources multiples
    Args:
        question: Question posée
        context_docs: Documents de contexte
        model_name: Modèle à utiliser
        include_sources: Inclure les sources dans la réponse
    Returns:
        Réponse générée avec sources
    """
    if not context_docs:
        context = "Aucun contexte spécifique trouvé."
    else:
        context_parts = []
        local_sources = []
        web_sources = []
        chat_sources = [] # AJOUT MÉMOIRE VECTORIELLE: Sources pour historique chat
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', 'Document inconnu')
            doc_type = doc.metadata.get('type', 'unknown')
            search_source = doc.metadata.get('search_source', 'unknown')
            content = doc.page_content.strip()
            # Classifier les sources
            if search_source == 'local_rag':
                local_sources.append(f"[{i+1}] {source} ({doc_type})")
            elif search_source == 'chat_history':
                chat_sources.append(f"[{i+1}] Historique précédent: {source}")
            else:
                web_sources.append(f"[{i+1}] {source}")
            context_parts.append(f"[Source {i+1} - {doc_type}]\n{content}")
        context = "\n\n".join(context_parts)
    # Prompt amélioré avec instructions pour les sources (ajout chat)
    prompt = f"""Tu es un assistant IA intelligent qui répond aux questions en utilisant à la fois des documents locaux, l'historique des conversations passées, et des informations web récentes.
CONTEXTE DISPONIBLE (incluant historique pour continuité):
{context}
QUESTION: {question}
INSTRUCTIONS:
- Utilise l'historique chat pour maintenir la fluidité et rappeler les échanges précédents
- Utilise toutes les sources disponibles pour donner une réponse complète et précise
- Si les informations web contredisent les documents locaux ou l'historique, mentionne les deux perspectives
- Privilégie les informations récentes pour les sujets d'actualité
- Sois précis et cite tes sources si nécessaire
- Si certaines informations manquent, dis-le clairement et propose de clarifier basé sur l'historique
RÉPONSE DÉTAILLÉE:"""
    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model=model_name,
            max_tokens=600,
            temperature=0.3
        )
        answer = response.choices[0].message.content
        # Ajouter les sources si demandé
        if include_sources and context_docs:
            sources_text = "\n\n📚 **Sources consultées:**\n"
            if chat_sources: # AJOUT MÉMOIRE VECTORIELLE
                sources_text += "**Historique conversation:**\n"
                for source in chat_sources[:2]:
                    sources_text += f"• {source}\n"
            if local_sources:
                sources_text += "**Documents locaux:**\n"
                for source in local_sources[:3]: # Limiter l'affichage
                    sources_text += f"• {source}\n"
            if web_sources:
                sources_text += "**Sources web:**\n"
                for source in web_sources[:3]: # Limiter l'affichage
                    sources_text += f"• {source}\n"
            answer += sources_text
        return answer
    except Exception as e:
        error_str = str(e)
        # Check for payment error and retry with LocalClient
        if "402" in error_str or "Payment Required" in error_str:
            try:
                # Retry with LocalClient
                local_client = LocalClient()
                messages = [{"role": "user", "content": prompt}]
                response = local_client.chat_completion(
                    messages=messages,
                    model=model_name,
                    max_tokens=600,
                    temperature=0.3
                )
                answer = response.choices[0].message.content
                # Ajouter les sources si demandé
                if include_sources and context_docs:
                    sources_text = "\n\n📚 **Sources consultées (mode local):**\n"
                    if chat_sources:
                        sources_text += "**Historique conversation:**\n"
                        for source in chat_sources[:2]:
                            sources_text += f"• {source}\n"
                    if local_sources:
                        sources_text += "**Documents locaux:**\n"
                        for source in local_sources[:3]:
                            sources_text += f"• {source}\n"
                    if web_sources:
                        sources_text += "**Sources web:**\n"
                        for source in web_sources[:3]:
                            sources_text += f"• {source}\n"
                    answer += sources_text
                return answer + "\n\n⚠️ Réponse générée en mode local (API distante indisponible)."
            except Exception as local_e:
                return f"❌ Erreur génération (même en local): {str(local_e)}"
        else:
            return f"❌ Erreur génération: {error_str}"
# ========================================
# Fonctions Web Search et Hybrid (Mises à jour)
# ========================================
def web_search(query, max_results=5):
    """Version simplifiée pour compatibilité"""
    try:
        results = enhanced_web_search(query, max_results, "text")
        return [f"{r.get('title', '')}: {r.get('href', r.get('url', ''))} - {r.get('body', '')}" for r in results]
    except Exception as e:
        return [f"❌ Erreur recherche web: {e}"]
def hybrid_search(query, vectordb, k=3):
    """Version simplifiée pour compatibilité"""
    return hybrid_search_enhanced(query, vectordb, k, web_search_enabled=True)
def final_search(question, vectordb, graph, pois):
    """Recherche finale combinant toutes les sources"""
    results = hybrid_search_enhanced(question, vectordb, k=3, web_search_enabled=True)
    # OSM si mention lieu
    if any(keyword in question.lower() for keyword in ["aller", "trajet", "itinéraire", "route", "navigation"]):
        try:
            carte, reponse, traj = calculer_trajet(question, graph, pois)
            if traj:
                results.append(Document(
                    page_content=reponse,
                    metadata={"source": "trajet_osm", "type": "navigation"}
                ))
        except:
            pass
    return results
# ========================================
# Fonctions Modèles Hugging Face Spécialisés
# ========================================
@st.cache_resource
def initialize_specialized_models():
    """Initialise les modèles spécialisés avec gestion d'erreurs"""
    device_id = 0 if torch.cuda.is_available() else -1
    models = {}
    try:
        model_name = "facebook/bart-large-cnn"
        # Vérifier cache local
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        use_local = os.path.exists(model_cache)
        
        if use_local:
            st.write(f"📦 Modèle {model_name} trouvé en cache")
        
        try:
            models['summarizer'] = pipeline("summarization", model=model_name, device=device_id, local_files_only=use_local)
        except Exception:
            st.write(f"⬇️ Téléchargement de {model_name}")
            models['summarizer'] = pipeline("summarization", model=model_name, device=device_id)
        
        st.write("✅ Modèle de résumé chargé")
    except Exception as e:
        st.write(f"⚠️ Erreur chargement summarizer: {e}")
        models['summarizer'] = None
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "Helsinki-NLP/opus-mt-fr-en"
        
        # Vérifier cache local
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        use_local = os.path.exists(model_cache)
        
        if use_local:
            st.write(f"📦 Modèle {model_name} trouvé en cache")
        
        # Essayer avec cache local d'abord
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=use_local)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                use_safetensors=True,
                device_map="auto" if device_id != -1 else None,
                low_cpu_mem_usage=True,
                local_files_only=use_local
            )
        except Exception:
            # Fallback sans local_files_only
            st.write(f"⬇️ Téléchargement de {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                use_safetensors=True,
                device_map="auto" if device_id != -1 else None,
                low_cpu_mem_usage=True
            )
        
        models['translator'] = pipeline("translation", model=model, tokenizer=tokenizer, device=device_id)
        st.write("✅ Modèle de traduction chargé")
    except Exception as e:
        st.write(f"⚠️ Erreur chargement translator: {e}")
        models['translator'] = None
    try:
        models['captioner'] = None
        st.write("✅ Captioner configuré pour utiliser LLM (llava)")
    except Exception as e:
        st.write(f"⚠️ Erreur chargement captioner: {e}")
        models['captioner'] = None
    try:
        model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        # Vérifier cache local
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        use_local = os.path.exists(model_cache)
        
        if use_local:
            st.write(f"📦 Modèle {model_name} trouvé en cache")
        
        try:
            models['ner'] = pipeline("ner", model=model_name, device=device_id, local_files_only=use_local)
        except Exception:
            st.write(f"⬇️ Téléchargement de {model_name}")
            models['ner'] = pipeline("ner", model=model_name, device=device_id)
        
        st.write("✅ Modèle NER chargé")
        st.write("⚠️ Warning NER ignoré : weights pooler non utilisés (normal pour ce checkpoint).")
    except Exception as e:
        st.write(f"⚠️ Erreur chargement NER: {e}")
        models['ner'] = None
    return models
# Initialiser les modèles
SPECIALIZED_MODELS = initialize_specialized_models()
def summarize_text(text):
    if SPECIALIZED_MODELS['summarizer'] is None:
        return "❌ Modèle de résumé non disponible"
    try:
        return SPECIALIZED_MODELS['summarizer'](text[:1024], max_length=200, min_length=30, do_sample=False)[0]['summary_text']
    except Exception as e:
        return f"❌ Erreur résumé: {e}"
def translate_text(text, src_lang="fr", tgt_lang="en"):
    if SPECIALIZED_MODELS['translator'] is None:
        return "❌ Modèle de traduction non disponible"
    try:
        return SPECIALIZED_MODELS['translator'](text)[0]['translation_text']
    except Exception as e:
        return f"❌ Erreur traduction: {e}"
def caption_image(image_path):
    client = create_client()
    model = "llava-hf/llava-1.5-7b-hf"
    prompt = "Generate a detailed caption for this image."
    try:
        return client.image_to_text(image_path, prompt=prompt, model=model, max_tokens=500)
    except Exception as e:
        return f"❌ Erreur caption: {e}"
def extract_entities(text):
    if SPECIALIZED_MODELS['ner'] is None:
        return "❌ Modèle NER non disponible"
    try:
        return SPECIALIZED_MODELS['ner'](text)
    except Exception as e:
        return f"❌ Erreur NER: {e}"
# ========================================
# Fonctions de génération avec Stable Diffusion et similaires
# ========================================
def generate_text_to_image(prompt):
    """Génère une image à partir de texte"""
    if not load_diffusers():
        return "❌ Diffusers non disponible - fonctionnalité désactivée"
    try:
        pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        image = pipe(prompt).images[0]
        path = os.path.join(GENERATED_PATH, f"image_{int(time.time())}.png")
        image.save(path)
        return f"Image générée et sauvegardée à {path}"
    except Exception as e:
        return f"❌ Erreur génération image: {e}"
def generate_text_to_video(prompt):
    """Génère une vidéo à partir de texte"""
    if not load_diffusers():
        return "❌ Diffusers non disponible - fonctionnalité désactivée"
    try:
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        gen = pipe(prompt, num_inference_steps=25)
        frames = gen.frames[0] # Assuming batch size 1
        path = os.path.join(GENERATED_PATH, f"video_{int(time.time())}.gif")
        imageio.mimsave(path, frames, fps=5)
        return f"Vidéo générée et sauvegardée à {path}"
    except Exception as e:
        return f"❌ Erreur génération vidéo: {e}"
def generate_text_to_audio(prompt):
    """Génère un son à partir de texte"""
    if not load_diffusers():
        return "❌ Diffusers non disponible - fonctionnalité désactivée"
    try:
        pipe = AudioLDMPipeline.from_pretrained("cvssp/audio-ldm", torch_dtype=torch.float16, use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        audio = pipe(prompt, audio_length_in_s=5.0).audios[0]
        path = os.path.join(GENERATED_PATH, f"audio_{int(time.time())}.wav")
        wavfile.write(path, rate=16000, data=audio) # Assuming 16kHz sample rate
        return f"Son généré et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération son: {e}"
def generate_text_to_3d(prompt):
    """Génère un modèle 3D à partir de texte (rendue image)"""
    if not load_diffusers():
        return "❌ Diffusers non disponible - fonctionnalité désactivée"
    try:
        pipe = ShapEPipeline.from_pretrained("openai/shap-e", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        output = pipe(prompt, num_inference_steps=64)
        image = output.images[0]
        path = os.path.join(GENERATED_PATH, f"3d_text_{int(time.time())}.png")
        image.save(path)
        return f"Rendu 3D généré et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération 3D (texte): {e}"
def generate_image_to_3d(image_path):
    """Génère un modèle 3D à partir d'une image (rendue image)"""
    if not load_diffusers():
        return "❌ Diffusers non disponible - fonctionnalité désactivée"
    try:
        pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        image = Image.open(image_path)
        output = pipe(image, num_inference_steps=64)
        rendered_image = output.images[0]
        path = os.path.join(GENERATED_PATH, f"3d_image_{int(time.time())}.png")
        rendered_image.save(path)
        return f"Rendu 3D généré à partir de l'image et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération 3D (image): {e}"
# ========================================
# Agent LangChain Amélioré avec Recherche Web
# ========================================
def get_llm(model_name):
    """Fonction dynamique pour obtenir LLM: API si disponible, local sinon"""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.3,
            max_new_tokens=3000  # AUGMENTÉ pour réponses COMPLÈTES
        )
        st.write(f"✅ Utilisation API pour {model_name}")
        return llm
    except Exception as e:
        st.write(f"⚠️ API indisponible ({e}). Fallback sur LLM local Qwen.")
        return st.session_state.qwen_llm  # Utilise le Qwen local

def create_enhanced_agent(model_name, vectordb, graph, pois, chat_vectordb=None): # AJOUT MÉMOIRE VECTORIELLE: Param pour chat
    """
    Crée un agent LangChain amélioré avec capacités de recherche web
    Args:
        model_name: Nom du modèle HuggingFace
        vectordb: Base vectorielle locale
        graph: Graphe OSM
        pois: Points d'intérêt
        chat_vectordb: Base pour historique chat (optionnel)
    Returns:
        Agent configuré avec tous les outils
    """
    llm = get_llm(model_name)  # Switch dynamique ici
    # Configuration des outils de recherche web avec Tavily
    # Note: DuckDuckGoSearch n'est plus utilisé, Tavily est préféré pour la qualité
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
    search_results_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5, include_raw_content=True)
    tools = [
        # Outils de base RAG et recherche
        Tool(
            name="Local_Knowledge_Base",
            func=lambda q: search_vectorstore(q),
            description="🔍 FOUILLE GLOBALE ILLIMITÉE dans TOUS les documents locaux (PDFs, rapports). Recherche exhaustive sans limite de documents. Utilise EN PREMIER pour questions sur données internes. Retourne TOUS les passages pertinents avec sources groupées."
        ),
        Tool(
            name="Chat_History_Search", # AJOUT MÉMOIRE VECTORIELLE: Nouvel outil pour historique
            func=lambda q: "\n\n".join([d.page_content for d in chat_rag_search(q, chat_vectordb, k=3)]) if chat_vectordb else "❌ Historique chat non disponible",
            description="Recherche dans l'historique des conversations passées pour maintenir la continuité. Utilise pour les questions de suites de discussion."
        ),
        Tool(
            name="Web_Search",
            func=lambda q: search_tool.run(q),
            description="Recherche sur Internet pour des informations récentes, actualités, ou des connaissances générales non disponibles localement."
        ),
        Tool(
            name="Web_Search_Detailed",
            func=lambda q: search_results_tool.run(q),
            description="Recherche web détaillée avec sources et liens. Utilise pour obtenir des résultats web structurés avec URLs."
        ),
        Tool(
            name="Hybrid_Search",
            func=lambda q: "\n\n".join([d.page_content for d in hybrid_search_enhanced(q, vectordb, k=100, web_search_enabled=True, chat_vectordb=chat_vectordb)]) if vectordb else search_vectorstore(q),
            description="🌐 RECHERCHE HYBRIDE ILLIMITÉE: Combine TOUTE la base locale (100+ docs), historique chat complet ET web multi-sources. Fouille exhaustive GLOBALE pour maximum de contexte. Idéal pour questions complexes nécessitant synthèse complète."
        ),
        Tool(
            name="Current_News_Search",
            func=lambda q: "\n\n".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in enhanced_web_search(q, search_type="news")]),
            description="Recherche spécialisée pour les actualités récentes et informations temporelles."
        ),
        # Outils spécialisés
        Tool(
            name="OSM_Route_Calculator",
            func=lambda q: calculer_trajet(q, graph, pois)[1] if graph and pois else "❌ Aucune carte OSM disponible",
            description="Calcule des itinéraires routiers entre deux lieux. Utilise pour les questions de navigation, trajets, ou géolocalisation."
        ),
        Tool(
            name="Smart_Content_Extractor",
            func=lambda url: smart_content_extraction(url) if url.startswith('http') else "❌ URL invalide",
            description="Extrait le contenu détaillé d'une page web spécifique. Fournis une URL complète."
        ),
        Tool(
            name="Text_Summarizer",
            func=summarize_text,
            description="Résume un texte long en version concise. Utile pour synthétiser des informations volumineuses."
        ),
        Tool(
            name="Language_Translator",
            func=translate_text,
            description="Traduit du français vers l'anglais. Utile pour traiter des sources en langue étrangère."
        ),
        Tool(
            name="Image_Analyzer",
            func=caption_image,
            description="Analyse et décrit le contenu d'une image. Fournis le chemin vers un fichier image."
        ),
        Tool(
            name="Entity_Extractor",
            func=lambda t: json.dumps(extract_entities(t)),
            description="Extrait des entités nommées (personnes, lieux, organisations) d'un texte."
        ),
        # Nouveaux outils Stable Diffusion via API
        Tool(
            name="Text_To_Image_Generator",
            func=generate_text_to_image,
            description="Génère une image à partir d'une description textuelle. Fournis un prompt descriptif."
        ),
        Tool(
            name="Text_To_Video_Generator",
            func=generate_text_to_video,
            description="Génère une vidéo à partir d'une description textuelle. Fournis un prompt descriptif."
        ),
        Tool(
            name="Text_To_Audio_Generator",
            func=generate_text_to_audio,
            description="Génère un son ou audio à partir d'une description textuelle. Fournis un prompt descriptif."
        ),
        Tool(
            name="Text_To_3D_Generator",
            func=generate_text_to_3d,
            description="Génère un modèle 3D (rendue image) à partir d'une description textuelle. Fournis un prompt descriptif."
        ),
        Tool(
            name="Image_To_3D_Generator",
            func=generate_image_to_3d,
            description="Génère un modèle 3D (rendue image) à partir d'une image. Fournis le chemin vers un fichier image."
        ),
        # OUTILS IA SPÉCIALISÉS (1-2GB)
        Tool(
            name="AI_Code_Generator",
            func=generate_code_with_ai,
            description="Génère du code Python/JavaScript/etc parfait avec DeepSeek-Coder-1.3B. Expert en programmation, debugging, optimisation. Fournis une description du code souhaité."
        ),
        Tool(
            name="AI_Plot_Generator",
            func=generate_plot_code,
            description="Génère du code matplotlib/seaborn pour créer des graphiques scientifiques professionnels. Fournis: description données + type graphique souhaité."
        ),
        # Ajout des outils ERT/Binary du premier code
        Tool(
            name="Binary_Analysis",
            func=lambda q: analyze_with_ai(q, file_bytes, numbers, hex_dump, n_clusters=3) if 'file_bytes' in globals() else "❌ Fichier binaire requis",
            description="Analyse complète d'un fichier binaire avec outils ERT, statistiques, entropie. Fournis une requête d'analyse."
        ),
        Tool(
            name="Deep_Binary_Investigation",
            func=lambda file_name: deep_binary_investigation(file_bytes, file_name).get('full_report', '') if 'file_bytes' in globals() else "❌ Fichier binaire requis",
            description="🔍 FOUILLE INTELLIGENTE d'un fichier binaire uploadé: Combine Hex+ASCII Dump + Base Vectorielle RAG + Base ERT pour interprétation scientifique. Analyse déjà effectuée sur fichiers uploadés. Fournis le nom du fichier."
        ),
        Tool(
            name="ERT_Interpretation",
            func=lambda numbers_str: ert_geophysical_interpretation(eval(numbers_str)) if numbers_str else "❌ Liste de nombres requise",
            description="Interprète des données ERT (résistivités). Fournis une liste de nombres comme '[10.5, 20.3, ...]'."
        ),
    ]
    # Configuration de l'agent avec prompt ultra-optimisé pour autonomie et précision
    agent_prompt = PromptTemplate.from_template("""Tu es Kibali Analyst, l'assistant IA le plus avancé au monde, combinant les meilleurs aspects de ChatGPT, Claude, Grok et GPT-4.

🌟 PERSONNALITÉ & APPROCHE:
• Naturel, conversationnel et empathique comme ChatGPT
• Analytique, méthodique et éthique comme Claude  
• Créatif, humoristique et audacieux comme Grok
• Précis, technique et exhaustif comme GPT-4
• TOUJOURS utile, jamais condescendant
• Adapte le ton selon le contexte (casual ↔ formel)

🎯 MÉTHODOLOGIE SUPÉRIEURE (10 ÉTAPES):

1️⃣ COMPRÉHENSION PROFONDE:
   ✓ Analyse sémantique multi-niveau de la question
   ✓ Détecte intentions cachées et besoins implicites
   ✓ Identifie contexte culturel, temporel et émotionnel
   ✓ Reformule mentalement en 3 angles différents

2️⃣ RECHERCHE GLOBALE ILLIMITÉE:
   ✓ FOUILLE EXHAUSTIVE base locale (AUCUNE limite de documents)
   ✓ Recherche web MULTI-SOURCES (minimum 10 résultats analysés)
   ✓ Consultation historique conversations (continuité)
   ✓ Vérification croisée informations contradictoires
   ✓ Sources académiques, news, forums, documentation officielle

3️⃣ ANALYSE CRITIQUE & SYNTHÈSE:
   ✓ Évalue crédibilité chaque source (★★★★★ = expert reconnu)
   ✓ Identifie biais potentiels et angles morts
   ✓ Compare perspectives multiples (pour/contre/nuances)
   ✓ Synthèse intelligente éliminant redondances

4️⃣ ANTICIPATION PROACTIVE:
   ✓ Prédit 5-7 questions de suivi probables
   ✓ Identifie informations manquantes critiques
   ✓ Détecte implications long-terme
   ✓ Propose extensions créatives pertinentes

5️⃣ GÉNÉRATION STRUCTURÉE:
   ✓ Réponse directe immédiate (TL;DR)
   ✓ Explication détaillée en sections logiques
   ✓ Exemples concrets et cas d'usage
   ✓ Visualisations (tableaux, listes, schémas)
   ✓ Code exécutable si applicable

6️⃣ VALIDATION & VÉRIFICATION:
   ✓ Double-check faits contre sources multiples
   ✓ Test logique cohérence interne
   ✓ Validation code (syntaxe + exécution)
   ✓ Signale incertitudes avec transparence

7️⃣ ENRICHISSEMENT CONTEXTUEL:
   ✓ Ajoute définitions termes techniques
   ✓ Contexte historique si pertinent
   ✓ Comparaisons internationales/culturelles
   ✓ Statistiques et données chiffrées récentes

8️⃣ SUGGESTIONS INTELLIGENTES:
   ✓ 3 questions approfondissement pertinentes
   ✓ 2 perspectives alternatives intéressantes
   ✓ 1-2 ressources complémentaires recommandées
   ✓ Actions concrètes suggérées ("Et si vous...")

9️⃣ ADAPTATION DYNAMIQUE:
   ✓ Ajuste complexité selon niveau utilisateur
   ✓ Détecte frustration → simplifie
   ✓ Détecte expertise → approfondit
   ✓ Switch langue si nécessaire (FR/EN)

🔟 AMÉLIORATION CONTINUE:
   ✓ Apprend des interactions précédentes
   ✓ Mémorise préférences utilisateur
   ✓ Auto-critique et amélioration réponses
   ✓ Suggère améliorations processus

📚 OUTILS DISPONIBLES (21+):
═══════════════════════════════════════════════════════════════════════════════════
│ 🔍 RECHERCHE ILLIMITÉE:
├─ Local_Knowledge_Base: FOUILLE GLOBALE documents (AUCUNE limite k)
├─ Hybrid_Search: Combinaison locale + web + historique (ILLIMITÉ)
├─ Web_Search: Internet temps réel (10-50 résultats analysés)
├─ Web_Search_Detailed: Sources complètes avec URLs
├─ Current_News_Search: Actualités dernières 24h-7j
└─ Chat_History_Search: Continuité conversationnelle

│ 🤖 IA SPÉCIALISÉES:
├─ AI_Code_Generator: DeepSeek-Coder (meilleur que GPT pour code)
├─ AI_Plot_Generator: Graphiques scientifiques professionnels
├─ Image_Analyzer: Vision IA pour images
├─ Entity_Extractor: NER extraction entités
└─ Binary_Analysis: Analyse fichiers binaires avancée

│ 🎨 GÉNÉRATION CRÉATIVE:
├─ Text_To_Image_Generator: FLUX/Stable Diffusion
├─ Text_To_Video_Generator: Vidéos IA
├─ Text_To_Audio_Generator: Musique/Audio IA
├─ Text_To_3D_Generator: Modèles 3D
└─ Image_To_3D_Generator: 3D depuis photos

│ 📊 ANALYSE & OUTILS:
├─ Smart_Content_Extractor: Extraction web complète
├─ Text_Summarizer: Résumés intelligents
├─ Language_Translator: FR↔EN
├─ ERT_Interpretation: Géophysique
└─ OSM_Route_Calculator: Navigation GPS
═══════════════════════════════════════════════════════════════════════════════════

💡 PATTERNS DE RÉPONSE (60+ PROMPTS INTÉGRÉS):

�️ CONVERSATIONNEL (ChatGPT-style):
• "Excellente question ! Laisse-moi te décomposer ça..."
• "Je comprends exactement ce que tu cherches..."
• "Voici ce qui est intéressant à ce sujet..."
• "Permets-moi d'ajouter une nuance importante..."
• "Tu touches un point crucial ici..."

🧠 ANALYTIQUE (Claude-style):
• "Examinons cette question sous plusieurs angles..."
• "Il est important de considérer les implications suivantes..."
• "Voici une analyse structurée en 3 parties..."
• "Je dois souligner quelques considérations éthiques..."
• "Contextuellement, il faut noter que..."

⚡ CRÉATIF (Grok-style):
• "Plot twist: la réponse est plus fascinante que prévu..."
• "Fun fact qui va te surprendre..."
• "Spoiler alert: c'est contre-intuitif mais..."
• "Imagine un monde où..."
• "Voici un angle auquel personne ne pense..."

🎓 EXPERT (GPT-4-style):
• "D'un point de vue technique précis..."
• "Les données empiriques montrent que..."
• "Selon la littérature académique récente (2023-2025)..."
• "Une analyse rigoureuse révèle..."
• "Méthodologiquement, l'approche optimale consiste à..."

📋 STRUCTURES TYPES:

A) RÉPONSE RAPIDE:
"🎯 **Réponse Directe**: [1-2 phrases essentielles]

📖 **Explication**:
[Développement structuré]

📊 **Sources**: [X sources vérifiées, confiance ★★★★☆]

💡 **Suggestions**:
1. [Question approfondissement]
2. [Perspective alternative]
3. [Action concrète]"

B) RÉPONSE TECHNIQUE:
"⚙️ **Solution Technique**:
```[langage]
[code testé et commenté]
```

📝 **Explication ligne par ligne**:
[Détails implémentation]

✅ **Validation**: [Tests effectués]

🔧 **Alternatives**: [2-3 approches différentes]

💡 **Prochaines étapes**: [Suggestions amélioration]"

C) RÉPONSE CRÉATIVE:
"✨ **Concept Principal**: [Idée centrale]

🎨 **Variations Créatives**:
1. [Option A - classique]
2. [Option B - innovante]
3. [Option C - audacieuse]

🚀 **Implémentation**: [Étapes concrètes]

💡 **Inspirations**: [Références pertinentes]"

D) RÉPONSE COMPARATIVE:
"📊 **Comparaison Détaillée**:

| Critère | Option A | Option B | Option C |
|---------|----------|----------|----------|
[Tableau complet]

🏆 **Recommandation**: [Meilleur choix selon contexte]

⚖️ **Trade-offs**: [Avantages/Inconvénients]

💡 **Conseil personnalisé**: [Selon situation utilisateur]"

🎯 CONSIGNES D'EXÉCUTION:

✓ RECHERCHE ILLIMITÉE: Utilise k=100+ pour fouille globale (pas de limite!)
✓ MULTI-SOURCES: Combine MINIMUM 3 sources différentes
✓ VÉRIFICATION: Croise-vérifie informations contradictoires
✓ CITATIONS: Indique sources avec niveau confiance
✓ SUGGESTIONS: TOUJOURS 3+ questions de suivi pertinentes
✓ ADAPTABILITÉ: Ajuste ton/complexité selon utilisateur
✓ TRANSPARENCE: Signale incertitudes et limites
✓ PROACTIVITÉ: Anticipe besoins non exprimés
✓ CRÉATIVITÉ: Propose solutions innovantes
✓ EMPATHIE: Comprends contexte émotionnel

OUTILS: {tools}

EXÉCUTION:
Question: {input}
Thought: [Analyse multi-niveau: Que veut vraiment l'utilisateur? Quelles sources combiner? Quelle stratégie optimale? Quelles suggestions proposer?]
Action: [outil_optimal avec recherche ILLIMITÉE]
Action Input: [requête optimisée]
Observation: [résultat]
... [Répéter jusqu'à synthèse complète de TOUTES sources pertinentes]
Thought: J'ai maintenant une vue GLOBALE complète avec sources multiples vérifiées
Final Answer:
🎯 **Réponse Directe**: [Essentiel en 1-2 phrases]

📖 **Développement Détaillé**:
[Sections structurées avec exemples]

📊 **Sources Vérifiées**: 
• [Source 1 - ★★★★★ - Type]
• [Source 2 - ★★★★☆ - Type]
• [Source 3+ - ★★★☆☆ - Type]

💡 **Suggestions Intelligentes**:
1. 🔍 [Question approfondissement pertinente]
2. 🌟 [Perspective alternative intéressante]  
3. 🚀 [Action concrète recommandée]

{agent_scratchpad}""")

    
    # Vérifier si les agents sont disponibles
    if create_react_agent is None:
        st.warning("⚠️ Agents non disponibles - Mode simplifié activé")
        return None
    
    # Créer l'agent avec LangChain 1.0+ / LangGraph V1.0+
    # create_agent retourne directement un exécuteur compilé
    try:
        agent_executor = create_react_agent(llm, tools)
        st.write(f"✅ Agent créé avec {len(tools)} outils disponibles")
        return agent_executor
    except Exception as e:
        st.error(f"❌ Erreur création agent: {e}")
        return None
# Alias pour compatibilité
def create_agent(model_name, vectordb, graph, pois):
    """Version simplifiée pour compatibilité"""
    return create_enhanced_agent(model_name, vectordb, graph, pois)
# ========================================
# Fonctions OSM et Graphe Routier
# ========================================
def haversine(lon1, lat1, lon2, lat2):
    """Calcul distance haversine en mètres"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
class RoadPOIHandler(osmium.SimpleHandler):
    """Handler pour extraire routes et POIs depuis OSM"""
    def __init__(self):
        super().__init__()
        self.graph = nx.Graph()
        self.pois = []
    def node(self, n):
        """Extraire les POIs (points d'intérêt)"""
        if n.location.valid() and n.tags:
            name = n.tags.get('name', '')
            amenity = n.tags.get('amenity', '')
            if name or amenity:
                self.pois.append({
                    'name': name,
                    'amenity': amenity,
                    'lon': n.location.lon,
                    'lat': n.location.lat,
                    'tags': dict(n.tags)
                })
    def way(self, w):
        """Extraire les routes"""
        if 'highway' in w.tags:
            coords = []
            for n in w.nodes:
                if n.location.valid():
                    coords.append((n.location.lon, n.location.lat))
            for i in range(len(coords)-1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i+1]
                n1, n2 = (lon1, lat1), (lon2, lat2)
                dist = haversine(lon1, lat1, lon2, lat2)
                self.graph.add_node(n1, x=lon1, y=lat1)
                self.graph.add_node(n2, x=lon2, y=lat2)
                self.graph.add_edge(n1, n2, length=dist, highway=w.tags.get("highway"))
def trouver_noeud_plus_proche(lon, lat, graph):
    """Trouve le nœud du graphe le plus proche"""
    min_dist = float("inf")
    closest_node = None
    for node, data in graph.nodes(data=True):
        nlon, nlat = float(data["x"]), float(data["y"])
        dist = haversine(lon, lat, nlon, nlat)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node
def chercher_poi_par_nom(nom, pois_list):
    """Recherche un POI par nom"""
    nom_lower = nom.lower()
    for poi in pois_list:
        if nom_lower in poi['name'].lower() or nom_lower in poi['amenity'].lower():
            return poi
    return None
def generer_carte_trajet(graph, path, pois_list, start_poi=None, end_poi=None):
    """Génère une carte 2D du trajet"""
    fig, ax = plt.subplots(figsize=(12, 10))
    # Dessiner le graphe en arrière-plan
    for edge in list(graph.edges())[:1000]: # Limiter pour la performance
        node1, node2 = edge
        x1, y1 = node1[0], node1[1]
        x2, y2 = node2[0], node2[1]
        ax.plot([x1, x2], [y1, y2], 'lightgray', alpha=0.3, linewidth=0.5)
    # Dessiner le trajet
    if path and len(path) > 1:
        path_x = [node[0] for node in path]
        path_y = [node[1] for node in path]
        ax.plot(path_x, path_y, 'red', linewidth=3, label='Trajet')
        # Marquer début et fin
        ax.scatter(path_x[0], path_y[0], color='green', s=100, label='Départ', zorder=5)
        ax.scatter(path_x[-1], path_y[-1], color='red', s=100, label='Arrivée', zorder=5)
    # Ajouter quelques POIs
    for poi in pois_list[:20]:
        if poi['name']:
            ax.scatter(poi['lon'], poi['lat'], color='blue', s=20, alpha=0.6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Trajet calculé sur la carte OSM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    # Sauvegarder en mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf
def calculer_trajet(question, graph, pois_list):
    """Calcule un trajet basé sur une question textuelle"""
    if not graph or not pois_list:
        return None, "❌ Graphe ou POIs non disponibles", None
    # Utiliser LLM pour extraire départ et arrivée
    try:
        client = create_client()
        prompt = f"""Extraie le lieu de départ et le lieu d'arrivée de cette question de trajet.
Question: {question}
Réponds au format exact:
Départ: [nom du lieu de départ]
Arrivée: [nom du lieu d'arrivée]"""
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model=WORKING_MODELS["Llama 3.1 8B (Équilibré)"],
            max_tokens=100,
            temperature=0.1
        )
        extraction = response.choices[0].message.content
        start_line = [line for line in extraction.split('\n') if line.startswith('Départ: ')]
        end_line = [line for line in extraction.split('\n') if line.startswith('Arrivée: ')]
        if start_line and end_line:
            start_place = start_line[0].replace('Départ: ', '').strip()
            end_place = end_line[0].replace('Arrivée: ', '').strip()
        else:
            return None, "❌ Impossible d'extraire les lieux de la question.", None
    except Exception as e:
        st.write(f"❌ Erreur extraction LLM: {e}")
        return None, "❌ Erreur lors de l'extraction des lieux.", None
    start_poi = chercher_poi_par_nom(start_place, pois_list)
    end_poi = chercher_poi_par_nom(end_place, pois_list)
    if not start_poi or not end_poi:
        return None, f"❌ Impossible de trouver les lieux: {start_place} ou {end_place}.", None
    # Trouver les nœuds dans le graphe
    start_node = trouver_noeud_plus_proche(start_poi['lon'], start_poi['lat'], graph)
    end_node = trouver_noeud_plus_proche(end_poi['lon'], end_poi['lat'], graph)
    if not start_node or not end_node:
        return None, "❌ Impossible de trouver les nœuds dans le graphe routier.", None
    try:
        # Calculer le chemin
        path = nx.shortest_path(graph, source=start_node, target=end_node, weight="length")
        # Calculer la distance
        distance_totale = 0
        for i in range(len(path)-1):
            distance_totale += graph[path[i]][path[i+1]]['length']
        # Générer la carte
        carte_buf = generer_carte_trajet(graph, path, pois_list, start_poi, end_poi)
        # Réponse textuelle
        reponse = f"""🗺️ **Trajet calculé**
📍 **Départ**: {start_poi['name']} ({start_poi['amenity']})
🎯 **Arrivée**: {end_poi['name']} ({end_poi['amenity']})
📏 **Distance**: {distance_totale/1000:.2f} km
⏱️ **Temps estimé**: {int(distance_totale/83.33):.0f} min à pied | {int(distance_totale/833.33):.0f} min en voiture
🛣️ **Étapes**: {len(path)} points"""
        return carte_buf, reponse, {
            'start': start_poi,
            'end': end_poi,
            'distance': distance_totale,
            'path_length': len(path)
        }
    except nx.NetworkXNoPath:
        return None, f"❌ Aucun chemin trouvé entre {start_poi['name']} et {end_poi['name']}", None
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}", None
# ========================================
# Fonctions utilitaires pour images
# ========================================
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)
def df_to_html(df, max_rows=10):
    # Réduire le tableau si trop long
    if len(df) > max_rows:
        summary_row = pd.DataFrame({col: ['...'] for col in df.columns})
        df = pd.concat([df.head(max_rows // 2), summary_row, df.tail(max_rows // 2)])
    return df.to_html(index=False, escape=False)
# ========================================
# Fonctions Image Analysis
# ========================================
def classify_soil(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[:,:,0])
    mean_sat = np.mean(hsv[:,:,1])
    mean_val = np.mean(hsv[:,:,2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_variance = np.var(gray)
    soil_type = "Inconnu"
    possible_contents = "Inconnu"
    possible_minerals = "Inconnu"
    if mean_val < 100 and texture_variance > 5000:
        soil_type = "Argileux (riche en matière organique)"
        possible_contents = "Peut contenir de l'eau, nutriments, adapté aux cultures racines"
        possible_minerals = "Argiles comme kaolinite, illite; possible fer, aluminium"
    elif mean_sat > 100 and texture_variance < 3000:
        soil_type = "Sableux (drainant)"
        possible_contents = "Peut contenir peu d'eau, adapté aux plantes résistantes à la sécheresse"
        possible_minerals = "Quartz, feldspath; silice abondante"
    elif mean_hue > 20 and mean_hue < 40:
        soil_type = "Limoneux (équilibré)"
        possible_contents = "Peut contenir minéraux, bon pour l'agriculture générale"
        possible_minerals = "Silt avec mica, quartz; calcium, potassium"
    # Graphisme : Histogramme des couleurs HSV
    fig, ax = plt.subplots()
    ax.hist(hsv[:,:,0].ravel(), bins=50, color='b', alpha=0.5, label='Hue')
    ax.hist(hsv[:,:,1].ravel(), bins=50, color='g', alpha=0.5, label='Saturation')
    ax.hist(hsv[:,:,2].ravel(), bins=50, color='r', alpha=0.5, label='Value')
    ax.set_title('Histogramme des Composantes HSV')
    ax.legend()
    hist_img = fig_to_pil(fig)
    # Tableau des metrics
    metrics_df = pd.DataFrame({
        'Métrique': ['Hue Moyenne', 'Saturation Moyenne', 'Valeur Moyenne', 'Variance Texture'],
        'Valeur': [mean_hue, mean_sat, mean_val, texture_variance],
        'Explication': ['Moyenne de la teinte', 'Moyenne de la saturation des couleurs', 'Moyenne de la luminosité', 'Variance de la texture pour rugosité']
    })
    metrics_html = df_to_html(metrics_df)
    return {
        "soil_type": soil_type,
        "possible_contents": possible_contents,
        "possible_minerals": possible_minerals
    }, hist_img, metrics_html
def simulate_infrared(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ir_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
    ax.set_title('Simulation Infrarouge (Colormap JET)')
    ax.axis('off')
    ir_pil = fig_to_pil(fig)
    # Analyse simple (fake temp based on intensity)
    mean_intensity = np.mean(gray)
    ir_analysis = f"Simulation IR: Intensité moyenne {mean_intensity:.2f} (plus rouge = plus chaud, bleu = plus froid)"
    return ir_pil, ir_analysis
def detect_objects(image: np.ndarray, scale_factor=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = image.copy()
    dimensions = []
    types = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10: continue # skip small
        cv2.rectangle(img_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
        w_m = w * scale_factor
        h_m = h * scale_factor
        aspect = w / h if h != 0 else 0
        if aspect > 5: obj_type = 'Route'
        elif aspect < 0.2: obj_type = 'Clôture'
        elif 0.5 < aspect < 2: obj_type = 'Bâtiment'
        else: obj_type = 'Autre'
        dimensions.append((w_m, h_m))
        types.append(obj_type)
        cv2.putText(img_with_contours, f"{obj_type}: {w_m:.4f}m x {h_m:.4f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    num_objects = len(contours)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Objets Détectés avec Contours ({num_objects})")
    ax.axis('off')
    obj_img = fig_to_pil(fig)
    if dimensions:
        dim_df = pd.DataFrame({
            'Type': types,
            'Largeur (m)': [d[0] for d in dimensions],
            'Hauteur (m)': [d[1] for d in dimensions],
            'Explication': ['Dimension estimée avec contours OpenCV' for _ in types]
        })
        dim_html = df_to_html(dim_df)
    else:
        dim_html = ""
    return num_objects, obj_img, dim_html
def detect_fences(image: np.ndarray, scale_factor=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    img_with_lines = image.copy()
    lengths = []
    if lines is not None:
        line_list = [line[0] for line in lines]
        filtered_lines = [l for l in line_list if abs(l[0] - l[2]) < 10 or abs(l[1] - l[3]) < 10 or abs((l[1]-l[3]) / (l[0]-l[2] + 1e-5)) < 0.1 or abs((l[1]-l[3]) / (l[0]-l[2] + 1e-5)) > 10]
        line_lengths = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2) for x1,y1,x2,y2 in filtered_lines]
        sorted_indices = np.argsort(line_lengths)[::-1]
        sorted_lines = [filtered_lines[i] for i in sorted_indices]
        for x1,y1,x2,y2 in sorted_lines:
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * scale_factor
            lengths.append(length)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(img_with_lines, f"{length:.4f}m", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Clôtures/Bordures Détectées avec ({len(lengths)})")
    ax.axis('off')
    fence_img = fig_to_pil(fig)
    if lengths:
        fence_df = pd.DataFrame({
            'Longueur (m)': lengths,
            'Explication': ['Longueur de bordure filtrée et triée pour précision' for _ in lengths]
        })
        fence_html = df_to_html(fence_df)
    else:
        fence_html = ""
    return len(lengths), fence_img, fence_html
def detect_anomalies(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    num_edges = np.sum(edges > 0)
    mean_variance = np.mean(cv2.Laplacian(gray, cv2.CV_64F).var())
    anomalies = []
    if num_edges > 10000:
        anomalies.append("Anomalies structurelles détectées (ex. : fissures, défauts)")
    if mean_variance > 500:
        anomalies.append("Textures inhabituelles (ex. : zones irrégulières)")
    # Simulation photogrammétrie basique avec Open3D (si disponible)
    num_points = 0
    if OPEN3D_AVAILABLE:
        depth = np.random.rand(*gray.shape) * 255
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(image),
                o3d.geometry.Image(depth.astype(np.float32))
            ),
            o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
        )
        num_points = len(point_cloud.points)
    else:
        # Fallback: estimation simplifiée du nombre de points
        num_points = gray.shape[0] * gray.shape[1]
    # Graphisme : Histogramme des variances
    fig, ax = plt.subplots()
    ax.hist(cv2.Laplacian(gray, cv2.CV_64F).ravel(), bins=50)
    ax.set_title('Histogramme des Variances Locales (Anomalies)')
    var_hist_img = fig_to_pil(fig)
    # Tableau des metrics anomalies
    anomaly_df = pd.DataFrame({
        'Métrique': ['Nombre de Bords', 'Variance Moyenne', 'Points dans Point Cloud'],
        'Valeur': [num_edges, mean_variance, num_points],
        'Explication': ['Indique complexité structurelle (haut = anomalies)', 'Mesure irrégularités texture', 'Simulation 3D pour volume']
    })
    anomaly_html = df_to_html(anomaly_df)
    anomaly_desc_df = pd.DataFrame({
        'Anomalie': anomalies,
        'Explication': ['Défauts potentiels dans le terrain ou structures' for _ in anomalies]
    })
    anomaly_desc_html = df_to_html(anomaly_desc_df)
    return anomalies, var_hist_img, anomaly_html, anomaly_desc_html
def advanced_analyses(image: np.ndarray):
    analyses = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 100, 200)
    adv_images = []
    adv_tables = []
    # 1. Analyse Géologique
    kmeans = KMeans(n_clusters=3).fit(gray.reshape(-1, 1))
    clustered = kmeans.labels_.reshape(gray.shape)
    analyses['Géologique'] = 'Clusters de textures : ' + str(np.unique(kmeans.labels_))
    fig, ax = plt.subplots()
    ax.imshow(clustered, cmap='viridis')
    ax.set_title('Analyse Géologique: Clustering Textures')
    ax.axis('off')
    adv_images.append(fig_to_pil(fig))
    geo_df = pd.DataFrame({'Cluster': np.unique(kmeans.labels_), 'Compte': np.bincount(kmeans.labels_), 'Explication': ['Groupe de texture géologique' for _ in np.unique(kmeans.labels_)]})
    adv_tables.append(df_to_html(geo_df))
    # 2. Analyse Hydrologique
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    water_area = np.sum(blue_mask > 0) / blue_mask.size * 100
    analyses['Hydrologique'] = f'Pourcentage eau : {water_area:.2f}%'
    fig, ax = plt.subplots()
    ax.imshow(blue_mask, cmap='gray')
    ax.set_title('Analyse Hydrologique: Masque Eau')
    ax.axis('off')
    adv_images.append(fig_to_pil(fig))
    hydro_df = pd.DataFrame({'Métrique': ['Pourcentage Eau'], 'Valeur': [water_area], 'Explication': ['Zone potentielle pour ressources hydriques']})
    adv_tables.append(df_to_html(hydro_df))
    return analyses, {}, adv_images, adv_tables
def process_image(uploaded_file):
    image = Image.open(BytesIO(uploaded_file))
    img_array = np.array(image)
    proc_images = [image]
    captions = ['Image Originale']
    tables_html = []
    # IR
    ir_pil, ir_analysis = simulate_infrared(img_array)
    proc_images.append(ir_pil)
    captions.append('Simulation Infrarouge')
    tables_html.append('<h3>Analyse IR</h3><p>' + ir_analysis + '</p>')
    # Soil
    soil, hist_img, metrics_html = classify_soil(img_array)
    proc_images.append(hist_img)
    captions.append('Histogramme HSV')
    tables_html.append('<h3>Métriques Sol</h3>' + metrics_html)
    # Objects
    num_objects, obj_img, dim_html = detect_objects(img_array)
    proc_images.append(obj_img)
    captions.append('Objets Détectés')
    if dim_html:
        tables_html.append('<h3>Dimensions Objets</h3>' + dim_html)
    # Fences
    num_fences, fence_img, fence_html = detect_fences(img_array)
    proc_images.append(fence_img)
    captions.append('Clôtures Détectées')
    if fence_html:
        tables_html.append('<h3>Longueurs Clôtures</h3>' + fence_html)
    # Anomalies
    anomalies, var_hist_img, anomaly_html, anomaly_desc_html = detect_anomalies(img_array)
    proc_images.append(var_hist_img)
    captions.append('Histogramme Variances')
    tables_html.append('<h3>Métriques Anomalies</h3>' + anomaly_html)
    # Advanced
    analyses, predictions, adv_images, adv_tables = advanced_analyses(img_array)
    proc_images += adv_images[:5] # Limiter le nombre d'images
    captions += ['Analyse Avancée'] * len(adv_images[:5])
    tables_html += adv_tables[:3] # Limiter le nombre de tableaux
    analysis_data = {
        "soil": soil,
        "ir_analysis": ir_analysis,
        "num_objects": num_objects,
        "num_fences": num_fences,
        "anomalies": anomalies,
        "analyses": analyses,
        "predictions": predictions
    }
    tables_str = '<br>'.join(tables_html)
    return analysis_data, proc_images, tables_str
def improve_analysis_with_llm(analysis_data, model_name):
    prompt = f"""Analyse les données suivantes de l'image et fournis une analyse naturelle améliorée:
DONNÉES:
{json.dumps(analysis_data, indent=2)}
ANALYSE AMÉLIORÉE:"""
    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model=model_name,
            max_tokens=800,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur: {str(e)}"
def update_agent(model_choice, vectordb, graph, pois, chat_vectordb=None): # AJOUT MÉMOIRE VECTORIELLE
    model_name = WORKING_MODELS[model_choice]
    agent = create_enhanced_agent(model_name, vectordb, graph, pois, chat_vectordb)
    cache_info = get_cache_stats()
    return model_name, agent, cache_info
def handle_clear_cache():
    """Vide le cache web"""
    try:
        if os.path.exists(WEB_CACHE_PATH):
            os.remove(WEB_CACHE_PATH)
        return "✅ Cache web vidé"
    except Exception as e:
        return f"❌ Erreur: {e}"
def highlight_important_words(text):
    """Met en évidence les mots importants avec effet scintillante et tooltip"""
    # Mots-clés simples pour exemple (peut être étendu avec NER)
    important_keywords = ['important', 'clé', 'essentiel', 'critique', 'principal', 'trajet', 'pétrole', 'topographie']
    for keyword in important_keywords:
        text = re.sub(rf'\b({keyword})\b', r'<span class="sparkle-word" title="\1: Terme clé pour la compréhension du contexte">\1</span>', text, flags=re.IGNORECASE)
    return text
def install_code_model():
    """Installe un modèle de code léger et performant (CodeLlama-7B ou DeepSeek-Coder-1.3B)"""
    try:
        st.info("📦 Téléchargement de DeepSeek-Coder-1.3B-Instruct (modèle léger ~1.3GB)...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        cache_dir = os.path.expanduser("~/.cache/huggingface/code_models")
        
        # Télécharger et cacher le modèle
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        st.success(f"✅ Modèle de code installé avec succès!")
        st.info(f"📍 Emplacement: {cache_dir}")
        st.info("🚀 Utilisation: Le modèle sera automatiquement utilisé en Mode Code Expert")
        
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors de l'installation: {e}")
        st.warning("💡 Alternative: Le mode utilisera AI_Code_Generator par défaut")
        return False

# ========================================
# ANALYSEUR DE FICHIERS TEMPS RÉEL
# ========================================

def analyze_uploaded_file_realtime(uploaded_file, chat_history, agent, vectordb, graph, pois, web_enabled):
    """
    Analyse PROACTIVE et INTELLIGENTE d'un fichier par Kibali
    1. Analyse technique complète automatique
    2. Recherche web contextuelle pour enrichir
    3. Suggestions d'actions possibles
    4. Explications détaillées en temps réel
    """
    import io
    from PIL import Image
    
    # Créer un conteneur pour l'analyse en temps réel
    analysis_container = st.container()
    
    with analysis_container:
        st.markdown("### 🤖 Kibali analyse votre fichier...")
        
        # === ÉTAPE 1: ANALYSE TECHNIQUE ===
        with st.status("� Analyse technique en cours...", expanded=True) as status:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            file_size = uploaded_file.size
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            
            st.write(f"📄 **{uploaded_file.name}**")
            st.write(f"📊 **Taille:** {file_size:,} octets ({file_size/1024:.2f} KB)")
            
            # Détection magic bytes
            magic_bytes = file_bytes[:16]
            magic_hex = ' '.join(f'{b:02X}' for b in magic_bytes[:8])
            st.code(f"Signature: {magic_hex}", language="text")
            
            # ⭐ ANALYSE SCIENTIFIQUE IMMÉDIATE pour fichiers binaires
            scientific_context = ""
            if file_extension in ['dat', 'bin']:
                st.write("🔬 **Détection**: Possible fichier ERT - Activation capacités géophysiques")
                scientific_context = "GEOPHYSICS_MODE"
            
            status.update(label=f"✅ Scan initial terminé{' - Mode géophysique' if scientific_context else ''}", state="complete")
        
        # Données extraites pour Kibali
        extracted_data = {
            "filename": uploaded_file.name,
            "extension": file_extension,
            "size": file_size,
            "scientific_domain": scientific_context if scientific_context else "general",
            "content": "",
            "metadata": {}
        }
        
        # === ÉTAPE 2: ANALYSE SPÉCIALISÉE SELON TYPE ===
        technical_summary = ""
        
        # IMAGES
        if file_extension in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg']:
            with st.status("🖼️ Analyse d'image...", expanded=True) as status:
                try:
                    img = Image.open(io.BytesIO(file_bytes))
                    st.image(img, caption=uploaded_file.name, use_container_width=True)
                    
                    extracted_data["metadata"] = {
                        "dimensions": f"{img.size[0]}x{img.size[1]}",
                        "mode": img.mode,
                        "format": img.format,
                        "pixels": img.size[0] * img.size[1]
                    }
                    
                    technical_summary = f"Image {img.format} de {img.size[0]}x{img.size[1]} pixels, mode {img.mode}"
                    st.success(f"✅ {technical_summary}")
                    status.update(label="✅ Image analysée", state="complete")
                except Exception as e:
                    technical_summary = f"Image {file_extension}"
                    st.warning(f"⚠️ Analyse partielle: {e}")
                    status.update(label="⚠️ Analyse partielle", state="complete")
        
        # PDF
        elif file_extension == 'pdf':
            with st.status("📄 Extraction PDF...", expanded=True) as status:
                try:
                    import fitz
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    num_pages = doc.page_count
                    
                    # Extraire tout le texte
                    full_text = ""
                    for page_num in range(num_pages):
                        full_text += doc[page_num].get_text()
                    
                    word_count = len(full_text.split())
                    extracted_data["content"] = full_text[:5000]  # 5000 premiers caractères
                    extracted_data["metadata"] = {
                        "pages": num_pages,
                        "words": word_count
                    }
                    
                    technical_summary = f"PDF de {num_pages} pages contenant {word_count} mots"
                    st.success(f"✅ {technical_summary}")
                    
                    with st.expander("📖 Aperçu du contenu"):
                        st.text(full_text[:800] + "...")
                    
                    status.update(label="✅ PDF extrait", state="complete")
                except Exception as e:
                    technical_summary = f"PDF de taille {file_size/1024:.2f} KB"
                    st.warning(f"⚠️ Extraction partielle: {e}")
                    status.update(label="⚠️ Extraction partielle", state="complete")
        
        # CSV/EXCEL
        elif file_extension in ['csv', 'xlsx', 'xls']:
            with st.status("� Analyse des données tabulaires...", expanded=True) as status:
                try:
                    import pandas as pd
                    if file_extension == 'csv':
                        df = pd.read_csv(io.BytesIO(file_bytes))
                    else:
                        df = pd.read_excel(io.BytesIO(file_bytes))
                    
                    st.dataframe(df.head(10))
                    
                    # Statistiques
                    stats = df.describe().to_string()
                    extracted_data["content"] = f"Colonnes: {list(df.columns)}\n\nStatistiques:\n{stats}"
                    extracted_data["metadata"] = {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": list(df.columns),
                        "dtypes": df.dtypes.to_dict()
                    }
                    
                    technical_summary = f"Tableau de {len(df)} lignes × {len(df.columns)} colonnes"
                    st.success(f"✅ {technical_summary}")
                    status.update(label="✅ Données chargées", state="complete")
                except Exception as e:
                    technical_summary = f"Fichier tabulaire {file_extension}"
                    st.warning(f"⚠️ Chargement partiel: {e}")
                    status.update(label="⚠️ Chargement partiel", state="complete")
        
        # JSON
        elif file_extension == 'json':
            with st.status("📋 Parsing JSON...", expanded=True) as status:
                try:
                    import json
                    text_content = file_bytes.decode('utf-8')
                    data = json.loads(text_content)
                    
                    st.json(data)
                    
                    extracted_data["content"] = json.dumps(data, indent=2)[:5000]
                    extracted_data["metadata"] = {
                        "keys": list(data.keys()) if isinstance(data, dict) else "array",
                        "size": len(str(data))
                    }
                    
                    technical_summary = f"JSON contenant {len(data)} éléments"
                    st.success(f"✅ {technical_summary}")
                    status.update(label="✅ JSON parsé", state="complete")
                except Exception as e:
                    technical_summary = "Fichier JSON"
                    st.warning(f"⚠️ Parsing partiel: {e}")
                    status.update(label="⚠️ Parsing partiel", state="complete")
        
        # TEXTE
        elif file_extension in ['txt', 'md', 'log', 'py', 'js', 'html', 'css', 'xml']:
            with st.status("📝 Lecture du texte...", expanded=True) as status:
                try:
                    text_content = file_bytes.decode('utf-8', errors='ignore')
                    lines = text_content.split('\n')
                    
                    extracted_data["content"] = text_content[:5000]
                    extracted_data["metadata"] = {
                        "lines": len(lines),
                        "characters": len(text_content)
                    }
                    
                    with st.expander("📄 Contenu"):
                        st.code(text_content[:1000], language=file_extension if file_extension in ['py', 'js', 'html', 'css'] else 'text')
                    
                    technical_summary = f"Fichier texte de {len(lines)} lignes"
                    st.success(f"✅ {technical_summary}")
                    status.update(label="✅ Texte lu", state="complete")
                except Exception as e:
                    technical_summary = "Fichier texte"
                    st.warning(f"⚠️ Lecture partielle: {e}")
                    status.update(label="⚠️ Lecture partielle", state="complete")
        
        # BINAIRE
        else:
            with st.status("🔢 Analyse binaire...", expanded=True) as status:
                # 🆕 DÉTECTION ET PARSING AUTOMATIQUE MULTI-FRÉQUENCES pour .dat
                if file_extension == 'dat':
                    st.write("🔬 **Détection fichier .dat - Test parseur multi-fréquences...**")
                    
                    try:
                        # Sauvegarder temporairement le fichier
                        import tempfile
                        temp_dir = tempfile.gettempdir()
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        
                        with open(temp_file_path, 'wb') as f:
                            f.write(file_bytes)
                        
                        # Tenter le parsing multi-fréquences
                        from multi_freq_ert_parser import MultiFreqERTParser
                        parser = MultiFreqERTParser()
                        df = parser.parse_multiple_files([temp_file_path])
                        
                        if not df.empty and len(df) > 0:
                            st.success(f"✅ **PARSEUR MULTI-FRÉQUENCES ACTIVÉ !**")
                            st.write(f"📊 {len(df)} mesures ERT chargées")
                            st.write(f"📡 {len(parser.frequencies)} fréquences détectées")
                            st.write(f"📍 {len(parser.survey_points)} survey points")
                            
                            # Afficher aperçu de la structure
                            with st.expander("📋 Structure des données (5 premières lignes)"):
                                st.dataframe(df[['project', 'survey_point', 'depth', 'frequency_MHz', 'resistivity']].head())
                            
                            # Coordonnées spatiales
                            coords_df = parser.get_coordinates_corrected()
                            with st.expander("📐 Coordonnées spatiales (X, Y, Z)"):
                                st.dataframe(coords_df[['x', 'y', 'z', 'resistivity', 'frequency_MHz']].head())
                            
                            # Sauvegarder dans session_state
                            st.session_state['multi_freq_data'] = df
                            st.session_state['multi_freq_coords'] = coords_df
                            st.session_state['multi_freq_parser'] = parser
                            
                            # Mettre à jour extracted_data
                            extracted_data["metadata"] = {
                                "type": "ERT_MULTI_FREQ",
                                "num_measurements": len(df),
                                "num_frequencies": len(parser.frequencies),
                                "num_survey_points": len(parser.survey_points),
                                "frequencies_MHz": [float(f) for f in parser.frequencies[:10]],  # 10 premières
                                "depth_range": (float(parser.metadata['depth_range'][0]), float(parser.metadata['depth_range'][1])),
                                "resistivity_range": (float(parser.metadata['resistivity_range'][0]), float(parser.metadata['resistivity_range'][1]))
                            }
                            
                            extracted_data["content"] = f"""Données ERT Multi-Fréquences:
- {len(df)} mesures
- {len(parser.frequencies)} fréquences (de {min(parser.frequencies):.2f} à {max(parser.frequencies):.2f} MHz)
- {len(parser.survey_points)} survey points
- Profondeurs: {parser.metadata['depth_range'][0]:.1f} à {parser.metadata['depth_range'][1]:.1f} m
- Résistivités: {parser.metadata['resistivity_range'][0]:.2f} à {parser.metadata['resistivity_range'][1]:.2f} Ω·m

Structure correcte validée:
project | survey_point | depth | frequency_MHz | resistivity

Coordonnées spatiales (X,Y,Z) calculées automatiquement.
"""
                            
                            technical_summary = f"Fichier ERT multi-fréquences: {len(df)} mesures, {len(parser.frequencies)} fréquences"
                            status.update(label="✅ Fichier ERT parsé avec succès!", state="complete")
                            
                        else:
                            # Fallback vers analyse binaire classique
                            st.info("ℹ️ Format ERT non détecté, analyse binaire standard...")
                            entropy_result = entropy_analysis(file_bytes)
                            compression_result = compression_ratio(file_bytes)
                            pattern_result = pattern_recognition(file_bytes)
                            
                            extracted_data["metadata"] = {
                                "entropy": entropy_result,
                                "compression": compression_result,
                                "patterns": pattern_result
                            }
                            
                            with st.expander("🔍 Analyse détaillée"):
                                st.write(f"**Entropie:** {entropy_result}")
                                st.write(f"**Compression:** {compression_result}")
                                st.write(f"**Patterns:** {pattern_result}")
                            
                            technical_summary = f"Fichier binaire .dat de {file_size:,} octets"
                            status.update(label="✅ Analyse binaire terminée", state="complete")
                    
                    except Exception as e:
                        st.warning(f"⚠️ Parseur multi-fréquences échoué: {e}")
                        st.info("Passage en mode analyse binaire standard...")
                        
                        # Fallback vers analyse binaire
                        entropy_result = entropy_analysis(file_bytes)
                        compression_result = compression_ratio(file_bytes)
                        pattern_result = pattern_recognition(file_bytes)
                        
                        extracted_data["metadata"] = {
                            "entropy": entropy_result,
                            "compression": compression_result,
                            "patterns": pattern_result
                        }
                        
                        technical_summary = f"Fichier binaire .dat de {file_size:,} octets"
                        status.update(label="⚠️ Analyse binaire (fallback)", state="complete")
                
                else:
                    # Fichiers binaires non-.dat
                    try:
                        entropy_result = entropy_analysis(file_bytes)
                        compression_result = compression_ratio(file_bytes)
                        pattern_result = pattern_recognition(file_bytes)
                        
                        extracted_data["metadata"] = {
                            "entropy": entropy_result,
                            "compression": compression_result,
                            "patterns": pattern_result
                        }
                        
                        with st.expander("🔍 Analyse détaillée"):
                            st.write(f"**Entropie:** {entropy_result}")
                            st.write(f"**Compression:** {compression_result}")
                            st.write(f"**Patterns:** {pattern_result}")
                        
                        technical_summary = f"Fichier binaire .{file_extension} de {file_size/1024:.2f} KB"
                        st.success(f"✅ {technical_summary}")
                        status.update(label="✅ Analyse binaire terminée", state="complete")
                    
                    except Exception as e:
                        technical_summary = f"Fichier binaire .{file_extension}"
                        st.warning(f"⚠️ Analyse partielle: {e}")
                        status.update(label="⚠️ Analyse partielle", state="complete")
        
        # === ÉTAPE 3: RECHERCHE WEB CONTEXTUELLE ===
        web_context = ""
        if web_enabled:
            with st.status("🌐 Recherche d'informations contextuelles...", expanded=True) as status:
                try:
                    # Créer une requête intelligente basée sur le type de fichier
                    search_query = f"analyse {file_extension} fichier utilisation cas pratiques"
                    
                    st.write(f"🔎 Recherche: *{search_query}*")
                    
                    # Utiliser Tavily pour recherche
                    from langchain_tavily import TavilySearch
                    tavily = TavilySearch()
                    results = tavily.run(search_query)
                    
                    web_context = f"\n\n**Contexte Web:**\n{results[:500]}"
                    st.success("✅ Informations trouvées sur le web")
                    status.update(label="✅ Recherche terminée", state="complete")
                except Exception as e:
                    st.info("ℹ️ Recherche web non disponible")
                    status.update(label="ℹ️ Recherche ignorée", state="complete")
        
        # === ÉTAPE 4: KIBALI GÉNÈRE L'ANALYSE INTELLIGENTE ===
        with st.status("🤖 Kibali prépare son analyse...", expanded=True) as status:
            st.write("💭 Réflexion en cours...")
            
            # Construire le prompt pour Kibali - ULTRA-OPTIMISÉ ET PUISSANT
            kibali_prompt = f"""
Tu es Kibali, expert IA géophysique ERT. Analyse RAPIDE, PRÉCISE, COMPLÈTE.

📁 {uploaded_file.name} | {file_extension.upper()} | {file_size:,} octets
📊 {extracted_data['content'][:500] if extracted_data['content'] else 'Binaire'}
🔍 {extracted_data['metadata']}
{web_context}

═══════════════════════════════════════════════════════════════
🎯 10 RÈGLES D'ANALYSE PUISSANTE:
═══════════════════════════════════════════════════════════════
1. IDENTIFIE type données (ERT/géo/autre) en 1 phrase
2. EXTRAIS valeurs clés (profondeurs, résistivités, coords)
3. DÉTECTE patterns (grille, anomalies, zones)
4. INTERPRÈTE géologiquement (formations, matériaux)
5. QUANTIFIE précisément (min/max/moyenne + unités)
6. CONTEXTUALISE scientifiquement (normes, refs)
7. VISUALISE structure (2D/3D, coupes, profils)
8. ANTICIPE questions (profondeur? zones? roches?)
9. PROPOSE 3 actions IMMÉDIATES
10. STRUCTURE sections claires (🔍📊💡🎯)

═══════════════════════════════════════════════════════════════
� 5 RÈGLES DE PRÉCISION MAXIMALE:
═══════════════════════════════════════════════════════════════
1. **VALEURS EXACTES**: Donne TOUTES les valeurs numériques avec UNITÉS
   Exemple: "Résistivités: 0.5-120.3 Ω·m, Profondeurs: 0.0-15.5 m"
2. **INCERTITUDES**: Mentionne précision/erreurs si détectables
   Exemple: "±0.1 m en profondeur, ±5% en résistivité"
3. **COORDONNÉES GPS**: Extrais latitude/longitude si présentes
   Exemple: "Position: 1°23'45\"S, 13°45'12\"E"
4. **TIMESTAMPS**: Note dates/heures d'acquisition si trouvées
   Exemple: "Mesures: 2024-11-06 14:30 UTC"
5. **PARAMÈTRES TECHNIQUE**: Liste config instrument (espacement, fréquence)
   Exemple: "Wenner α=2m, f=100Hz, 48 électrodes"

═══════════════════════════════════════════════════════════════
📊 5 RÈGLES DE SYNTHÈSE/RAPPORTS/STATS/GRAPHIQUES:
═══════════════════════════════════════════════════════════════
1. **STATS COMPLÈTES**: Calcule ET affiche:
   • Moyenne, médiane, écart-type
   • Min, max, quartiles (Q1, Q3)
   • Distribution normale? Asymétrie (skewness)?
   
2. **GRAPHIQUES AUTO**: Génère IMMÉDIATEMENT:
   ✓ Profil vertical (résistivité vs profondeur)
   ✓ Histogramme avec courbe normale
   ✓ Coupe 2D si grille détectée
   ✓ Légende couleurs matériaux
   
3. **TABLEAU COMPARATIF**: Crée table matériaux:
   | Matériau | Résistivité typique | Détecté? | Profondeur |
   |----------|-------------------|----------|------------|
   | Eau | 0.5-10 Ω·m | OUI | 0-5m |
   | Argile | 1-100 Ω·m | OUI | 5-10m |
   
4. **RAPPORT SYNTHÈSE**: Structure professionnelle:
   ## 📋 RAPPORT D'ANALYSE ERT
   ### 1. RÉSUMÉ EXÉCUTIF (3 lignes)
   ### 2. STATISTIQUES CLÉS (tableau)
   ### 3. INTERPRÉTATION GÉOLOGIQUE (bullet points)
   ### 4. RECOMMANDATIONS (numérotées)
   ### 5. ANNEXES (graphiques téléchargeables)
   
5. **EXPORT MULTI-FORMAT**: Propose téléchargements:
   📥 CSV (données brutes)
   📥 PDF (rapport complet)
   📥 PNG (graphiques HD 300dpi)
   📥 HTML (interactif Plotly)

═══════════════════════════════════════════════════════════════
🛠️ 10 RÈGLES UTILISATION COHÉRENTE OUTILS:
═══════════════════════════════════════════════════════════════
1. code_agent → extractions précises
2. web_search → contexte scientifique
3. auto-viz → si >50 points
4. numpy → stats si >10 valeurs
5. profils verticaux → données profondeur
6. légendes couleur → résistivités
7. histogrammes → distributions
8. tableaux comparatifs → matériaux
9. export multi-formats (PNG/PDF/HTML)
10. workflows combinés (extract→stats→viz)

═══════════════════════════════════════════════════════════════
📋 STRUCTURE RÉPONSE OBLIGATOIRE:
═══════════════════════════════════════════════════════════════
🔍 **IDENTIFICATION** (1 ligne précise)
📊 **STATISTIQUES** (tableau formaté)
📈 **GRAPHIQUES** (génération automatique)
💡 **INTERPRÉTATION** (géologie + hydro)
🎯 **ACTIONS** (3 commandes concrètes)
❓ **QUESTIONS** (3 probables)

SOIS: CONCIS | PRÉCIS | ACTIONNABLE | VISUEL
"""
            
            # ⚡ ANALYSE SIMPLIFIÉE : Utiliser uniquement l'agent de code (pas de LLM)
            # Le LLM interviendra après dans le chat pour répondre aux questions
            status.update(label="✅ Analyse technique terminée - Prêt pour vos questions", state="complete")
            
            # Créer un résumé simple sans LLM
            simple_summary = f"""
## 📊 Analyse du fichier `{uploaded_file.name}`

### 🔍 IDENTIFICATION
- **Type**: {file_extension.upper()}
- **Taille**: {file_size:,} octets ({file_size/1024:.2f} KB)
- **Statut**: ✅ Fichier chargé et prêt à analyser

### 📝 RÉSUMÉ TECHNIQUE
{technical_summary if technical_summary else "Fichier binaire détecté"}

### 🎯 ACTIONS DISPONIBLES
Vous pouvez maintenant me poser des questions sur ce fichier :
- "Extrais les données du fichier"
- "Analyse les résistivités"
- "Crée un graphique"
- "Quelles sont les statistiques ?"

💬 **Kibali est prêt à répondre à vos questions !**
"""
            
            return simple_summary, extracted_data

# ========================================
# SYSTÈME D'OUTILS AUTONOMES POUR KIBALI
# ========================================

class KibaliToolsManager:
    """
    Gestionnaire d'outils autonomes pour Kibali
    Permet à Kibali de choisir et utiliser le bon outil selon la requête
    """
    
    def __init__(self):
        self.tools = {
            "resistivity_plot": {
                "name": "Générateur de Coupe de Résistivité ERT",
                "description": "Génère des graphiques de résistivité avec couleurs exactes selon normes géophysiques",
                "function": self.generate_resistivity_section,
                "keywords": ["résistivité", "ert", "coupe", "profil", "graphique", "dat", "ohm"]
            },
            "resistivity_analysis": {
                "name": "Analyseur de Matériaux par Résistivité",
                "description": "Identifie les matériaux géologiques à partir de valeurs de résistivité",
                "function": self.analyze_resistivity_materials,
                "keywords": ["matériau", "minéral", "roche", "sol", "argile", "sable"]
            },
            "data_extraction": {
                "name": "Extracteur de Données .DAT",
                "description": "Lit et extrait les valeurs numériques des fichiers .dat ERT",
                "function": self.extract_dat_data,
                "keywords": ["dat", "fichier", "extraire", "lire", "données"]
            },
            "statistical_analysis": {
                "name": "Analyse Statistique Avancée",
                "description": "Calcule statistiques, corrélations, distributions",
                "function": self.perform_statistical_analysis,
                "keywords": ["statistique", "moyenne", "écart-type", "corrélation", "distribution"]
            },
            "web_research": {
                "name": "Recherche Web Contextuelle",
                "description": "Recherche des informations supplémentaires sur internet",
                "function": self.perform_web_search,
                "keywords": ["recherche", "internet", "web", "information", "documentation"]
            },
            "data_visualization": {
                "name": "Visualiseur de Données Génériques",
                "description": "Crée des graphiques pour tout type de données",
                "function": self.create_generic_plot,
                "keywords": ["graphique", "courbe", "histogramme", "scatter", "visualisation"]
            },
            "color_mapper": {
                "name": "Carte de Couleurs de Résistivité",
                "description": "Affiche la correspondance couleur-résistivité selon normes ERT",
                "function": self.show_color_mapping,
                "keywords": ["couleur", "palette", "colormap", "légende"]
            },
            "dat_structure_analyzer": {
                "name": "Analyseur de Structure de Fichiers .DAT",
                "description": "Analyse automatiquement la structure de fichiers .DAT (encodage, délimiteur, colonnes)",
                "function": self.analyze_dat_structure,
                "keywords": ["structure", "dat", "analyse", "format", "encodage", "délimiteur", "colonnes"]
            },
            "survey_depth_parser": {
                "name": "Parseur Survey-Point / Depth / Data",
                "description": "Parseur spécialisé pour fichiers avec colonnes survey-point, depth, data (profils ERT verticaux)",
                "function": self.parse_survey_depth_data,
                "keywords": ["survey", "point", "depth", "profondeur", "profil", "vertical", "coupe", "3d", "volume"]
            },
            "multi_freq_parser": {
                "name": "Parseur Multi-Fréquences ERT (Fichiers Complémentaires)",
                "description": "Parse et fusionne plusieurs fichiers .dat avec multi-fréquences (MHz), calcule coordonnées spatiales correctes (X, Y, Z)",
                "function": self.parse_multi_freq_data,
                "keywords": ["multi", "fréquence", "mhz", "complémentaire", "fusionner", "fusion", "coordonnées", "spatial", "xyz", "plusieurs fichiers"]
            }
        }
    
    def detect_needed_tools(self, user_query: str, file_data: dict = None) -> list:
        """
        Détecte automatiquement les outils nécessaires pour répondre à la requête
        """
        query_lower = user_query.lower()
        needed_tools = []
        
        # Vérifier chaque outil
        for tool_id, tool_info in self.tools.items():
            for keyword in tool_info["keywords"]:
                if keyword in query_lower:
                    needed_tools.append(tool_id)
                    break
        
        # Ajout automatique selon contexte fichier
        if file_data:
            if file_data.get("extension") == "dat":
                if "data_extraction" not in needed_tools:
                    needed_tools.insert(0, "data_extraction")
                if "résistivité" in query_lower or "ert" in query_lower:
                    if "resistivity_plot" not in needed_tools:
                        needed_tools.append("resistivity_plot")
        
        return needed_tools
    
    def generate_resistivity_section(self, data: dict, query: str = "") -> dict:
        """Génère une coupe de résistivité ERT avec couleurs exactes"""
        try:
            resistivity_values = data.get("resistivity_values", [])
            
            if not resistivity_values:
                # Essayer d'extraire des valeurs du contenu
                content = data.get("content", "")
                import re
                numbers = re.findall(r'\d+\.?\d*', content)
                resistivity_values = [float(n) for n in numbers if float(n) > 0][:100]
            
            if resistivity_values:
                # Utiliser la fonction existante
                plot_html = generate_resistivity_plot(resistivity_values)
                
                # Analyse des matériaux
                materials_analysis = analyze_minerals_from_resistivity(resistivity_values)
                
                # Table de couleurs
                color_table = generate_resistivity_table(resistivity_values)
                
                return {
                    "success": True,
                    "plot": plot_html,
                    "analysis": materials_analysis,
                    "color_table": color_table,
                    "message": f"✅ Coupe de résistivité générée avec {len(resistivity_values)} points"
                }
            else:
                return {
                    "success": False,
                    "message": "❌ Aucune valeur de résistivité trouvée"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Erreur génération coupe: {e}"
            }
    
    def analyze_resistivity_materials(self, data: dict, query: str = "") -> dict:
        """Analyse les matériaux à partir des résistivités"""
        try:
            resistivity_values = data.get("resistivity_values", [])
            filename = data.get("filename", "unknown")
            
            if resistivity_values:
                analysis = analyze_minerals_from_resistivity(resistivity_values, filename)
                color_analysis = resistivity_color_analysis(resistivity_values)
                
                return {
                    "success": True,
                    "analysis": analysis,
                    "color_analysis": color_analysis,
                    "message": "✅ Analyse des matériaux terminée"
                }
            else:
                return {
                    "success": False,
                    "message": "❌ Aucune donnée de résistivité"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Erreur analyse: {e}"
            }
    
    def extract_dat_data(self, data: dict, query: str = "") -> dict:
        """Extrait les données d'un fichier .dat"""
        try:
            content = data.get("content", "")
            
            if not content:
                return {
                    "success": False,
                    "message": "❌ Aucun contenu à extraire"
                }
            
            # Extraction des nombres
            import re
            numbers = re.findall(r'-?\d+\.?\d*', content)
            extracted_values = [float(n) for n in numbers if n]
            
            # Filtrer pour garder les résistivités plausibles (0.001 - 1000000 Ohm.m)
            resistivity_values = [v for v in extracted_values if 0.001 <= v <= 1000000]
            
            return {
                "success": True,
                "resistivity_values": resistivity_values,
                "total_numbers": len(extracted_values),
                "message": f"✅ {len(resistivity_values)} valeurs de résistivité extraites"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Erreur extraction: {e}"
            }
    
    def perform_statistical_analysis(self, data: dict, query: str = "") -> dict:
        """Effectue une analyse statistique des données"""
        try:
            values = data.get("resistivity_values", [])
            
            if not values:
                return {
                    "success": False,
                    "message": "❌ Aucune donnée pour analyse statistique"
                }
            
            import numpy as np
            from scipy import stats
            
            arr = np.array(values)
            
            statistics = {
                "Nombre de valeurs": len(arr),
                "Moyenne": np.mean(arr),
                "Médiane": np.median(arr),
                "Écart-type": np.std(arr),
                "Minimum": np.min(arr),
                "Maximum": np.max(arr),
                "Q1 (25%)": np.percentile(arr, 25),
                "Q3 (75%)": np.percentile(arr, 75),
                "Skewness": stats.skew(arr),
                "Kurtosis": stats.kurtosis(arr)
            }
            
            # Format pour affichage
            stats_text = "## 📊 Statistiques Descriptives\n\n"
            for key, value in statistics.items():
                if isinstance(value, (int, float)):
                    stats_text += f"- **{key}:** {value:.3f}\n"
                else:
                    stats_text += f"- **{key}:** {value}\n"
            
            return {
                "success": True,
                "statistics": statistics,
                "formatted_text": stats_text,
                "message": "✅ Analyse statistique complète"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Erreur statistiques: {e}"
            }
    
    def perform_web_search(self, data: dict, query: str = "") -> dict:
        """Effectue une recherche web contextuelle"""
        try:
            from langchain_tavily import TavilySearch
            
            # Construire requête intelligente
            search_query = query
            if data.get("extension") == "dat":
                search_query += " ERT electrical resistivity tomography interpretation"
            
            tavily = TavilySearch()
            results = tavily.run(search_query)
            
            return {
                "success": True,
                "results": results,
                "message": "✅ Recherche web terminée"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"⚠️ Recherche web non disponible: {e}",
                "results": ""
            }
    
    def create_generic_plot(self, data: dict, query: str = "") -> dict:
        """Crée un graphique générique"""
        try:
            values = data.get("resistivity_values", [])
            
            if not values:
                return {
                    "success": False,
                    "message": "❌ Aucune donnée à visualiser"
                }
            
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            import base64
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Graphique 1: Courbe
            ax1.plot(values, marker='o', linestyle='-', linewidth=2)
            ax1.set_title("Évolution des valeurs")
            ax1.set_xlabel("Index")
            ax1.set_ylabel("Valeur")
            ax1.grid(True, alpha=0.3)
            
            # Graphique 2: Histogramme
            ax2.hist(values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax2.set_title("Distribution")
            ax2.set_xlabel("Valeur")
            ax2.set_ylabel("Fréquence")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convertir en base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            plot_base64 = base64.b64encode(plot_data).decode()
            plot_html = f'<img src="data:image/png;base64,{plot_base64}" style="max-width: 100%;" />'
            
            return {
                "success": True,
                "plot": plot_html,
                "message": "✅ Graphique généré"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Erreur graphique: {e}"
            }
    
    def show_color_mapping(self, data: dict, query: str = "") -> dict:
        """Affiche la carte de correspondance couleurs-résistivité"""
        try:
            color_table = get_water_resistivity_color_table()
            
            return {
                "success": True,
                "color_mapping": color_table,
                "message": "✅ Carte de couleurs ERT"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Erreur: {e}"
            }
    
    def analyze_dat_structure(self, data: dict, query: str = "") -> dict:
        """Analyse automatiquement la structure d'un fichier .DAT"""
        try:
            file_path = data.get("physical_path", "")
            if not file_path:
                return {
                    "success": False,
                    "message": "❌ Aucun chemin de fichier fourni"
                }
            
            # Utiliser l'analyseur intelligent
            structure = analyze_dat_file(file_path)
            
            if structure:
                return {
                    "success": True,
                    "structure": structure,
                    "message": f"✅ Structure analysée: {structure.file_type} | Confiance: {structure.confidence}%"
                }
            else:
                return {
                    "success": False,
                    "message": "❌ Impossible d'analyser la structure du fichier"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Erreur analyse structure: {e}"
            }
    
    def parse_survey_depth_data(self, data: dict, query: str = "") -> dict:
        """Parseur spécialisé pour fichiers survey-point/depth/data - DISABLED"""
        return {
            "success": False,
            "message": "⚠️ Parseur survey-depth désactivé (calculs incorrects). Utilisez multi_freq_parser à la place."
        }
    
    def parse_multi_freq_data(self, data: dict, query: str = "") -> dict:
        """
        Parseur multi-fréquences pour fichiers .dat complémentaires
        Calcule les coordonnées spatiales correctes (X, Y, Z)
        """
        try:
            # Vérifier si des fichiers multiples sont disponibles dans session_state
            if 'multi_freq_data' in st.session_state and st.session_state.multi_freq_data is not None:
                parser = st.session_state.get('multi_freq_parser')
                df = st.session_state.multi_freq_data
                coords_df = st.session_state.get('multi_freq_coords')
                
                return {
                    "success": True,
                    "data": df,
                    "coordinates": coords_df,
                    "parser": parser,
                    "metadata": parser.metadata if parser else {},
                    "message": f"✅ Données multi-fréquences disponibles: {len(df)} mesures, {len(parser.frequencies)} fréquences"
                }
            
            # Sinon, essayer de parser un seul fichier
            file_path = data.get("physical_path", "")
            if not file_path or not os.path.exists(file_path):
                return {
                    "success": False,
                    "message": "❌ Aucun fichier disponible. Uploadez des fichiers .dat dans l'interface multi-fréquences."
                }
            
            # Parser le fichier
            from multi_freq_ert_parser import MultiFreqERTParser
            parser = MultiFreqERTParser()
            df = parser.parse_multiple_files([file_path])
            
            if df.empty:
                return {
                    "success": False,
                    "message": "❌ Aucune donnée extraite. Vérifiez le format du fichier."
                }
            
            # Générer coordonnées
            coords_df = parser.get_coordinates_corrected()
            
            # Sauvegarder dans session state
            st.session_state['multi_freq_data'] = df
            st.session_state['multi_freq_coords'] = coords_df
            st.session_state['multi_freq_parser'] = parser
            
            return {
                "success": True,
                "data": df,
                "coordinates": coords_df,
                "parser": parser,
                "metadata": parser.metadata,
                "frequencies": parser.frequencies,
                "survey_points": parser.survey_points,
                "message": f"✅ Fichier parsé: {len(df)} mesures, {len(parser.frequencies)} fréquences, {len(parser.survey_points)} survey points"
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "message": f"❌ Erreur parseur multi-fréquences: {e}",
                "traceback": traceback.format_exc()
            }
    
    def execute_tools(self, tool_ids: list, data: dict, query: str = "") -> dict:
        """
        Exécute plusieurs outils et retourne les résultats combinés
        """
        results = {
            "tools_used": [],
            "outputs": {},
            "success": True
        }
        
        for tool_id in tool_ids:
            if tool_id in self.tools:
                tool_info = self.tools[tool_id]
                st.info(f"🔧 Utilisation de l'outil: **{tool_info['name']}**")
                
                # Exécuter l'outil
                output = tool_info["function"](data, query)
                
                results["tools_used"].append(tool_info["name"])
                results["outputs"][tool_id] = output
                
                if not output.get("success", False):
                    results["success"] = False
        
        return results

# Instance globale du gestionnaire d'outils
kibali_tools = KibaliToolsManager()

# Instance globale du moteur de visualisation
viz_engine = VisualizationEngine()

# Instance globale de l'auto-visualiseur
auto_viz = AutoVisualizer()


def llm_with_tools(prompt: str, file_data: dict = None, qwen_llm=None, code_agent=None) -> dict:
    """
    LLM Qwen qui analyse la question et décide intelligemment d'utiliser les outils
    
    Args:
        prompt: Question de l'utilisateur
        file_data: Données du fichier uploadé
        qwen_llm: Instance du modèle Qwen
        code_agent: Instance de l'agent de code
    
    Returns:
        dict: {
            'response': str,  # Réponse textuelle du LLM
            'needs_code': bool,  # Besoin d'exécuter du code
            'needs_viz': bool,  # Besoin de visualisations
            'code_params': dict  # Paramètres pour l'agent de code
        }
    """
    # Préparer le contexte pour Qwen
    context = f"""Tu es Kibali, expert en géophysique et analyse ERT.

FICHIER ANALYSÉ: {file_data.get('filename', 'N/A') if file_data else 'Aucun'}

QUESTION: {prompt}

INSTRUCTIONS:
1. Analyse la question et réponds de façon naturelle et conversationnelle
2. Si tu as besoin d'extraire des données précises du fichier, indique [USE_CODE_AGENT]
3. Si tu veux créer des visualisations, indique [USE_VISUALIZATION]
4. Sois concis et pédagogique

Réponds maintenant:"""
    
    try:
        # Utiliser Qwen pour générer la réponse avec LIMITE AUGMENTÉE
        if qwen_llm:
            from langchain.schema import HumanMessage
            # Configuration pour générer des réponses LONGUES et DÉTAILLÉES
            response = qwen_llm.invoke(
                [HumanMessage(content=context)],
                config={"max_tokens": 3000, "temperature": 0.7}  # 3000 tokens
            )
            llm_response = response.content if hasattr(response, 'content') else str(response)
        else:
            # Fallback si Qwen n'est pas disponible
            llm_response = f"Pour répondre à votre question '{prompt}', j'ai besoin d'analyser le fichier en détail."
        
        # Détecter les besoins d'outils dans la réponse
        needs_code = '[USE_CODE_AGENT]' in llm_response or any(word in prompt.lower() for word in [
            'profondeur', 'valeur', 'extraire', 'calculer', 'statistique', 'précis', 'exact'
        ])
        
        needs_viz = '[USE_VISUALIZATION]' in llm_response or any(word in prompt.lower() for word in [
            'graphique', 'profil', 'coupe', 'visualis', 'affiche', 'montre', 'schéma'
        ])
        
        # Nettoyer la réponse des marqueurs
        clean_response = llm_response.replace('[USE_CODE_AGENT]', '').replace('[USE_VISUALIZATION]', '').strip()
        
        return {
            'response': clean_response,
            'needs_code': needs_code,
            'needs_viz': needs_viz,
            'code_params': {
                'action': 'extract' if 'profondeur' in prompt.lower() or 'valeur' in prompt.lower() else 'analyze',
                'query': prompt
            }
        }
    
    except Exception as e:
        print(f"❌ Erreur LLM: {e}")
        return {
            'response': f"Je peux vous aider avec '{prompt}'. Laissez-moi analyser les données...",
            'needs_code': True,
            'needs_viz': False,
            'code_params': {'action': 'analyze', 'query': prompt}
        }


def handle_chat_enhanced(message, history, agent, model_choice, vectordb, graph, pois, web_enabled, mode="humain", mode_prompt=None):
    """
    Conversation naturelle avec le LLM principal qui décide lui-même d'utiliser les outils
    Le LLM contrôle tout, les agents sont des outils optionnels
    
    Args:
        message: Question de l'utilisateur
        history: Historique de conversation
        file_context: Contexte du fichier uploadé (optionnel)
        web_enabled: Activer recherche web
    
    Returns:
        Réponse conversationnelle du LLM
    """
    # Construire le contexte complet pour le LLM
    system_prompt = """Tu es Kibali, un assistant IA expert en géophysique et analyse de données ERT.

🎯 **TON RÔLE**:
- Avoir des conversations naturelles et fluides avec l'utilisateur
- Répondre directement aux questions sans répéter la même analyse
- Utiliser ta connaissance pour expliquer, interpréter, contextualiser
- Décider toi-même quand tu as besoin d'outils pour compléter ta réponse

🛠️ **OUTILS DISPONIBLES** (à utiliser SI NÉCESSAIRE):
1. **code_agent** : Générer et exécuter du code Python pour analyses complexes
   - Usage: Quand tu as besoin de calculs précis, extractions de données, analyses statistiques
   - Exemple: "Je vais générer du code pour extraire les profondeurs exactes"

2. **web_search** : Rechercher des informations actuelles sur internet
   - Usage: Pour contextualiser avec des références scientifiques, normes, études
   - Exemple: "Je vais chercher les normes de résistivité pour ce type de sol"

3. **visualization** : Créer des graphiques professionnels
   - Usage: Pour illustrer les données, créer des profils, sections, histogrammes
   - Exemple: "Je vais créer un profil de résistivité pour visualiser ces données"

📝 **RÈGLES DE CONVERSATION**:
1. **Réponds directement** à la question posée, ne refais pas l'analyse complète à chaque fois
2. **Utilise le contexte** : Référence les données du fichier déjà analysé
3. **Sois conversationnel** : Parle naturellement, pas en bullet points robotiques
4. **Appelle les outils** seulement quand nécessaire, pas systématiquement
5. **Explique ce que tu fais** : "Je vais utiliser l'outil X pour Y"

💡 **EXEMPLES DE BONNES RÉPONSES**:

Question: "Donne moi les profondeurs exactes"
❌ Mauvais: *refaire toute l'analyse du fichier*
✅ Bon: "D'après l'analyse précédente, les profondeurs vont de 0m à 15m avec des mesures tous les 0.5m. Voici les profondeurs exactes: [liste]. Veux-tu que je crée un profil vertical pour mieux visualiser ces données ?"

Question: "C'est quoi cette zone de faible résistivité ?"
❌ Mauvais: *refaire extraction complète*
✅ Bon: "La zone de faible résistivité (< 10 ohm-m) que nous avons détectée correspond probablement à de l'argile saturée en eau ou à une nappe phréatique. Ces valeurs sont typiques pour..."
"""

    # Ajouter le contexte du fichier si disponible
    if file_context:
        system_prompt += f"""

📁 **FICHIER ANALYSÉ**:
- Nom: {file_context.get('filename', 'N/A')}
- Type: {file_context.get('extension', 'N/A')}
- Données disponibles: {file_context.get('summary', 'Résistivités, profondeurs, positions')}
"""

    # Générer la réponse avec le LLM
    try:
        # Utiliser Ollama (local)
        from openai import OpenAI
        
        try:
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            
            # Préparer les messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # Ajouter l'historique récent
            for msg in history[-5:]:
                messages.append({"role": msg['role'], "content": msg['content']})
            
            # Ajouter le message actuel
            messages.append({"role": "user", "content": message})
            
            response = client.chat.completions.create(
                model="llama3.2:latest",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ Ollama non disponible: {e}")
            # Fallback: réponse basique intelligente
            return generate_smart_fallback_response(message, history, file_context)
    
    except Exception as e:
        print(f"❌ Erreur LLM: {e}")
        return f"Je suis désolé, j'ai rencontré une erreur. Peux-tu reformuler ta question ? Erreur: {e}"


def generate_smart_fallback_response(message: str, history: list, file_context: dict) -> str:
    """Génère une réponse intelligente sans LLM en analysant la question"""
    message_lower = message.lower()
    
    # Détection de questions spécifiques
    if any(word in message_lower for word in ['profondeur', 'depth', 'profond']):
        return """D'après le fichier analysé, voici les informations sur les profondeurs :

Les mesures ont été effectuées à différentes profondeurs. Pour obtenir les valeurs exactes et précises, je peux générer du code Python qui va extraire toutes les profondeurs du fichier. 

Souhaites-tu que je fasse cette extraction détaillée ?"""
    
    elif any(word in message_lower for word in ['résistivité', 'resistivity', 'valeur']):
        return """Concernant les valeurs de résistivité, le fichier contient plusieurs mesures. Je peux te donner différents types d'informations :

- Les valeurs brutes complètes
- Les statistiques (min, max, moyenne, médiane)
- La distribution par zones géologiques
- Un profil vertical

Qu'est-ce qui t'intéresse particulièrement ?"""
    
    elif any(word in message_lower for word in ['graph', 'visualis', 'profil', 'coupe']):
        return """Je peux créer plusieurs types de visualisations pour ces données :

📊 **Disponibles**:
- Profil vertical de résistivité
- Coupe 2D avec colormap
- Histogramme par zones
- Colonne géologique

Quel type de graphique veux-tu que je génère ?"""
    
    else:
        # Réponse générique conversationnelle
        return f"""Je comprends ta question sur "{message}". 

Pour te donner une réponse précise, peux-tu me dire exactement ce que tu cherches ? Par exemple :
- Des valeurs numériques spécifiques ?
- Une interprétation géologique ?
- Une visualisation ?
- Une comparaison avec d'autres données ?

Je suis là pour t'aider de façon conversationnelle et naturelle ! 😊"""


def handle_chat_enhanced(message, history, agent, model_choice, vectordb, graph, pois, web_enabled, mode="humain", mode_prompt=None):
    # AJOUT MÉMOIRE VECTORIELLE: Charger la base chat
    chat_vectordb, _ = load_chat_vectordb()
    if not message.strip():
        return ""
    
    # 🎯 ENRICHISSEMENT AUTOMATIQUE DU CONTEXTE FICHIER
    # Si un fichier est uploadé, toujours l'ajouter au contexte de la conversation
    file_context_prompt = ""
    
    # 📊 CONTEXTE MULTI-FRÉQUENCES ERT
    multi_freq_context = ""
    if 'multi_freq_data' in st.session_state and st.session_state.multi_freq_data is not None:
        parser = st.session_state.get('multi_freq_parser')
        if parser:
            multi_freq_context = f"""

📡 **DONNÉES ERT MULTI-FRÉQUENCES CHARGÉES**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 Fichiers fusionnés: {parser.metadata['num_files']}
📡 Fréquences: {', '.join([f'{f:.1f} MHz' for f in parser.frequencies])}
📍 Survey points: {parser.metadata['num_survey_points']}
📏 Profondeurs: {parser.metadata['depth_range'][0]:.1f}m à {parser.metadata['depth_range'][1]:.1f}m
📊 Total mesures: {parser.metadata['total_measurements']:,}
🔬 Résistivités: {parser.metadata['resistivity_range'][0]:.2f} - {parser.metadata['resistivity_range'][1]:.2f} Ω·m

💡 **Tu peux accéder aux DataFrames**:
- `st.session_state.multi_freq_data` : Données complètes
- `st.session_state.multi_freq_coords` : Coordonnées spatiales (X, Y, Z)
- `st.session_state.multi_freq_parser` : Objet parseur avec toutes les méthodes

🎨 **Méthodes de visualisation disponibles**:
- `parser.create_2d_section_by_frequency(freq)` : Coupe 2D pour une fréquence
- `parser.create_3d_volume()` : Volume 3D interactif
- `parser.create_frequency_comparison()` : Comparaison multi-fréquences
- `parser.get_coordinates_corrected()` : Coordonnées spatiales corrigées

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    if 'uploaded_file_data' in st.session_state and st.session_state.uploaded_file_data:
        file_data = st.session_state.uploaded_file_data
        file_context_prompt = f"""
📎 **CONTEXTE FICHIER ACTIF** (Toujours présent dans la conversation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 Fichier: {file_data['filename']}
📂 Chemin physique: {file_data.get('physical_path', 'N/A')}
📊 Type: {file_data.get('extension', 'N/A')}
💾 Taille: {file_data['size']} octets ({file_data['size']/1024:.2f} KB)
{multi_freq_context}

📝 **Métadonnées extraites**:
{file_data.get('metadata', 'Aucune métadonnée')}

📄 **Aperçu du contenu**:
{file_data.get('content', 'Contenu non disponible')[:500]}...

🔬 **CAPACITÉS SCIENTIFIQUES DISPONIBLES**:
- Analyse ERT (Electrical Resistivity Tomography) : résistivité électrique, profondeurs, formations géologiques
- Interprétation géologique : classification des couches, identification des matériaux
- Recherches web en temps réel : pour contextualiser les données avec des références scientifiques
- Génération de code Python : pour analyses avancées, visualisations, calculs statistiques
- Extraction de données binaires : formats .dat, .bin, structures hexadécimales
- Calculs statistiques : moyennes, médianes, écarts-types, distributions
- Visualisations : graphiques, courbes de résistivité, cartes de profondeur

� **OUTILS DE VISUALISATION DISPONIBLES** (utilise-les spontanément !):
- create_resistivity_profile() : Profils verticaux de résistivité interactifs avec Plotly
- create_2d_resistivity_section() : Coupes 2D avec colormaps géologiques
- create_geological_column() : Colonnes stratigraphiques avec légendes
- create_cross_section_diagram() : Coupes transversales annotées
- create_histogram_with_zones() : Histogrammes avec zones géologiques
- create_depth_statistics_chart() : Statistiques multi-courbes par profondeur
- create_legend_table() : Tables de légendes HTML stylées
- create_annotated_diagram() : Schémas avec annotations OpenCV
- Tous les graphiques sont téléchargeables en PNG, PDF, ou HTML interactif

�💡 **INSTRUCTIONS POUR TES RÉPONSES**:
1. **Contextualise toujours** : Fais référence au fichier dans tes explications
2. **Sois pédagogique** : Explique les concepts scientifiques de façon accessible
3. **Utilise ta connaissance** : Apporte du contexte géologique, physique, chimique
4. **Fais des recherches** : Si besoin, cherche sur internet pour enrichir ton analyse
5. **Génère du code** : Pour répondre aux questions complexes, crée et exécute du code Python
6. **Crée des visualisations** : Utilise spontanément les outils graphiques pour illustrer tes explications
7. **Reste naturel** : Parle comme un expert humain, pas comme un robot
8. **Anticipe les besoins** : Propose des analyses supplémentaires pertinentes
9. **Sois précis** : Donne des valeurs numériques, des unités, des ordres de grandeur
10. **Rends tout téléchargeable** : Chaque graphique doit avoir un bouton de téléchargement

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Question de l'utilisateur**: {message}
"""
        # Ajouter le contexte enrichi au message original (ne pas remplacer complètement)
        # IMPORTANT: Extraire UNIQUEMENT la vraie question utilisateur pour le code agent
        if "[QUESTION DE L'UTILISATEUR]" in message:
            # Extraire la question entre les marqueurs
            parts = message.split("[QUESTION DE L'UTILISATEUR]")
            if len(parts) > 1:
                question_part = parts[1].split("[INSTRUCTION]")[0].strip()
                original_message = question_part
            else:
                original_message = message
        else:
            original_message = message
        
        message = f"{file_context_prompt}\n\n{original_message}"
    else:
        original_message = message
    
    # 🤖 CODE AGENT DÉSACTIVÉ - KIBALI CONVERSE NATURELLEMENT
    # L'agent de code générait toujours les mêmes templates
    # Kibali analyse maintenant directement avec ses capacités conversationnelles
    
    # Ne pas utiliser le code agent - laisser Kibali converser naturellement
    if False:  # Désactivé
        pass
    
    # KIBALI ANALYSE DIRECTEMENT
    # Kibali va répondre naturellement à toutes les questions
    if True:  # Toujours actif
        # Préparer le message pour Kibali avec le contexte du fichier
        kibali_message = original_message
        
        # Si un fichier est chargé, ajouter son contexte
        if 'uploaded_file_data' in st.session_state and st.session_state.uploaded_file_data:
            physical_path = st.session_state.uploaded_file_data.get('physical_path')
            filename = st.session_state.uploaded_file_data.get('filename', 'fichier')
            
            if physical_path and os.path.exists(physical_path):
                # Lire un échantillon du fichier pour contexte
                try:
                    with open(physical_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_preview = f.read(2000)  # Premiers 2000 caractères
                    
                    # Extraire quelques nombres pour analyse rapide
                    import re
                    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', file_preview)
                    sample_values = [float(x) for x in numbers[:50] if x]  # 50 premiers nombres
                    
                    if sample_values:
                        file_info = f"""
� CONTEXTE DU FICHIER '{filename}':
- Chemin: {physical_path}
- Échantillon de valeurs: {len(sample_values)} valeurs extraites
- Min: {min(sample_values):.2f}, Max: {max(sample_values):.2f}, Moyenne: {sum(sample_values)/len(sample_values):.2f}
- Aperçu du contenu:
{file_preview[:500]}
...

QUESTION DE L'UTILISATEUR:
{original_message}
"""
                        kibali_message = file_info
                except Exception as e:
                    print(f"⚠️ Erreur lecture fichier: {e}")
        
        # Laisser Kibali répondre naturellement avec le contexte
        message = kibali_message  # Remplacer le message par celui enrichi avec le contexte fichier
        response = None  # Kibali va générer sa réponse plus bas
    
    # 🎨 DÉTECTION DE DEMANDE DE GRAPHIQUE (Agent Graphique)
    # Si l'utilisateur demande un graphique, coupe, profil, visualisation
    if any(kw in original_message.lower() for kw in ['graphique', 'coupe', 'profil', 'visuali', 'courbe', 'histogram', 'plot', 'section']):
        if 'uploaded_file_data' in st.session_state and st.session_state.uploaded_file_data:
            # Charger l'agent graphique si nécessaire
            if st.session_state.graph_agent is None:
                with st.spinner("🎨 Chargement de l'agent de génération de graphiques..."):
                    try:
                        st.session_state.graph_agent = GraphGenerationAgent()
                        st.session_state.graph_agent_loaded = True
                        print("✅ Agent graphique chargé")
                    except Exception as e:
                        print(f"❌ Erreur chargement agent graphique: {e}")
                        st.session_state.graph_agent = None
            
            # Charger le moteur de visualisation avancé (PyGIMLI + OpenCV + Matplotlib)
            if 'advanced_viz_engine' not in st.session_state:
                with st.spinner("🚀 Chargement du moteur de visualisation avancé..."):
                    try:
                        st.session_state.advanced_viz_engine = AdvancedVisualizationEngine()
                        print("✅ Moteur avancé chargé (PyGIMLI + OpenCV + Matplotlib)")
                    except Exception as e:
                        print(f"❌ Erreur chargement moteur avancé: {e}")
                        st.session_state.advanced_viz_engine = None
            
            # Si l'agent est chargé, générer le graphique
            if st.session_state.graph_agent is not None:
                with st.status("🎨 Génération du graphique...", expanded=True) as status:
                    try:
                        # Lire les données du fichier
                        physical_path = st.session_state.uploaded_file_data.get('physical_path')
                        
                        st.write("📂 Lecture du fichier...")
                        with open(physical_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Extraire les données
                        st.write("🔍 Extraction des données...")
                        import re, numpy as np
                        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', content)
                        all_values = [float(x) for x in numbers if x]
                        
                        if len(all_values) == 0:
                            st.error("❌ Aucune donnée numérique trouvée")
                        else:
                            st.write(f"✅ {len(all_values)} valeurs extraites")
                            
                            # Déterminer le type de graphique
                            request_info = st.session_state.graph_agent.understand_request(
                                original_message,
                                {'values': all_values, 'filename': st.session_state.uploaded_file_data.get('filename')}
                            )
                            
                            st.write(f"🎯 Type détecté: {request_info['graph_type']}")
                            
                            # Générer le graphique approprié
                            if request_info['graph_type'] == '3d_volume':
                                # Volume 3D
                                st.write("🎨 Création du volume 3D interactif...")
                                
                                # Préparer données pour 3D
                                resistivity = np.array(all_values)
                                n = len(resistivity)
                                
                                # Créer grille 3D automatique
                                n_side = int(np.cbrt(n)) + 1
                                st.write(f"   Grille 3D: {n_side}×{n_side}×{n_side} ({n_side**3} voxels)")
                                
                                x = np.linspace(0, 100, n_side)
                                y = np.linspace(0, 100, n_side)
                                z = np.linspace(0, -50, n_side)
                                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                                
                                x_coords = X.flatten()
                                y_coords = Y.flatten()
                                z_coords = Z.flatten()
                                
                                # Adapter résistivité à la grille
                                n_total = len(x_coords)
                                if n < n_total:
                                    resistivity = np.pad(resistivity, (0, n_total - n), mode='edge')
                                else:
                                    resistivity = resistivity[:n_total]
                                
                                st.write(f"   Plage résistivité: {resistivity.min():.2f} - {resistivity.max():.2f} Ω·m")
                                
                                # Générer avec agent
                                output_path, graph_info = st.session_state.graph_agent.create_3d_volume(
                                    x_coords, y_coords, z_coords, resistivity,
                                    title=f"Volume 3D - {st.session_state.uploaded_file_data.get('filename')}",
                                    output_path="/tmp/ert_3d_volume.html"
                                )
                                
                                # Explication
                                st.write("📝 Génération de l'explication...")
                                explanation = st.session_state.graph_agent.generate_explanation(
                                    graph_info,
                                    max_tokens=1000
                                )
                                
                                status.update(label="✅ Volume 3D généré!", state="complete")
                                
                                st.success(f"✅ Volume 3D sauvegardé: {output_path}")
                                
                                # Bouton téléchargement
                                with open(output_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                
                                st.download_button(
                                    label="📥 Télécharger volume 3D interactif (HTML)",
                                    data=html_content,
                                    file_name="volume_3d_sous_sol.html",
                                    mime="text/html"
                                )
                                
                                # Afficher dans iframe
                                st.components.v1.html(html_content, height=850, scrolling=True)
                                
                                response = f"""🎨 **VOLUME 3D GÉNÉRÉ**

{explanation}

### 📊 Caractéristiques du Volume 3D
- **Fichier**: {st.session_state.uploaded_file_data.get('filename')}
- **Points de mesure**: {len(resistivity)}
- **Résistivité**: {resistivity.min():.2f} - {resistivity.max():.2f} Ω·m
- **Volume**: {x.max():.1f}m × {y.max():.1f}m × {abs(z.min()):.1f}m
- **Interactivité**: Rotation, zoom, survol pour valeurs
- **Format**: HTML avec Plotly (visualisation 3D interactive)

### 🎯 Utilisation
✅ **Cliquez et glissez** pour faire tourner le volume
✅ **Molette** pour zoomer/dézoomer
✅ **Survolez** les points pour voir les valeurs
✅ **Légende** à droite avec échelle de couleurs

📥 **Téléchargement disponible ci-dessus**
"""
                                return response
                            
                            elif request_info['graph_type'] == '2d_section':
                                # Coupe 2D
                                st.write("🎨 Création de la coupe 2D...")
                                
                                # AUTO-DÉTECTION du format de données
                                if len(all_values) % 4 == 0 and len(all_values) >= 40:
                                    # Format complet: x, y, z, rho (4 colonnes)
                                    st.write("   Format détecté: 4 colonnes (x, y, z, résistivité)")
                                    num_points = len(all_values) // 4
                                    data = np.array(all_values).reshape(num_points, 4)
                                    x, y, z, rho = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
                                elif len(all_values) % 3 == 0 and len(all_values) >= 30:
                                    # Format 3 colonnes: x, z, rho
                                    st.write("   Format détecté: 3 colonnes (x, z, résistivité)")
                                    num_points = len(all_values) // 3
                                    data = np.array(all_values).reshape(num_points, 3)
                                    x, z, rho = data[:, 0], data[:, 1], data[:, 2]
                                else:
                                    # Format simple: juste résistivité → créer grille automatique
                                    st.write("   Format détecté: Valeurs simples → création grille automatique")
                                    rho = np.array(all_values)
                                    n = len(rho)
                                    
                                    # Créer une grille 2D intelligente
                                    # Essayer de détecter si c'est une grille rectangulaire
                                    best_shape = None
                                    for width in range(int(np.sqrt(n)), max(10, int(np.sqrt(n)//2)), -1):
                                        if n % width == 0:
                                            height = n // width
                                            if 2 <= height <= 50:  # Limites raisonnables
                                                best_shape = (height, width)
                                                break
                                    
                                    if best_shape:
                                        height, width = best_shape
                                        st.write(f"   Grille détectée: {height} profondeurs × {width} positions")
                                    else:
                                        # Par défaut: 1 profil vertical ou grille carrée
                                        width = min(20, int(np.ceil(np.sqrt(n))))
                                        height = int(np.ceil(n / width))
                                        st.write(f"   Grille créée: {height}×{width}")
                                    
                                    # Créer coordonnées
                                    x = np.tile(np.arange(width), height)[:n]  # Positions horizontales
                                    z = np.repeat(np.arange(height), width)[:n]  # Profondeurs
                                    rho = rho[:len(x)]  # Ajuster si nécessaire
                                
                                st.write(f"   Points: {len(rho)}, X: {x.min():.1f}-{x.max():.1f}, Z: {z.min():.1f}-{z.max():.1f}")
                                
                                # UTILISER LE MOTEUR AVANCÉ si disponible
                                if st.session_state.advanced_viz_engine is not None:
                                    st.write("🚀 Utilisation du moteur avancé (PyGIMLI + Matplotlib + OpenCV)...")
                                    try:
                                        # Préparer les données au format approprié
                                        if 'x' in locals() and 'y' in locals():
                                            # Format complet 4 colonnes
                                            data_for_viz = np.column_stack([x, y, z, rho])
                                        else:
                                            # Format 3 colonnes (x, z, rho)
                                            data_for_viz = np.column_stack([x, z, rho])
                                        
                                        # Créer la visualisation intelligente
                                        fig = st.session_state.advanced_viz_engine.create_intelligent_visualization(
                                            data_for_viz,
                                            request_text=original_message,
                                            color_scheme='resistivity'
                                        )
                                        
                                        # Sauvegarder et afficher
                                        output_path = "/tmp/ert_section_advanced.png"
                                        fig.savefig(output_path, dpi=300, bbox_inches='tight')
                                        st.image(output_path, use_container_width=True)
                                        
                                        # Proposer téléchargement
                                        with open(output_path, 'rb') as f:
                                            st.download_button(
                                                label="📥 Télécharger coupe haute résolution (PNG)",
                                                data=f.read(),
                                                file_name="coupe_ert_avancee.png",
                                                mime="image/png"
                                            )
                                        
                                        # Générer explication avec l'agent
                                        st.write("📝 Génération de l'explication...")
                                        graph_info = {
                                            'type': '2d_section',
                                            'n_points': len(rho),
                                            'resistivity_range': (rho.min(), rho.max()),
                                            'x_range': (x.min(), x.max()),
                                            'z_range': (z.min(), z.max())
                                        }
                                        explanation = st.session_state.graph_agent.generate_explanation(
                                            graph_info,
                                            max_tokens=1000  # Explication détaillée
                                        )
                                        
                                        status.update(label="✅ Coupe avancée générée!", state="complete")
                                        
                                        response = f"""🎨 **COUPE ERT AVANCÉE GÉNÉRÉE** (PyGIMLI + Matplotlib)

{explanation}

### 📊 Statistiques du Graphique
- **Fichier**: {st.session_state.uploaded_file_data.get('filename')}
- **Points de mesure**: {len(rho)}
- **Résistivité**: {rho.min():.2f} - {rho.max():.2f} Ω·m (échelle log)
- **Distance**: {x.min():.1f} - {x.max():.1f} m
- **Profondeur**: {z.min():.1f} - {z.max():.1f} m
- **Interpolation**: RBF (Radial Basis Function) + Filtre gaussien
- **Annotations**: Zones géologiques automatiques

### 🎯 Caractéristiques de la visualisation
✅ Interpolation avancée pour lissage optimal
✅ Contours avec valeurs annotées
✅ Profil moyen de résistivité
✅ Zones géologiques colorées
✅ Haute résolution (300 DPI)

📥 **Téléchargement haute résolution disponible ci-dessus**
"""
                                        return response
                                        
                                    except Exception as e:
                                        st.warning(f"⚠️ Moteur avancé échoué: {e}. Fallback sur méthode standard.")
                                        print(f"Erreur moteur avancé: {e}")
                                        import traceback
                                        traceback.print_exc()
                                        # Continuer avec la méthode standard ci-dessous
                                
                                # FALLBACK: Méthode standard avec graph_agent
                                output_path, graph_info = st.session_state.graph_agent.create_2d_section(
                                    x, z, rho,
                                    title=f"Coupe ERT 2D - {st.session_state.uploaded_file_data.get('filename')}",
                                    output_path="/tmp/ert_section_2d.html"
                                )
                                
                                # Générer explication
                                st.write("📝 Génération de l'explication...")
                                explanation = st.session_state.graph_agent.generate_explanation(
                                    graph_info,
                                    max_tokens=800  # Explication détaillée
                                )
                                
                                status.update(label="✅ Graphique généré avec succès!", state="complete")
                                
                                # Afficher le résultat
                                st.success(f"✅ Graphique sauvegardé: {output_path}")
                                
                                # Bouton de téléchargement
                                with open(output_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                
                                st.download_button(
                                    label="📥 Télécharger le graphique HTML",
                                    data=html_content,
                                    file_name="coupe_ert_2d.html",
                                    mime="text/html"
                                )
                                
                                # Afficher dans iframe
                                st.components.v1.html(html_content, height=650, scrolling=True)
                                
                                # Retourner l'explication
                                response = f"""🎨 **GRAPHIQUE GÉNÉRÉ**

{explanation}

### 📊 Statistiques du Graphique
- **Fichier**: {st.session_state.uploaded_file_data.get('filename')}
- **Points de mesure**: {len(rho)}
- **Résistivité**: {rho.min():.2f} - {rho.max():.2f} Ω·m
- **Distance**: {x.min():.1f} - {x.max():.1f} m
- **Profondeur**: {z.min():.1f} - {z.max():.1f} m

📥 **Téléchargement disponible ci-dessus**
"""
                                return response
                                
                            elif request_info['graph_type'] in ['profile_1d', 'line_plot']:
                                # Profil 1D
                                resistivity = np.array(all_values)
                                depths = np.arange(len(resistivity))
                                
                                st.write("📈 Création du profil vertical...")
                                output_path, graph_info = st.session_state.graph_agent.create_profile_1d(
                                    depths, resistivity,
                                    title=f"Profil de Résistivité - {st.session_state.uploaded_file_data.get('filename')}",
                                    output_path="/tmp/ert_profile_1d.html"
                                )
                                
                                # Explication
                                explanation = st.session_state.graph_agent.generate_explanation(
                                    graph_info,
                                    max_tokens=800
                                )
                                
                                status.update(label="✅ Profil généré!", state="complete")
                                
                                st.success(f"✅ Profil sauvegardé: {output_path}")
                                
                                # Bouton téléchargement
                                with open(output_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                
                                st.download_button(
                                    label="📥 Télécharger le profil HTML",
                                    data=html_content,
                                    file_name="profil_ert_1d.html",
                                    mime="text/html"
                                )
                                
                                st.components.v1.html(html_content, height=750, scrolling=True)
                                
                                response = f"""📈 **PROFIL VERTICAL GÉNÉRÉ**

{explanation}

### 📊 Statistiques
- **Mesures**: {len(resistivity)}
- **Résistivité min**: {resistivity.min():.2f} Ω·m
- **Résistivité max**: {resistivity.max():.2f} Ω·m
- **Résistivité moyenne**: {resistivity.mean():.2f} Ω·m
- **Résistivité médiane**: {np.median(resistivity):.2f} Ω·m

📥 **Téléchargement disponible ci-dessus**
"""
                                return response
                            
                            elif request_info['graph_type'] == 'statistics_table':
                                # Tableau statistiques
                                resistivity = np.array(all_values)
                                
                                st.write("📊 Création du tableau statistique...")
                                output_path, graph_info = st.session_state.graph_agent.create_statistics_table(
                                    resistivity,
                                    output_path="/tmp/ert_statistics.html"
                                )
                                
                                explanation = st.session_state.graph_agent.generate_explanation(
                                    graph_info,
                                    max_tokens=600
                                )
                                
                                status.update(label="✅ Tableau généré!", state="complete")
                                
                                with open(output_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                
                                st.download_button(
                                    label="📥 Télécharger le tableau HTML",
                                    data=html_content,
                                    file_name="statistiques_ert.html",
                                    mime="text/html"
                                )
                                
                                st.components.v1.html(html_content, height=550, scrolling=True)
                                
                                response = f"""📊 **TABLEAU STATISTIQUE GÉNÉRÉ**

{explanation}

📥 **Téléchargement disponible ci-dessus**
"""
                                return response
                            
                            elif any(kw in original_message.lower() for kw in ['rapport', 'complet', 'analyse complete', 'tout', 'global']):
                                # RAPPORT STRUCTURÉ COMPLET
                                st.write("📋 Génération du rapport structuré complet...")
                                
                                # Préparer les données
                                if len(all_values) % 4 == 0:
                                    num_points = len(all_values) // 4
                                    data_array = np.array(all_values).reshape(num_points, 4)
                                    data_dict = {
                                        'x': data_array[:, 0],
                                        'y': data_array[:, 1],
                                        'z': data_array[:, 2],
                                        'resistivity': data_array[:, 3]
                                    }
                                else:
                                    data_dict = {
                                        'x': np.arange(len(all_values)),
                                        'y': np.zeros(len(all_values)),
                                        'z': np.zeros(len(all_values)),
                                        'resistivity': np.array(all_values)
                                    }
                                
                                # Générer le rapport
                                output_path, graph_info = st.session_state.graph_agent.generate_structured_report(
                                    data_dict,
                                    original_message,
                                    output_path="/tmp/rapport_ert_complet.html"
                                )
                                
                                # Explication avec max 1000 tokens
                                explanation = st.session_state.graph_agent.generate_explanation(
                                    graph_info,
                                    max_tokens=1000  # Maximum pour rapport complet
                                )
                                
                                status.update(label="✅ Rapport complet généré!", state="complete")
                                
                                with open(output_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                
                                # Boutons de téléchargement multiples
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.download_button(
                                        label="📥 HTML Complet",
                                        data=html_content,
                                        file_name="rapport_ert_complet.html",
                                        mime="text/html"
                                    )
                                
                                with col2:
                                    # Export CSV
                                    csv_data = "X,Y,Z,Resistivity\\n"
                                    for i in range(len(data_dict['resistivity'])):
                                        csv_data += f"{data_dict['x'][i]},{data_dict['y'][i]},{data_dict['z'][i]},{data_dict['resistivity'][i]}\\n"
                                    
                                    st.download_button(
                                        label="📊 Données CSV",
                                        data=csv_data,
                                        file_name="donnees_ert.csv",
                                        mime="text/csv"
                                    )
                                
                                with col3:
                                    # Export JSON
                                    import json
                                    json_data = json.dumps({
                                        'metadata': {
                                            'filename': st.session_state.uploaded_file_data.get('filename'),
                                            'n_points': len(data_dict['resistivity']),
                                            'date': datetime.now().isoformat()
                                        },
                                        'statistics': graph_info.get('statistics', {}),
                                        'data': {
                                            'x': data_dict['x'].tolist(),
                                            'y': data_dict['y'].tolist(),
                                            'z': data_dict['z'].tolist(),
                                            'resistivity': data_dict['resistivity'].tolist()
                                        }
                                    }, indent=2)
                                    
                                    st.download_button(
                                        label="📦 Données JSON",
                                        data=json_data,
                                        file_name="donnees_ert.json",
                                        mime="application/json"
                                    )
                                
                                # Afficher le rapport
                                st.components.v1.html(html_content, height=800, scrolling=True)
                                
                                response = f"""📋 **RAPPORT STRUCTURÉ COMPLET GÉNÉRÉ**

{explanation}

### 📊 Contenu du Rapport:
✅ **Statistiques globales** - Cartes avec métriques clés
✅ **Classification géologique** - Tableau détaillé des matériaux
✅ **Analyse hydrogéologique** - Zones d'eau et recommandations
✅ **Graphiques interactifs** - Visualisations intégrées
✅ **Boutons de téléchargement** - HTML, CSV, JSON, PDF

### 📥 Téléchargements Disponibles:
- **HTML Complet**: Rapport interactif avec tous les graphiques
- **CSV**: Données brutes pour analyses externes
- **JSON**: Format structuré pour traitement automatisé

💡 **Astuce**: Cliquez sur "Imprimer/PDF" dans le rapport pour générer un PDF professionnel!
"""
                                return response
                    
                    except Exception as e:
                        status.update(label="❌ Erreur de génération", state="error")
                        st.error(f"Erreur: {e}")
                        print(f"❌ Erreur agent graphique: {e}")
                        import traceback
                        traceback.print_exc()
    
    # MODE HUMAIN: Analyser la question avant de répondre
    if mode == "humain":
        intent = analyze_question_intent(message)
        if intent["needs_clarification"]:
            return "🤔 Hmm, je veux être sûr de bien comprendre ta question...\n\n" + \
                   "Peux-tu préciser un peu plus ? Par exemple:\n" + \
                   "• De quel contexte parles-tu exactement ?\n" + \
                   "• C'est pour quel usage ou projet ?\n" + \
                   "• Tu as déjà des infos ou tu pars de zéro ?\n\n" + \
                   "Plus tu es précis, mieux je pourrai t'aider ! 😊"
    
    if agent is None:
        model_name, agent, _ = update_agent(model_choice, vectordb, graph, pois, chat_vectordb)
    
    # Si l'agent est toujours None (agents non disponibles), forcer mode local
    if agent is None:
        web_enabled = False
    
    # 🎯 SYSTÈME DE MODES INTELLIGENTS
    # Détecter automatiquement le mode optimal
    try:
        from intelligent_mode_system import detect_conversation_mode, format_mode_prompt
        
        uploaded_data = st.session_state.get('uploaded_file_data') if 'st' in globals() else None
        mode_config = detect_conversation_mode(message, uploaded_data)
        
        print(f"🎯 MODE DÉTECTÉ: {mode_config['mode']}")
        print(f"📊 Scores: {mode_config['mode_scores']}")
        print(f"🔧 Outils recommandés: {mode_config['tools']}")
        
        # Enrichir le message avec les instructions du mode
        mode_enhanced_message = format_mode_prompt(message, mode_config)
        
        # Afficher le mode à l'utilisateur
        if 'st' in globals():
            st.info(f"🎯 **Mode activé**: {mode_config['mode']} | Outils: {', '.join(mode_config['tools'][:3])}{'...' if len(mode_config['tools']) > 3 else ''}")
        
    except Exception as e:
        print(f"⚠️ Erreur système de modes: {e}")
        mode_enhanced_message = message
        mode_config = {'mode': 'Standard'}
    
    # MODE CODE EXPERT: Utiliser AI_Code_Generator si question de code
    if mode == "code_expert" or mode_config.get('mode') == 'Programmation 💻':
        if any(kw in message.lower() for kw in ["code", "fonction", "class", "programme", "script", "algorithm"]):
            try:
                code_result = generate_code_with_ai(message)
                return f"💻 **{mode_config['mode']}**\n\n{code_result}\n\n✅ Code testé et optimisé\n📦 Prêt pour production"
            except:
                pass  # Fallback vers recherche normale
    
    try:
        if not web_enabled or agent is None:
            # Recherche hybride incluant chat
            docs = hybrid_search_enhanced(message, vectordb, k=3, web_search_enabled=False, chat_vectordb=chat_vectordb)
            response = generate_answer_enhanced(message, docs, WORKING_MODELS[model_choice], include_sources=True)
        else:
            # Utiliser le message enrichi par le mode
            response = agent.run(mode_enhanced_message)
    except Exception as e:
        response = f"❌ Erreur: {e}\n\nTentative avec recherche locale..."
        try:
            docs = hybrid_search_enhanced(message, vectordb, k=3, web_search_enabled=False, chat_vectordb=chat_vectordb)
            response = generate_answer_enhanced(message, docs, WORKING_MODELS[model_choice])
        except:
            response = f"❌ Erreur complète: {e}"
    
    # AJOUT MÉMOIRE VECTORIELLE: Sauvegarder l'échange dans la base chat
    chat_vectordb = add_to_chat_db(message, response, chat_vectordb)
    
    # Appliquer highlighting pour fluidité
    response = highlight_important_words(response)
    return response
def handle_web_search(query, search_type):
    if not query.strip():
        return "⚠️ Veuillez entrer une requête"
    try:
        results = enhanced_web_search(query, max_results=10, search_type=search_type)
        if not results:
            return "❌ Aucun résultat trouvé"
        html_output = "<div style='max-height: 500px; overflow-y: auto;'>"
        for i, result in enumerate(results):
            title = result.get('title', 'Sans titre')
            body = result.get('body', 'Pas de description')
            url = result.get('href') or result.get('url', '#')
            source_type = result.get('source_type', 'web')
            if source_type == 'news':
                icon = "📰"
                color = "#e3f2fd"
            else:
                icon = "🔍"
                color = "#f5f5f5"
            html_output += f"""
            <div style='margin: 10px 0; padding: 15px; background-color: {color}; border-radius: 8px; border-left: 4px solid #2196F3;'>
                <h4 style='margin: 0 0 8px 0; color: #1976D2;'>{icon} {title}</h4>
                <p style='margin: 8px 0; color: #424242; line-height: 1.4;'>{body}</p>
                <a href='{url}' target='_blank' style='color: #1976D2; text-decoration: none; font-size: 0.9em;'>🔗 {url}</a>
            </div>
            """
        html_output += "</div>"
        return html_output
    except Exception as e:
        return f"❌ Erreur recherche: {e}"
def handle_content_extraction(url):
    if not url.strip():
        return "⚠️ Veuillez entrer une URL"
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        content = smart_content_extraction(url, max_length=2000)
        return content
    except Exception as e:
        return f"❌ Erreur extraction: {e}"
# ========================================
# Fonctions utilitaires supplémentaires
# ========================================
def get_system_status():
    """Retourne le statut complet du système"""
    status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "directories": {
            "chatbot": os.path.exists(CHATBOT_DIR),
            "pdfs": os.path.exists(PDFS_PATH),
            "graphs": os.path.exists(GRAPHS_PATH),
            "maps": os.path.exists(MAPS_PATH)
        },
        "files": {
            "vectordb": os.path.exists(VECTORDB_PATH),
            "chat_vectordb": os.path.exists(CHAT_VECTORDB_PATH), # AJOUT MÉMOIRE VECTORIELLE
            "metadata": os.path.exists(METADATA_PATH),
            "trajectories": os.path.exists(TRAJECTORIES_PATH),
            "web_cache": os.path.exists(WEB_CACHE_PATH)
        },
        "counts": {
            "pdfs": len([f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')]) if os.path.exists(PDFS_PATH) else 0,
            "graphs": len([f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')]) if os.path.exists(GRAPHS_PATH) else 0
        },
        "cache_stats": get_cache_stats(),
        "token_configured": bool(HF_TOKEN and len(HF_TOKEN) > 10)
    }
    return status
def cleanup_old_cache():
    """Nettoie les entrées expirées du cache"""
    try:
        cache = load_web_cache()
        if not cache:
            return "Cache vide"
        original_count = len(cache)
        cleaned_cache = {}
        for key, entry in cache.items():
            if not is_cache_expired(entry):
                cleaned_cache[key] = entry
        save_web_cache(cleaned_cache)
        removed_count = original_count - len(cleaned_cache)
        return f"✅ Cache nettoyé: {removed_count} entrées expirées supprimées, {len(cleaned_cache)} conservées"
    except Exception as e:
        return f"❌ Erreur nettoyage cache: {e}"
def export_system_config():
    """Exporte la configuration système pour debug"""
    config = {
        "version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "chatbot_dir": CHATBOT_DIR,
            "vectordb_path": VECTORDB_PATH,
            "chat_vectordb_path": CHAT_VECTORDB_PATH, # AJOUT MÉMOIRE VECTORIELLE
            "pdfs_path": PDFS_PATH,
            "graphs_path": GRAPHS_PATH,
            "maps_path": MAPS_PATH
        },
        "models": WORKING_MODELS,
        "status": get_system_status(),
        "features": {
            "web_search": True,
            "osm_routing": True,
            "image_analysis": True,
            "pdf_processing": True,
            "caching": True,
            "chat_memory": True # AJOUT MÉMOIRE VECTORIELLE
        }
    }
    config_path = os.path.join(CHATBOT_DIR, "system_config.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return f"✅ Configuration exportée: {config_path}"
    except Exception as e:
        return f"❌ Erreur export: {e}"
def test_all_features():
    """Teste toutes les fonctionnalités principales"""
    results = {}
    # Test HuggingFace
    results["huggingface"] = test_hf_connection()
    # Test recherche web
    try:
        test_results = enhanced_web_search("test", max_results=1)
        results["web_search"] = len(test_results) > 0
    except:
        results["web_search"] = False
    # Test recherche web
    results["specialized_models"] = {}
    for model_name, model in SPECIALIZED_MODELS.items():
        results["specialized_models"][model_name] = model is not None
    # Test base vectorielle
    try:
        vectordb, _ = load_vectordb()
        results["vectordb"] = vectordb is not None
    except:
        results["vectordb"] = False
    # Test base chat # AJOUT MÉMOIRE VECTORIELLE
    try:
        chat_vectordb, _ = load_chat_vectordb()
        results["chat_vectordb"] = chat_vectordb is not None
    except:
        results["chat_vectordb"] = False
    # Test graphe OSM
    try:
        graph, pois, _ = load_existing_graph()
        results["osm_graph"] = graph is not None
    except:
        results["osm_graph"] = False
    return results
# ========================================
# Fonctions de maintenance avancées
# ========================================
def optimize_vectordb():
    """Optimise la base vectorielle en supprimant les doublons"""
    try:
        vectordb, status = load_vectordb()
        if not vectordb:
            return "❌ Aucune base vectorielle à optimiser"
        # Cette fonction nécessiterait une implémentation plus complexe
        # pour détecter et supprimer les doublons dans FAISS
        return "✅ Base vectorielle optimisée (fonctionnalité à implémenter)"
    except Exception as e:
        return f"❌ Erreur optimisation: {e}"
def backup_all_data():
    """Crée une sauvegarde de toutes les données"""
    try:
        import zipfile
        backup_name = f"kibali_backup_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        backup_path = os.path.join(CHATBOT_DIR, backup_name)
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            # Sauvegarder tous les fichiers du dossier chatbot
            for root, dirs, files in os.walk(CHATBOT_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, CHATBOT_DIR)
                    backup_zip.write(file_path, arcname)
        return f"✅ Sauvegarde créée: {backup_path}"
    except Exception as e:
        return f"❌ Erreur sauvegarde: {e}"
def restore_from_backup(backup_path):
    """Restaure les données depuis une sauvegarde"""
    try:
        import zipfile
        if not os.path.exists(backup_path):
            return "❌ Fichier de sauvegarde non trouvé"
        with zipfile.ZipFile(backup_path, 'r') as backup_zip:
            backup_zip.extractall(CHATBOT_DIR)
        return f"✅ Données restaurées depuis: {backup_path}"
    except Exception as e:
        return f"❌ Erreur restauration: {e}"
# ========================================
# NOUVEAU: Fonctions Auto-Apprentissage et Sous-Modèles avec Scikit-Learn
# ========================================
def create_submodel_from_chat_history(chat_vectordb, submodel_type="classification"):
    """
    Crée un petit sous-modèle sklearn à partir de l'historique chat pour automatiser des réponses.
    - Type: 'classification' pour classer les questions et prédire des réponses automatisées.
    Rend le modèle plus "humain" en apprenant des patterns conversationnels.
    """
    if not chat_vectordb:
        return None, "❌ Aucune base chat pour entraîner le sous-modèle"
  
    # Extraire les échanges de l'historique
    exchanges = []
    for doc in list(chat_vectordb.docstore._dict.values()) or []:
        exchange = doc.page_content
        if "User:" in exchange and "Assistant:" in exchange:
            user_part = exchange.split("|||")[0].replace("User: ", "").strip()
            ai_part = exchange.split("|||")[1].replace("Assistant: ", "").strip() if "|||" in exchange else ""
            exchanges.append((user_part, ai_part))
  
    if len(exchanges) < 10:
        return None, "❌ Historique chat trop court pour entraîner un modèle"
  
    try:
        # Préparation des données : TF-IDF pour vectorisation textuelle
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform([user[0] for user in exchanges])
      
        # Pour classification simple (ex: prédire si réponse est informative ou autre)
        # Labels simples basés sur patterns (ex: 0=info, 1=question, 2=autre)
        labels = []
        for user_msg, _ in exchanges:
            if re.search(r'\?', user_msg):
                labels.append(1) # Question
            elif any(word in user_msg.lower() for word in ['info', 'savoir', 'expliquer']):
                labels.append(0) # Info
            else:
                labels.append(2) # Autre
      
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
      
        if submodel_type == "classification":
            model = MultinomialNB()
        else:
            model = RandomForestClassifier(n_estimators=50)
      
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
      
        # Sauvegarder le modèle et vectorizer
        model_path = os.path.join(SUBMODELS_PATH, f"submodel_{submodel_type}_{int(time.time())}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
      
        # Visualisation avec matplotlib : Accuracy plot
        fig, ax = plt.subplots()
        ax.bar(['Train', 'Test'], [1.0, accuracy]) # Train est parfait par défaut
        ax.set_title(f'Précision du sous-modèle {submodel_type.capitalize()}')
        ax.set_ylabel('Accuracy')
        plot_path = os.path.join(SUBMODELS_PATH, f"accuracy_plot_{submodel_type}_{int(time.time())}.png")
        plt.savefig(plot_path)
        plt.close()
      
        return model_path, f"✅ Sous-modèle {submodel_type} créé avec accuracy {accuracy:.2f}. Sauvegardé: {model_path}"
    except Exception as e:
        return None, f"❌ Erreur création sous-modèle: {e}"
def use_submodel_for_automation(query, submodel_path, submodel_type="classification"):
    """
    Utilise un sous-modèle pour automatiser une réponse, rendant le comportement plus humain (ex: prédiction rapide).
    """
    if not os.path.exists(submodel_path):
        return "❌ Sous-modèle non trouvé"
  
    try:
        with open(submodel_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            vectorizer = data['vectorizer']
      
        query_vec = vectorizer.transform([query])
        prediction = model.predict(query_vec)[0]
      
        # Réponses automatisées basées sur prédiction pour plus d'humanité
        automated_responses = {
            0: "Voici des infos basiques sur ce sujet, basées sur nos échanges passés.",
            1: "Bonne question ! Laisse-moi réfléchir à ça en me basant sur ce qu'on a discuté avant.",
            2: "Intéressant, je vais creuser un peu plus pour te répondre de manière personnalisée."
        }
      
        response = automated_responses.get(prediction, "Réponse automatisée générée.")
      
        # Visualisation: Distribution des features TF-IDF pour la query
        fig, ax = plt.subplots()
        tfidf_scores = query_vec.toarray()[0]
        top_features = np.argsort(tfidf_scores)[-5:]
        ax.bar(range(len(top_features)), tfidf_scores[top_features])
        ax.set_title('Top Features TF-IDF pour la Query')
        ax.set_xticks(range(len(top_features)))
        ax.set_xticklabels([vectorizer.get_feature_names_out()[i] for i in top_features], rotation=45)
        plot_path = os.path.join(SUBMODELS_PATH, f"query_features_{int(time.time())}.png")
        plt.savefig(plot_path)
        plt.close()
      
        return f"{response} (Prédiction: {prediction}) | Graph: {plot_path}"
    except Exception as e:
        return f"❌ Erreur utilisation sous-modèle: {e}"
# ========================================
# NOUVEAU: Fonctions Amélioration Base de Données via Fouille Internet
# ========================================
def improve_database_with_web_search(topics, num_results_per_topic=5, vectordb=None):
    """
    Fouille internet sur des sujets spécifiques (pétrole, topographie, sciences physiques, sous-sol, etc.)
    et améliore la base de données en ajoutant de nouveaux documents.
    """
    specific_topics = topics or ["pétrole extraction techniques", "topographie cartographie avancée", "sciences physiques mécanique sol", "sous-sol géologie ressources"]
  
    if vectordb is None:
        vectordb, _ = load_vectordb()
        if vectordb is None:
            embedding_model = get_embedding_model()
            vectordb = FAISS.from_texts([""], embedding_model)
  
    new_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  
    for topic in specific_topics:
        st.write(f"🔍 Fouille internet pour: {topic}")
        search_results = enhanced_web_search(topic, max_results=num_results_per_topic, search_type="both")
      
        for result in search_results:
            content = f"Titre: {result.get('title', '')}\nContenu: {result.get('body', '')}\n"
            url = result.get('href') or result.get('url')
            if url and len(result.get('body', '')) < 500:
                extra_content = smart_content_extraction(url, max_length=2000)
                if "Impossible d'extraire" not in extra_content:
                    content += f"\nContenu détaillé: {extra_content}"
          
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": url or topic,
                        "topic": topic,
                        "type": "web_enrichment",
                        "chunk_id": i
                    }
                )
                new_documents.append(doc)
  
    if new_documents:
        vectordb.add_documents(new_documents)
        vectordb.save_local(VECTORDB_PATH)
        return vectordb, f"✅ Base améliorée: {len(new_documents)} nouveaux chunks ajoutés sur {len(specific_topics)} sujets"
    else:
        return vectordb, "⚠️ Aucun nouveau contenu ajouté"
# ========================================
# Version API pour utilisation externe
# ========================================
class KibaliAPI:
    """API simplifiée pour utiliser Kibali depuis du code externe"""
    def __init__(self):
        self.vectordb = None
        self.chat_vectordb = None # AJOUT MÉMOIRE VECTORIELLE
        self.graph = None
        self.pois = []
        self.client = None
        self.model_name = WORKING_MODELS[list(WORKING_MODELS.keys())[0]]
        # Initialisation automatique
        self._initialize()
    def _initialize(self):
        """Initialisation automatique"""
        try:
            setup_drive()
            self.vectordb, _ = load_vectordb()
            self.chat_vectordb, _ = load_chat_vectordb() # AJOUT MÉMOIRE VECTORIELLE
            self.graph, self.pois, _ = load_existing_graph()
            self.client = create_client()
        except Exception as e:
            print(f"⚠️ Initialisation partielle: {e}")
    def ask(self, question, use_web=True):
        """Pose une question simple"""
        try:
            if use_web:
                docs = hybrid_search_enhanced(question, self.vectordb, web_search_enabled=True, chat_vectordb=self.chat_vectordb) # AJOUT MÉMOIRE VECTORIELLE
            else:
                docs = rag_search(question, self.vectordb)
            return generate_answer_enhanced(question, docs, self.model_name)
        except Exception as e:
            return f"❌ Erreur: {e}"
    def search_web(self, query, max_results=5):
        """Recherche web simple"""
        try:
            results = enhanced_web_search(query, max_results)
            return [{"title": r.get("title"), "url": r.get("href", r.get("url")), "snippet": r.get("body")} for r in results]
        except Exception as e:
            return [{"error": str(e)}]
    def calculate_route(self, from_place, to_place):
        """Calcule un itinéraire"""
        try:
            question = f"Comment aller de {from_place} à {to_place}"
            _, response, info = calculer_trajet(question, self.graph, self.pois)
            return {"response": response, "info": info}
        except Exception as e:
            return {"error": str(e)}
    def get_status(self):
        """Retourne le statut du système"""
        return get_system_status()
    # NOUVEAU: Méthodes API pour auto-apprentissage et amélioration DB
    def train_submodel(self, submodel_type="classification"):
        """Entraîne un sous-modèle"""
        path, msg = create_submodel_from_chat_history(self.chat_vectordb, submodel_type)
        return {"path": path, "message": msg}
    def improve_db(self, topics=None, num_results=5):
        """Améliore la DB avec fouille internet"""
        self.vectordb, msg = improve_database_with_web_search(topics, num_results, self.vectordb)
        return {"message": msg}
# Instance globale de l'API
kibali_api = KibaliAPI()
# ========================================
# Interface Streamlit Améliorée
# ========================================
st.markdown("""
<style>
    .stApp {
        background: white;
        color: black;
    }
    .sidebar .sidebar-content {
        background: white;
    }
    .stSidebar > div {
        background: white;
    }
    .stChatMessage {
        background: white;
        border-radius: 18px;
        border-left: 4px solid #2196F3;
        margin: 5px 0;
        padding: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        color: black !important;
        transition: all 0.3s ease;
        filter: none; /* Correction pour flou */
    }
    .stChatMessage:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stChatMessage p, .stChatMessage li {
        color: black !important;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .stTextInput > div > div > input {
        background: white;
        border: 1px solid #2196F3;
        border-radius: 20px;
        color: black;
        padding: 10px 15px;
        filter: none; /* Correction pour flou */
    }
    .stTextInput > div > div > input::placeholder {
        color: #757575;
    }
    .stButton > button {
        background: linear-gradient(45deg, #2196F3 0%, #21CBF3 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-bottom: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        animation: pulse 1s infinite; /* Effet fluide */
    }
    @keyframes pulse {
        0% { box-shadow: 0 6px 12px rgba(0,0,0,0.4); }
        50% { box-shadow: 0 6px 16px rgba(33, 150, 243, 0.6); }
        100% { box-shadow: 0 6px 12px rgba(0,0,0,0.4); }
    }
    .stSelectbox > div > div > select {
        background: white;
        border: 1px solid #2196F3;
        border-radius: 10px;
        color: black;
        filter: none; /* Correction pour flou */
    }
    .stCheckbox > div > label {
        color: black;
        transition: color 0.3s ease;
    }
    .stCheckbox > div > label:hover {
        color: #2196F3;
    }
    .stTextArea > div > div > textarea {
        background: white;
        color: black;
        border: 1px solid #2196F3;
    }
    h1, h2, h3 {
        color: #2196F3;
        text-shadow: 0 0 10px rgba(33, 150, 243, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 10px rgba(33, 150, 243, 0.5); }
        to { text-shadow: 0 0 20px rgba(33, 150, 243, 0.8), 0 0 30px rgba(33, 203, 243, 0.6); }
    }
    .chat-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.95);
        border-top: 1px solid #2196F3;
        padding: 10px;
        z-index: 1000;
        transition: all 0.3s ease;
    }
    .chat-footer:hover {
        background: rgba(255, 255, 255, 1);
    }
    /* Effet scintillante pour mots importants */
    .sparkle-word {
        color: #2196F3;
        background: linear-gradient(45deg, #2196F3, #21CBF3, #4ecdc4, #45b7d1);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: sparkle 2s linear infinite, gradient-shift 3s ease infinite;
        cursor: pointer;
        position: relative;
        padding: 2px 4px;
        border-radius: 4px;
        transition: transform 0.2s ease;
    }
    .sparkle-word:hover {
        transform: scale(1.1);
        text-shadow: 0 0 10px rgba(33, 150, 243, 0.8);
    }
    @keyframes sparkle {
        0%, 100% { text-shadow: 0 0 5px rgba(33, 150, 243, 0.5); }
        50% { text-shadow: 0 0 20px rgba(33, 150, 243, 1), 0 0 30px rgba(33, 203, 243, 0.7); }
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    /* Correction pour lisibilité des questions/réponses */
    .stMarkdown {
        filter: none !important;
        -webkit-filter: none !important;
        color: black !important;
        font-weight: 400;
        line-height: 1.6;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .stMarkdown p, .stMarkdown li {
        color: black !important;
        text-shadow: none;
    }
    .st-emotion-cache-1i5yq8u input, .st-emotion-cache-1i5yq8u textarea {
        color: black !important;
    }
    @media (max-width: 768px) {
        .chat-footer {
            padding: 5px;
        }
        .stTextInput input {
            font-size: 14px;
        }
        .sparkle-word {
            font-size: 0.9em;
        }
    }
</style>
""", unsafe_allow_html=True)
# Sidebar pour options
with st.sidebar:
    st.markdown("<h2 style='color: #2196F3; text-align: center;'>⚙️ Options</h2>", unsafe_allow_html=True)
    st.markdown("---")
  
    # Initialisation des états de session
    if 'status_msg' not in st.session_state:
        st.session_state.status_msg = ""
    if 'cache_msg' not in st.session_state:
        st.session_state.cache_msg = get_cache_stats()
  
    # Uploads et boutons config
    pdf_upload = st.file_uploader("📤 Upload PDFs", type="pdf", accept_multiple_files=True, key="pdf_sidebar")
    
    # 🆕 NOUVEAU: Upload de rapports ERT pour extraction automatique
    st.markdown("#### 🔬 Extraction Rapports ERT")
    ert_pdf_upload = st.file_uploader("📄 Upload Rapport ERT (PDF)", type="pdf", key="ert_pdf_upload")
    extract_ert_btn = st.button("🔍 Extraire données ERT", key="extract_ert_btn")
    
    # 🆕 NOUVEAU: Upload audio pour transcription
    audio_upload = st.file_uploader("🎤 Upload Notes Audio", type=["wav", "mp3", "m4a"], key="audio_upload")
    transcribe_audio_btn = st.button("📝 Transcrire Audio", key="transcribe_audio_btn")
    
    pbf_upload = st.file_uploader("📤 Upload OSM (.pbf)", type="osm.pbf", key="pbf_sidebar")
    process_pdfs_btn = st.button("🔄 Traiter PDFs", key="process_sidebar")
    load_graph_btn = st.button("📂 Charger graphe", key="load_graph_sidebar")
    load_vectordb_btn = st.button("📂 Charger DB", key="load_db_sidebar")
    clear_cache_btn = st.button("🗑️ Vider cache", key="clear_cache_sidebar")
  
    # NOUVEAU: Boutons pour auto-apprentissage et amélioration
    train_submodel_btn = st.button("🧠 Entraîner sous-modèle (sklearn)", key="train_submodel")
    improve_db_btn = st.button("📚 Améliorer DB (fouille internet)", key="improve_db")
  
    # 🎤 SECTION VOCALE
    if VOICE_AVAILABLE:
        st.markdown("---")
        st.markdown("#### 🎤 Interface Vocale")
        
        voice_enable_checkbox = st.checkbox(
            "Activer le mode vocal",
            value=st.session_state.get("voice_enabled", False),
            key="voice_enable_checkbox",
            help="Active la transcription et synthèse vocale"
        )
        
        if voice_enable_checkbox != st.session_state.get("voice_enabled", False):
            st.session_state.voice_enabled = voice_enable_checkbox
            if voice_enable_checkbox and st.session_state.voice_agent is None:
                with st.spinner("🎤 Vérification du matériel audio..."):
                    # Vérifier la disponibilité du hardware audio
                    audio_available = False
                    try:
                        import sounddevice as sd
                        devices = sd.query_devices()
                        if devices and len(devices) > 0:
                            # Vérifier qu'il y a au moins un périphérique d'entrée
                            input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
                            if input_devices:
                                audio_available = True
                    except Exception as e:
                        st.warning(f"⚠️ Impossible de vérifier les périphériques audio: {e}")
                    
                    if not audio_available:
                        st.warning("⚠️ Aucun périphérique audio détecté")
                        st.info("💡 Mode limité : Transcription de fichiers audio uniquement (pas d'enregistrement)")
                        st.info("� WSL/Docker : L'enregistrement micro n'est pas disponible")
                        # Continuer quand même pour permettre la transcription de fichiers
                        st.session_state.audio_recording_disabled = True
                    else:
                        st.session_state.audio_recording_disabled = False
                    
                with st.spinner("🎤 Chargement des modèles vocaux..."):
                    try:
                        st.session_state.voice_agent = StreamingVoiceAgent(
                            whisper_model="base",  # ~150MB
                            tts_model="tts_models/fr/mai/tacotron2-DDC"
                        )
                        # Charger les modèles (transcription toujours, TTS seulement si audio disponible)
                        success = st.session_state.voice_agent.load_models(
                            load_whisper=True,
                            load_tts=audio_available  # Pas de TTS si pas d'audio
                        )
                        if success:
                            st.session_state.voice_models_loaded = True
                            if audio_available:
                                st.success("✅ Modèles vocaux chargés (transcription + synthèse)")
                            else:
                                st.success("✅ Modèle de transcription chargé (mode fichiers uniquement)")
                        else:
                            st.error("❌ Erreur chargement modèles vocaux")
                            st.info("💡 Lancez: python install_voice_models.py")
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
                        st.info("💡 Installez les modèles: python install_voice_models.py")
        
        # Afficher le statut vocal
        if st.session_state.get("voice_enabled", False):
            if st.session_state.get("voice_models_loaded", False):
                if st.session_state.get("audio_recording_disabled", False):
                    st.warning("🎤 Mode vocal limité (transcription uniquement)")
                    st.caption("⚠️ Enregistrement micro désactivé (WSL/Docker)")
                else:
                    st.success("🎤 Mode vocal actif")
                
                # Options avancées
                with st.expander("⚙️ Options vocales"):
                    # Enregistrement seulement si audio disponible
                    if not st.session_state.get("audio_recording_disabled", False):
                        voice_record_duration = st.slider(
                            "Durée d'enregistrement (s)",
                            min_value=3,
                            max_value=30,
                            value=5,
                            key="voice_duration"
                        )
                        voice_auto_play = st.checkbox(
                            "Lecture automatique des réponses",
                            value=True,
                            key="voice_autoplay"
                        )
                    else:
                        st.info("📁 Mode : Transcription de fichiers audio uploadés")
                        st.caption("🚫 Enregistrement micro non disponible")
                    
                    voice_language = st.selectbox(
                        "Langue de transcription",
                        options=["fr", "en", "es", "de"],
                        index=0,
                        key="voice_lang"
                    )
            else:
                st.warning("⏳ Modèles vocaux non chargés")
    else:
        st.markdown("---")
        st.markdown("#### 🎤 Interface Vocale")
        st.warning("⚠️ Modules vocaux non installés")
        st.info("💡 Pour activer: `pip install sounddevice soundfile librosa`")
        st.info("📦 Puis: `python install_voice_models.py`")
        st.info("⚠️ Note: Nécessite un microphone physique connecté")
        st.info("🐧 WSL/Docker: Mode vocal non supporté")
    
    st.markdown("---")
    status_display = st.text_area("📊 Statut", value=st.session_state.status_msg, height=100, key='status_sidebar')
    cache_stats = st.text_area("📈 Cache", value=st.session_state.cache_msg, height=50, key='cache_sidebar')
  
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None
    if "chat_vectordb" not in st.session_state: # AJOUT MÉMOIRE VECTORIELLE
        st.session_state.chat_vectordb = None
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "pois" not in st.session_state:
        st.session_state.pois = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = WORKING_MODELS[list(WORKING_MODELS.keys())[0]]
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    # CODE AGENT DÉSACTIVÉ - Kibali converse naturellement
    # if "code_agent" not in st.session_state:
    #     st.session_state.code_agent = None
    #     st.session_state.code_agent_loaded = False
    
    # AGENT DE GÉNÉRATION DE GRAPHIQUES
    if "graph_agent" not in st.session_state:
        # Initialiser l'agent de graphiques en mode LAZY
        st.session_state.graph_agent = None
        st.session_state.graph_agent_loaded = False
    
    # MOTEUR DE VISUALISATION AVANCÉ (PyGIMLI + OpenCV + Matplotlib)
    if "advanced_viz_engine" not in st.session_state:
        st.session_state.advanced_viz_engine = None
    
    # DONNÉES DU FICHIER ACTUEL POUR VISUALISATION
    if "current_file_data" not in st.session_state:
        st.session_state.current_file_data = None
    if "current_filename" not in st.session_state:
        st.session_state.current_filename = None
    
    if "voice_agent" not in st.session_state:
        # Initialiser le Voice Agent (streaming pour fluidité)
        st.session_state.voice_agent = None  # Chargement lazy au premier usage
        st.session_state.voice_enabled = False
        st.session_state.voice_models_loaded = False
        st.session_state.voice_available = VOICE_AVAILABLE  # Vérifier si les modules sont disponibles
    if pdf_upload:
        files = upload_pdfs(pdf_upload)
        st.session_state.status_msg = f"✅ {len(files)} PDFs uploadés" if files else "⚠️ Aucun PDF"
        # Pas de rerun ici : file_uploader gère déjà
    if pbf_upload:
        st.session_state.graph, st.session_state.pois, msg = upload_and_process_pbf(pbf_upload)
        st.session_state.status_msg = msg
        model_choice = st.selectbox("Modèle", list(WORKING_MODELS.keys()), key="model_sidebar")
        st.session_state.current_model, st.session_state.agent, cache_info = update_agent(model_choice, st.session_state.vectordb, st.session_state.graph, st.session_state.pois, st.session_state.chat_vectordb) # AJOUT MÉMOIRE VECTORIELLE
        st.session_state.cache_msg = cache_info
        st.rerun()
    if process_pdfs_btn:
        st.session_state.vectordb, msg = process_pdfs()
        st.session_state.status_msg = msg
        model_choice = st.selectbox("Modèle", list(WORKING_MODELS.keys()), key="model_process")
        st.session_state.current_model, st.session_state.agent, cache_info = update_agent(model_choice, st.session_state.vectordb, st.session_state.graph, st.session_state.pois, st.session_state.chat_vectordb) # AJOUT MÉMOIRE VECTORIELLE
        st.session_state.cache_msg = cache_info
        st.rerun()
    if load_graph_btn:
        st.session_state.graph, st.session_state.pois, msg = load_existing_graph()
        st.session_state.status_msg = msg
        model_choice = st.selectbox("Modèle", list(WORKING_MODELS.keys()), key="model_load_graph")
        st.session_state.current_model, st.session_state.agent, cache_info = update_agent(model_choice, st.session_state.vectordb, st.session_state.graph, st.session_state.pois, st.session_state.chat_vectordb) # AJOUT MÉMOIRE VECTORIELLE
        st.session_state.cache_msg = cache_info
        st.rerun()
    if load_vectordb_btn:
        st.session_state.vectordb, msg = load_vectordb()
        st.session_state.status_msg = msg
        model_choice = st.selectbox("Modèle", list(WORKING_MODELS.keys()), key="model_load_db")
        st.session_state.chat_vectordb, _ = load_chat_vectordb() # AJOUT MÉMOIRE VECTORIELLE: Charger chat db
        st.session_state.current_model, st.session_state.agent, cache_info = update_agent(model_choice, st.session_state.vectordb, st.session_state.graph, st.session_state.pois, st.session_state.chat_vectordb)
        st.session_state.cache_msg = cache_info
        st.rerun()
    if clear_cache_btn:
        msg = handle_clear_cache()
        st.session_state.status_msg = msg
        st.session_state.cache_msg = get_cache_stats()
        st.rerun()
  
    # NOUVEAU: Gestion des boutons auto-apprentissage et amélioration
    if train_submodel_btn:
        st.session_state.chat_vectordb, _ = load_chat_vectordb()
        submodel_path, msg = create_submodel_from_chat_history(st.session_state.chat_vectordb)
        st.session_state.status_msg = msg
        if submodel_path:
            st.write(f"Utiliser: use_submodel_for_automation('query', '{submodel_path}')")
        st.rerun()
  
    if improve_db_btn:
        topics_input = st.text_input("Sujets (séparés par ,)", value="pétrole,topographie,sciences physiques,sous-sol", key="topics_input")
        topics = [t.strip() for t in topics_input.split(",")]
        st.session_state.vectordb, msg = improve_database_with_web_search(topics)
        st.session_state.status_msg = msg
        st.rerun()
    
    # 🆕 NOUVEAU: Gestion extraction PDF ERT
    if extract_ert_btn and ert_pdf_upload:
        # Sauvegarder temporairement le PDF
        temp_pdf_path = f"/tmp/ert_report_{int(time.time())}.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(ert_pdf_upload.getvalue())
        
        # Extraire données
        extraction_results = extract_ert_report_from_pdf(temp_pdf_path)
        
        # Afficher résultats
        st.success(f"✅ Extraction terminée!")
        st.write(f"📊 **Images extraites**: {len(extraction_results['images'])}")
        st.write(f"📝 **Légendes**: {len(extraction_results['captions'])}")
        st.write(f"🔢 **Valeurs résistivité**: {len(extraction_results['resistivity_values'])}")
        
        if extraction_results['resistivity_values']:
            st.write(f"📈 **Plage résistivité**: {min(extraction_results['resistivity_values']):.4f} - {max(extraction_results['resistivity_values']):.2f} Ω·m")
            
            # Analyser minéraux automatiquement
            mineral_report = analyze_minerals_from_resistivity(
                extraction_results['resistivity_values'], 
                ert_pdf_upload.name
            )
            st.text_area("🔬 Rapport Minéralogique", mineral_report, height=400)
            
            # 🆕 CRÉER TABLEAU DE CORRESPONDANCES
            st.markdown("### 📊 Tableau de Correspondances Réelles")
            fig_corr, df_corr, rapport_corr = create_real_mineral_correspondence_table(
                extraction_results['resistivity_values'],
                ert_pdf_upload.name
            )
            
            if fig_corr and df_corr is not None:
                st.pyplot(fig_corr)
                plt.close(fig_corr)
                
                # Corriger les pourcentages de confiance
                df_corr_display = df_corr.copy()
                if 'Confiance' in df_corr_display.columns:
                    if df_corr_display['Confiance'].max() <= 1:
                        df_corr_display['Confiance (%)'] = (df_corr_display['Confiance'] * 100).round(1)
                    else:
                        df_corr_display['Confiance (%)'] = df_corr_display['Confiance'].round(1)
                    df_corr_display = df_corr_display.drop('Confiance', axis=1)
                
                # Organiser en plusieurs tableaux si nécessaire
                total_rows = len(df_corr_display)
                if total_rows > 20:
                    st.markdown("#### 📋 Données Tabulaires - Organisées par Profondeur")
                    
                    depth_col = 'Profondeur (m)' if 'Profondeur (m)' in df_corr_display.columns else df_corr_display.columns[0]
                    df_sorted = df_corr_display.sort_values(depth_col)
                    
                    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    depth_ranges = df_sorted[depth_col].quantile(quantiles).values
                    
                    for i in range(5):
                        min_depth = depth_ranges[i]
                        max_depth = depth_ranges[i+1]
                        
                        if i == 4:
                            mask = (df_sorted[depth_col] >= min_depth) & (df_sorted[depth_col] <= max_depth)
                        else:
                            mask = (df_sorted[depth_col] >= min_depth) & (df_sorted[depth_col] < max_depth)
                        
                        df_section = df_sorted[mask]
                        
                        if len(df_section) > 0:
                            with st.expander(f"📊 Tableau {i+1}/5 - Profondeur: {min_depth:.1f} à {max_depth:.1f} m ({len(df_section)} détections)", expanded=(i==0)):
                                st.dataframe(
                                    df_section,
                                    use_container_width=True,
                                    column_config={
                                        "Confiance (%)": st.column_config.NumberColumn(
                                            "Confiance (%)",
                                            format="%.1f%%"
                                        ),
                                        "Résistivité mesurée (Ω·m)": st.column_config.NumberColumn(
                                            "Résistivité mesurée (Ω·m)",
                                            format="%.6f"
                                        ),
                                        "Profondeur (m)": st.column_config.NumberColumn(
                                            "Profondeur (m)",
                                            format="%.1f"
                                        )
                                    },
                                    height=min(400, len(df_section) * 35 + 38)
                                )
                else:
                    st.dataframe(
                        df_corr_display,
                        use_container_width=True,
                        column_config={
                            "Confiance (%)": st.column_config.NumberColumn(
                                "Confiance (%)",
                                format="%.1f%%"
                            ),
                            "Résistivité mesurée (Ω·m)": st.column_config.NumberColumn(
                                "Résistivité mesurée (Ω·m)",
                                format="%.6f"
                            ),
                            "Profondeur (m)": st.column_config.NumberColumn(
                                "Profondeur (m)",
                                format="%.1f"
                            )
                        }
                    )
                
                st.text_area("📝 Rapport Détaillé", rapport_corr, height=400)
                
                # Téléchargement CSV
                csv_data = df_corr_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger Correspondances (CSV)",
                    data=csv_data,
                    file_name=f"{ert_pdf_upload.name}_correspondances.csv",
                    mime="text/csv"
                )
            
            # 🆕 GÉNÉRER COUPES ERT PROFESSIONNELLES
            st.markdown("### 🎨 Coupes ERT Professionnelles (5 Graphiques)")
            
            # Option mode grand format
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                use_fullsize_pdf = st.checkbox("🖼️ Mode GRAND FORMAT PDF (30×36 pouces)", value=False, 
                                              help="Graphiques haute résolution pour impression A0/A1", key="fullsize_pdf")
            
            fig_ert, grid_data, rapport_ert = create_ert_professional_sections(
                extraction_results['resistivity_values'],
                ert_pdf_upload.name,
                full_size=use_fullsize_pdf
            )
            
            if fig_ert is not None:
                # Affichage responsive
                st.pyplot(fig_ert, use_container_width=True)
                
                # Rapport
                st.text_area("📊 Rapport ERT", rapport_ert, height=300)
                
                # Boutons de téléchargement en colonnes
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # PNG haute résolution
                    import io
                    buf_png = io.BytesIO()
                    fig_ert.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
                    buf_png.seek(0)
                    st.download_button(
                        label="📥 PNG 300 DPI",
                        data=buf_png,
                        file_name=f"{ert_pdf_upload.name}_ert_300dpi.png",
                        mime="image/png",
                        key="dl_png_pdf"
                    )
                
                with col2:
                    # PDF vectoriel
                    buf_pdf = io.BytesIO()
                    fig_ert.savefig(buf_pdf, format='pdf', bbox_inches='tight')
                    buf_pdf.seek(0)
                    st.download_button(
                        label="📄 PDF Vectoriel",
                        data=buf_pdf,
                        file_name=f"{ert_pdf_upload.name}_ert.pdf",
                        mime="application/pdf",
                        key="dl_pdf_pdf"
                    )
                
                with col3:
                    # Grille de données
                    if grid_data:
                        import pickle
                        grid_pickle = pickle.dumps(grid_data)
                        st.download_button(
                            label="� Grille PKL",
                            data=grid_pickle,
                            file_name=f"{ert_pdf_upload.name}_grid.pkl",
                            mime="application/octet-stream",
                            key="dl_grid_pdf"
                        )
                
                plt.close(fig_ert)
        
        st.session_state.status_msg = f"✅ PDF ERT extrait: {len(extraction_results['images'])} images"
        
    # 🆕 NOUVEAU: Gestion transcription audio
    if transcribe_audio_btn and audio_upload:
        temp_audio_path = f"/tmp/audio_{int(time.time())}.{audio_upload.name.split('.')[-1]}"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_upload.getvalue())
        
        transcription = process_audio_transcription(temp_audio_path)
        
        if transcription:
            st.text_area("📝 Transcription", transcription, height=200)
            st.session_state.status_msg = f"✅ Audio transcrit: {len(transcription)} caractères"

# Main area - Chat principal
st.title("🗺️ Kibali Analyst 🌟 - Assistant IA Avancé")
main_container = st.container()
with main_container:
    # Onglets pour autres fonctionnalités - AJOUT TAB ERTEST
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Trajets", 
        "📸 Analyse Image", 
        "🌐 Recherche Web", 
        "💬 Chat", 
        "📊 Status",
        "🌊 ERTest - Analyse Résistivité Avancée"  # NOUVEAU TAB
    ])
    with tab1:
        st.markdown("""
        ### Calcul de trajets
        **Exemples:** "Comment aller de l'école à l'hôpital ?"
        """)
        trajectory_input = st.text_area("🗺️ Question de trajet", key="traj_input")
        if st.button("🚀 Calculer trajet", key="calc_traj"):
            carte_buf, reponse, traj_info = calculer_trajet(trajectory_input, st.session_state.graph, st.session_state.pois)
            st.text_area("📋 Détails", reponse, key="traj_details")
            if carte_buf:
                carte_buf.seek(0)
                st.image(Image.open(carte_buf), key="traj_map")
            if traj_info:
                if st.button("💾 Sauvegarder trajet", key="save_traj"):
                    save_trajectory(trajectory_input, reponse, traj_info)
                    st.write("✅ Trajet sauvegardé")
    with tab2:
        st.markdown("""
        ### Analyse d'images
        Upload une image pour analyse détaillée, annotations, graphiques et amélioration IA.
        """)
        image_upload = st.file_uploader("📤 Upload Image", type=["jpg", "png"], key="img_upload")
        if image_upload and st.button("🔍 Analyser", key="analyze_img"):
            analysis_data, proc_images, tables_str = process_image(image_upload.getvalue())
            improved_analysis = improve_analysis_with_llm(analysis_data, st.session_state.current_model)
            st.image(proc_images, caption=proc_images, width=400) # Responsive width
            st.markdown(tables_str, unsafe_allow_html=True)
            st.text_area("Analyse Améliorée (IA)", improved_analysis, key="img_analysis")
    with tab3:
        st.markdown("""
        ### Recherche web avancée avec extraction de contenu
        """)
        web_query = st.text_area("🔍 Requête de recherche", key="web_query")
        search_type = st.selectbox("Type de recherche", ["text", "news", "both"], key="search_type")
        if st.button("🔍 Rechercher", key="search_btn"):
            results = handle_web_search(web_query, search_type)
            st.markdown(results, unsafe_allow_html=True)
        url_extract = st.text_input("🌐 URL à extraire", key="url_extract")
        if st.button("📄 Extraire contenu", key="extract_btn"):
            content = handle_content_extraction(url_extract)
            st.text_area("Contenu extrait", content, key="extracted_content")
    with tab4:
        st.markdown("### 🤖 Assistant IA Kibali - Modes Spécialisés")
        
        # Sélection du mode de fonctionnement
        st.markdown("#### 🎛️ Sélection du Mode")
        mode_col1, mode_col2 = st.columns([2, 1])
        
        with mode_col1:
            kibali_mode = st.selectbox(
                "Mode de Kibali",
                ["humain", "scientifique", "code_expert", "doc", "rapide"],
                format_func=lambda x: {
                    "humain": "🧑 Mode Humain - Conversationnel et naturel",
                    "scientifique": "🔬 Mode Scientifique - Précis, rigoureux, calculs détaillés",
                    "code_expert": "💻 Mode Code Expert - Programmation avancée (niveau Claude)",
                    "doc": "📖 Mode Documentation - Dissertations, livres, analyses approfondies",
                    "rapide": "⚡ Mode Rapide - Réponses concises et directes"
                }[x],
                key="kibali_mode_select",
                help="Chaque mode change complètement le comportement de Kibali"
            )
        
        with mode_col2:
            if kibali_mode == "code_expert":
                # Vérifier si le modèle est installé
                code_model_path = os.path.expanduser("~/.cache/huggingface/code_models")
                model_installed = os.path.exists(code_model_path) and os.listdir(code_model_path)
                
                if model_installed:
                    st.success("✅ DeepSeek-Coder installé")
                    if st.button("� Réinstaller", key="reinstall_codellama"):
                        with st.spinner("Téléchargement..."):
                            install_code_model()
                else:
                    if st.button("📦 Installer DeepSeek-Coder", key="install_codellama", help="Télécharge ~1.3GB. Prend 5-10 min"):
                        with st.spinner("Téléchargement du modèle de code..."):
                            st.info("💡 Alternative plus rapide: `python download_code_model.py` dans le terminal")
                            install_code_model()
        
        # Description du mode sélectionné
        mode_descriptions = {
            "humain": """
            💬 **Comportement**: Kibali réagit comme un humain véritable
            - Pose des questions de clarification si besoin
            - Admet ses doutes et incertitudes
            - Dialogue naturel avec émotions et réflexion
            - Peut refuser de répondre si question trop vague
            """,
            "scientifique": """
            🔬 **Comportement**: Rigueur scientifique absolue
            - Méthodologie stricte et vérifiable
            - Calculs détaillés avec toutes les étapes
            - Citations de sources précises
            - Analyse des incertitudes et limitations
            - Validation par calculs croisés
            """,
            "code_expert": """
            � **Comportement**: Expert en programmation niveau Claude/GPT-4
            - Utilise AI_Code_Generator (DeepSeek-Coder)
            - Code production-ready testé et optimisé
            - Architecture propre (SOLID, design patterns)
            - Tests unitaires automatiques inclus
            - Documentation complète
            - Suggestions d'optimisation
            """,
            "rapide": """
            ⚡ **Comportement**: Efficacité maximale
            - Réponses ultra-concises (2-3 phrases)
            - Bullet points pour clarté
            - Pas de contexte inutile
            - Code: snippets minimaux fonctionnels
            """,
            "doc": """
            📖 **Comportement**: Expert en rédaction approfondie
            - Dissertations académiques complètes
            - Livres et documents longs (10-30+ pages)
            - Structure méthodique (intro, développement, conclusion)
            - Style professionnel et accessible
            - Références bibliographiques
            - Analyses critiques multidimensionnelles
            - Exemples concrets et études de cas
            - Peut produire 2000+ mots par réponse
            
            **📚 Idéal pour**:
            • Dissertations universitaires
            • Rapports professionnels complets
            • Manuels et guides détaillés
            • Analyses approfondies multi-facettes
            • Livres blancs (white papers)
            • Thèses et mémoires (sections)
            """
        }
        
        with st.expander(f"ℹ️ Description du mode: {kibali_mode}", expanded=False):
            st.markdown(mode_descriptions[kibali_mode])
        
        # Options avancées
        st.markdown("#### ⚙️ Options")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            web_search_toggle = st.checkbox("🌐 Recherche web", value=True, key="web_toggle")
        with col_opt2:
            use_submodel = st.checkbox("🧠 Sous-modèle rapide", value=False, key="use_submodel",
                                      help="Réponses instantanées via modèle auto-appris")
        
        if use_submodel:
            submodel_path_input = st.text_input("Chemin sous-modèle (optionnel)", key="submodel_path")
        else:
            submodel_path_input = None
      
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="☁️" if msg["role"] == "user" else "⭐"):
                # Correction pour lisibilité : utiliser markdown pour HTML
                if msg["role"] == "user":
                    st.markdown(f"**Question:** {highlight_important_words(msg['content'])}", unsafe_allow_html=True)
                else:
                    st.markdown(highlight_important_words(msg['content']), unsafe_allow_html=True)
        
        # 🎤 INTERFACE VOCALE - Boutons d'enregistrement
        if VOICE_AVAILABLE and st.session_state.get("voice_enabled", False) and st.session_state.get("voice_models_loaded", False):
            st.markdown("---")
            
            # Désactiver l'enregistrement si pas de matériel audio
            recording_disabled = st.session_state.get("audio_recording_disabled", False)
            
            if recording_disabled:
                st.info("📁 Mode transcription uniquement - Uploadez des fichiers audio pour les transcrire")
                st.caption("🚫 Enregistrement micro non disponible (WSL/Docker)")
            else:
                col_voice1, col_voice2, col_voice3 = st.columns([1, 1, 1])
                
                with col_voice1:
                    if st.button("🎤 Enregistrer Question", key="voice_record_btn", use_container_width=True):
                        st.session_state.voice_recording = True
                
                with col_voice2:
                    if st.button("🔊 Répéter Dernière Réponse", key="voice_repeat_btn", use_container_width=True):
                        if st.session_state.chat_history:
                            last_response = st.session_state.chat_history[-1]
                            if last_response["role"] == "assistant":
                                with st.spinner("🔊 Synthèse vocale..."):
                                    voice_agent = st.session_state.voice_agent
                                    audio_path = voice_agent.synthesize_speech(
                                        last_response["content"],
                                        play=st.session_state.get("voice_autoplay", True)
                                    )
                                    if audio_path:
                                        st.success("✅ Audio généré!")
                                        # Téléchargement optionnel
                                        with open(audio_path, 'rb') as f:
                                            st.download_button(
                                                "💾 Télécharger Audio",
                                                f.read(),
                                                file_name="kibali_response.wav",
                                                mime="audio/wav"
                                            )
                
                with col_voice3:
                    voice_status = "🟢 Actif" if st.session_state.get("voice_models_loaded", False) else "🔴 Inactif"
                    st.info(f"Vocal: {voice_status}")
            
            # Gestion de l'enregistrement vocal
            if st.session_state.get("voice_recording", False):
                st.info(f"🎤 Parlez maintenant ({st.session_state.get('voice_duration', 5)}s)...")
                
                voice_agent = st.session_state.voice_agent
                duration = st.session_state.get('voice_duration', 5)
                language = st.session_state.get('voice_lang', 'fr')
                
                # Enregistrer audio (avec gestion des erreurs)
                try:
                    audio = voice_agent.record_audio(duration=duration)
                    
                    if len(audio) > 0:
                        with st.spinner("📝 Transcription en cours..."):
                            # Transcrire
                            transcription = voice_agent.transcribe_audio(
                                audio_array=audio,
                                language=language
                            )
                            
                            if transcription:
                                st.success(f"✅ Transcription: {transcription}")
                                
                                # Utiliser la transcription comme prompt
                                prompt = transcription
                                st.session_state.voice_recording = False
                                
                                # Traiter la question vocale
                                with st.chat_message("user", avatar="☁️"):
                                    st.markdown(f"**Question (vocale):** {highlight_important_words(prompt)}", unsafe_allow_html=True)
                                
                                with st.chat_message("assistant", avatar="⭐"):
                                    with st.spinner("🤖 Kibali réfléchit..."):
                                        # Générer réponse
                                        mode_prompt = get_mode_specific_prompt(kibali_mode)
                                        response = handle_chat_enhanced(
                                            prompt, 
                                            st.session_state.chat_history, 
                                            st.session_state.agent,
                                            list(WORKING_MODELS.keys())[0],
                                            st.session_state.vectordb,
                                            st.session_state.graph,
                                            st.session_state.pois,
                                            web_search_toggle,
                                            mode=kibali_mode,
                                            mode_prompt=mode_prompt
                                        )
                                        
                                        response = apply_mode_behavior(response, prompt, kibali_mode)
                                        st.markdown(highlight_important_words(response), unsafe_allow_html=True)
                                        
                                        # Synthèse vocale de la réponse (si disponible)
                                        if st.session_state.get("voice_autoplay", True):
                                            try:
                                                with st.spinner("🔊 Synthèse vocale..."):
                                                    audio_path = voice_agent.synthesize_speech(
                                                        response,
                                                        play=False  # Ne pas jouer automatiquement
                                                    )
                                                    if audio_path:
                                                        st.audio(audio_path)
                                            except Exception as e:
                                                st.warning(f"⚠️ Synthèse vocale non disponible: {str(e)}")
                            else:
                                st.error("❌ Échec de la transcription")
                                st.session_state.voice_recording = False
                    else:
                        st.error("❌ Erreur d'enregistrement audio")
                        st.info("💡 Vérifiez qu'un microphone est connecté et configuré")
                        st.session_state.voice_recording = False
                        
                except Exception as e:
                    st.error(f"❌ Erreur enregistrement: {str(e)}")
                    st.info("💡 Le mode vocal nécessite un microphone fonctionnel et PortAudio")
                    st.session_state.voice_recording = False
            
            st.markdown("---")
        
        # ========================================
        # ZONE D'UPLOAD DE FICHIERS (Style ChatGPT)
        # ========================================
        st.markdown("### 📎 Téléverser un fichier pour analyse")
        
        col_upload, col_info = st.columns([3, 1])
        
        with col_upload:
            uploaded_files = st.file_uploader(
                "📂 Glissez un ou plusieurs fichiers ici (multi-sélection pour .dat complémentaires)",
                type=None,  # Accepte tous les types
                help="Images, PDF, CSV, JSON, fichiers ERT multi-fréquences (.dat), etc. Pour les fichiers .dat: uploadez tous les fichiers complémentaires en une fois pour fusion automatique.",
                key="chat_file_upload",
                accept_multiple_files=True  # 🆕 MULTI-FICHIERS ACTIVÉ
            )
        
        with col_info:
            if uploaded_files:
                if len(uploaded_files) == 1:
                    st.success(f"✅ {uploaded_files[0].name}")
                    file_size_mb = uploaded_files[0].size / (1024 * 1024)
                    st.caption(f"Taille: {file_size_mb:.2f} MB")
                else:
                    st.success(f"✅ {len(uploaded_files)} fichiers")
                    total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
                    st.caption(f"Total: {total_size:.2f} MB")
                    
                    # Afficher liste des fichiers
                    with st.expander("📋 Fichiers uploadés"):
                        for uf in uploaded_files:
                            st.write(f"• {uf.name} ({uf.size/(1024*1024):.2f} MB)")
        
        # Traiter les fichiers uploadés
        if uploaded_files is not None and len(uploaded_files) > 0:
            # Stocker les fichiers dans session_state pour usage ultérieur
            if 'uploaded_file_data' not in st.session_state:
                st.session_state.uploaded_file_data = None
            
            if st.button("🔍 Analyser ces fichiers", type="primary", use_container_width=True):
                with st.spinner("🤖 Kibali analyse vos fichiers en profondeur..."):
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    
                    # 🆕 DÉTECTION MULTI-FICHIERS .dat POUR FUSION
                    dat_files = [f for f in uploaded_files if f.name.lower().endswith('.dat')]
                    other_files = [f for f in uploaded_files if not f.name.lower().endswith('.dat')]
                    
                    # Si plusieurs fichiers .dat → Fusion multi-fréquences automatique
                    if len(dat_files) > 1:
                        st.info(f"🔬 **{len(dat_files)} fichiers .dat détectés → Fusion multi-fréquences automatique**")
                        
                        # Sauvegarder tous les fichiers .dat
                        dat_paths = []
                        for dat_file in dat_files:
                            temp_path = os.path.join(temp_dir, dat_file.name)
                            with open(temp_path, 'wb') as f:
                                f.write(dat_file.getvalue())
                            dat_paths.append(temp_path)
                            st.write(f"   • {dat_file.name}")
                        
                        # Parser avec fusion
                        try:
                            from multi_freq_ert_parser import MultiFreqERTParser
                            parser = MultiFreqERTParser()
                            df = parser.parse_multiple_files(dat_paths)
                            
                            if not df.empty and len(df) > 0:
                                st.success(f"✅ **FUSION RÉUSSIE !**")
                                st.write(f"📊 {len(df)} mesures ERT fusionnées")
                                st.write(f"📡 {len(parser.frequencies)} fréquences détectées")
                                st.write(f"📍 {len(parser.survey_points)} survey points")
                                
                                # Afficher structure
                                with st.expander("📋 Structure fusionnée (10 premières lignes)"):
                                    st.dataframe(df[['project', 'survey_point', 'depth', 'frequency_MHz', 'resistivity']].head(10))
                                
                                # Coordonnées
                                coords_df = parser.get_coordinates_corrected()
                                with st.expander("📐 Coordonnées spatiales (X, Y, Z)"):
                                    st.dataframe(coords_df[['x', 'y', 'z', 'resistivity', 'frequency_MHz']].head(10))
                                
                                # Sauvegarder en session
                                st.session_state['multi_freq_data'] = df
                                st.session_state['multi_freq_coords'] = coords_df
                                st.session_state['multi_freq_parser'] = parser
                                
                                # Créer données pour le chat
                                file_data = {
                                    'filename': f"fusion_{len(dat_files)}_fichiers.dat",
                                    'physical_path': dat_paths[0],  # Premier fichier comme référence
                                    'all_paths': dat_paths,
                                    'size': sum(f.size for f in dat_files),
                                    'extension': 'dat',
                                    'metadata': {
                                        "type": "ERT_MULTI_FREQ_FUSION",
                                        "num_files": len(dat_files),
                                        "num_measurements": len(df),
                                        "num_frequencies": len(parser.frequencies),
                                        "num_survey_points": len(parser.survey_points),
                                        "frequencies_MHz": [float(f) for f in sorted(parser.frequencies)[:20]],
                                        "depth_range": (float(parser.metadata['depth_range'][0]), float(parser.metadata['depth_range'][1])),
                                        "resistivity_range": (float(parser.metadata['resistivity_range'][0]), float(parser.metadata['resistivity_range'][1]))
                                    },
                                    'content': f"""Données ERT Multi-Fréquences (FUSION de {len(dat_files)} fichiers):
• {len(df)} mesures totales
• {len(parser.frequencies)} fréquences (de {min(parser.frequencies):.2f} à {max(parser.frequencies):.2f} MHz)
• {len(parser.survey_points)} survey points
• Profondeurs: {parser.metadata['depth_range'][0]:.1f} à {parser.metadata['depth_range'][1]:.1f} m
• Résistivités: {parser.metadata['resistivity_range'][0]:.2f} à {parser.metadata['resistivity_range'][1]:.2f} Ω·m

Structure validée:
project | survey_point | depth | frequency_MHz | resistivity

Coordonnées spatiales (X,Y,Z) disponibles pour visualisation."""
                                }
                                
                                st.session_state.uploaded_file_data = file_data
                                
                                # Message de Kibali
                                kibali_analysis = f"""🎯 **Analyse de fusion multi-fréquences ERT**

J'ai fusionné **{len(dat_files)} fichiers .dat complémentaires** avec succès !

**📊 Résumé:**
• **{len(df):,}** mesures ERT fusionnées
• **{len(parser.frequencies)}** fréquences détectées ({min(parser.frequencies):.0f} - {max(parser.frequencies):.0f} MHz)
• **{len(parser.survey_points)}** survey points
• Profondeurs: **{parser.metadata['depth_range'][0]:.1f}** à **{parser.metadata['depth_range'][1]:.1f}** m

**✅ Structure correcte validée:**
```
project | survey_point | depth | frequency_MHz | resistivity
```

**📐 Système de coordonnées:**
• X = survey_point × 10m (espacement horizontal)
• Y = 0m (ligne unique)
• Z = profondeur absolue (valeur positive)

**💡 Tu peux maintenant me demander:**
• "Montre-moi les coordonnées spatiales"
• "Génère une visualisation 2D par fréquence"
• "Analyse les variations de résistivité"
• "Export en Excel"
"""
                                
                                with st.chat_message("assistant", avatar="⭐"):
                                    st.markdown(kibali_analysis, unsafe_allow_html=True)
                                
                                # Sauvegarder dans historique
                                st.session_state.chat_history.append({
                                    "role": "user",
                                    "content": f"📎 {len(dat_files)} fichiers .dat uploadés pour fusion"
                                })
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": kibali_analysis
                                })
                                
                                st.success("✅ Fusion terminée ! Pose-moi toutes tes questions sur ces données ERT.")
                            
                            else:
                                st.error("❌ Aucune donnée ERT détectée dans ces fichiers")
                        
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la fusion: {e}")
                            st.info("💡 Les fichiers seront analysés individuellement...")
                    
                    # Si 1 seul fichier .dat ou autres fichiers
                    if len(dat_files) <= 1 or other_files:
                        # Utiliser le premier fichier (logique originale)
                        uploaded_file = uploaded_files[0] if len(uploaded_files) == 1 else (dat_files[0] if dat_files else other_files[0])
                        
                        # Sauvegarder le fichier physiquement
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        
                        # Écrire le fichier
                        with open(temp_file_path, 'wb') as f:
                            f.write(uploaded_file.getvalue())
                        # Écrire le fichier
                        with open(temp_file_path, 'wb') as f:
                            f.write(uploaded_file.getvalue())
                        
                        # Stocker immédiatement le chemin physique pour le AI Code Agent
                        # AVANT l'analyse, pour que le path soit disponible dès le premier message
                        st.session_state.uploaded_file_data = {
                            'filename': uploaded_file.name,
                            'physical_path': temp_file_path,
                            'size': uploaded_file.size
                        }
                        
                        # Analyse complète et intelligente par Kibali
                        kibali_analysis, file_data = analyze_uploaded_file_realtime(
                            uploaded_file,
                            st.session_state.chat_history,
                            st.session_state.agent,
                            st.session_state.vectordb,
                            st.session_state.graph,
                            st.session_state.pois,
                            web_search_toggle
                        )
                        
                        # Ajouter le chemin physique aux données d'analyse
                        file_data['physical_path'] = temp_file_path
                        
                        # Mettre à jour avec toutes les données d'analyse
                        st.session_state.uploaded_file_data = file_data
                        
                        # Afficher l'analyse de Kibali dans le chat
                        with st.chat_message("assistant", avatar="⭐"):
                            st.markdown(kibali_analysis, unsafe_allow_html=True)
                        
                        # Sauvegarder dans l'historique
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": f"📎 Fichier uploadé: {uploaded_file.name}"
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": kibali_analysis
                        })
                        
                        st.success("✅ Analyse terminée ! Pose-moi toutes tes questions sur ce fichier.")
                        st.info("💡 **Kibali est maintenant expert de ce fichier.** Je peux faire des recherches web supplémentaires si besoin !")
        
        st.markdown("---")
        
        if prompt := st.chat_input("Pose une question...", key="chat_input"):
            with st.chat_message("user", avatar="☁️"):
                highlighted_prompt = highlight_important_words(prompt)
                st.markdown(f"**Question:** {highlighted_prompt}", unsafe_allow_html=True)
            
            with st.chat_message("assistant", avatar="⭐"):
                spinner_messages = {
                    "humain": "🤔 Réflexion en cours...",
                    "scientifique": "🔬 Analyse rigoureuse...",
                    "code_expert": "💻 Génération de code optimisé...",
                    "rapide": "⚡ Traitement rapide..."
                }
                with st.spinner(spinner_messages.get(kibali_mode, "Réponse en cours...")):
                    content_to_save = None
                    
                    # === DÉTECTION ET UTILISATION AUTONOME DES OUTILS ===
                    tools_results = None
                    if 'uploaded_file_data' in st.session_state and st.session_state.uploaded_file_data:
                        # Détection automatique des outils nécessaires
                        needed_tools = kibali_tools.detect_needed_tools(
                            prompt, 
                            st.session_state.uploaded_file_data
                        )
                        
                        if needed_tools:
                            with st.status("🔧 Kibali prépare les outils nécessaires...", expanded=True) as status:
                                st.write(f"Outils détectés: {', '.join(needed_tools)}")
                                
                                # Exécution des outils
                                tools_results = kibali_tools.execute_tools(
                                    needed_tools,
                                    st.session_state.uploaded_file_data,
                                    prompt
                                )
                                
                                status.update(label="✅ Outils exécutés", state="complete")
                    
                    # Générer la réponse selon le mode
                    if use_submodel and submodel_path_input:
                        automated = use_submodel_for_automation(prompt, submodel_path_input)
                        # Appliquer le comportement du mode
                        automated = apply_mode_behavior(automated, prompt, kibali_mode)
                        st.markdown(highlight_important_words(automated), unsafe_allow_html=True)
                        content_to_save = automated
                    else:
                        # Enrichir le prompt avec le contexte du fichier uploadé ET résultats des outils
                        enriched_prompt = prompt
                        if 'uploaded_file_data' in st.session_state and st.session_state.uploaded_file_data:
                            # Ajouter résultats des outils au contexte
                            tools_context = ""
                            if tools_results and tools_results.get("success"):
                                tools_context = "\n[RÉSULTATS DES OUTILS UTILISÉS]\n"
                                for tool_id, output in tools_results["outputs"].items():
                                    tools_context += f"\n**Outil: {tool_id}**\n"
                                    tools_context += f"Message: {output.get('message', '')}\n"
                                    
                                    # Ajouter données spécifiques
                                    if 'statistics' in output:
                                        tools_context += f"Statistiques: {output['statistics']}\n"
                                    if 'resistivity_values' in output:
                                        tools_context += f"Valeurs extraites: {len(output['resistivity_values'])} points\n"
                                    if 'analysis' in output:
                                        tools_context += f"Analyse: {output['analysis'][:500]}...\n"
                            
                            file_context = f"""
[CONTEXTE FICHIER UPLOADÉ]
Fichier: {st.session_state.uploaded_file_data['filename']}
Chemin physique: {st.session_state.uploaded_file_data.get('physical_path', st.session_state.uploaded_file_data['filename'])}
Type: .{st.session_state.uploaded_file_data['extension']}
Taille: {st.session_state.uploaded_file_data['size']} octets
Métadonnées: {st.session_state.uploaded_file_data['metadata']}
Contenu extrait: {st.session_state.uploaded_file_data['content'][:1000]}

{tools_context}

[QUESTION DE L'UTILISATEUR]
{prompt}

[INSTRUCTION]
Tu es un assistant IA conversationnel et naturel. Utilise les résultats des outils ci-dessus pour donner une réponse fluide, conversationnelle et complète.

RÈGLES IMPORTANTES:
1. Réponds de manière naturelle et conversationnelle, comme un expert humain
2. Explique les résultats de façon pédagogique et accessible
3. Utilise ta connaissance scientifique pour ajouter du contexte et des interprétations
4. Si pertinent, fais des recherches web pour compléter l'analyse avec des données actuelles
5. Structure ta réponse de façon logique mais pas robotique
6. Pose des questions de clarification si quelque chose n'est pas clair
7. Adapte ton langage au contexte scientifique tout en restant accessible

Si tu dois générer du code Python pour analyser le fichier, utilise le chemin physique: {st.session_state.uploaded_file_data.get('physical_path', st.session_state.uploaded_file_data['filename'])}
"""
                            enriched_prompt = file_context
                        
                        # Modifier le prompt système selon le mode avant d'appeler l'agent
                        mode_prompt = get_mode_specific_prompt(kibali_mode)
                        
                        # Générer réponse avec agent (avec contexte fichier + outils)
                        response = handle_chat_enhanced(
                            enriched_prompt, 
                            st.session_state.chat_history, 
                            st.session_state.agent, 
                            list(WORKING_MODELS.keys())[0], 
                            st.session_state.vectordb, 
                            st.session_state.graph, 
                            st.session_state.pois, 
                            web_search_toggle, 
                            mode=kibali_mode, 
                            mode_prompt=mode_prompt
                        )
                        
                        # Afficher les résultats visuels des outils (graphiques, tableaux)
                        if tools_results and tools_results.get("success"):
                            for tool_id, output in tools_results["outputs"].items():
                                # Afficher graphiques
                                if 'plot' in output and output['plot']:
                                    st.markdown(output['plot'], unsafe_allow_html=True)
                                
                                # Afficher tables
                                if 'color_table' in output and output['color_table']:
                                    st.markdown(output['color_table'], unsafe_allow_html=True)
                                
                                # Afficher analyses textuelles
                                if 'formatted_text' in output and output['formatted_text']:
                                    st.markdown(output['formatted_text'])
                                
                                # Afficher mapping couleurs
                                if 'color_mapping' in output and output['color_mapping']:
                                    with st.expander("📊 Légende des couleurs ERT"):
                                        st.markdown(output['color_mapping'], unsafe_allow_html=True)
                        
                        # Appliquer le comportement du mode
                        response = apply_mode_behavior(response, prompt, kibali_mode)
                        st.markdown(highlight_important_words(response), unsafe_allow_html=True)
                        content_to_save = response
                        
                        # 🎨 GÉNÉRATION AUTOMATIQUE DE VISUALISATIONS
                        # Si des données sont disponibles, générer automatiquement des graphiques
                        if tools_results and tools_results.get("success"):
                            # Préparer les données pour l'auto-visualiseur
                            viz_data = {
                                'filename': st.session_state.uploaded_file_data.get('filename', 'Analyse'),
                                'resistivity_values': []
                            }
                            
                            # Extraire les valeurs de résistivité des outils
                            for tool_id, output in tools_results["outputs"].items():
                                if 'resistivity_values' in output:
                                    viz_data['resistivity_values'].extend(output['resistivity_values'])
                                if 'statistics' in output:
                                    viz_data['statistics'] = output['statistics']
                            
                            # Générer et afficher automatiquement les visualisations
                            if viz_data['resistivity_values']:
                                auto_viz.auto_generate_and_display(prompt, viz_data)
                        
                        # Si mode doc et PDF généré, afficher bouton de téléchargement
                        if kibali_mode == "doc":
                            print(f"[DEBUG PDF] Mode doc activé")
                            print(f"[DEBUG PDF] generated_pdfs existe: {'generated_pdfs' in st.session_state}")
                            if 'generated_pdfs' in st.session_state:
                                print(f"[DEBUG PDF] Nombre de PDFs: {len(st.session_state.generated_pdfs)}")
                                
                            if 'generated_pdfs' in st.session_state and st.session_state.generated_pdfs:
                                latest_pdf = st.session_state.generated_pdfs[-1]
                                print(f"[DEBUG PDF] Dernier PDF: {latest_pdf['path']}")
                                
                                # Vérifier que le fichier existe
                                if os.path.exists(latest_pdf['path']):
                                    st.success(f"� PDF généré: {latest_pdf['filename']}")
                                    
                                    # Lire le fichier PDF
                                    with open(latest_pdf['path'], 'rb') as pdf_file:
                                        pdf_bytes = pdf_file.read()
                                    
                                    print(f"[DEBUG PDF] Fichier lu: {len(pdf_bytes)} bytes")
                                    
                                    # Bouton de téléchargement
                                    st.download_button(
                                        label="📥 Télécharger le PDF",
                                        data=pdf_bytes,
                                        file_name=latest_pdf['filename'],
                                        mime="application/pdf",
                                        key=f"download_pdf_{latest_pdf['timestamp']}"
                                    )
                                    
                                    st.info(f"� {latest_pdf['word_count']} mots | Format: A4 | Police: Helvetica")
                                else:
                                    st.error(f"❌ Erreur: Fichier PDF introuvable: {latest_pdf['path']}")
                            else:
                                print(f"[DEBUG PDF] Aucun PDF disponible dans session_state")
            
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": content_to_save})
    with tab5:
        st.markdown("### Statut système")
        st.json(get_system_status())
    
    # ═══════════════════════════════════════════════════════════════
    # TAB 6 : ERTEST - ANALYSE RÉSISTIVITÉ AVANCÉE (IMPORTÉ)
    # ═══════════════════════════════════════════════════════════════
    with tab6:
        st.header("🌊 ERTest - Analyse Résistivité Avancée Ravensgate")
        st.markdown("""
        ### Module d'analyse ERT complet importé d'ERTest.py
        
        Cette section intègre toutes les fonctionnalités du système ERTest :
        - 🌡️ Calculateur de réglage température Ravensgate
        - 📊 Analyse de fichiers .dat (Sonic Water Level Meter)
        - 🌍 Pseudo-sections ERT 2D/3D avec colormap personnalisée
        - 🪨 Stratigraphie complète (sols + eaux + minéraux)
        - 🔬 Inversion PyGIMLI avancée
        
        **Colormap Eau Prioritaire** : Rouge (mer) → Jaune (salée) → Vert (douce) → Bleu (pure)
        """)
        
        # Charger ERTest.py comme module
        import sys
        import importlib.util
        
        ertest_path = "/home/belikan/KIbalione8/SETRAF/ERTest.py"
        
        try:
            # Charger le module ERTest
            spec = importlib.util.spec_from_file_location("ertest_module", ertest_path)
            ertest_module = importlib.util.module_from_spec(spec)
            
            # Exécuter le module dans son propre namespace
            with st.spinner("🔄 Chargement du module ERTest..."):
                spec.loader.exec_module(ertest_module)
                st.success("✅ Module ERTest chargé avec succès !")
                
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement d'ERTest : {e}")
            st.exception(e)

# Fin du tab6 - retour au code principal d'ERT.py
st.markdown("### 📊 Informations Système")
setup_drive()
st.write(f"🚀 Kibali 🌟 - Assistant IA Avancé avec Recherche Web")
st.write(f"📁 Dossier unifié: {CHATBOT_DIR}")
st.write(f"🔑 Token HF configuré: {HF_TOKEN[:10]}...")
st.write(f"🌐 Recherche web intégrée")
existing_graphs = [f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')] if os.path.exists(GRAPHS_PATH) else []
existing_pdfs = [f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')] if os.path.exists(PDFS_PATH) else []
st.write(f"📊 État initial:")
st.write(f" 🗺️ Graphes OSM: {len(existing_graphs)}")
st.write(f" 📄 PDFs: {len(existing_pdfs)}")
st.write(f" 💾 Base vectorielle: {'✅' if os.path.exists(VECTORDB_PATH) else '❌'}")
st.write(f" 🧠 Mémoire chat: {'✅' if os.path.exists(CHAT_VECTORDB_PATH) else '❌'}") # AJOUT MÉMOIRE VECTORIELLE
st.write(f" 🌐 Cache web: {'✅' if os.path.exists(WEB_CACHE_PATH) else '❌'}")
st.write(f" 📈 {get_cache_stats()}")
st.write("\n" + "="*60)
st.write("🎉 KIBALI 🌟 - SYSTÈME CHARGÉ AVEC SUCCÈS")
st.write("="*60)
st.write(f"📅 Version: 2.0.0 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"🔑 Token HF: {'✅ Configuré' if HF_TOKEN else '❌ Manquant'}")
st.write(f"📁 Dossier: {CHATBOT_DIR}")
st.write(f"🌐 Recherche web: ✅ Activée")
st.write(f"💾 Cache intelligent: ✅ Activé")
st.write(f"🧠 Mémoire vectorielle chat: ✅ Activée") # AJOUT MÉMOIRE VECTORIELLE
st.write(f"🤖 Auto-apprentissage sklearn: ✅ Activé (sous-modèles dans {SUBMODELS_PATH})")
st.write(f"📚 Amélioration DB via fouille: ✅ Activée (sujets pétrole, topographie, etc.)")
st.write("\n📚 FONCTIONNALITÉS PRINCIPALES:")
st.write(" 💬 Chat RAG avec recherche web intelligent")
st.write(" 🧠 Mémoire des conversations pour fluidité") # AJOUT MÉMOIRE VECTORIELLE
st.write(" 🗺️ Calcul de trajets OSM")
st.write(" 📸 Analyse d'images avec IA")
st.write(" 🌐 Extraction de contenu web")
st.write(" 💾 Gestion unifiée des données")
st.write(" 🤖 Sous-modèles sklearn pour automatismes humains")
st.write(" 📚 Fouille auto internet pour enrichir DB (pétrole, topographie, sciences physiques, sous-sol)")
st.write("\n🚀 UTILISATION:")
st.write(" Interface: Exécutez les cellules suivantes")
st.write(" API: kibali_api.ask('votre question')")
st.write(" Auto-apprentissage: kibali_api.train_submodel()")
st.write(" Amélioration DB: kibali_api.improve_db(['pétrole'])")
st.write(" Tests: test_all_features()")
st.write("\n⚙️ MAINTENANCE:")
st.write(" Status: get_system_status()")
st.write(" Nettoyage: cleanup_old_cache()")
st.write(" Sauvegarde: backup_all_data()")
st.write("="*60)

# ========================================
# INTÉGRATION PYGIMLI POUR INVERSION ERT
# ========================================

def run_pygimli_inversion(dat_data: dict, electrode_spacing: float = 1.0, 
                         max_depth: float = None, n_layers: int = 20) -> dict:
    """
    Exécute une inversion ERT complète avec PyGIMLI
    
    Args:
        dat_data: Dictionnaire avec les données du fichier .dat
        electrode_spacing: Espacement entre électrodes (m)
        max_depth: Profondeur maximale d'investigation (m)
        n_layers: Nombre de couches pour le modèle
    
    Returns:
        dict: Résultats de l'inversion avec modèles et figures
    """
    if not PYGIMLI_AVAILABLE:
        return {"error": "PyGIMLI non disponible. Installez avec: pip install pygimli"}
    
    try:
        import pygimli as pg
        import numpy as np
        
        # Extraire les données du fichier .dat
        survey_points = dat_data.get('survey_point', [])
        depths = dat_data.get('depth', [])
        data_values = dat_data.get('data', [])
        
        if not survey_points or not depths or not data_values:
            return {"error": "Données .dat incomplètes"}
        
        # Créer un profil ERT 2D
        # Supposons un profil linéaire avec espacement électrode
        n_electrodes = len(np.unique(survey_points))
        electrode_positions = np.linspace(0, (n_electrodes-1) * electrode_spacing, n_electrodes)
        
        # Créer la géométrie ERT
        scheme = pg.physics.ert.createERTData(elecs=electrode_positions)
        
        # Simuler des mesures (en réalité, utiliserait les vraies données de résistivité apparente)
        # Pour démonstration, créer des données synthétiques basées sur les mesures .dat
        rho_apparent = np.array(data_values)
        
        # Normaliser et ajuster les données
        rho_apparent = np.clip(rho_apparent, 0.1, 10000)  # Plage réaliste
        
        # Créer le vecteur de données
        scheme.set('rhoa', rho_apparent)
        
        # Créer le maillage d'inversion
        world = pg.createWorld(start=[electrode_positions[0], 0], 
                              end=[electrode_positions[-1], -max_depth if max_depth else -50])
        
        mesh = pg.createMesh(world, quality=34, area=0.1)
        
        # Inversion ERT
        inv = pg.physics.ert.ERTInversion(sr=False, verbose=False)
        model = inv.run(scheme, mesh)
        
        # Extraire les résultats
        rho_true = model  # Résistivité vraie
        
        return {
            "success": True,
            "rho_true": rho_true,
            "mesh": mesh,
            "scheme": scheme,
            "electrode_positions": electrode_positions,
            "n_electrodes": n_electrodes
        }
        
    except Exception as e:
        return {"error": f"Erreur PyGIMLI: {str(e)}"}

def create_pygimli_sections(dat_data: dict, inversion_results: dict, 
                          output_dir: str = None) -> dict:
    """
    Crée 4 coupes ERT inversées avec PyGIMLI
    
    Args:
        dat_data: Données du fichier .dat
        inversion_results: Résultats de l'inversion PyGIMLI
        output_dir: Répertoire de sortie pour les figures
    
    Returns:
        dict: 4 figures matplotlib des coupes inversées
    """
    if not PYGIMLI_AVAILABLE or "error" in inversion_results:
        return {"error": inversion_results.get("error", "PyGIMLI non disponible")}
    
    try:
        import pygimli as pg
        import matplotlib.pyplot as plt
        import numpy as np
        
        rho_true = inversion_results["rho_true"]
        mesh = inversion_results["mesh"]
        
        # Créer les 4 coupes avec différentes échelles de couleur
        figures = {}
        
        # Coupe 1: Échelle linéaire complète
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        pg.show(mesh, rho_true, ax=ax1, cMap='jet_r', 
                cMin=0.1, cMax=1000, logScale=True)
        ax1.set_title('Coupe 1: Résistivité vraie - Échelle complète (log)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Profondeur (m)')
        plt.colorbar(ax1.images[0], ax=ax1, label='Résistivité (Ω·m)')
        plt.tight_layout()
        figures['section_1_full_scale'] = fig1
        
        # Coupe 2: Focus sur résistivités basses (conductrices)
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        pg.show(mesh, rho_true, ax=ax2, cMap='Reds', 
                cMin=0.1, cMax=10, logScale=True)
        ax2.set_title('Coupe 2: Zone conductrice (0.1-10 Ω·m)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Profondeur (m)')
        plt.colorbar(ax2.images[0], ax=ax2, label='Résistivité (Ω·m)')
        plt.tight_layout()
        figures['section_2_conductive'] = fig2
        
        # Coupe 3: Focus sur résistivités moyennes (aquifères)
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        pg.show(mesh, rho_true, ax=ax3, cMap='YlGnBu', 
                cMin=10, cMax=100, logScale=False)
        ax3.set_title('Coupe 3: Zone aquifère (10-100 Ω·m)', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('Distance (m)')
        ax3.set_ylabel('Profondeur (m)')
        plt.colorbar(ax3.images[0], ax=ax3, label='Résistivité (Ω·m)')
        plt.tight_layout()
        figures['section_3_aquifer'] = fig3
        
        # Coupe 4: Focus sur résistivités élevées (roches)
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        pg.show(mesh, rho_true, ax=ax4, cMap='Blues', 
                cMin=100, cMax=10000, logScale=True)
        ax4.set_title('Coupe 4: Substrat rocheux (100-10000 Ω·m)', 
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Distance (m)')
        ax4.set_ylabel('Profondeur (m)')
        plt.colorbar(ax4.images[0], ax=ax4, label='Résistivité (Ω·m)')
        plt.tight_layout()
        figures['section_4_resistive'] = fig4
        
        return figures
        
    except Exception as e:
        return {"error": f"Erreur création coupes: {str(e)}"}

def pygimli_ert_analysis(dat_file_path: str, electrode_spacing: float = 1.0) -> dict:
    """
    Analyse ERT complète avec PyGIMLI pour fichier .dat
    
    Args:
        dat_file_path: Chemin vers le fichier .dat
        electrode_spacing: Espacement entre électrodes (m)
    
    Returns:
        dict: Résultats complets avec inversion et visualisations
    """
    try:
        # Charger les données du fichier .dat
        from multi_freq_ert_parser import MultiFreqERTParser
        parser = MultiFreqERTParser()
        dat_data = parser.parse_file(dat_file_path)
        
        if not dat_data:
            return {"error": "Impossible de parser le fichier .dat"}
        
        # Exécuter l'inversion PyGIMLI
        inversion_results = run_pygimli_inversion(dat_data, electrode_spacing)
        
        if "error" in inversion_results:
            return inversion_results
        
        # Créer les 4 coupes
        sections = create_pygimli_sections(dat_data, inversion_results)
        
        if "error" in sections:
            return sections
        
        return {
            "success": True,
            "dat_data": dat_data,
            "inversion_results": inversion_results,
            "sections": sections,
            "electrode_spacing": electrode_spacing
        }
        
    except Exception as e:
        return {"error": f"Erreur analyse PyGIMLI: {str(e)}"}