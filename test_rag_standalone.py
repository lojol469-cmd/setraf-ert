#!/usr/bin/env python3
"""
Test STANDALONE du système RAG pour SETRAF (sans Streamlit)
"""

import os
import sys
import numpy as np

# Configuration pour éviter les erreurs CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Chemins
RAG_DOCUMENTS_PATH = "/home/belikan/KIbalione8/SETRAF/rag_documents"
VECTOR_DB_PATH = "/home/belikan/KIbalione8/SETRAF/vector_db"

class StandaloneRAG:
    """Version standalone du système RAG (sans Streamlit)"""
    
    def __init__(self):
        self.vectorstore = None
        self.embeddings = None
        self.documents = []
        self.initialized = False
        
    def initialize_embeddings(self):
        """Charge le modèle d'embeddings"""
        try:
            print("🔄 Chargement du modèle d'embeddings...")
            from sentence_transformers import SentenceTransformer
            
            # Charger le modèle léger
            self.embeddings = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder="/home/belikan/.cache/huggingface",
                device='cpu'
            )
            
            # S'assurer qu'il est bien sur CPU
            self.embeddings = self.embeddings.to('cpu')
            self.embeddings.eval()
            self.embeddings.max_seq_length = 256
            
            # Test rapide
            test_embed = self.embeddings.encode(["test"], show_progress_bar=False)
            print(f"✅ Modèle chargé (dimension: {test_embed.shape[1]})")
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement embeddings : {str(e)}")
            return False
    
    def load_or_create_vectorstore(self):
        """Charge ou crée la base vectorielle"""
        try:
            import faiss
            import pickle
            
            os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            db_file = os.path.join(VECTOR_DB_PATH, "ert_knowledge_light.faiss")
            docs_file = os.path.join(VECTOR_DB_PATH, "ert_documents_light.pkl")
            
            if os.path.exists(db_file) and os.path.exists(docs_file):
                print("🔄 Chargement de la base vectorielle existante...")
                self.vectorstore = faiss.read_index(db_file)
                with open(docs_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['texts']
                print(f"✅ Base chargée : {len(self.documents)} documents")
                self.initialized = True
                return True
            else:
                print("🔄 Création de la base vectorielle...")
                return self.create_vectorstore()
                
        except Exception as e:
            print(f"❌ Erreur base vectorielle : {str(e)}")
            return False
    
    def create_vectorstore(self):
        """Crée une nouvelle base vectorielle"""
        try:
            import faiss
            import pickle
            
            # Documents par défaut
            default_docs = [
                {
                    "title": "Résistivité ERT",
                    "content": """
                    ÉCHELLE RÉSISTIVITÉ ERT:
                    0.01-1 Ω·m : EAU DE MER / MINÉRAUX
                    1-10 Ω·m : EAU SAUMÂTRE / ARGILES
                    10-100 Ω·m : EAU DOUCE / SOLS FINS
                    100-1000 Ω·m : SABLES SATURÉS
                    1000-10000 Ω·m : ROCHES SÉDIMENTAIRES
                    >10000 Ω·m : SOCLE CRISTALLIN
                    """
                },
                {
                    "title": "Méthodes ERT",
                    "content": """
                    MÉTHODES ERT:
                    PSEUDO-SECTIONS: Représentation 2D rapide des données brutes
                    INVERSION: Reconstruction 3D des valeurs réelles de résistivité
                    CLASSIFICATION: Regroupement par zones de résistivité similaire
                    """
                }
            ]
            
            # Découper en chunks
            texts = []
            for doc in default_docs:
                content = doc["content"].strip()
                if len(content) > 100:
                    texts.append(content)
            
            print(f"🔄 Génération des embeddings pour {len(texts)} documents...")
            
            # Générer les embeddings
            embeddings_array = self.embeddings.encode(texts, show_progress_bar=True)
            
            # Créer l'index FAISS
            dimension = embeddings_array.shape[1]
            self.vectorstore = faiss.IndexFlatL2(dimension)
            self.vectorstore.add(embeddings_array.astype('float32'))
            
            # Sauvegarder
            db_file = os.path.join(VECTOR_DB_PATH, "ert_knowledge_light.faiss")
            docs_file = os.path.join(VECTOR_DB_PATH, "ert_documents_light.pkl")
            
            faiss.write_index(self.vectorstore, db_file)
            with open(docs_file, 'wb') as f:
                pickle.dump({'texts': texts, 'metadatas': [{}]*len(texts)}, f)
            
            self.documents = texts
            self.initialized = True
            print(f"✅ Base créée et sauvegardée : {len(texts)} documents")
            return True
            
        except Exception as e:
            print(f"❌ Erreur création base : {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_knowledge_base(self, query, k=2):
        """Recherche dans la base vectorielle"""
        try:
            if not self.vectorstore or not self.embeddings or not self.initialized:
                print("❌ Base non initialisée")
                return []
            
            # Encoder la requête
            query_embedding = self.embeddings.encode([query], show_progress_bar=False)
            
            # Rechercher
            distances, indices = self.vectorstore.search(query_embedding.astype('float32'), k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'content': self.documents[idx][:300],
                        'distance': float(distances[0][i]),
                        'relevance_score': max(0, 1.0 - float(distances[0][i]))
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Erreur recherche : {str(e)}")
            return []
    
    def get_enhanced_context(self, query):
        """Obtient un contexte enrichi"""
        results = self.search_knowledge_base(query, k=2)
        
        if not results:
            return ""
        
        context_parts = ["=== CONTEXTE RAG ==="]
        for i, result in enumerate(results):
            context_parts.append(f"\nRésultat {i+1} (score: {result['relevance_score']:.2f}):")
            context_parts.append(result['content'])
        
        return "\n".join(context_parts)


def test_rag_system():
    """Teste le système RAG standalone"""
    print("\n" + "="*60)
    print("🧪 TEST SYSTÈME RAG OPTIMISÉ")
    print("="*60 + "\n")
    
    try:
        # Créer l'instance
        rag = StandaloneRAG()
        print("✅ Instance RAG créée\n")
        
        # Initialiser les embeddings
        if not rag.initialize_embeddings():
            print("❌ Échec initialisation embeddings")
            return False
        print()
        
        # Charger/créer la base vectorielle
        if not rag.load_or_create_vectorstore():
            print("❌ Échec chargement base vectorielle")
            return False
        print()
        
        # Test de recherche
        print("🔍 Test de recherche...")
        query = "résistivité de l'eau douce"
        results = rag.search_knowledge_base(query, k=2)
        
        if results:
            print(f"✅ {len(results)} résultat(s) trouvé(s):")
            for i, result in enumerate(results):
                print(f"\n  Résultat {i+1}:")
                print(f"    Score: {result['relevance_score']:.3f}")
                print(f"    Distance: {result['distance']:.3f}")
                print(f"    Contenu: {result['content'][:100]}...")
        else:
            print("❌ Aucun résultat trouvé")
            return False
        print()
        
        # Test de contexte enrichi
        print("📝 Test de contexte enrichi...")
        context = rag.get_enhanced_context("argile résistivité")
        if context:
            print(f"✅ Contexte généré ({len(context)} caractères)")
            print(f"\nAperçu:\n{context[:200]}...\n")
        else:
            print("❌ Échec génération contexte")
            return False
        
        print("="*60)
        print("🎉 TOUS LES TESTS RÉUSSIS !")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR FATALE : {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1)
