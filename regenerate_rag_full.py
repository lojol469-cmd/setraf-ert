#!/usr/bin/env python3.13
"""
Script pour régénérer complètement la base vectorielle RAG avec TOUS les PDFs
"""

import os
import sys
import pickle
import numpy as np

# Configuration
SETRAF_PATH = os.path.dirname(os.path.abspath(__file__))
RAG_DOCUMENTS_PATH = os.path.join(SETRAF_PATH, "rag_documents")
VECTOR_DB_PATH = os.path.join(SETRAF_PATH, "vector_db")

print("=" * 80)
print("🔄 RÉGÉNÉRATION COMPLÈTE DE LA BASE VECTORIELLE RAG")
print("=" * 80)

# 1. Extraction des PDFs
print("\n📖 ÉTAPE 1: Extraction des PDFs")
print("-" * 80)

try:
    from pypdf import PdfReader
    
    pdf_files = [f for f in os.listdir(RAG_DOCUMENTS_PATH) if f.endswith('.pdf')]
    print(f"📄 {len(pdf_files)} PDF(s) trouvé(s): {', '.join(pdf_files)}")
    
    all_texts = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(RAG_DOCUMENTS_PATH, pdf_file)
        print(f"\n📖 Traitement: {pdf_file}")
        
        reader = PdfReader(pdf_path)
        text = ""
        
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if len(page_text.strip()) > 50:
                text += page_text + "\n\n"
        
        all_texts.append({
            "title": f"PDF: {pdf_file}",
            "content": text,
            "pages": len(reader.pages),
            "source": pdf_file
        })
        print(f"  ✅ {len(reader.pages)} pages, {len(text)} caractères")
    
except Exception as e:
    print(f"❌ Erreur extraction: {e}")
    sys.exit(1)

# 2. Documents par défaut
print("\n📚 ÉTAPE 2: Ajout documents par défaut")
print("-" * 80)

default_docs = [
    {
        "title": "Résistivité ERT - Échelle rapide",
        "content": """
        ÉCHELLE RÉSISTIVITÉ ERT:
        0.01-1 Ω·m : EAU DE MER / MINÉRAUX CONDUCTEURS
        1-10 Ω·m : EAU SAUMÂTRE / ARGILES SATURÉES
        10-100 Ω·m : EAU DOUCE / SOLS FINS / AQUIFÈRE ARGILEUX
        100-1000 Ω·m : SABLES SATURÉS / GRAVIERS / AQUIFÈRE PRODUCTIF
        1000-10000 Ω·m : ROCHES SÉDIMENTAIRES / SOCLE ALTÉRÉ
        >10000 Ω·m : SOCLE CRISTALLIN / GRANITE / GNEISS
        
        MÉTHODES D'ACQUISITION:
        - Wenner: Bonne pénétration verticale
        - Schlumberger: Compromis résolution/profondeur
        - Dipôle-dipôle: Haute résolution latérale
        - Pôle-pôle: Grande profondeur d'investigation
        """,
        "source": "default"
    },
    {
        "title": "Interprétation géophysique ERT",
        "content": """
        ANALYSE DES PSEUDO-SECTIONS:
        - Représentation 2D des résistivités apparentes
        - Identification des anomalies conductrices/résistantes
        - Corrélation avec la géologie locale
        
        INVERSION DE DONNÉES:
        - Transformation pseudo-section → vraie résistivité
        - Modèle 2D/3D du sous-sol
        - Contraintes géologiques et hydrogéologiques
        
        APPLICATIONS HYDROGÉOLOGIQUES:
        - Détection d'aquifères (10-100 Ω·m)
        - Cartographie du socle rocheux (>1000 Ω·m)
        - Identification des argiles (1-10 Ω·m)
        - Évaluation de la profondeur des formations
        """,
        "source": "default"
    }
]

all_texts.extend(default_docs)
print(f"✅ {len(default_docs)} documents par défaut ajoutés")
print(f"📊 TOTAL: {len(all_texts)} documents à chunker")

# 3. Découpage en chunks
print("\n✂️ ÉTAPE 3: Découpage en chunks (512 caractères)")
print("-" * 80)

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )
    
    texts = []
    metadatas = []
    
    for doc in all_texts:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                texts.append(chunk.strip())
                metadatas.append({
                    "title": doc["title"],
                    "source": doc.get("source", "unknown")
                })
    
    print(f"✅ {len(texts)} chunks générés")
    print(f"📏 Longueur moyenne: {sum(len(t) for t in texts) // len(texts)} caractères")
    
except Exception as e:
    print(f"❌ Erreur chunking: {e}")
    sys.exit(1)

# 4. Génération des embeddings
print("\n🧠 ÉTAPE 4: Génération des embeddings (all-MiniLM-L6-v2)")
print("-" * 80)

try:
    from sentence_transformers import SentenceTransformer
    
    embeddings_path = os.path.join(SETRAF_PATH, "models/embeddings/sentence-transformers--all-MiniLM-L6-v2")
    
    if not os.path.exists(embeddings_path):
        print(f"❌ Modèle d'embeddings non trouvé: {embeddings_path}")
        sys.exit(1)
    
    print(f"📂 Chargement depuis: {embeddings_path}")
    embeddings_model = SentenceTransformer(embeddings_path, device='cpu')
    embeddings_model.eval()
    
    print(f"🔄 Encodage de {len(texts)} chunks...")
    
    # Traitement par batch pour la mémoire
    batch_size = 32
    embeddings_list = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embeddings_model.encode(batch_texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings_list.append(batch_embeddings)
        print(f"  ✅ Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
    
    embeddings_array = np.vstack(embeddings_list)
    print(f"✅ Embeddings générés: shape {embeddings_array.shape}")
    
except Exception as e:
    print(f"❌ Erreur embeddings: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Création de l'index FAISS
print("\n🗄️ ÉTAPE 5: Création index FAISS")
print("-" * 80)

try:
    import faiss
    
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array.astype('float32'))
    
    print(f"✅ Index FAISS créé: {index.ntotal} vecteurs, dimension {dimension}")
    
except Exception as e:
    print(f"❌ Erreur FAISS: {e}")
    sys.exit(1)

# 6. Sauvegarde
print("\n💾 ÉTAPE 6: Sauvegarde de la base vectorielle")
print("-" * 80)

try:
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    
    db_file = os.path.join(VECTOR_DB_PATH, "ert_knowledge_light.faiss")
    docs_file = os.path.join(VECTOR_DB_PATH, "ert_documents_light.pkl")
    
    # Sauvegarder FAISS
    faiss.write_index(index, db_file)
    print(f"✅ Index FAISS sauvegardé: {db_file}")
    
    # Sauvegarder documents
    with open(docs_file, 'wb') as f:
        pickle.dump({
            'texts': texts,
            'metadatas': metadatas
        }, f)
    print(f"✅ Documents sauvegardés: {docs_file}")
    
except Exception as e:
    print(f"❌ Erreur sauvegarde: {e}")
    sys.exit(1)

# 7. Vérification
print("\n✅ ÉTAPE 7: Vérification")
print("-" * 80)

# Test de recherche
test_query = "résistivité de l'eau"
print(f"🔍 Test de recherche: '{test_query}'")

query_embedding = embeddings_model.encode([test_query], convert_to_numpy=True)
distances, indices = index.search(query_embedding.astype('float32'), k=3)

print(f"\n📊 Top 3 résultats:")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
    print(f"\n{i}. Distance: {dist:.4f}")
    print(f"   Chunk #{idx}: {texts[idx][:150]}...")

print("\n" + "=" * 80)
print("🎉 RÉGÉNÉRATION COMPLÈTE TERMINÉE!")
print("=" * 80)
print(f"📚 {len(texts)} chunks indexés")
print(f"📊 {len(pdf_files)} PDFs traités")
print(f"🎯 Dimension: {dimension}")
print(f"💾 Taille index: {os.path.getsize(db_file) / 1024:.1f} KB")
print(f"💾 Taille docs: {os.path.getsize(docs_file) / 1024:.1f} KB")
print("\n✅ Le système RAG est maintenant prêt à être utilisé dans Streamlit!")
