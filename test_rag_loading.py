#!/usr/bin/env python3.13
"""Test du chargement de la base vectorielle RAG"""

import os
import sys
import pickle
import faiss

SETRAF_PATH = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(SETRAF_PATH, "vector_db")

print("=" * 80)
print("🧪 TEST DE CHARGEMENT DE LA BASE VECTORIELLE")
print("=" * 80)

# Vérifier les fichiers
db_file = os.path.join(VECTOR_DB_PATH, "ert_knowledge_light.faiss")
docs_file = os.path.join(VECTOR_DB_PATH, "ert_documents_light.pkl")

print(f"\n📂 Fichiers:")
print(f"  - FAISS: {db_file}")
print(f"  - Exists: {os.path.exists(db_file)}")
if os.path.exists(db_file):
    print(f"  - Taille: {os.path.getsize(db_file) / 1024:.1f} KB")

print(f"\n  - PKL: {docs_file}")
print(f"  - Exists: {os.path.exists(docs_file)}")
if os.path.exists(docs_file):
    print(f"  - Taille: {os.path.getsize(docs_file) / 1024:.1f} KB")

# Charger l'index FAISS
print("\n🔄 Chargement FAISS...")
try:
    index = faiss.read_index(db_file)
    print(f"✅ Index chargé: {index.ntotal} vecteurs")
except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)

# Charger les documents
print("\n🔄 Chargement documents...")
try:
    with open(docs_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"📦 Type de données: {type(data)}")
    
    if isinstance(data, dict):
        print(f"📋 Clés du dict: {data.keys()}")
        texts = data.get('texts', [])
        metadatas = data.get('metadatas', [])
    else:
        texts = data
        metadatas = []
    
    print(f"\n✅ {len(texts)} chunks chargés")
    
    # Statistiques
    total_chars = sum(len(t) for t in texts)
    total_words = sum(len(t.split()) for t in texts)
    avg_chars = total_chars // len(texts) if texts else 0
    
    print(f"\n📊 STATISTIQUES:")
    print(f"  - Chunks: {len(texts)}")
    print(f"  - Caractères totaux: {total_chars:,}")
    print(f"  - Mots totaux: {total_words:,}")
    print(f"  - Moyenne par chunk: {avg_chars} chars")
    
    # Afficher les 3 premiers chunks
    print(f"\n📄 PREMIERS CHUNKS:")
    for i, text in enumerate(texts[:3], 1):
        preview = text[:150].replace('\n', ' ')
        print(f"\n{i}. ({len(text)} chars) {preview}...")
    
    # Vérifier cohérence
    print(f"\n🔍 VÉRIFICATION:")
    if index.ntotal == len(texts):
        print(f"✅ Cohérence FAISS/Documents: {index.ntotal} vecteurs = {len(texts)} chunks")
    else:
        print(f"⚠️ INCOHÉRENCE: {index.ntotal} vecteurs != {len(texts)} chunks")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ TEST TERMINÉ")
print("=" * 80)
