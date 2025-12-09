#!/usr/bin/env python3
"""
🔍 Script de vérification de l'installation SETRAF
Vérifie que tous les modèles et dépendances sont présents
"""

import os
import sys
from pathlib import Path

def check_directory(path, name, required=True):
    """Vérifie qu'un dossier existe"""
    exists = os.path.exists(path)
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {name}: {path}")
    if exists and os.path.isdir(path):
        size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
        size_gb = size / (1024**3)
        if size_gb > 0.1:
            print(f"   └─ Taille: {size_gb:.2f} GB")
    return exists

def check_file(path, name, required=True):
    """Vérifie qu'un fichier existe"""
    exists = os.path.exists(path)
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {name}: {path}")
    if exists:
        size = os.path.getsize(path)
        size_mb = size / (1024**2)
        if size_mb > 1:
            print(f"   └─ Taille: {size_mb:.1f} MB")
    return exists

def check_python_package(package_name):
    """Vérifie qu'un package Python est installé"""
    try:
        __import__(package_name)
        print(f"✅ Python package: {package_name}")
        return True
    except ImportError:
        print(f"❌ Python package: {package_name} (MANQUANT)")
        return False

def main():
    print("=" * 70)
    print("🔍 VÉRIFICATION DE L'INSTALLATION SETRAF")
    print("=" * 70)
    
    # Déterminer le chemin de base
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n📁 Dossier SETRAF: {script_dir}\n")
    
    # Vérifier la structure des dossiers
    print("📂 STRUCTURE DES DOSSIERS")
    print("-" * 70)
    
    all_checks = []
    
    # Modèles
    all_checks.append(check_directory(
        os.path.join(script_dir, "models"), 
        "Dossier models/", 
        required=True
    ))
    all_checks.append(check_directory(
        os.path.join(script_dir, "models/mistral-7b"), 
        "  └─ Mistral-7B", 
        required=True
    ))
    
    mistral_files = [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "config.json",
        "tokenizer.json"
    ]
    
    for fname in mistral_files:
        fpath = os.path.join(script_dir, "models/mistral-7b", fname)
        all_checks.append(check_file(fpath, f"     ├─ {fname}", required=True))
    
    all_checks.append(check_directory(
        os.path.join(script_dir, "models/clip"), 
        "  └─ CLIP", 
        required=False
    ))
    
    all_checks.append(check_directory(
        os.path.join(script_dir, "models/embeddings/sentence-transformers--all-MiniLM-L6-v2"), 
        "  └─ Embeddings (all-MiniLM-L6-v2)", 
        required=True
    ))
    
    # Dossiers ML
    all_checks.append(check_directory(
        os.path.join(script_dir, "ml_models"), 
        "Dossier ml_models/", 
        required=True
    ))
    
    all_checks.append(check_directory(
        os.path.join(script_dir, "vector_db"), 
        "Dossier vector_db/", 
        required=True
    ))
    
    all_checks.append(check_directory(
        os.path.join(script_dir, "rag_documents"), 
        "Dossier rag_documents/", 
        required=True
    ))
    
    # Fichiers de base vectorielle
    faiss_file = os.path.join(script_dir, "vector_db/ert_knowledge_light.faiss")
    docs_file = os.path.join(script_dir, "vector_db/ert_documents_light.pkl")
    
    if os.path.exists(faiss_file):
        check_file(faiss_file, "  └─ Index FAISS", required=False)
        check_file(docs_file, "  └─ Documents", required=False)
    else:
        print("  ⚠️  Base vectorielle sera créée au premier lancement")
    
    # Vérifier les packages Python
    print("\n🐍 PACKAGES PYTHON")
    print("-" * 70)
    
    packages = [
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "sklearn",
        "transformers",
        "torch",
        "sentence_transformers",
        "faiss",
        "joblib",
        "pygimli",
        "plotly"
    ]
    
    for pkg in packages:
        # Gestion des noms spéciaux
        import_name = pkg
        if pkg == "sklearn":
            import_name = "sklearn"
        elif pkg == "faiss":
            import_name = "faiss"
        
        all_checks.append(check_python_package(import_name))
    
    # Résumé
    print("\n" + "=" * 70)
    success_count = sum(all_checks)
    total_count = len(all_checks)
    
    if success_count == total_count:
        print("✅ INSTALLATION COMPLÈTE")
        print(f"   Tous les composants sont présents ({success_count}/{total_count})")
        print("\n🚀 Vous pouvez lancer l'application avec:")
        print(f"   cd {script_dir}")
        print("   streamlit run ERTest.py")
        return 0
    else:
        print("⚠️  INSTALLATION INCOMPLÈTE")
        print(f"   {success_count}/{total_count} composants présents")
        print(f"   {total_count - success_count} composants manquants")
        
        if not os.path.exists(os.path.join(script_dir, "models/mistral-7b")):
            print("\n❌ Modèle Mistral-7B manquant !")
            print("   Copiez-le depuis le cache HuggingFace:")
            print("   cp -rL ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/*/ models/mistral-7b/")
        
        return 1
    
    print("=" * 70)

if __name__ == "__main__":
    sys.exit(main())
