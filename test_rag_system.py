#!/usr/bin/env python3.13
"""
Script de test du système RAG pour vérifier que tous les PDFs sont correctement traités
"""

import os
import sys

# Ajouter le dossier SETRAF au path
SETRAF_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SETRAF_PATH)

def test_rag_system():
    """Teste le système RAG sans Streamlit"""
    print("=" * 80)
    print("🧪 TEST DU SYSTÈME RAG - EXTRACTION COMPLÈTE DES PDFs")
    print("=" * 80)
    
    # Configuration des chemins
    RAG_DOCUMENTS_PATH = os.path.join(SETRAF_PATH, "rag_documents")
    
    print(f"\n📁 Dossier PDFs: {RAG_DOCUMENTS_PATH}")
    
    # Lister les PDFs
    if os.path.exists(RAG_DOCUMENTS_PATH):
        pdf_files = [f for f in os.listdir(RAG_DOCUMENTS_PATH) if f.endswith('.pdf')]
        print(f"\n📄 {len(pdf_files)} fichier(s) PDF trouvé(s):")
        for pdf in pdf_files:
            pdf_path = os.path.join(RAG_DOCUMENTS_PATH, pdf)
            size_kb = os.path.getsize(pdf_path) / 1024
            print(f"  - {pdf} ({size_kb:.1f} KB)")
    else:
        print(f"\n❌ Dossier non trouvé: {RAG_DOCUMENTS_PATH}")
        return
    
    # Test extraction avec pypdf
    print("\n" + "=" * 80)
    print("🔍 TEST D'EXTRACTION DES PDFs")
    print("=" * 80)
    
    try:
        from pypdf import PdfReader
        
        total_pages = 0
        total_chars = 0
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(RAG_DOCUMENTS_PATH, pdf_file)
            print(f"\n📖 Traitement: {pdf_file}")
            
            try:
                reader = PdfReader(pdf_path)
                n_pages = len(reader.pages)
                total_pages += n_pages
                
                text = ""
                for page_num in range(n_pages):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if len(page_text.strip()) > 50:
                        text += page_text + "\n\n"
                
                total_chars += len(text)
                
                print(f"  ✅ {n_pages} pages extraites")
                print(f"  ✅ {len(text)} caractères extraits")
                print(f"  ✅ {len(text.split())} mots extraits")
                
                # Afficher un extrait
                if text:
                    preview = text[:200].replace('\n', ' ')
                    print(f"  📝 Extrait: {preview}...")
                
            except Exception as e:
                print(f"  ❌ Erreur: {str(e)[:100]}")
        
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ DE L'EXTRACTION")
        print("=" * 80)
        print(f"📄 Total PDFs traités: {len(pdf_files)}")
        print(f"📚 Total pages extraites: {total_pages}")
        print(f"📝 Total caractères: {total_chars:,}")
        print(f"📏 Moyenne par PDF: {total_chars // len(pdf_files) if pdf_files else 0:,} chars")
        
        # Estimation du nombre de chunks (512 chars par chunk)
        chunk_size = 512
        estimated_chunks = total_chars // chunk_size
        print(f"\n🔢 Chunks estimés (512 chars): ~{estimated_chunks}")
        
    except ImportError as e:
        print(f"\n❌ Module manquant: {e}")
        return
    
    print("\n" + "=" * 80)
    print("✅ TEST TERMINÉ")
    print("=" * 80)

if __name__ == "__main__":
    test_rag_system()
