#!/usr/bin/env python3
"""
Script de test du système RAG pour géophysique ERT
Teste l'initialisation, la recherche vectorielle et la génération d'explications
"""

import sys
import os
sys.path.append('/home/belikan/KIbalione8/SETRAF')

def test_rag_system():
    """Test complet du système RAG"""
    print("🧪 TEST DU SYSTÈME RAG POUR GÉOPHYSIQUE ERT")
    print("=" * 50)

    try:
        # Import des modules
        print("📦 Import des modules...")
        from ERTest import ERTKnowledgeBase, initialize_rag_system
        print("✅ Modules importés")

        # Initialisation
        print("\n🔄 Initialisation du système RAG...")
        kb = ERTKnowledgeBase()

        # Test embeddings
        print("🧠 Test des embeddings...")
        if kb.initialize_embeddings():
            print("✅ Embeddings initialisés")
        else:
            print("❌ Échec initialisation embeddings")
            return False

        # Test base vectorielle
        print("📚 Test de la base vectorielle...")
        if kb.load_or_create_vectorstore():
            print(f"✅ Base vectorielle chargée : {len(kb.documents)} documents")
        else:
            print("❌ Échec chargement base vectorielle")
            return False

        # Test recherche vectorielle
        print("🔍 Test de recherche vectorielle...")
        query = "résistivité géophysique ERT eau"
        results = kb.search_knowledge_base(query, k=3)

        if results:
            print(f"✅ Recherche réussie : {len(results)} résultats")
            for i, result in enumerate(results):
                print(f"  {i+1}. Pertinence: {result['relevance_score']:.2f}")
                print(f"     Contenu: {result['content'][:100]}...")
        else:
            print("❌ Aucun résultat de recherche")
            return False

        # Test recherche web (si activée)
        if kb.web_search_enabled:
            print("🌐 Test de recherche web...")
            try:
                web_results = kb.search_web(query, max_results=2)
                if web_results:
                    print(f"✅ Recherche web réussie : {len(web_results)} résultats")
                    for i, result in enumerate(web_results):
                        print(f"  {i+1}. {result['title'][:50]}...")
                else:
                    print("⚠️ Aucun résultat web (API peut être inactive)")
            except Exception as e:
                print(f"⚠️ Erreur recherche web : {str(e)}")
        else:
            print("🌐 Recherche web désactivée")

        # Test contexte enrichi
        print("📝 Test de génération de contexte enrichi...")
        enhanced_context = kb.get_enhanced_context(query, use_web=False)
        if enhanced_context:
            print(f"✅ Contexte généré : {len(enhanced_context)} caractères")
            print(f"Aperçu: {enhanced_context[:200]}...")
        else:
            print("❌ Échec génération contexte")
            return False

        print("\n" + "=" * 50)
        print("🎉 TEST RAG RÉUSSI !")
        print("Le système est prêt à enrichir les explications LLM.")
        return True

    except ImportError as e:
        print(f"❌ Erreur d'import : {str(e)}")
        print("Vérifiez que toutes les dépendances sont installées:")
        print("pip install sentence-transformers faiss-cpu langchain pypdf requests")
        return False

    except Exception as e:
        print(f"❌ Erreur inattendue : {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_system()
    exit(0 if success else 1)</content>
<parameter name="filePath">/home/belikan/KIbalione8/SETRAF/test_rag.py