#!/usr/bin/env python3
"""
Test rapide du système RAG optimisé pour SETRAF
"""

import sys
import os
sys.path.append('/home/belikan/KIbalione8/SETRAF')

def test_rag_lightweight():
    """Test rapide du système RAG optimisé"""
    print("🧪 Test du système RAG optimisé...")

    try:
        # Importer les classes nécessaires
        from ERTest import ERTKnowledgeBase

        # Créer une instance
        kb = ERTKnowledgeBase()
        print("✅ Instance RAG créée")

        # Tester l'initialisation rapide
        print("🔄 Test initialisation...")
        success = kb.initialize_embeddings()
        if success:
            print("✅ Embeddings chargés rapidement")
        else:
            print("❌ Échec chargement embeddings")
            return False

        # Tester la création/chargement de la base
        print("🔄 Test base vectorielle...")
        success = kb.load_or_create_vectorstore()
        if success:
            print(f"✅ Base vectorielle OK : {len(kb.documents)} documents")
        else:
            print("❌ Échec base vectorielle")
            return False

        # Tester une recherche rapide
        print("🔄 Test recherche rapide...")
        results = kb.search_knowledge_base("résistivité ERT", k=1)
        if results and len(results) > 0:
            print(f"✅ Recherche OK : {len(results)} résultats")
            print(f"   Score: {results[0]['relevance_score']:.2f}")
        else:
            print("❌ Échec recherche")
            return False

        # Tester le contexte enrichi
        print("🔄 Test contexte enrichi...")
        context = kb.get_enhanced_context("eau douce résistivité", use_web=False)
        if context and len(context) > 0:
            print(f"✅ Contexte généré : {len(context)} caractères")
        else:
            print("❌ Échec contexte")
            return False

        print("🎉 Test RAG réussi ! Système optimisé opérationnel.")
        return True

    except Exception as e:
        print(f"❌ Erreur test RAG : {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        print("🔄 Test contexte enrichi...")
        context = kb.get_enhanced_context("eau douce résistivité", use_web=False)
        if context:
            print(f"✅ Contexte généré : {len(context)} caractères")
        else:
            print("❌ Échec contexte")

        print("🎉 Test RAG réussi ! Système optimisé opérationnel.")
        return True

    except Exception as e:
        print(f"❌ Erreur test RAG : {str(e)}")
        return False

if __name__ == "__main__":
    test_rag_lightweight()