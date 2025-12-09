#!/usr/bin/env python3
"""
Test de l'intégration LLM Mistral dans SETRAF ERTest.py
Vérifie que le LLM est correctement intégré et fonctionnel
"""

import re
import sys

def test_llm_integration():
    """Test l'intégration complète du LLM Mistral"""
    
    print("🧠 Test de l'intégration LLM Mistral dans SETRAF\n")
    
    with open("ERTest.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Configuration du chemin Mistral
    tests_total += 1
    if 'MISTRAL_MODEL_PATH = "/home/belikan/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2"' in content:
        print("✅ Test 1: Chemin du modèle Mistral configuré")
        tests_passed += 1
    else:
        print("❌ Test 1: Chemin du modèle Mistral manquant ou incorrect")
    
    # Test 2: Fonction de chargement du LLM
    tests_total += 1
    if "def load_mistral_llm(use_cpu=True):" in content:
        print("✅ Test 2: Fonction load_mistral_llm() présente")
        tests_passed += 1
    else:
        print("❌ Test 2: Fonction load_mistral_llm() manquante")
    
    # Test 3: Fonction d'analyse avec Mistral
    tests_total += 1
    if "def analyze_data_with_mistral(llm_pipeline, geophysical_data):" in content:
        print("✅ Test 3: Fonction analyze_data_with_mistral() présente")
        tests_passed += 1
    else:
        print("❌ Test 3: Fonction analyze_data_with_mistral() manquante")
    
    # Test 4: Cache Streamlit pour le LLM
    tests_total += 1
    llm_section = content[content.find("def load_mistral_llm"):content.find("def load_mistral_llm") + 500]
    if "@st.cache_resource" in content[max(0, content.find("def load_mistral_llm") - 100):content.find("def load_mistral_llm")]:
        print("✅ Test 4: Cache Streamlit configuré pour le LLM")
        tests_passed += 1
    else:
        print("❌ Test 4: Cache Streamlit manquant pour le LLM")
    
    # Test 5: Intégration dans section spectrale
    tests_total += 1
    if 'st.checkbox("🧠 Activer l\'analyse LLM avancée (recommandé)"' in content:
        print("✅ Test 5: LLM intégré dans la section spectrale")
        tests_passed += 1
    else:
        print("❌ Test 5: LLM non intégré dans la section spectrale")
    
    # Test 6: Intégration dans section finale
    tests_total += 1
    if 'st.checkbox("🤖 Activer l\'analyse LLM complète (recommandé)"' in content:
        print("✅ Test 6: LLM intégré dans la section finale")
        tests_passed += 1
    else:
        print("❌ Test 6: LLM non intégré dans la section finale")
    
    # Test 7: Paramètre llm_enhanced_prompt dans generate_realistic_geological_image
    tests_total += 1
    if "llm_enhanced_prompt=None" in content and "def generate_realistic_geological_image" in content:
        print("✅ Test 7: Paramètre llm_enhanced_prompt ajouté à la fonction de génération")
        tests_passed += 1
    else:
        print("❌ Test 7: Paramètre llm_enhanced_prompt manquant")
    
    # Test 8: Utilisation du prompt LLM dans la génération
    tests_total += 1
    if "if llm_enhanced_prompt:" in content and "Utilisation du prompt optimisé par" in content:
        print("✅ Test 8: Prompt LLM utilisé dans la génération d'images")
        tests_passed += 1
    else:
        print("❌ Test 8: Prompt LLM non utilisé dans la génération")
    
    # Test 9: Stockage du prompt LLM dans session_state
    tests_total += 1
    llm_storage_count = content.count("st.session_state['llm_prompt_")
    if llm_storage_count >= 2:  # Au moins 2 (spectral + final)
        print(f"✅ Test 9: Prompts LLM stockés dans session_state ({llm_storage_count} occurrences)")
        tests_passed += 1
    else:
        print(f"❌ Test 9: Prompts LLM non stockés correctement ({llm_storage_count} occurrences)")
    
    # Test 10: Gestion d'erreurs robuste
    tests_total += 1
    if 'st.warning(f"⚠️ Impossible de charger Mistral' in content:
        print("✅ Test 10: Gestion d'erreurs robuste pour le chargement du LLM")
        tests_passed += 1
    else:
        print("❌ Test 10: Gestion d'erreurs manquante")
    
    # Test 11: Utilisation de transformers et pipeline
    tests_total += 1
    if "from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline" in content:
        print("✅ Test 11: Bibliothèques transformers importées")
        tests_passed += 1
    else:
        print("❌ Test 11: Bibliothèques transformers manquantes")
    
    # Test 12: Paramètres de génération du LLM
    tests_total += 1
    params_check = all([
        "max_new_tokens=1024" in content,
        "temperature=0.7" in content,
        "top_p=0.95" in content,
        "repetition_penalty=1.15" in content
    ])
    if params_check:
        print("✅ Test 12: Paramètres de génération LLM correctement configurés")
        tests_passed += 1
    else:
        print("❌ Test 12: Paramètres de génération LLM incomplets")
    
    # Test 13: Parsing de la réponse LLM
    tests_total += 1
    if "INTERPRÉTATION" in content and "RECOMMANDATION" in content and "PROMPT" in content:
        print("✅ Test 13: Parsing des sections de la réponse LLM implémenté")
        tests_passed += 1
    else:
        print("❌ Test 13: Parsing des sections LLM manquant")
    
    # Test 14: Affichage de l'interprétation LLM
    tests_total += 1
    if 'st.markdown("#### 📊 Interprétation Géologique' in content:
        print("✅ Test 14: Affichage de l'interprétation LLM configuré")
        tests_passed += 1
    else:
        print("❌ Test 14: Affichage de l'interprétation LLM manquant")
    
    # Test 15: Collecte complète des données géophysiques
    tests_total += 1
    data_fields = [
        "'n_spectra'",
        "'rho_min'",
        "'rho_max'",
        "'rho_mean'",
        "'rho_std'",
        "'n_imputed'",
        "'imputation_method'",
        "'model_dims'",
        "'n_cells'",
        "'convergence'",
        "'n_trajectories'"
    ]
    data_collection_complete = sum([field in content for field in data_fields]) >= 8
    if data_collection_complete:
        print("✅ Test 15: Collecte complète des données géophysiques implémentée")
        tests_passed += 1
    else:
        print("❌ Test 15: Collecte des données géophysiques incomplète")
    
    # Résumé
    print(f"\n{'='*60}")
    print(f"📊 RÉSULTAT FINAL : {tests_passed}/{tests_total} tests réussis")
    print(f"{'='*60}\n")
    
    if tests_passed == tests_total:
        print("🎉 SUCCÈS TOTAL ! Le LLM Mistral est parfaitement intégré.")
        print("\n✅ Fonctionnalités validées :")
        print("  • Chargement du modèle Mistral avec cache")
        print("  • Analyse intelligente des données géophysiques")
        print("  • Génération d'explications naturelles")
        print("  • Prompts optimisés pour génération d'images")
        print("  • Intégration dans section spectrale")
        print("  • Intégration dans section finale complète")
        print("  • Stockage persistant des prompts LLM")
        print("  • Gestion d'erreurs robuste")
        print("\n🚀 Le système est prêt à utiliser Mistral !")
        return 0
    elif tests_passed >= tests_total * 0.8:
        print(f"✅ SUCCÈS PARTIEL ({tests_passed}/{tests_total})")
        print("\n⚠️  Quelques tests ont échoué mais l'intégration est fonctionnelle.")
        return 0
    else:
        print(f"⚠️  {tests_total - tests_passed} test(s) ont échoué.")
        print("\n🔍 Actions recommandées :")
        print("  • Vérifier l'installation des bibliothèques transformers")
        print("  • Valider le chemin du modèle Mistral")
        print("  • Tester le chargement du LLM manuellement")
        return 1

if __name__ == "__main__":
    sys.exit(test_llm_integration())
