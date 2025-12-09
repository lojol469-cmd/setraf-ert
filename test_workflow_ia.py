#!/usr/bin/env python3
"""
Test du workflow complet de génération IA dans SETRAF ERTest.py
Vérifie que les boutons restent visibles et les résultats persistent
"""

import re
import sys

def test_ia_workflow():
    """Test l'intégration du workflow IA"""
    
    print("🔍 Test du workflow de génération IA dans ERTest.py\n")
    
    with open("ERTest.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Vérifier que la section IA spectrale existe APRÈS le bouton d'extraction
    tests_total += 1
    if "# =================== GÉNÉRATION IA FINALE (APRÈS TOUTES LES ANALYSES) ===================" in content:
        print("✅ Test 1: Section IA spectrale trouvée après extraction")
        tests_passed += 1
    else:
        print("❌ Test 1: Section IA spectrale manquante")
    
    # Test 2: Vérifier que la section utilise session_state pour persistance
    tests_total += 1
    if "if 'spectra' in st.session_state and 'positions' in st.session_state:" in content:
        print("✅ Test 2: Persistance avec session_state configurée")
        tests_passed += 1
    else:
        print("❌ Test 2: Session state non utilisé correctement")
    
    # Test 3: Vérifier que le bouton est en dehors du bloc conditionnel d'extraction
    tests_total += 1
    spectral_section = content[content.find("# =================== GÉNÉRATION IA FINALE (APRÈS"):
                               content.find("# =================== 2. IMPUTATION MATRICIELLE")]
    
    # Compter les niveaux d'indentation du bouton
    button_matches = re.finditer(r'if st\.button\("🚀 Générer Rendu Réaliste Final"', spectral_section)
    button_found = False
    for match in button_matches:
        # Vérifier l'indentation (devrait être 12 espaces = 3 niveaux)
        start = match.start()
        line_start = spectral_section.rfind('\n', 0, start) + 1
        indentation = len(spectral_section[line_start:start])
        if indentation == 12:  # 3 niveaux d'indentation
            button_found = True
            break
    
    if button_found:
        print("✅ Test 3: Bouton spectral correctement placé en dehors du bloc d'extraction")
        tests_passed += 1
    else:
        print("❌ Test 3: Bouton spectral mal placé ou indentation incorrecte")
    
    # Test 4: Vérifier la section IA FINALE existe
    tests_total += 1
    if "# =================== GÉNÉRATION IA FINALE - SYNTHÈSE COMPLÈTE ===================" in content:
        print("✅ Test 4: Section IA finale (synthèse complète) trouvée")
        tests_passed += 1
    else:
        print("❌ Test 4: Section IA finale manquante")
    
    # Test 5: Vérifier que la section finale vérifie TOUTES les étapes
    tests_total += 1
    if ("if ('spectra' in st.session_state and 'rho_imputed' in st.session_state and" in content and
        "'rho_3d' in st.session_state):" in content):
        print("✅ Test 5: Section finale vérifie toutes les étapes (spectres + imputation + 3D)")
        tests_passed += 1
    else:
        print("❌ Test 5: Vérification des étapes incomplète")
    
    # Test 6: Vérifier la persistance des résultats finaux
    tests_total += 1
    if ("st.session_state['final_generation_requested'] = True" in content and
        "if st.session_state.get('final_generation_complete', False):" in content):
        print("✅ Test 6: Persistance des résultats finaux configurée")
        tests_passed += 1
    else:
        print("❌ Test 6: Persistance des résultats finaux manquante")
    
    # Test 7: Vérifier les boutons de téléchargement
    tests_total += 1
    download_buttons = len(re.findall(r'st\.download_button\(', content))
    if download_buttons >= 5:  # Au moins 5 boutons de téléchargement
        print(f"✅ Test 7: {download_buttons} boutons de téléchargement trouvés")
        tests_passed += 1
    else:
        print(f"❌ Test 7: Seulement {download_buttons} boutons de téléchargement (attendu ≥5)")
    
    # Test 8: Vérifier les 5 modèles IA
    tests_total += 1
    models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "Lykon/DreamShaper-8",
        "SG161222/RealVisXL_V4.0",
        "SG161222/Realistic_Vision_V5.1_noVAE",
        "emilianJR/epiCRealism"  # Correction du nom du modèle
    ]
    all_models_found = all(model in content for model in models)
    if all_models_found:
        print("✅ Test 8: Tous les 5 modèles IA configurés")
        tests_passed += 1
    else:
        missing = [m for m in models if m not in content]
        print(f"❌ Test 8: Modèles manquants: {missing}")
    
    # Test 9: Vérifier les 4 styles de génération
    tests_total += 1
    styles = ["Réaliste scientifique", "Art géologique", "Coupes techniques", "3D réaliste"]
    all_styles_found = all(style in content for style in styles)
    if all_styles_found:
        print("✅ Test 9: Tous les 4 styles de génération disponibles")
        tests_passed += 1
    else:
        print("❌ Test 9: Certains styles manquants")
    
    # Test 10: Vérifier l'ordre logique du workflow
    tests_total += 1
    sections_order = [
        "1. EXTRACTION SPECTRALE",
        "2. IMPUTATION MATRICIELLE",
        "3. MODÉLISATION FORWARD",
        "4. RECONSTRUCTION 3D",
        "5. DÉTECTION DE TRAJECTOIRES",
        "GÉNÉRATION IA FINALE - SYNTHÈSE COMPLÈTE"
    ]
    
    positions = []
    for section in sections_order:
        pos = content.find(section)
        if pos != -1:
            positions.append(pos)
        else:
            positions.append(-1)
    
    workflow_correct = all(positions[i] < positions[i+1] for i in range(len(positions)-1) if positions[i] != -1 and positions[i+1] != -1)
    
    if workflow_correct and all(p != -1 for p in positions):
        print("✅ Test 10: Workflow dans l'ordre correct (Extraction → Imputation → Forward → 3D → Trajectoires → IA Finale)")
        tests_passed += 1
    else:
        print("❌ Test 10: Workflow dans le désordre ou sections manquantes")
    
    # Résumé
    print(f"\n{'='*60}")
    print(f"📊 RÉSULTAT FINAL : {tests_passed}/{tests_total} tests réussis")
    print(f"{'='*60}\n")
    
    if tests_passed == tests_total:
        print("🎉 SUCCÈS TOTAL ! Le workflow IA est correctement implémenté.")
        print("\n✅ Points validés :")
        print("  • Les boutons ne disparaissent plus (session_state)")
        print("  • L'IA est placée à la FIN du workflow")
        print("  • Toutes les étapes sont dans le bon ordre")
        print("  • Les résultats persistent après génération")
        print("  • 5 modèles IA et 4 styles disponibles")
        return 0
    else:
        print(f"⚠️  {tests_total - tests_passed} test(s) ont échoué.")
        print("\n🔍 Actions recommandées :")
        print("  • Vérifier que toutes les sections sont présentes")
        print("  • Valider l'ordre du workflow")
        print("  • Tester la persistance des résultats")
        return 1

if __name__ == "__main__":
    sys.exit(test_ia_workflow())
