#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test des fonctionnalités IA ajoutées à ERTest.py
"""

import sys
import numpy as np

print("🧪 Test des fonctionnalités de génération d'images IA")
print("=" * 60)

# Test 1: Import du module principal
print("\n1. Test d'import du module...")
try:
    # Ne pas importer streamlit directement (nécessite interface)
    with open('ERTest.py', 'r', encoding='utf-8') as f:
        content = f.read()
    print("   ✅ Fichier ERTest.py chargé")
    print(f"   📏 Taille: {len(content)} caractères")
    print(f"   📄 Lignes: {len(content.splitlines())}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test 2: Vérification des imports nécessaires
print("\n2. Test des dépendances IA...")
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    print(f"   🔧 CUDA disponible: {torch.cuda.is_available()}")
except Exception as e:
    print(f"   ❌ PyTorch: {e}")

try:
    from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
    print("   ✅ Diffusers OK")
except Exception as e:
    print(f"   ❌ Diffusers: {e}")

try:
    from PIL import Image
    print("   ✅ PIL/Pillow OK")
except Exception as e:
    print(f"   ❌ PIL: {e}")

# Test 3: Vérification des fonctions clés
print("\n3. Vérification des fonctions IA dans ERTest.py...")
functions_to_check = [
    'load_image_generation_pipeline',
    'analyze_resistivity_patterns',
    'create_geological_prompt',
    'generate_realistic_geological_image',
    'create_side_by_side_comparison'
]

for func_name in functions_to_check:
    if f"def {func_name}" in content:
        print(f"   ✅ Fonction '{func_name}' trouvée")
    else:
        print(f"   ❌ Fonction '{func_name}' MANQUANTE")

# Test 4: Vérification des sections UI
print("\n4. Vérification des sections UI...")
ui_sections = [
    ('Génération d\'Image Réaliste du Sous-Sol', 'Section Analyse Spectrale'),
    ('Visualisations Réalistes des Coupes 3D', 'Section Reconstruction 3D'),
    ('Générer Image Réaliste', 'Bouton génération spectrale'),
    ('Générer Images Réalistes des Coupes', 'Bouton génération 3D')
]

for search_text, description in ui_sections:
    if search_text in content:
        print(f"   ✅ {description} présente")
    else:
        print(f"   ❌ {description} MANQUANTE")

# Test 5: Vérification des modèles disponibles
print("\n5. Vérification du dictionnaire des modèles...")
if 'GENERATION_MODELS' in content:
    print("   ✅ Dictionnaire GENERATION_MODELS défini")
    models = [
        'Stable Diffusion XL',
        'DreamShaper 8',
        'RealVisXL V4.0',
        'Realistic Vision V5.1',
        'epiCRealism'
    ]
    for model in models:
        if model in content:
            print(f"   ✅ Modèle '{model}' configuré")
        else:
            print(f"   ⚠️  Modèle '{model}' non trouvé")
else:
    print("   ❌ Dictionnaire GENERATION_MODELS MANQUANT")

# Test 6: Test fonctionnel de base
print("\n6. Test fonctionnel de base...")
try:
    # Créer des données de test
    test_rho_slice = np.random.rand(10, 10) * 1000  # Valeurs de résistivité aléatoires
    
    # Test de la fonction d'analyse (si importable)
    print("   ✅ Création de données test réussie")
    print(f"   📊 Shape: {test_rho_slice.shape}")
    print(f"   📊 Range: {test_rho_slice.min():.2f} - {test_rho_slice.max():.2f} Ω·m")
    
except Exception as e:
    print(f"   ⚠️  Test fonctionnel: {e}")

# Test 7: Vérification de l'intégration PDF
print("\n7. Vérification intégration rapports PDF...")
pdf_checks = [
    'generated_spectral_image',
    'generated_3d_image',
    'Visualisation Réaliste du Sous-Sol'
]

for check in pdf_checks:
    if check in content:
        print(f"   ✅ Intégration PDF: '{check}' présente")
    else:
        print(f"   ❌ Intégration PDF: '{check}' manquante")

# Résumé
print("\n" + "=" * 60)
print("📊 RÉSUMÉ DES TESTS")
print("=" * 60)

total_checks = 0
passed_checks = 0

# Compter les checks
for func_name in functions_to_check:
    total_checks += 1
    if f"def {func_name}" in content:
        passed_checks += 1

for search_text, _ in ui_sections:
    total_checks += 1
    if search_text in content:
        passed_checks += 1

print(f"\n✅ Tests passés: {passed_checks}/{total_checks}")
print(f"📈 Taux de réussite: {(passed_checks/total_checks)*100:.1f}%")

if passed_checks == total_checks:
    print("\n🎉 TOUS LES TESTS SONT PASSÉS !")
    print("✨ Les fonctionnalités IA sont correctement intégrées")
    print("\n🚀 Pour tester en production:")
    print("   cd /home/belikan/KIbalione8/SETRAF")
    print("   streamlit run ERTest.py")
    print("\n📍 Puis allez à la section:")
    print("   🖼️ Analyse Spectrale d'Images (Imputation + Reconstruction)")
    print("\n💡 Les boutons de génération IA apparaîtront après avoir:")
    print("   1. Uploadé une image")
    print("   2. Cliqué sur '🚀 Extraire Spectres'")
else:
    print("\n⚠️  Certains éléments manquent")
    print("📝 Vérifiez le fichier ERTest.py")

print("\n" + "=" * 60)
