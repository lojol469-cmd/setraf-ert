#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour le système de gestion d'ID des documents et fichiers .dat
"""

import sys
import os

# Ajouter le chemin du module
sys.path.insert(0, '/home/belikan/KIbalione8/SETRAF')

def test_id_system():
    """Test du système de génération d'ID et vérification d'existence"""
    
    print("=" * 60)
    print("🧪 TEST DU SYSTÈME DE GESTION D'ID")
    print("=" * 60)
    
    # Test 1 : Génération d'ID pour document
    print("\n📝 Test 1 : Génération d'ID pour document")
    print("-" * 60)
    
    from ERTest import ERTKnowledgeBase
    
    kb = ERTKnowledgeBase()
    
    # Simuler un contenu de document
    test_content = """
    ÉCHELLE RAPIDE RÉSISTIVITÉ ERT:
    0.01-1 Ω·m : EAU DE MER / MINÉRAUX
    1-10 Ω·m : EAU SAUMÂTRE / ARGILES
    """
    
    doc_id = kb._generate_document_id(test_content)
    print(f"✅ ID généré : {doc_id}")
    print(f"   Longueur : {len(doc_id)} caractères")
    
    # Test 2 : Vérifier que le même contenu génère le même ID
    print("\n🔄 Test 2 : Reproductibilité des ID")
    print("-" * 60)
    
    doc_id2 = kb._generate_document_id(test_content)
    
    if doc_id == doc_id2:
        print(f"✅ SUCCÈS : Même contenu = Même ID")
        print(f"   ID 1: {doc_id}")
        print(f"   ID 2: {doc_id2}")
    else:
        print(f"❌ ÉCHEC : Les ID devraient être identiques")
        print(f"   ID 1: {doc_id}")
        print(f"   ID 2: {doc_id2}")
    
    # Test 3 : Contenu différent = ID différent
    print("\n🔀 Test 3 : Unicité des ID")
    print("-" * 60)
    
    test_content_different = test_content + "\nLigne supplémentaire"
    doc_id3 = kb._generate_document_id(test_content_different)
    
    if doc_id != doc_id3:
        print(f"✅ SUCCÈS : Contenu différent = ID différent")
        print(f"   ID original : {doc_id}")
        print(f"   ID modifié  : {doc_id3}")
    else:
        print(f"❌ ÉCHEC : Les ID devraient être différents")
    
    # Test 4 : Vérification d'existence (nouveau document)
    print("\n🆕 Test 4 : Vérification d'existence (nouveau document)")
    print("-" * 60)
    
    check_result = kb.check_document_exists(test_content)
    
    if not check_result['exists']:
        print(f"✅ SUCCÈS : Document correctement identifié comme nouveau")
        print(f"   Doc ID : {check_result['doc_id']}")
        print(f"   Existe : {check_result['exists']}")
    else:
        print(f"❌ ÉCHEC : Le document ne devrait pas exister")
    
    # Test 5 : Génération d'ID pour fichier .dat
    print("\n📂 Test 5 : Génération d'ID pour fichier .dat")
    print("-" * 60)
    
    # Simuler un contenu de fichier .dat
    test_file_bytes = b"""2025/12/09 10:30:00
Survey Point: 1
Depth From: 0.0
Depth To: 5.0
Data: 12.5
"""
    
    file_id = kb._generate_dat_file_id(test_file_bytes, "test_file.dat")
    print(f"✅ ID fichier .dat généré : {file_id}")
    print(f"   Longueur : {len(file_id)} caractères")
    
    # Test 6 : Vérification d'existence fichier .dat
    print("\n🔍 Test 6 : Vérification d'existence fichier .dat")
    print("-" * 60)
    
    check_result_dat = kb.check_dat_file_exists(test_file_bytes, "test_file.dat")
    
    if not check_result_dat['exists']:
        print(f"✅ SUCCÈS : Fichier .dat correctement identifié comme nouveau")
        print(f"   File ID : {check_result_dat['file_id']}")
        print(f"   Existe  : {check_result_dat['exists']}")
    else:
        print(f"❌ ÉCHEC : Le fichier ne devrait pas exister")
    
    # Test 7 : Chargement et sauvegarde des registres
    print("\n💾 Test 7 : Persistance des registres")
    print("-" * 60)
    
    initial_doc_count = len(kb.document_ids)
    initial_dat_count = len(kb.dat_file_registry)
    
    print(f"   Documents dans le registre : {initial_doc_count}")
    print(f"   Fichiers .dat dans le registre : {initial_dat_count}")
    
    # Sauvegarder
    kb._save_id_registry()
    print(f"✅ Registres sauvegardés")
    
    # Créer une nouvelle instance et recharger
    kb2 = ERTKnowledgeBase()
    
    if len(kb2.document_ids) == initial_doc_count and len(kb2.dat_file_registry) == initial_dat_count:
        print(f"✅ SUCCÈS : Registres rechargés correctement")
        print(f"   Documents rechargés : {len(kb2.document_ids)}")
        print(f"   Fichiers .dat rechargés : {len(kb2.dat_file_registry)}")
    else:
        print(f"❌ ÉCHEC : Problème de rechargement des registres")
    
    # Résumé des tests
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    print("✅ Tous les tests de base ont été exécutés")
    print("📝 Vérifiez visuellement les résultats ci-dessus")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        test_id_system()
        print("\n✨ Script de test terminé avec succès\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur lors des tests : {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
