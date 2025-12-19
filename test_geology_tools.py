#!/usr/bin/env python3
"""
TEST DES OUTILS GÉOLOGIQUES KIBALI
Script de validation de l'intégration des outils
"""

import sys
import os
sys.path.append('/home/belikan/KIbalione8/SETRAF')

def test_geology_tools():
    """Test complet des outils géologiques"""
    print("🧪 TEST DES OUTILS GÉOLOGIQUES KIBALI")
    print("=" * 50)

    try:
        # Import des outils
        from tools import geology_tools_orchestrator
        print("✅ Import des outils réussi")

        # Données de test
        resistivity_values = [10.5, 25.3, 45.7, 120.8, 89.2, 156.4, 234.1, 78.9, 312.6, 145.7]
        depths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        print(f"📊 Données de test: {len(resistivity_values)} mesures de résistivité")
        print(f"📏 Profondeurs: {depths[0]} - {depths[-1]} m")

        # Test de l'analyse complète
        print("\n🔬 Test de l'analyse géologique complète...")
        analysis = geology_tools_orchestrator.perform_complete_geology_analysis(
            resistivity_values=resistivity_values,
            depths=depths,
            soil_composition="argiles et sables",
            location_context="site test géophysique"
        )

        print("✅ Analyse complète réussie")
        print(f"📈 Statistiques calculées: {len(analysis.get('statistiques_resistivite', {}))} métriques")
        print(f"🎯 Clusters identifiés: {len(analysis.get('analyse_clusters', {}))}")
        print(f"🔍 Anomalies détectées: {analysis.get('anomalies_detectees', {}).get('total_points', 0)} points")

        # Test de génération de réponse
        print("\n📝 Test de génération de réponse d'expert...")
        response = geology_tools_orchestrator.generate_expert_response(
            analysis_results=analysis,
            user_query="Quelle est la formation géologique principale?",
            response_length="detailed"
        )

        print("✅ Génération de réponse réussie")
        print(f"📄 Longueur de la réponse: {len(response)} caractères")

        # Vérification du format
        if "📊 ANALYSE GÉOLOGIQUE COMPLÈTE (KIBALI)" in response:
            print("✅ Format d'expert respecté")
        else:
            print("⚠️ Format d'expert non détecté")

        # Test des outils individuels
        print("\n🛠️ Test des outils individuels...")

        # Test statistiques
        stats = geology_tools_orchestrator.stats_tool.calculate_resistivity_statistics(resistivity_values)
        print(f"📊 Statistiques: moyenne={stats.get('mean', 'N/A'):.1f} Ω·m")

        # Test interprétation
        interp = geology_tools_orchestrator.interpretation_tool.interpret_resistivity_profile(stats)
        formations = interp.get("formations_probables", [])
        if formations:
            print(f"🪨 Formation principale: {formations[0]['formation']}")
        else:
            print("🪨 Aucune formation identifiée")

        print("\n🎉 TOUS LES TESTS RÉUSSIS !")
        print("✅ Les outils géologiques KIBALI sont opérationnels")

        return True

    except Exception as e:
        print(f"❌ ERREUR LORS DES TESTS: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_geology_tools()
    sys.exit(0 if success else 1)