#!/usr/bin/env python3
"""
DÉMONSTRATION DES OUTILS GÉOLOGIQUES KIBALI
Exemple d'utilisation complète du système d'outils
"""

import sys
import os
sys.path.append('/home/belikan/KIbalione8/SETRAF')

def demo_geology_tools():
    """Démonstration complète des capacités des outils géologiques"""
    print("🚀 DÉMONSTRATION DES OUTILS GÉOLOGIQUES KIBALI")
    print("=" * 60)

    try:
        from tools import geology_tools_orchestrator, get_resistivity_category, GEOLOGICAL_FORMATIONS

        # Données d'exemple représentatives d'un profil ERT réel
        resistivity_values = [
            15.2, 22.8, 18.9, 45.6, 78.3, 124.7, 89.4, 156.8,
            234.1, 312.6, 145.7, 98.3, 67.2, 34.5, 12.8, 8.9,
            156.4, 278.9, 445.6, 123.4, 87.6, 145.2, 198.7, 267.8
        ]

        depths = [i * 0.5 for i in range(len(resistivity_values))]  # Profondeurs de 0 à 11.5m

        print(f"📊 Profil ERT simulé: {len(resistivity_values)} mesures")
        print(f"📏 Profondeur d'investigation: {depths[-1]:.1f} mètres")
        print(f"🎯 Résistivité: {min(resistivity_values):.1f} - {max(resistivity_values):.1f} Ω·m")
        print()

        # 1. ANALYSE STATISTIQUE
        print("1️⃣ ANALYSE STATISTIQUE DÉTAILLÉE")
        print("-" * 40)
        stats = geology_tools_orchestrator.stats_tool.calculate_resistivity_statistics(resistivity_values)
        print(f"📈 Moyenne: {stats['mean']:.1f} Ω·m")
        print(f"📊 Médiane: {stats['median']:.1f} Ω·m")
        print(f"📏 Écart-type: {stats['std']:.1f} Ω·m")
        print(f"📊 Coefficient de variation: {stats['cv']:.3f}")
        print(f"🎯 Plage: {stats['min']:.1f} - {stats['max']:.1f} Ω·m")
        print()

        # 2. CLASSIFICATION GÉOLOGIQUE
        print("2️⃣ CLASSIFICATION GÉOLOGIQUE")
        print("-" * 40)
        categories = {}
        for rho in resistivity_values:
            cat = get_resistivity_category(rho)
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(resistivity_values)) * 100
            formation_info = GEOLOGICAL_FORMATIONS[cat]
            print(f"🪨 {cat.replace('_', ' ').title()}: {count} mesures ({percentage:.1f}%)")
            print(f"   💧 Potentiel aquifère: {formation_info['potentiel_aquifere']}")
        print()

        # 3. ANALYSE PAR PROFONDEUR
        print("3️⃣ ANALYSE PAR PROFONDEUR")
        print("-" * 40)
        depth_analysis = geology_tools_orchestrator.stats_tool.analyze_depth_distribution(depths, resistivity_values)
        mean_gradient = depth_analysis.get('mean_gradient', 'N/A')
        correlation = depth_analysis.get('correlation', 'N/A')
        print(f"📏 Gradient vertical moyen: {mean_gradient if isinstance(mean_gradient, str) else f'{mean_gradient:.3f}'} Ω·m/m")
        print(f"📊 Corrélations profondeur-résistivité: {correlation if isinstance(correlation, str) else f'{correlation:.3f}'}")
        print()

        # 4. DÉTECTION D'ANOMALIES
        print("4️⃣ DÉTECTION D'ANOMALIES")
        print("-" * 40)
        anomalies = geology_tools_orchestrator.stats_tool.detect_anomalies(resistivity_values)
        print(f"🔍 {anomalies.get('total_points', len(resistivity_values))} points analysés")
        anomalies_count = anomalies.get('anomalies_count', len(anomalies.get('anomalies_indices', [])))
        print(f"⚠️ {anomalies_count} anomalies détectées")
        anomalies_indices = anomalies.get('anomalies_indices', [])
        if anomalies_indices:
            print(f"📍 Indices des anomalies: {anomalies_indices[:5]}...")  # Premiers 5
        print()

        # 5. ANALYSE PAR CLUSTERING
        print("5️⃣ ANALYSE PAR CLUSTERING")
        print("-" * 40)
        clusters = geology_tools_orchestrator.stats_tool.cluster_analysis(resistivity_values, depths)
        print(f"🎯 {len(clusters)} clusters identifiés:")
        for cluster_name, cluster_data in clusters.items():
            resistivity_center = cluster_data.get('resistivity_center', 'N/A')
            depth_center = cluster_data.get('depth_center', 'N/A')
            interpretation = cluster_data.get('geological_interpretation', 'N/A')
            print(f"   • {cluster_name.upper()}: {interpretation}")
            if isinstance(resistivity_center, (int, float)) and isinstance(depth_center, (int, float)):
                print(f"     📍 Centre: ρ={resistivity_center:.1f} Ω·m, z={depth_center:.1f} m")
            else:
                print(f"     📍 Centre: ρ={resistivity_center}, z={depth_center}")
        print()

        # 6. INTERPRÉTATION GÉOLOGIQUE
        print("6️⃣ INTERPRÉTATION GÉOLOGIQUE EXPERTE")
        print("-" * 40)
        interpretation = geology_tools_orchestrator.interpretation_tool.interpret_resistivity_profile(stats)

        formations = interpretation.get("formations_probables", [])
        if formations:
            top_formation = formations[0]
            print(f"🏆 FORMATION PRINCIPALE: {top_formation['formation'].upper()}")
            print(f"📝 Description: {top_formation['description']}")
            print(f"🎯 Implications: {top_formation['implications']}")

        hetero = interpretation.get("heterogeneite_geologique", {})
        print(f"🔄 Hétérogénéité: {hetero.get('niveau_homogeneite', 'N/A')}")

        aquifer = interpretation.get("potentiel_aquifere", {})
        print(f"💧 Potentiel aquifère: {aquifer.get('potentiel', 'N/A')}")
        print()

        # 7. ANALYSE COMPLÈTE INTÉGRÉE
        print("7️⃣ ANALYSE COMPLÈTE INTÉGRÉE")
        print("-" * 40)
        complete_analysis = geology_tools_orchestrator.perform_complete_geology_analysis(
            resistivity_values=resistivity_values,
            depths=depths,
            soil_composition="formations sédimentaires variées",
            location_context="site d'étude géophysique en zone tempérée"
        )
        print("✅ Analyse complète réalisée avec succès")
        print(f"📊 {len(complete_analysis)} sections d'analyse générées")
        print()

        # 8. RÉPONSE D'EXPERT FORMATÉE
        print("8️⃣ RÉPONSE D'EXPERT FORMATÉE")
        print("-" * 40)
        expert_response = geology_tools_orchestrator.generate_expert_response(
            analysis_results=complete_analysis,
            user_query="Pouvez-vous analyser ce profil ERT et identifier les formations géologiques présentes ?",
            response_length="detailed"
        )

        print("📄 RÉPONSE GÉNÉRÉE:")
        print("-" * 20)
        # Afficher seulement les premières lignes pour la démo
        lines = expert_response.split('\n')
        for i, line in enumerate(lines[:15]):  # Premières 15 lignes
            print(line)
        if len(lines) > 15:
            print(f"... ({len(lines) - 15} lignes supplémentaires)")
        print()

        print("🎉 DÉMONSTRATION TERMINÉE AVEC SUCCÈS !")
        print("✅ Tous les outils géologiques KIBALI fonctionnent correctement")
        print("\n💡 Les outils sont maintenant intégrés dans ERTest.py et disponibles")
        print("   dans le chat IA pour des analyses géologiques expertes en temps réel.")

        return True

    except Exception as e:
        print(f"❌ ERREUR LORS DE LA DÉMONSTRATION: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_geology_tools()
    sys.exit(0 if success else 1)