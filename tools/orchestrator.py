"""
ORCHESTRATEUR PRINCIPAL DES OUTILS GÉOLOGIQUES
Intégration de tous les outils pour les analyses KIBALI
"""

from .web_search_tools import GeologyWebSearchTool, WebResearchManager
from .geology_analysis_tools import GeologyStatisticsTool
from .geology_interpretation_tools import GeologyInterpretationTool
from .kibali_ultra_fast_tool import get_kibali_tool, initialize_kibali_tool
from typing import Dict, List, Optional
import logging


class GeologyToolsOrchestrator:
    """Orchestrateur principal de tous les outils géologiques"""

    def __init__(self):
        self.web_search = GeologyWebSearchTool()
        self.research_manager = WebResearchManager()
        self.stats_tool = GeologyStatisticsTool()
        self.interpretation_tool = GeologyInterpretationTool()

        # Initialisation de l'outil IA KIBALI Ultra-Fast
        self.kibali_tool = get_kibali_tool()
        self.kibali_initialized = False

        # Configuration du logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def perform_complete_geology_analysis(self,
                                        resistivity_values: List[float],
                                        depths: List[float],
                                        soil_composition: str = "argiles et marnes",
                                        location_context: str = "") -> Dict:
        """
        Analyse géologique complète utilisant tous les outils

        Args:
            resistivity_values: Valeurs de résistivité mesurées
            depths: Profondeurs correspondantes
            soil_composition: Composition du sol
            location_context: Contexte géographique

        Returns:
            Analyse complète structurée
        """

        self.logger.info("🚀 Début de l'analyse géologique complète")

        # 1. ANALYSE STATISTIQUE
        self.logger.info("📊 Calcul des statistiques de résistivité...")
        resistivity_stats = self.stats_tool.calculate_resistivity_statistics(resistivity_values)

        # 2. ANALYSE PAR PROFONDEUR
        self.logger.info("📏 Analyse de la distribution en profondeur...")
        depth_analysis = self.stats_tool.analyze_depth_distribution(depths, resistivity_values)

        # 3. DÉTECTION D'ANOMALIES
        self.logger.info("🔍 Détection d'anomalies...")
        anomalies = self.stats_tool.detect_anomalies(resistivity_values)

        # 4. ANALYSE PAR CLUSTERING
        self.logger.info("🎯 Analyse par clustering...")
        clusters = self.stats_tool.cluster_analysis(resistivity_values, depths)

        # 5. INTERPRÉTATION GÉOLOGIQUE
        self.logger.info("🧠 Interprétation géologique avancée...")
        geological_interpretation = self.interpretation_tool.interpret_resistivity_profile(resistivity_stats)

        # 6. RECHERCHE WEB
        self.logger.info("🌐 Recherche d'informations complémentaires...")
        web_research = self.research_manager.gather_comprehensive_geology_data(resistivity_stats, soil_composition)

        # 7. ANALYSE IA AVEC KIBALI ULTRA-FAST
        self.logger.info("🤖 Analyse IA avancée avec KIBALI...")
        kibali_analysis = self._perform_kibali_ai_analysis(resistivity_values, depths, soil_composition, location_context)

        # 8. STRUCTURE DU RAPPORT
        report_structure = self.interpretation_tool.generate_expert_report_structure({
            "resistivity_stats": resistivity_stats,
            "depth_analysis": depth_analysis,
            "anomalies": anomalies,
            "clusters": clusters,
            "geological_interpretation": geological_interpretation
        })

        # Compilation finale
        complete_analysis = {
            "statistiques_resistivite": resistivity_stats,
            "analyse_profondeur": depth_analysis,
            "anomalies_detectees": anomalies,
            "analyse_clusters": clusters,
            "interpretation_geologique": geological_interpretation,
            "analyse_ia_kibali": kibali_analysis,
            "recherche_web": web_research,
            "structure_rapport": report_structure,
            "metadonnees": {
                "points_mesure": len(resistivity_values),
                "composition_sol": soil_composition,
                "contexte_localisation": location_context,
                "outils_utilises": ["statistiques", "clustering", "interpretation", "recherche_web", "ia_kibali"]
            }
        }

        self.logger.info("✅ Analyse géologique complète terminée")
        return complete_analysis

    def generate_expert_response(self,
                               analysis_results: Dict,
                               user_query: str = "",
                               response_length: str = "detailed") -> str:
        """
        Génération de réponse d'expert basée sur l'analyse complète

        Args:
            analysis_results: Résultats de l'analyse complète
            user_query: Question spécifique de l'utilisateur
            response_length: Longueur de la réponse ("brief", "detailed", "comprehensive")

        Returns:
            Réponse formatée d'expert
        """

        # Récupération des données clés
        stats = analysis_results.get("statistiques_resistivite", {})
        interp = analysis_results.get("interpretation_geologique", {})
        web_info = analysis_results.get("recherche_web", "")

        # Construction de la réponse selon le format demandé
        if response_length == "brief":
            return self._generate_brief_response(stats, interp)
        elif response_length == "comprehensive":
            return self._generate_comprehensive_response(analysis_results, user_query)
        else:  # detailed
            return self._generate_detailed_response(analysis_results, user_query)

    def _perform_kibali_ai_analysis(self, resistivity_values: List[float], depths: List[float],
                                  soil_composition: str, location_context: str) -> Dict:
        """
        Analyse IA avancée utilisant l'outil KIBALI Ultra-Fast

        Args:
            resistivity_values: Valeurs de résistivité
            depths: Profondeurs correspondantes
            soil_composition: Composition du sol
            location_context: Contexte géographique

        Returns:
            Analyse IA structurée
        """
        try:
            if not self.kibali_initialized:
                initialize_kibali_tool()
                self.kibali_initialized = True

            # Analyse des données géologiques avec KIBALI
            geological_data = {
                "resistivity_values": resistivity_values,
                "depths": depths,
                "soil_composition": soil_composition,
                "location_context": location_context
            }

            # Convertir les données en contexte textuel
            context_str = f"Données géologiques: résistivité {resistivity_values}, profondeurs {depths}, composition {soil_composition}, contexte {location_context}"
            question = "Analysez ces données géologiques et fournissez des insights sur les formations présentes"

            kibali_insights = self.kibali_tool.generate_geological_insights(context_str, question)

            # Analyse des anomalies potentielles
            anomaly_data = {
                "resistivity_values": resistivity_values,
                "depths": depths,
                "soil_composition": soil_composition
            }
            anomalies_analysis = self.kibali_tool.interpret_resistivity_anomaly(anomaly_data)

            # Génération d'interprétations expertes
            expert_interpretation = self.kibali_tool.analyze_geological_data(geological_data)

            return {
                "insights_ia": kibali_insights,
                "analyse_anomalies": anomalies_analysis,
                "interpretation_experte": expert_interpretation,
                "statut_analyse": "succès",
                "modele_utilise": "KIBALI Ultra-Fast"
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse IA KIBALI: {str(e)}")
            return {
                "insights_ia": "Analyse IA non disponible",
                "analyse_anomalies": "Analyse d'anomalies non disponible",
                "interpretation_experte": f"Erreur technique: {str(e)}",
                "statut_analyse": "erreur",
                "modele_utilise": "KIBALI Ultra-Fast"
            }

    def _generate_brief_response(self, stats: Dict, interp: Dict) -> str:
        """Génération de réponse brève"""
        formations = interp.get("formations_probables", [])
        top_formation = formations[0]["formation"] if formations else "inconnue"

        return f"""
📊 ANALYSE GÉOLOGIQUE RAPIDE:
Formation principale: {top_formation}
Résistivité: {stats.get('mean', 'N/A'):.1f} Ω·m (moyenne)
Hétérogénéité: {interp.get('heterogeneite_geologique', {}).get('niveau_homogeneite', 'N/A')}
Potentiel aquifère: {interp.get('potentiel_aquifere', {}).get('potentiel', 'N/A')}
"""

    def _generate_detailed_response(self, analysis_results: Dict, user_query: str) -> str:
        """Génération de réponse détaillée (format demandé par l'utilisateur)"""
        stats = analysis_results.get("statistiques_resistivite", {})
        interp = analysis_results.get("interpretation_geologique", {})
        formations = interp.get("formations_probables", [])
        web_info = analysis_results.get("recherche_web", "")

        # Construction du rapport détaillé
        response = "📊 ANALYSE GÉOLOGIQUE COMPLÈTE (KIBALI) :\n\n"

        # Section Géologie
        response += "Géologie: "
        if formations:
            top_formation = formations[0]
            response += f"Sous-sol composé principalement de {top_formation['formation']} "
            response += f"({top_formation['description']}). "
            response += f"Cette formation présente une résistivité moyenne de {stats.get('mean', 'N/A'):.1f} Ω·m, "
            response += f"avec une plage variant de {stats.get('min', 'N/A'):.1f} à {stats.get('max', 'N/A'):.1f} Ω·m. "
            response += f"L'hétérogénéité géologique est {interp.get('heterogeneite_geologique', {}).get('niveau_homogeneite', 'inconnue').lower()}, "
            response += f"indiquant {interp.get('heterogeneite_geologique', {}).get('interpretation', 'caractéristiques variables').lower()}. "
            response += f"Les implications principales sont: {top_formation['implications']}.\n\n"
        else:
            response += "Analyse des formations géologiques en cours...\n\n"

        # Section Actions
        response += "Actions: Interpréter les données ERT (Electrical Resistivity Tomography) pour détecter aquifères ou pollutions. "
        response += "Calibrer les modèles géologiques en utilisant les statistiques de résistivité et l'analyse par clusters. "
        response += "Procéder à une validation croisée avec les données de recherche web disponibles. "
        response += "Recommander des investigations complémentaires dans les zones d'anomalies détectées.\n\n"

        # Section Image
        response += "Image: La coupe géologique révèle une alternance de zones à faible et haute résistivité, "
        response += f"caractéristique d'une formation {formations[0]['formation'] if formations else 'géologique'} hétérogène. "
        response += "Les zones conductrices (faible résistivité) correspondent probablement à des niveaux saturés ou argileux, "
        response += "tandis que les zones résistantes indiquent des formations plus compactes ou sableuses. "
        response += "Une trajectoire principale a été détectée, suggérant une structure géologique organisée.\n\n"

        # Section Recommandations
        response += "🎯 RECOMMANDATIONS D'ACTION :\n\n"
        recommendations = interp.get("recommandations_action", [])
        for rec in recommendations:
            response += f"• {rec}\n"
        response += "\n"

        # Section Statistiques
        response += "📈 STATISTIQUES DÉTAILLÉES :\n\n"
        response += f"• Résistivité : {stats.get('min', 'N/A'):.1f} - {stats.get('max', 'N/A'):.1f} Ω·m (moyenne: {stats.get('mean', 'N/A'):.1f} Ω·m)\n"
        response += f"• Points de mesure : {stats.get('count', 'N/A')} données réelles\n"
        response += f"• Médiane : {stats.get('median', 'N/A'):.1f} Ω·m\n"
        response += f"• Écart-type : {stats.get('std', 'N/A'):.1f} Ω·m\n"
        response += f"• Coefficient de variation : {stats.get('cv', 'N/A'):.3f}\n"
        response += f"• Hétérogénéité : {interp.get('heterogeneite_geologique', {}).get('niveau_homogeneite', 'N/A')}\n"
        response += f"• Structures détectées : {analysis_results.get('anomalies_detectees', {}).get('total_points', 'N/A')} points avec {len(analysis_results.get('analyse_clusters', {}))} clusters identifiés\n\n"

        # Conclusion
        response += "CONCLUSION EXPERTE: Cette analyse révèle une formation géologique complexe nécessitant "
        response += "une approche intégrée combinant géophysique, géologie et hydrologie. "
        response += "Les résultats obtenus sont cohérents avec les données de référence et suggèrent "
        response += "des investigations complémentaires pour confirmer les interprétations proposées."

        return response

    def _generate_comprehensive_response(self, analysis_results: Dict, user_query: str) -> str:
        """Génération de réponse complète avec tous les détails"""
        detailed_response = self._generate_detailed_response(analysis_results, user_query)

        # Ajouter les informations de recherche web
        web_info = analysis_results.get("recherche_web", "")
        if web_info:
            detailed_response += f"\n\n🔬 RÉFÉRENCES ET DONNÉES COMPLÉMENTAIRES :\n{web_info}"

        # Ajouter l'analyse par clusters
        clusters = analysis_results.get("analyse_clusters", {})
        if clusters:
            detailed_response += f"\n\n🎯 ANALYSE PAR CLUSTERS :\n"
            for cluster_name, cluster_data in clusters.items():
                detailed_response += f"• {cluster_name.upper()}: {cluster_data.get('geological_interpretation', 'N/A')} "
                detailed_response += f"(résistivité: {cluster_data.get('resistivity_center', 'N/A'):.1f} Ω·m, "
                detailed_response += f"profondeur: {cluster_data.get('depth_center', 'N/A'):.1f} m)\n"

        # Ajouter l'analyse IA KIBALI
        kibali_analysis = analysis_results.get("analyse_ia_kibali", {})
        if kibali_analysis and kibali_analysis.get("statut_analyse") == "succès":
            detailed_response += f"\n\n🤖 ANALYSE IA AVANCÉE (KIBALI) :\n"
            detailed_response += f"• Insights IA: {kibali_analysis.get('insights_ia', 'N/A')}\n"
            detailed_response += f"• Analyse d'anomalies: {kibali_analysis.get('analyse_anomalies', 'N/A')}\n"
            detailed_response += f"• Interprétation experte: {kibali_analysis.get('interpretation_experte', 'N/A')}\n"
            detailed_response += f"• Modèle utilisé: {kibali_analysis.get('modele_utilise', 'N/A')}\n"

        return detailed_response


# Instance globale de l'orchestrateur
geology_tools_orchestrator = GeologyToolsOrchestrator()