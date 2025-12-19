"""
OUTILS DE RECHERCHE WEB POUR ANALYSES GÉOLOGIQUES
Utilise Tavily pour des recherches spécialisées en géologie et ERT
"""

import os
from typing import List, Dict, Optional
from tavily import TavilyClient


class GeologyWebSearchTool:
    """Outil de recherche web spécialisé en géologie"""

    def __init__(self):
        self.api_key = os.getenv('TAVILY_API_KEY')
        self.client = None
        if self.api_key:
            self.client = TavilyClient(api_key=self.api_key)
        else:
            print("⚠️ TAVILY_API_KEY non trouvée - recherche web limitée")

    def search_geology_info(self, query: str, max_results: int = 3) -> str:
        """
        Recherche spécialisée en géologie et ERT

        Args:
            query: Terme de recherche géologique
            max_results: Nombre maximum de résultats

        Returns:
            Résultats formatés pour analyse
        """
        if not self.client:
            return "🔍 Recherche web non disponible - clé API manquante"

        try:
            # Recherche avec domaines spécialisés
            response = self.client.search(
                query=f"géologie ERT analyse {query}",
                search_depth="advanced",
                max_results=max_results,
                include_domains=[
                    ".edu", ".gov", ".org", "researchgate.net",
                    "science.org", "geology.com", "usgs.gov"
                ]
            )

            # Formater les résultats
            results = []
            for result in response.get('results', []):
                formatted_result = f"""
📚 **{result.get('title', 'Sans titre')}**
{result.get('content', '')[:400]}...
🔗 Source: {result.get('url', '')}
---
"""
                results.append(formatted_result)

            return "\n".join(results) if results else "Aucun résultat trouvé"

        except Exception as e:
            return f"❌ Erreur recherche web: {str(e)[:100]}"

    def search_ert_methodology(self, specific_topic: str = "") -> str:
        """Recherche méthodologie ERT spécifique"""
        query = f"méthodologie ERT tomographie électrique {specific_topic}"
        return self.search_geology_info(query, max_results=2)

    def search_aquifer_detection(self) -> str:
        """Recherche méthodes de détection d'aquifères"""
        return self.search_geology_info("détection aquifères ERT géologie", max_results=3)

    def search_pollution_detection(self) -> str:
        """Recherche méthodes de détection de pollution"""
        return self.search_geology_info("détection pollution sols ERT géologie", max_results=3)

    def search_soil_properties(self, soil_type: str) -> str:
        """Recherche propriétés géologiques d'un type de sol"""
        return self.search_geology_info(f"propriétés {soil_type} géologie résistivité", max_results=2)


class WebResearchManager:
    """Gestionnaire centralisé des recherches web"""

    def __init__(self):
        self.search_tool = GeologyWebSearchTool()

    def gather_comprehensive_geology_data(self, resistivity_data: Dict, soil_composition: str) -> str:
        """
        Collecte complète de données géologiques pour analyse

        Args:
            resistivity_data: Données de résistivité (min, max, moyenne, etc.)
            soil_composition: Composition du sol (argile, marne, etc.)

        Returns:
            Données de recherche formatées
        """
        research_results = []

        # Recherche propriétés du sol
        soil_research = self.search_tool.search_soil_properties(soil_composition)
        research_results.append(f"🔬 PROPRIÉTÉS GÉOLOGIQUES ({soil_composition.upper()}):\n{soil_research}")

        # Recherche méthodologie ERT
        methodology_research = self.search_tool.search_ert_methodology("calibration modèles")
        research_results.append(f"📊 MÉTHODOLOGIE ERT:\n{methodology_research}")

        # Recherche détection aquifères si résistivité basse
        if resistivity_data.get('mean', 50) < 50:
            aquifer_research = self.search_tool.search_aquifer_detection()
            research_results.append(f"💧 DÉTECTION AQUIFÈRES:\n{aquifer_research}")

        # Recherche pollution si résistivité variable
        resistivity_range = resistivity_data.get('max', 100) - resistivity_data.get('min', 10)
        if resistivity_range > 50:
            pollution_research = self.search_tool.search_pollution_detection()
            research_results.append(f"⚠️ DÉTECTION POLLUTION:\n{pollution_research}")

        return "\n\n".join(research_results)


# Instance globale
web_research_manager = WebResearchManager()