"""
OUTILS D'INTERPRÉTATION GÉOLOGIQUE AVANCÉE
Modèles d'interprétation et recommandations expertes pour ERT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class GeologicalFormation(Enum):
    """Types de formations géologiques"""
    CLAY = "argile"
    MARL = "marne"
    SAND = "sable"
    LIMESTONE = "calcaire"
    SANDSTONE = "grès"
    GRANITE = "granite"
    BASALT = "basalte"
    AQUIFER = "aquifère"
    POLLUTED_ZONE = "zone_polluée"


class GeologyInterpretationTool:
    """Outil d'interprétation géologique avancée"""

    def __init__(self):
        # Base de connaissances des propriétés géologiques
        self.geological_properties = {
            GeologicalFormation.CLAY: {
                "resistivity_range": (1, 20),
                "description": "Argiles très conductrices, souvent saturées d'eau",
                "implications": "Aquifères potentiels, sensibilité à la pollution",
                "ert_interpretation": "Zones conductrices, anomalies basses fréquentes"
            },
            GeologicalFormation.MARL: {
                "resistivity_range": (5, 50),
                "description": "Marnes argileuses, mélange argile-calcaire",
                "implications": "Étanchéité variable, stockage possible",
                "ert_interpretation": "Conductivité moyenne, transitions graduelles"
            },
            GeologicalFormation.SAND: {
                "resistivity_range": (50, 200),
                "description": "Sables propres, bonne perméabilité",
                "implications": "Aquifères productifs, drainage rapide",
                "ert_interpretation": "Résistivité moyenne à élevée, homogène"
            },
            GeologicalFormation.LIMESTONE: {
                "resistivity_range": (100, 1000),
                "description": "Calcaires compacts, karstiques possibles",
                "implications": "Aquifères karstiques, instabilité possible",
                "ert_interpretation": "Très résistant, anomalies locales possibles"
            },
            GeologicalFormation.SANDSTONE: {
                "resistivity_range": (200, 2000),
                "description": "Grès consolidés, résistants",
                "implications": "Réservoirs pétroliers, aquifères fracturés",
                "ert_interpretation": "Très résistant, structures stratifiées"
            },
            GeologicalFormation.GRANITE: {
                "resistivity_range": (1000, 10000),
                "description": "Roches magmatiques dures",
                "implications": "Substratum stable, faible perméabilité",
                "ert_interpretation": "Extrêmement résistant, très homogène"
            }
        }

    def interpret_resistivity_profile(self, resistivity_stats: Dict) -> Dict:
        """
        Interprétation complète d'un profil de résistivité

        Args:
            resistivity_stats: Statistiques de résistivité

        Returns:
            Interprétation géologique détaillée
        """
        mean_res = resistivity_stats.get('mean', 50)
        std_res = resistivity_stats.get('std', 10)
        min_res = resistivity_stats.get('min', 1)
        max_res = resistivity_stats.get('max', 100)

        # Identification des formations probables
        probable_formations = self._identify_formations(mean_res, std_res)

        # Analyse de l'hétérogénéité
        heterogeneity = self._analyze_heterogeneity(std_res, mean_res, max_res - min_res)

        # Détection d'aquifères
        aquifer_potential = self._assess_aquifer_potential(mean_res, min_res)

        # Risques de pollution
        pollution_risk = self._assess_pollution_risk(std_res, max_res - min_res)

        return {
            "formations_probables": probable_formations,
            "heterogeneite_geologique": heterogeneity,
            "potentiel_aquifere": aquifer_potential,
            "risque_pollution": pollution_risk,
            "recommandations_action": self._generate_recommendations(probable_formations, heterogeneity)
        }

    def _identify_formations(self, mean_res: float, std_res: float) -> List[Dict]:
        """Identification des formations géologiques probables"""
        formations = []

        for formation, properties in self.geological_properties.items():
            res_range = properties["resistivity_range"]
            if res_range[0] <= mean_res <= res_range[1]:
                confidence = self._calculate_confidence(mean_res, res_range)
                formations.append({
                    "formation": formation.value,
                    "confiance": confidence,
                    "description": properties["description"],
                    "implications": properties["implications"]
                })

        # Trier par confiance
        formations.sort(key=lambda x: x["confiance"], reverse=True)
        return formations[:3]  # Top 3 formations

    def _calculate_confidence(self, resistivity: float, res_range: Tuple[float, float]) -> float:
        """Calcule le niveau de confiance pour une formation"""
        range_center = (res_range[0] + res_range[1]) / 2
        range_width = res_range[1] - res_range[0]

        distance_from_center = abs(resistivity - range_center)
        confidence = max(0, 1 - (distance_from_center / (range_width / 2)))

        return round(confidence * 100, 1)

    def _analyze_heterogeneity(self, std_res: float, mean_res: float, res_range: float) -> Dict:
        """Analyse de l'hétérogénéité géologique"""
        cv = std_res / mean_res if mean_res != 0 else 0  # Coefficient de variation

        if cv < 0.3:
            homogeneity = "Très homogène"
            interpretation = "Formation géologique uniforme, faible variabilité"
        elif cv < 0.7:
            homogeneity = "Modérément hétérogène"
            interpretation = "Variabilité moyenne, transitions géologiques possibles"
        else:
            homogeneity = "Très hétérogène"
            interpretation = "Forte variabilité, multiples formations ou perturbations"

        return {
            "niveau_homogeneite": homogeneity,
            "coefficient_variation": round(cv, 3),
            "plage_resistivite": round(res_range, 1),
            "interpretation": interpretation
        }

    def _assess_aquifer_potential(self, mean_res: float, min_res: float) -> Dict:
        """Évaluation du potentiel aquifère"""
        # Les basses résistivités indiquent souvent des aquifères
        if min_res < 20:
            potential = "Élevé"
            confidence = 85
            indicators = "Résistivités très basses détectées, saturation probable"
        elif mean_res < 50:
            potential = "Moyen"
            confidence = 65
            indicators = "Résistivités conductrices, aquifères possibles"
        elif mean_res < 100:
            potential = "Faible"
            confidence = 30
            indicators = "Résistivités modérées, aquifères limités"
        else:
            potential = "Très faible"
            confidence = 10
            indicators = "Résistivités élevées, formations imperméables"

        return {
            "potentiel": potential,
            "confiance": confidence,
            "indicateurs": indicators
        }

    def _assess_pollution_risk(self, std_res: float, res_range: float) -> Dict:
        """Évaluation du risque de pollution"""
        # Forte variabilité peut indiquer pollution
        if res_range > 100 and std_res > 20:
            risk = "Élevé"
            confidence = 80
            indicators = "Forte variabilité résistivité, anomalies possibles"
        elif res_range > 50 or std_res > 15:
            risk = "Moyen"
            confidence = 60
            indicators = "Variabilité modérée, surveillance recommandée"
        else:
            risk = "Faible"
            confidence = 20
            indicators = "Formation homogène, faible risque apparent"

        return {
            "risque": risk,
            "confiance": confidence,
            "indicateurs": indicators
        }

    def _generate_recommendations(self, formations: List[Dict], heterogeneity: Dict) -> List[str]:
        """Génération de recommandations d'action"""
        recommendations = []

        # Recommandations basées sur les formations identifiées
        if formations:
            top_formation = formations[0]["formation"]

            if "argile" in top_formation or "marne" in top_formation:
                recommendations.append("📏 Calibration fine du modèle géologique requise - sensibilité élevée aux conditions d'humidité")
                recommendations.append("💧 Investigation complémentaire pour aquifères - forages de vérification recommandés")

            if "sable" in top_formation:
                recommendations.append("🏗️ Évaluation stabilité des sols - risques de liquéfaction possibles")
                recommendations.append("💦 Tests de perméabilité - potentiel aquifère significatif")

            if "calcaire" in top_formation or "grès" in top_formation:
                recommendations.append("⚠️ Inspection karstique - risques de cavités et instabilité")
                recommendations.append("🔍 Sondages géophysiques complémentaires - GPR recommandé")

        # Recommandations basées sur l'hétérogénéité
        hetero_level = heterogeneity.get("niveau_homogeneite", "")

        if "Très hétérogène" in hetero_level:
            recommendations.append("🔬 Étude géologique détaillée - hétérogénéité importante détectée")
            recommendations.append("📊 Augmentation densité points de mesure - couverture spatiale insuffisante")

        if "Très homogène" in hetero_level:
            recommendations.append("✅ Validation rapide possible - formation géologique cohérente")
            recommendations.append("📈 Extension modélisation - extrapolation fiable possible")

        # Recommandations générales
        recommendations.extend([
            "📋 Rapport d'interprétation détaillé à produire",
            "🎯 Points d'intérêt identifiés pour investigations complémentaires",
            "📊 Calibration croisée avec données géologiques existantes"
        ])

        return recommendations

    def generate_expert_report_structure(self, analysis_data: Dict) -> str:
        """
        Génération de la structure complète du rapport d'expert

        Args:
            analysis_data: Données d'analyse complètes

        Returns:
            Structure formatée du rapport
        """
        structure = f"""
📊 ANALYSE GÉOLOGIQUE COMPLÈTE (KIBALI) :

Géologie: [Description détaillée de la géologie du sous-sol basée sur {analysis_data.get('formations_probables', [])}]

Actions: [Méthodologie détaillée d'interprétation ERT et calibration modèles]

Image: [Description détaillée de la coupe géologique avec alternances de zones détectées]

🎯 RECOMMANDATIONS D'ACTION :
{chr(10).join(f"• {rec}" for rec in analysis_data.get('recommandations_action', []))}

📈 STATISTIQUES DÉTAILLÉES :
• Résistivité: {analysis_data.get('resistivity_stats', {}).get('min', 'N/A')} - {analysis_data.get('resistivity_stats', {}).get('max', 'N/A')} Ω·m
• Moyenne: {analysis_data.get('resistivity_stats', {}).get('mean', 'N/A'):.1f} Ω·m
• Écart-type: {analysis_data.get('resistivity_stats', {}).get('std', 'N/A'):.1f} Ω·m
• Points de mesure: {analysis_data.get('resistivity_stats', {}).get('count', 'N/A')}

CONCLUSION EXPERTE: [Synthèse professionnelle des résultats et implications]
"""

        return structure


# Instance globale
geology_interpretation_tool = GeologyInterpretationTool()