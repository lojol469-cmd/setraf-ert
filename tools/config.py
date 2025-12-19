"""
CONFIGURATION DES OUTILS GÉOLOGIQUES KIBALI
Paramètres et constantes pour tous les outils
"""

import os
from typing import Dict, Any

# Configuration API
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')

# Seuils de résistivité pour classification géologique (Ω·m)
RESISTIVITY_THRESHOLDS = {
    'minéraux_métalliques': (0.001, 1.0),
    'eau_mer_argiles_marines': (0.1, 10.0),
    'eau_douce_sols_fins': (10.0, 100.0),
    'sables_satures_aquifere': (100.0, 1000.0),
    'roches_sedimentaires': (1000.0, 10000.0),
    'socle_cristallin': (10000.0, float('inf'))
}

# Couleurs associées aux formations
GEOLOGICAL_COLORS = {
    'minéraux_métalliques': 'black',
    'eau_mer_argiles_marines': 'blue',
    'eau_douce_sols_fins': 'green',
    'sables_satures_aquifere': 'yellow',
    'roches_sedimentaires': 'orange',
    'socle_cristallin': 'red'
}

# Descriptions des formations géologiques
GEOLOGICAL_FORMATIONS = {
    'minéraux_métalliques': {
        'description': 'Minéraux conducteurs (graphite, sulfures métalliques)',
        'implications': 'Ressources minérales potentielles, nécessite analyse chimique',
        'potentiel_aquifere': 'très faible',
        'recommandations': ['Analyse chimique des minéraux', 'Évaluation économique du gisement']
    },
    'eau_mer_argiles_marines': {
        'description': 'Eaux salées ou saumâtres, argiles marines saturées',
        'implications': 'Zone d\'interface eau douce/eau salée, risque de contamination',
        'potentiel_aquifere': 'faible à moyen',
        'recommandations': ['Analyse qualité eau', 'Étude de salinisation', 'Pompage contrôlé']
    },
    'eau_douce_sols_fins': {
        'description': 'Eaux douces, sols fins argileux ou limoneux',
        'implications': 'Aquifère de porosité, stockage important, perméabilité variable',
        'potentiel_aquifere': 'moyen à élevé',
        'recommandations': ['Forage d\'exploration', 'Test de pompage', 'Protection de la ressource']
    },
    'sables_satures_aquifere': {
        'description': 'Sables saturés, aquifère productif',
        'implications': 'Excellente perméabilité, débit potentiel élevé',
        'potentiel_aquifere': 'élevé',
        'recommandations': ['Forage de production', 'Dimensionnement du captage', 'Étude d\'impact']
    },
    'roches_sedimentaires': {
        'description': 'Calcaire, grès, schiste - formations sédimentaires consolidées',
        'implications': 'Aquifère fissuré, karstique possible, débit variable',
        'potentiel_aquifere': 'moyen (fissures)',
        'recommandations': ['Étude structurale', 'Test de fracturation', 'Analyse karstique']
    },
    'socle_cristallin': {
        'description': 'Roche mère cristalline (granite, gneiss, basalte)',
        'implications': 'Faible perméabilité, aquifère uniquement en zones fracturées',
        'potentiel_aquifere': 'faible',
        'recommandations': ['Étude de fracturation', 'Forage profond', 'Recherche d\'eau de subsurface']
    }
}

# Paramètres d'analyse statistique
STATISTICAL_PARAMS = {
    'anomaly_threshold': 2.0,  # Écart-type pour détection d'anomalies
    'min_samples_cluster': 3,  # Nombre minimum d'échantillons par cluster
    'max_clusters': 5,  # Nombre maximum de clusters
    'confidence_level': 0.95,  # Niveau de confiance pour intervalles
    'heterogeneity_threshold': 0.3  # Seuil pour hétérogénéité (coefficient de variation)
}

# Paramètres de recherche web
WEB_SEARCH_PARAMS = {
    'max_results': 3,  # Nombre maximum de résultats par recherche
    'search_timeout': 10,  # Timeout en secondes
    'retry_attempts': 2,  # Nombre de tentatives
    'cache_duration': 3600  # Durée du cache en secondes (1 heure)
}

# Paramètres d'interprétation géologique
GEOLOGICAL_INTERPRETATION_PARAMS = {
    'depth_weight': 0.3,  # Poids de la profondeur dans l'interprétation
    'resistivity_weight': 0.7,  # Poids de la résistivité
    'formation_confidence_threshold': 0.6,  # Seuil de confiance pour classification
    'aquifer_potential_weights': {
        'resistivity': 0.4,
        'thickness': 0.3,
        'heterogeneity': 0.3
    }
}

# Messages d'erreur et avertissements
ERROR_MESSAGES = {
    'no_api_key': '⚠️ TAVILY_API_KEY non trouvée - recherche web limitée',
    'invalid_data': '❌ Données de résistivité invalides ou insuffisantes',
    'analysis_failed': '❌ Échec de l\'analyse géologique',
    'web_search_failed': '⚠️ Recherche web indisponible',
    'clustering_failed': '⚠️ Analyse par clusters impossible'
}

# Templates de réponse d'expert
RESPONSE_TEMPLATES = {
    'brief': """
📊 ANALYSE GÉOLOGIQUE RAPIDE:
Formation principale: {formation}
Résistivité: {mean:.1f} Ω·m (moyenne)
Hétérogénéité: {heterogeneity}
Potentiel aquifère: {aquifer_potential}
""",

    'detailed_header': "📊 ANALYSE GÉOLOGIQUE COMPLÈTE (KIBALI) :\n\n",

    'conclusion': """
CONCLUSION EXPERTE: Cette analyse révèle une formation géologique complexe nécessitant
une approche intégrée combinant géophysique, géologie et hydrologie.
Les résultats obtenus sont cohérents avec les données de référence et suggèrent
des investigations complémentaires pour confirmer les interprétations proposées."""
}

# Configuration de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

def get_resistivity_category(resistivity: float) -> str:
    """Détermine la catégorie géologique basée sur la résistivité"""
    for category, (min_val, max_val) in RESISTIVITY_THRESHOLDS.items():
        if min_val <= resistivity < max_val:
            return category
    return 'socle_cristallin'  # Par défaut

def get_formation_info(formation_key: str) -> Dict[str, Any]:
    """Récupère les informations détaillées d'une formation"""
    return GEOLOGICAL_FORMATIONS.get(formation_key, {})

def validate_api_keys() -> Dict[str, bool]:
    """Valide la disponibilité des clés API"""
    return {
        'tavily': bool(TAVILY_API_KEY)
    }

# Validation au chargement du module
api_status = validate_api_keys()
if not api_status['tavily']:
    print(ERROR_MESSAGES['no_api_key'])