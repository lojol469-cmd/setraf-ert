# OUTILS GÉOLOGIQUES KIBALI

## Vue d'ensemble

Le système d'outils géologiques KIBALI fournit une suite complète d'outils spécialisés pour l'analyse géophysique avancée, l'interprétation géologique et la recherche web intégrée. Ces outils permettent à KIBALI de fonctionner comme un expert géologue IA capable de fournir des analyses détaillées et professionnelles.

## Architecture

```
tools/
├── __init__.py              # Module principal et orchestrateur
├── web_search_tools.py      # Outils de recherche web (Tavily API)
├── geology_analysis_tools.py # Outils d'analyse statistique
├── geology_interpretation_tools.py # Outils d'interprétation géologique
└── orchestrator.py          # Orchestrateur principal
```

## Outils Disponibles

### 1. GeologyWebSearchTool
- **Fonction**: Recherche d'informations géologiques sur le web
- **API**: Tavily API pour recherche intelligente
- **Méthodes principales**:
  - `search_geology_info()`: Recherche ciblée sur des termes géologiques
  - `gather_comprehensive_geology_data()`: Recherche approfondie avec contexte

### 2. WebResearchManager
- **Fonction**: Gestionnaire de recherche web avancée
- **Capacités**: Agrégation et synthèse d'informations multiples
- **Utilisation**: Recherche contextuelle pour enrichir les analyses

### 3. GeologyStatisticsTool
- **Fonction**: Analyse statistique des données de résistivité
- **Méthodes**:
  - `calculate_resistivity_statistics()`: Statistiques complètes (moyenne, médiane, écart-type, etc.)
  - `analyze_depth_distribution()`: Analyse par profondeur
  - `detect_anomalies()`: Détection d'anomalies par méthodes statistiques
  - `cluster_analysis()`: Clustering K-means pour identification de groupes

### 4. GeologyInterpretationTool
- **Fonction**: Interprétation géologique experte
- **Capacités**:
  - Classification automatique des formations géologiques
  - Évaluation du potentiel aquifère
  - Analyse de l'hétérogénéité géologique
  - Génération de recommandations d'action
  - Structure de rapport d'expert

### 5. GeologyToolsOrchestrator
- **Fonction**: Orchestrateur principal intégrant tous les outils
- **Méthodes clés**:
  - `perform_complete_geology_analysis()`: Analyse complète intégrée
  - `generate_expert_response()`: Génération de réponses d'expert formatées

## Utilisation dans ERTest.py

### Import
```python
from tools import geology_tools_orchestrator
```

### Analyse complète
```python
# Données d'exemple
resistivity_values = [10.5, 25.3, 45.7, 120.8, 89.2]
depths = [0.5, 1.0, 1.5, 2.0, 2.5]

# Analyse complète
analysis = geology_tools_orchestrator.perform_complete_geology_analysis(
    resistivity_values=resistivity_values,
    depths=depths,
    soil_composition="argiles et sables",
    location_context="site géophysique"
)
```

### Génération de réponse d'expert
```python
response = geology_tools_orchestrator.generate_expert_response(
    analysis_results=analysis,
    user_query="Quelle formation géologique prédomine?",
    response_length="detailed"  # "brief", "detailed", "comprehensive"
)
```

## Format de Réponse d'Expert

Les réponses suivent le format standardisé :

```
📊 ANALYSE GÉOLOGIQUE COMPLÈTE (KIBALI) :

Géologie: [Analyse détaillée des formations...]

Actions: [Actions recommandées...]

Image: [Description visuelle...]

🎯 RECOMMANDATIONS D'ACTION :
• [Recommandation 1]
• [Recommandation 2]

📈 STATISTIQUES DÉTAILLÉES :
• [Statistiques détaillées]

CONCLUSION EXPERTE: [Conclusion professionnelle...]
```

## Configuration

### Variables d'environnement requises
- `TAVILY_API_KEY`: Clé API pour la recherche web (optionnel mais recommandé)

### Installation des dépendances
```bash
pip install tavily-py numpy pandas scikit-learn
```

## Intégration dans le Chat IA

Le système s'intègre automatiquement dans le chat IA d'ERTest.py :

1. **Détection automatique**: Les questions contenant des mots-clés géologiques déclenchent l'analyse KIBALI
2. **Extraction des données**: Utilise automatiquement les données chargées dans `st.session_state['dataframe']`
3. **Réponse enrichie**: Combine l'analyse KIBALI avec le contexte RAG et la génération LLM

## Exemple de Question Géologique

Questions qui déclenchent l'analyse KIBALI :
- "Quelle est la formation géologique principale ?"
- "Y a-t-il des aquifères dans ce sous-sol ?"
- "Analyse la résistivité de mes données ERT"
- "Quelles anomalies géologiques détectez-vous ?"

## Métriques et Performances

- **Précision**: Analyse statistique rigoureuse avec validation croisée
- **Vitesse**: Optimisé pour les grands jeux de données
- **Robustesse**: Gestion d'erreur complète et fallbacks
- **Évolutivité**: Architecture modulaire pour ajout d'outils

## Développement et Extension

### Ajout d'un nouvel outil
1. Créer une classe spécialisée dans un fichier dédié
2. L'intégrer dans `GeologyToolsOrchestrator`
3. L'ajouter aux imports dans `__init__.py`
4. Tester l'intégration

### Personnalisation des réponses
- Modifier les templates dans `orchestrator.py`
- Ajuster les seuils dans les outils d'analyse
- Étendre la base de connaissances géologique

## Support et Maintenance

- **Logs**: Logging complet dans tous les outils
- **Tests**: Script de test automatisé (`test_geology_tools.py`)
- **Documentation**: Mise à jour automatique des docstrings
- **Monitoring**: Métriques de performance intégrées

---

**Version**: 1.0.0
**Auteur**: Système KIBALI Geological Analysis
**Date**: Décembre 2024