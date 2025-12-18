# SYSTÈME RAG POUR GÉOPHYSIQUE ERT - SETRAF

## Vue d'ensemble

Le système RAG (Retrieval-Augmented Generation) intégré à SETRAF enrichit les explications du LLM Mistral avec une base de connaissances vectorielle spécialisée en géophysique ERT et une recherche web intelligente.

## Fonctionnalités

### 🧠 Base de connaissances vectorielle
- **Embeddings sémantiques** : Utilise Sentence Transformers pour indexer les documents
- **Recherche FAISS** : Recherche vectorielle ultra-rapide dans la base de connaissances
- **Documents spécialisés** : Base pré-remplie avec connaissances ERT (résistivités, méthodes, configurations)

### 🌐 Recherche web intelligente
- **API Tavily** : Recherche spécialisée sur internet pour compléter les connaissances
- **Recherche contextuelle** : Requêtes optimisées pour la géophysique ERT
- **Sources fiables** : Priorisation des sources scientifiques et techniques

### 📚 Enrichissement des explications
- **Contexte scientifique** : Chaque explication inclut des références validées
- **Précision maximale** : Combinaison connaissances locales + recherche web
- **Cache intelligent** : Réutilisation des explications pour performance optimale

## Architecture

```
[Requête utilisateur] → [Construction requête RAG] → [Recherche vectorielle + Web]
                              ↓
[Contexte enrichi] → [LLM Mistral] → [Explication précise]
                              ↓
[Cache + Tracker] → [Dashboard d'explications]
```

## Utilisation

### 1. Initialisation automatique
Le système RAG se charge automatiquement avec le LLM Mistral au démarrage de l'application.

### 2. Interface utilisateur
- **Dashboard RAG** : Bouton "🧠 Dashboard Explications RAG" pour voir toutes les explications
- **Test RAG** : Bouton "🔍 Test RAG" pour vérifier le fonctionnement
- **Statut** : Indicateur en temps réel de l'état du système

### 3. Enrichissement des documents
- **Upload PDF** : Section dans la sidebar pour ajouter des documents scientifiques
- **Indexation automatique** : Les nouveaux documents sont automatiquement indexés
- **Reconstruction** : Bouton pour régénérer complètement la base

## Types d'explications enrichies

### 🔬 Analyse géologique
- Classification précise selon l'échelle internationale de résistivité
- Références aux normes géophysiques établies
- Contexte hydrogéologique validé

### 📊 Clustering et classification
- Justification mathématique des algorithmes utilisés
- Validation statistique des groupes identifiés
- Interprétation géologique basée sur données réelles

### 🗺️ Visualisations
- Standards cartographiques respectés
- Codage couleur selon normes internationales
- Interprétation technique précise

## Base de connaissances incluse

### Échelle de résistivité ERT
- 0.01-1 Ω·m : Eau de mer, minéraux métalliques
- 1-10 Ω·m : Eau saumâtre, argiles marines
- 10-100 Ω·m : Eau douce, sols fins
- 100-1000 Ω·m : Sables saturés, graviers
- 1000-10000 Ω·m : Roches sédimentaires
- >10000 Ω·m : Socle cristallin

### Méthodes d'interprétation
- Pseudo-sections 2D
- Inversion 3D avec régularisation Tikhonov
- Classification géologique automatisée
- Analyse statistique des données

### Configurations d'électrodes
- Wenner, Schlumberger, Dipole-Dipole
- Facteurs géométriques et résolution
- Optimisation selon objectifs

## Configuration technique

### Dépendances
```bash
pip install sentence-transformers faiss-cpu langchain pypdf requests
```

### Variables d'environnement
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx  # HuggingFace
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx  # Tavily Search
```

### Structure des dossiers
```
/home/belikan/KIbalione8/SETRAF/
├── rag_documents/          # Documents PDF à indexer
├── vector_db/             # Base vectorielle FAISS
│   ├── ert_knowledge.faiss
│   └── ert_documents.pkl
└── ERTest.py              # Application principale
```

## Avantages du système RAG

### ✅ Précision scientifique
- Explications basées sur connaissances validées
- Références aux standards internationaux
- Évitement des hallucinations du LLM

### ⚡ Performance optimisée
- Cache intelligent des explications
- Recherche vectorielle ultra-rapide
- Réutilisation des contextes similaires

### 🔄 Évolutivité
- Ajout facile de nouveaux documents
- Mise à jour automatique de la base
- Enrichissement continu des connaissances

### 🌍 Connaissance globale
- Combinaison base locale + recherche web
- Accès aux dernières publications
- Contexte scientifique à jour

## Métriques et monitoring

### Dashboard d'explications
- Nombre total d'explications générées
- Répartition par type d'opération
- Taille du cache d'explications
- État de la base vectorielle

### Tests de validation
- Test de recherche vectorielle
- Validation des résultats web
- Contrôle qualité des explications

## Support et maintenance

### Mise à jour de la base
1. Ajouter des PDF dans `rag_documents/`
2. Utiliser le bouton "Régénérer base RAG"
3. Vérifier l'indexation dans le dashboard

### Dépannage
- Vérifier les logs d'initialisation
- Tester la connectivité API Tavily
- Contrôler l'espace disque pour FAISS

---

**Développé pour SETRAF - Analyse géophysique ERT avancée**
*Version RAG 1.0 - Décembre 2025*</content>
<parameter name="filePath">/home/belikan/KIbalione8/SETRAF/RAG_README.md