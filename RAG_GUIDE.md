# 🧠 Guide RAG - SETRAF Application

## ✅ Système RAG Optimisé et Fonctionnel

### 🎯 Qu'est-ce que le RAG ?

Le système RAG (Retrieval-Augmented Generation) enrichit les explications du LLM avec des connaissances géophysiques précises stockées dans une base vectorielle.

### 🚀 Fonctionnalités

1. **Base de connaissances vectorielle** : Index FAISS avec embeddings optimisés (all-MiniLM-L6-v2)
2. **Recherche sémantique rapide** : Similarité cosinus pour retrouver les informations pertinentes
3. **Upload de documents PDF** : Intégration de vos propres documents scientifiques
4. **Recherche web optionnelle** : Intégration Tavily API pour contexte temps réel
5. **Cache d'explications** : Performance optimisée avec mise en cache

### 📤 Ajouter vos documents PDF

#### Dans la sidebar :
1. Allez dans la section **"📤 Ajouter des documents PDF"**
2. Cliquez sur **"Choisir un fichier PDF"** et sélectionnez votre document
3. Cliquez sur **"📚 Indexer le document"**
4. Le document est automatiquement :
   - Extrait (2 premières pages)
   - Découpé en chunks de 512 caractères
   - Encodé en vecteurs 384D
   - Indexé dans FAISS

#### Dossier manuel :
Vous pouvez aussi copier directement vos PDFs dans :
```bash
/home/belikan/KIbalione8/SETRAF/rag_documents/
```
Puis cliquer sur **"🔄 Régénérer base RAG"**

### 🔍 Utilisation

#### Activation automatique :
- Le RAG s'active automatiquement si disponible
- Les explications LLM utilisent le contexte RAG sans action nécessaire

#### Dashboard RAG :
1. Cliquez sur **"🧠 Dashboard Explications RAG"**
2. Consultez les statistiques :
   - Nombre de documents indexés
   - Explications en cache
   - Historique des requêtes

#### Test du système :
- Cliquez sur **"🔍 Test RAG"** pour vérifier que le système fonctionne

### ⚙️ Configuration

#### Paramètres disponibles (sidebar) :
- **Recherche Web** : Toggle ON/OFF pour activer Tavily
- **Mode de recherche** : Vectorielle seule / Hybride (vecteur + web)

#### Performances :
- ⚡ Chargement modèle : ~2 secondes
- ⚡ Recherche vectorielle : <100ms
- ⚡ Génération contexte : <200ms
- 💾 Cache automatique pour réutilisation

### 🏗️ Architecture Technique

```
ERTKnowledgeBase
├── initialize_embeddings()      → Charge all-MiniLM-L6-v2 (384D)
├── load_or_create_vectorstore() → FAISS IndexFlatL2
├── search_knowledge_base()      → Recherche par similarité
├── search_web()                 → Tavily API (optionnel)
└── get_enhanced_context()       → Context enrichi final
```

#### Fichiers générés :
```
/home/belikan/KIbalione8/SETRAF/
├── rag_documents/                    # Vos PDFs sources
├── vector_db/
│   ├── ert_knowledge_light.faiss    # Index vectoriel
│   └── ert_documents_light.pkl      # Métadonnées documents
```

### 🧪 Test Standalone

Un script de test est disponible :
```bash
cd /home/belikan/KIbalione8/SETRAF
python test_rag_standalone.py
```

Résultats attendus :
```
✅ Modèle chargé (dimension: 384)
✅ Base créée : 2+ documents
✅ Recherche OK : résultats pertinents
✅ Contexte généré : ~600+ caractères
```

### 📊 Documents par défaut

Le système inclut par défaut :
1. **Échelle de résistivité ERT** : Valeurs typiques pour différents matériaux
2. **Méthodes ERT** : Pseudo-sections, inversion, classification

### 🔧 Dépannage

#### Erreur "meta tensor" :
✅ **Corrigé** : Le modèle charge maintenant directement sur CPU sans `.to()`

#### RAG non initialisé :
- Vérifiez que le modèle sentence-transformers est installé
- Consultez les logs Streamlit pour les erreurs
- Utilisez le script de test standalone pour diagnostic

#### Aucun résultat de recherche :
- Vérifiez que la base contient des documents
- Le seuil de pertinence est à 1.5 (distance L2)
- Régénérez la base si nécessaire

### 🌐 Intégration API Web

#### Tavily Search :
- API Key configurée dans `.env` ou code
- Timeout : 3 secondes
- Mode : "basic" (rapide)
- Résultats : 1 seul pour performance

#### Activer/Désactiver :
Toggle dans sidebar : **"Recherche Web"**

### 📈 Métriques de Performance

| Opération | Temps | Ressources |
|-----------|-------|------------|
| Init embeddings | ~2s | CPU only |
| Charge vectorstore | <0.5s | ~10MB RAM |
| Recherche (k=2) | <100ms | Minimal |
| Web search | <3s | Network |
| Context total | <300ms | Optimisé |

### 💡 Bonnes Pratiques

1. **Documents courts** : Limitez à 50 pages max par PDF
2. **Pertinence** : Uploadez uniquement docs géophysique/ERT
3. **Format** : PDFs avec texte extractible (non scannés)
4. **Régénération** : Après ajout de plusieurs docs, régénérez la base
5. **Cache** : Laissez le cache actif pour performance

### 🎓 Exemple d'utilisation

```
Utilisateur: "Quelle est la résistivité de l'argile ?"

RAG Process:
1. Encode query → vecteur 384D
2. Recherche FAISS → Top 2 chunks pertinents
3. Extract: "1-10 Ω·m : EAU SAUMÂTRE / ARGILES"
4. Context enrichi → LLM
5. LLM génère: "L'argile a typiquement une résistivité entre 1 et 10 Ω·m..."
```

### ✨ Avantages

- ✅ **Précision** : Réponses basées sur vraies données scientifiques
- ✅ **Rapidité** : Recherche vectorielle ultra-rapide (<100ms)
- ✅ **Évolutif** : Ajoutez vos propres documents
- ✅ **Hybride** : Combine connaissances locales + web
- ✅ **Cache** : Réutilisation intelligente des explications

---

**Version système** : RAG Optimized v1.0  
**Dernière mise à jour** : 9 décembre 2025  
**Test validé** : ✅ Tous les tests passent
