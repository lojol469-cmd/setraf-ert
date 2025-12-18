# 📚 Système RAG - Guide de Régénération

## ✅ Base Vectorielle Actuelle

- **374 chunks** indexés
- **27,708 mots** de connaissances géophysiques
- **2 PDFs** traités (FicheERT.pdf + 001.PDF TEXTE.pdf)

## 🔄 Comment Régénérer la Base RAG

### Option 1: Depuis Streamlit (Recommandé)

1. Lancez l'application : `streamlit run ERTest.py`
2. Dans la sidebar, section **"📚 Système RAG"**
3. Cliquez sur **"🔄 Régénérer base RAG"**
4. Attendez le chargement (5-10 secondes)
5. Vérifiez l'affichage : `✅ RAG Actif: 374 chunks | 27,708 mots`

### Option 2: Script Python (Plus Rapide)

```bash
cd /home/belikan/KIbalione8/SETRAF
python3.13 regenerate_rag_full.py
```

Ce script va :
- Extraire TOUS les PDFs du dossier `rag_documents/`
- Découper en chunks de 512 caractères
- Générer les embeddings (all-MiniLM-L6-v2)
- Créer l'index FAISS
- Sauvegarder dans `vector_db/`

## 📤 Ajouter de Nouveaux PDFs

### Méthode 1: Upload dans l'interface

1. Dans la sidebar, section **"📤 Ajouter des documents PDF"**
2. Cliquez sur **"Choisir un fichier PDF"**
3. Sélectionnez votre PDF
4. Cliquez sur **"📚 Indexer le document"**
5. Le PDF est automatiquement ajouté à la base vectorielle

### Méthode 2: Copie manuelle

```bash
# Copier votre PDF dans le dossier
cp mon_document.pdf /home/belikan/KIbalione8/SETRAF/rag_documents/

# Régénérer la base complète
cd /home/belikan/KIbalione8/SETRAF
python3.13 regenerate_rag_full.py
```

## 🧪 Tester le Système RAG

### Test 1: Vérifier le chargement

```bash
cd /home/belikan/KIbalione8/SETRAF
python3.13 test_rag_loading.py
```

Résultat attendu :
```
✅ 374 chunks chargés
📊 27,708 mots totaux
✅ Cohérence FAISS/Documents: 374 vecteurs = 374 chunks
```

### Test 2: Tester l'extraction

```bash
cd /home/belikan/KIbalione8/SETRAF
python3.13 test_rag_system.py
```

### Test 3: Dans l'interface Streamlit

1. Cliquez sur **"🧠 Dashboard Explications RAG"**
2. Allez dans l'onglet **"📚 Base de Connaissances"**
3. Vérifiez : `✅ 374 chunks indexés`
4. Dans l'onglet **"🔍 Tester la Recherche"**
5. Tapez une question : `"Quelle est la résistivité de l'argile ?"`
6. Vérifiez que 5 chunks pertinents sont retournés

## 🔧 Dépannage

### Problème : "2 chunks seulement"

**Solution** : La session Streamlit a chargé l'ancien cache

```bash
# Supprimer le cache et régénérer
cd /home/belikan/KIbalione8/SETRAF
rm -f vector_db/ert_knowledge_light.faiss vector_db/ert_documents_light.pkl
python3.13 regenerate_rag_full.py

# Puis dans Streamlit, cliquez sur "🔄 Régénérer base RAG"
```

### Problème : "Aucun résultat RAG"

**Vérifications** :

1. Vérifier que les fichiers existent :
```bash
ls -lh /home/belikan/KIbalione8/SETRAF/vector_db/
```

2. Vérifier le contenu :
```bash
python3.13 test_rag_loading.py
```

3. Vérifier que le modèle d'embeddings existe :
```bash
ls -lh /home/belikan/KIbalione8/SETRAF/models/embeddings/sentence-transformers--all-MiniLM-L6-v2/
```

### Problème : "Recherche web (Tavily) ne marche pas"

**Vérifier l'API Key** :

Dans `ERTest.py`, ligne 59 :
```python
TAVILY_API_KEY = "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"
```

**Activer la recherche web** :

Dans la sidebar, cocher **"🌐 Recherche web (Tavily)"**

## 📊 Architecture du Système

```
SETRAF/
├── rag_documents/           # PDFs sources
│   ├── FicheERT.pdf        (6 pages, 8,940 chars)
│   └── 001.PDF TEXTE.pdf   (33 pages, 157,495 chars)
│
├── vector_db/               # Base vectorielle
│   ├── ert_knowledge_light.faiss  (561 KB - 374 vecteurs)
│   └── ert_documents_light.pkl    (182 KB - 374 chunks)
│
├── models/embeddings/       # Modèle d'embeddings
│   └── sentence-transformers--all-MiniLM-L6-v2/
│
└── ERTest.py                # Application principale
```

## 🎯 Configuration RAG

### Paramètres actuels

- **Chunk size** : 512 caractères
- **Chunk overlap** : 50 caractères
- **Nombre de résultats (k)** : 5 chunks
- **Dimension embeddings** : 384
- **Index FAISS** : IndexFlatL2 (recherche exacte)
- **Recherche web** : Tavily API (2 résultats max)

### Modifier les paramètres

Pour augmenter le nombre de chunks retournés :

Dans `ERTest.py`, ligne ~680 :
```python
def get_enhanced_context(self, query, use_web=False):
    # Modifier k=5 pour avoir plus de chunks
    vector_results = self.search_knowledge_base(query, k=5)  # Changez ici
```

## ✅ Checklist de Vérification

- [ ] `python3.13 test_rag_loading.py` affiche 374 chunks
- [ ] Dans Streamlit sidebar : `✅ RAG Actif: 374 chunks | 27,708 mots`
- [ ] Dashboard RAG → Base de Connaissances → 374 chunks indexés
- [ ] Test recherche RAG retourne 5 résultats pertinents
- [ ] Recherche web Tavily activée et fonctionnelle
- [ ] LLM génère des analyses détaillées de 30+ lignes

---

**Dernière mise à jour** : 9 décembre 2025
**Status** : ✅ 374 chunks opérationnels
