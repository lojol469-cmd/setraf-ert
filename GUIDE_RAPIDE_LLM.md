# 🚀 GUIDE RAPIDE D'UTILISATION - LLM MISTRAL

## ⚡ DÉMARRAGE RAPIDE (5 étapes)

### 1️⃣ Lancer SETRAF
```bash
cd /home/belikan/KIbalione8/SETRAF
streamlit run ERTest.py
```

### 2️⃣ Aller à l'onglet "Analyse Spectrale d'Images"
- Cliquer sur l'onglet 🖼️ en haut

### 3️⃣ Uploader votre image géophysique
- Format accepté : PNG, JPG, TIFF
- Cliquer sur "Browse files"

### 4️⃣ Activer l'IA Mistral (NOUVEAU !)
```
✅ Cocher "🧠 Activer l'analyse LLM avancée (recommandé)"
```
**Attendre 10-30 secondes** (chargement unique du modèle)

### 5️⃣ Lire les résultats intelligents
- 📊 **Interprétation Géologique** : Que révèlent vos données ?
- 🎯 **Recommandations Pratiques** : Que faire concrètement ?
- 🎨 **Générer l'image** : Rendu photo-réaliste optimisé

---

## 🎯 DEUX MODES D'UTILISATION

### MODE 1 : Analyse Rapide (après extraction spectrale)
```
1. Extraire spectres
2. ✅ Activer LLM
3. Lire l'analyse
4. Générer image
```
⏱️ **Temps total** : 1-2 minutes

### MODE 2 : Analyse Complète (toutes les étapes)
```
1. Extraire spectres
2. Imputation matricielle
3. Modélisation forward
4. Reconstruction 3D
5. Détection trajectoires
6. ✅ Activer LLM complet
7. Lire l'analyse globale
8. Générer rendu final
```
⏱️ **Temps total** : 5-10 minutes

---

## 📖 CE QUE FAIT LE LLM

### 🧠 Analyse Intelligente
```
Mistral lit vos données :
- Résistivités mesurées (min/max/moyenne)
- Structures détectées
- Qualité des données
```

### 💬 Explications en Langage Naturel
```
"Les valeurs de résistivité mesurées (15-850 Ω·m) 
indiquent la présence de trois formations distinctes.
La zone de faible résistivité suggère un aquifère 
peu profond..."
```

### 🎨 Optimisation des Images
```
Prompt standard :
"geological cross-section"

Prompt LLM optimisé :
"underground geological cross-section showing 
three distinct layers: surface clay (blue), 
sandy layer (yellow), sandstone bedrock (red), 
clear boundaries at 5m and 15m depth..."
```

---

## ✅ AVANTAGES IMMÉDIATS

| Sans LLM | Avec LLM |
|----------|----------|
| Texte générique | **Analyse personnalisée** |
| Pas de recommandations | **Actions concrètes** |
| Image moyenne | **Rendu photo-réaliste** |
| Interprétation manuelle | **Analyse automatique** |

---

## 🔧 PARAMÈTRES (optionnels)

### Modèle IA de Génération
- **Stable Diffusion XL** : Haute résolution (1024×1024)
- **RealVisXL V4.0** : Ultra-réaliste
- **DreamShaper 8** : Artistique
- **Realistic Vision V5.1** : Réaliste scientifique
- **epiCRealism** : Photo-réaliste

### Style Artistique
- **Réaliste scientifique** : Pour rapports techniques
- **Art géologique** : Pour présentations
- **Coupes techniques** : Pour publications
- **3D réaliste** : Pour visualisation 3D

---

## 💡 CONSEILS D'UTILISATION

### ✅ RECOMMANDÉ
- Toujours activer le LLM (meilleurs résultats)
- Lire l'interprétation AVANT de générer l'image
- Suivre les recommandations pratiques
- Télécharger l'image finale pour vos rapports

### ⚠️ À ÉVITER
- Ne pas désactiver le LLM (sauf si problème de mémoire)
- Ne pas ignorer les recommandations
- Ne pas fermer la fenêtre pendant la génération

---

## 📊 EXEMPLE CONCRET

### Données en entrée
```
Résistivité : 10-1000 Ω·m
Spectres : 250
Trajectoires : 3 détectées
```

### Sortie LLM (exemple)
```
📊 INTERPRÉTATION GÉOLOGIQUE :
"Les mesures révèlent un système aquifère multicouche 
avec une zone conductrice en surface (10-50 Ω·m) 
correspondant à des argiles saturées, une formation 
intermédiaire (100-300 Ω·m) de sables fins, et un 
socle résistif (>500 Ω·m) en profondeur. Les trois 
structures linéaires détectées marquent les interfaces 
entre ces formations."

🎯 RECOMMANDATIONS :
- Effectuer des forages d'exploration à 5-10m de profondeur
- Cibler les zones à résistivité 10-50 Ω·m pour l'eau
- Réaliser des essais de pompage pour confirmer
- Intégrer avec données hydrogéologiques existantes

🎨 PROMPT OPTIMISÉ :
"Underground geological cross-section with three distinct 
layers: saturated clay formation with low resistivity 
(blue tones), sandy intermediate layer (yellow-green), 
deep resistive bedrock (red-orange), clear stratigraphic 
boundaries, realistic textures, scientific accuracy"
```

---

## 🎬 RÉSULTAT FINAL

### Sans LLM
🖼️ Image standard générique

### Avec LLM
🖼️ **Image photo-réaliste personnalisée**
📊 **Interprétation experte**
🎯 **Recommandations concrètes**
📝 **Rapport professionnel**

---

## ⏱️ TEMPS DE TRAITEMENT

| Étape | Temps |
|-------|-------|
| Chargement LLM (1ère fois) | 10-30s |
| Chargements suivants | Instantané |
| Analyse LLM simple | 5-15s |
| Analyse LLM complète | 10-30s |
| Génération d'image | 30s-2min |

---

## 🐛 PROBLÈMES COURANTS

### "Impossible de charger Mistral"
✅ **Solution** : Le système continue sans LLM (mode standard)

### "Mémoire insuffisante"
✅ **Solution** : Fermer d'autres applications, le mode CPU est optimisé

### "Analyse trop lente"
✅ **Solution** : Normal la première fois (mise en cache), ensuite instantané

---

## 📞 BESOIN D'AIDE ?

Consultez la documentation complète :
- `INTEGRATION_LLM_MISTRAL.md` : Guide détaillé
- `RESUME_INTEGRATION_LLM.md` : Résumé technique
- `test_llm_integration.py` : Tests automatisés

---

## 🎉 PROFITEZ DE L'IA !

**SETRAF avec Mistral LLM = Votre assistant géophysicien intelligent**

Analysez, interprétez, visualisez vos données en quelques clics !

---

**Développé pour SETRAF - Subaquifère ERT Analysis Tool**  
**Intelligence Artificielle Avancée - Décembre 2025**
