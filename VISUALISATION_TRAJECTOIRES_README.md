# 🔬 Visualisation Réaliste des Trajectoires de Neutrinos

## 🎯 Objectif

Créer des **coupes géologiques réalistes** montrant les **cavités, failles et structures cachées** détectées par l'analyse RANSAC inspirée de la physique des neutrinos.

---

## 🌟 Fonctionnalités

### 1. **Détection des Structures Linéaires**
- Algorithme RANSAC (RANdom SAmple Consensus)
- Inspiré de la détection de trajectoires de neutrinos
- Identifie les discontinuités géologiques

### 2. **Révélation des Anomalies**
Le système détecte :
- 🕳️ **Cavités et vides** (grottes, karsts)
- 🪨 **Failles géologiques** (fractures, cassures)
- 💧 **Écoulements souterrains** (rivières cachées)
- 📏 **Couches inclinées** (pendages géologiques)

### 3. **Génération IA Spécialisée**
- **Prompt ultra-spécifique** pour les trajectoires
- **Emphase configurable** : cavités, failles, ou toutes structures
- **3 styles** : Réaliste scientifique, Coupes techniques, Art géologique

---

## 🔧 Utilisation

### **Étape 1 : Détecter les Trajectoires**
```
1. Complétez toutes les étapes précédentes
2. Allez à "5. Détection de Trajectoires (RANSAC)"
3. Cliquez "🚀 Détecter Trajectoires"
4. Attendez l'analyse (peut prendre 1-2 minutes)
```

### **Étape 2 : Générer la Visualisation**
```
5. Descendez à "Visualisation Réaliste des Trajectoires & Cavités"
6. Choisissez :
   - Modèle IA (RealVisXL recommandé pour précision)
   - Style de coupe (Réaliste scientifique pour rapports)
   - Emphase (Cavités, Failles, ou Toutes structures)
7. Cliquez "🚀 Générer Coupe Réaliste des Trajectoires"
8. Attendez 30s-2min (génération complexe)
```

### **Étape 3 : Analyser les Résultats**
```
9. Examinez la coupe réaliste générée
10. Identifiez les zones sombres (cavités) et lignes brillantes (failles)
11. Lisez les recommandations d'exploration
12. Téléchargez l'image pour vos rapports
```

---

## �� Interprétation des Images

### **Zones Sombres/Noires** 🕳️
- **Signification** : Cavités, vides, grottes
- **Résistivité** : Très faible (< 10 Ω·m)
- **Action** : Investigations spéléologiques, mesures de stabilité

### **Lignes Brillantes/Fractures** 🪨
- **Signification** : Failles, discontinuités
- **Résistivité** : Contraste élevé
- **Action** : Cartographie précise, évaluation risques sismiques

### **Zones Bleutées Continues** 💧
- **Signification** : Écoulements souterrains
- **Résistivité** : Faible à moyenne (10-100 Ω·m)
- **Action** : Études hydrogéologiques, forages d'exploration

### **Zones Claires/Orangées** 🏔️
- **Signification** : Roches compactes, socle
- **Résistivité** : Élevée (> 500 Ω·m)
- **Action** : Fondations possibles, stabilité confirmée

---

## 🧮 Algorithme RANSAC

### **Principe**
```
1. Sélection aléatoire d'échantillons (points de mesure)
2. Ajustement d'un modèle linéaire
3. Comptage des "inliers" (points conformes)
4. Répétition itérative
5. Sélection du meilleur modèle (score RANSAC)
```

### **Paramètres Configurables**
- **Échantillons min** : Nombre de points pour ajuster une ligne (2-10)
- **Seuil résiduel** : Distance max pour être "inlier" (0.1-5.0)
- **Essais max** : Nombre d'itérations RANSAC (100-10000)

### **Inspiration Physique**
Cette méthode est directement inspirée de la **détection de trajectoires de neutrinos** dans les détecteurs de particules (IceCube, Super-Kamiokande), où on cherche des alignements de signaux dans un bruit de fond important.

---

## 🎨 Prompt IA Généré

### **Structure du Prompt**
```python
f"""Geological cross-section showing {emphasis}.
{n_trajectories} linear structures detected by neutrino-inspired RANSAC.
Resistivity range: {rho_min} to {rho_max} ohm-meters.
Highlighted pathways indicate subsurface anomalies:
- Dark zones for low resistivity (water-filled cavities)
- Bright fractures for geological discontinuities.
Scientific accuracy, realistic textures."""
```

### **Adaptation Dynamique**
Le prompt s'adapte automatiquement à :
- Nombre de trajectoires détectées
- Plage de résistivités mesurées
- Type d'emphase choisi par l'utilisateur
- Profondeur des structures

---

## 📈 Statistiques Affichées

### **Métrique 1 : Trajectoires Détectées**
- Nombre total de structures linéaires
- Indique la complexité du sous-sol

### **Métrique 2 : Score Moyen RANSAC**
- Qualité moyenne des trajectoires
- Score élevé = structures bien définies
- Score faible = incertitude, validation terrain requise

### **Métrique 3 : Points d'Intérêt**
- Nombre total d'inliers (tous les trajectoires)
- Densité des anomalies détectées

---

## 🎯 Cas d'Usage

### **1. Exploration Spéléologique**
- Détection de grottes et karsts
- Cartographie de réseaux souterrains
- Évaluation de la stabilité

### **2. Géotechnique**
- Identification de failles pour risques de construction
- Évaluation de la stabilité de fondations
- Détection de zones de glissement potentielles

### **3. Hydrogéologie**
- Localisation d'écoulements souterrains
- Cartographie d'aquifères fracturés
- Planification de forages

### **4. Archéologie**
- Détection de structures enterrées
- Localisation de cavités artificielles (tunnels, cryptes)
- Cartographie non-invasive

### **5. Risques Naturels**
- Évaluation de risques karstiques
- Détection de vides sous routes/bâtiments
- Cartographie de zones instables

---

## 🔬 Validation Scientifique

### **Comparaison avec Méthodes Classiques**
| Méthode | Précision | Coût | Rapidité |
|---------|-----------|------|----------|
| RANSAC + IA | ★★★★☆ | € | ★★★★★ |
| Radar géologique | ★★★★★ | €€€ | ★★★☆☆ |
| Sismique réfraction | ★★★★☆ | €€€€ | ★★☆☆☆ |
| Forages exploratoires | ★★★★★ | €€€€€ | ★☆☆☆☆ |

### **Avantages de la Méthode**
- ✅ Non-invasive
- ✅ Rapide (quelques minutes)
- ✅ Coût réduit
- ✅ Visualisation intuitive
- ✅ Basée sur données réelles

### **Limites**
- ⚠️ Résolution dépend de la qualité des données
- ⚠️ Nécessite validation terrain pour confirmation
- ⚠️ Profondeur d'investigation limitée

---

## 💾 Format des Résultats

### **Image Générée**
- **Format** : PNG haute résolution
- **Taille** : 512×512 ou 1024×1024 (selon modèle)
- **Utilisation** : Rapports, présentations, publications

### **Données Stockées**
```python
st.session_state['trajectories'] = [
    {
        'depth': int,           # Profondeur de la trajectoire
        'model': LinearRegression,  # Modèle ajusté
        'inliers': np.array,    # Masque des inliers
        'x_coords': np.array,   # Coordonnées X
        'y_coords': np.array,   # Coordonnées Y
        'score': float          # Score RANSAC
    },
    ...
]
```

---

## 🚀 Workflow Complet

```
Image géophysique
    ↓
Extraction spectrale RGB → Résistivité
    ↓
Imputation matricielle (combler trous)
    ↓
Modélisation forward (simulation physique)
    ↓
Reconstruction 3D (volume complet)
    ↓
Détection RANSAC (trajectoires linéaires) ← VOUS ÊTES ICI
    ↓
Génération IA (visualisation réaliste) ← NOUVELLE FONCTIONNALITÉ
    ↓
Coupe montrant cavités, failles, structures
    ↓
Recommandations d'exploration
```

---

## 📚 Références Scientifiques

### **RANSAC**
- Fischler & Bolles (1981) - "Random Sample Consensus"
- Applications en vision par ordinateur et géophysique

### **Détection de Neutrinos**
- IceCube Collaboration - Reconstruction de trajectoires
- Super-Kamiokande - Détection d'alignements dans bruit de fond

### **Tomographie Électrique**
- Loke & Barker (1996) - ERT inversion
- Binley & Kemna (2005) - DC resistivity methods

---

## 🎓 Formation et Support

### **Tutoriels Disponibles**
- Guide d'utilisation : `GUIDE_RAPIDE_LLM.md`
- Documentation LLM : `INTEGRATION_LLM_MISTRAL.md`
- Tests : `test_llm_integration.py`

### **Support Technique**
- Vérifier les logs d'erreur dans l'expander "Détails de l'erreur"
- Ajuster les paramètres RANSAC si peu de trajectoires détectées
- Essayer différents modèles IA pour meilleurs résultats

---

**Développé pour SETRAF - Subaquifère ERT Analysis Tool**  
**Visualisation Avancée des Structures Souterraines**  
**Décembre 2025**
