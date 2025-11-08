# 📚 Index de la documentation SETRAF

**Version** : 2.0  
**Date** : 8 novembre 2025  
**Système** : SETRAF-ERT (Subaquifère ERT Analysis Platform)

---

## 📖 Documentation principale

### 🎯 Pour démarrer
| Document | Description | Taille |
|----------|-------------|--------|
| [README.md](README.md) | Vue d'ensemble du projet, installation, utilisation | 7.4K |
| [KERNEL-README.md](KERNEL-README.md) | Guide du Mini Kernel OS, commandes, monitoring | 6.7K |
| [SYSTEM-STATUS.md](SYSTEM-STATUS.md) | Architecture, statut des services, configuration | 7.2K |

### 🔐 Authentification OTP
| Document | Description | Taille |
|----------|-------------|--------|
| [GUIDE_OTP.md](GUIDE_OTP.md) | Guide utilisateur complet de l'OTP | 5.1K |
| [WORKFLOWS_OTP.md](WORKFLOWS_OTP.md) | Workflows visuels et diagrammes | 19K |
| [CORRECTIONS_OTP.md](CORRECTIONS_OTP.md) | Documentation technique des corrections | 9.7K |
| [RESUME_CORRECTIONS_OTP.md](RESUME_CORRECTIONS_OTP.md) | Résumé rapide des modifications | 3.8K |

### ⚙️ Commandes et scripts
| Document | Description | Taille |
|----------|-------------|--------|
| [COMMANDES.md](COMMANDES.md) | Cheat sheet des commandes du kernel | 4.9K |
| [test_otp.sh](test_otp.sh) | Script de test automatique de l'API OTP | Exécutable |

---

## 🗂️ Structure de la documentation

```
SETRAF/
├── 📘 README.md                    → Vue d'ensemble
├── 🔧 KERNEL-README.md             → Guide du kernel OS
├── 📊 SYSTEM-STATUS.md             → Architecture et statut
│
├── 🔐 OTP Authentication
│   ├── GUIDE_OTP.md                → Guide utilisateur
│   ├── WORKFLOWS_OTP.md            → Diagrammes visuels
│   ├── CORRECTIONS_OTP.md          → Documentation technique
│   └── RESUME_CORRECTIONS_OTP.md   → Résumé rapide
│
├── ⚙️ Commandes et tests
│   ├── COMMANDES.md                → Cheat sheet
│   └── test_otp.sh                 → Tests automatiques
│
└── 📚 INDEX.md                     → Ce fichier
```

---

## 🎓 Guides par profil utilisateur

### 👨‍💼 Chef de projet / Manager
**Je veux comprendre le système rapidement**
1. [README.md](README.md) - Vue d'ensemble (5 min)
2. [SYSTEM-STATUS.md](SYSTEM-STATUS.md) - Architecture (5 min)
3. [RESUME_CORRECTIONS_OTP.md](RESUME_CORRECTIONS_OTP.md) - Dernières corrections (2 min)

**Total** : ~12 minutes

### 👨‍💻 Développeur / Mainteneur
**Je veux comprendre le code et l'architecture**
1. [SYSTEM-STATUS.md](SYSTEM-STATUS.md) - Architecture complète
2. [CORRECTIONS_OTP.md](CORRECTIONS_OTP.md) - Détails techniques OTP
3. [KERNEL-README.md](KERNEL-README.md) - Fonctionnement du kernel
4. [COMMANDES.md](COMMANDES.md) - Toutes les commandes

**Total** : ~30 minutes

### 👨‍🔬 Scientifique / Utilisateur final
**Je veux utiliser l'application**
1. [GUIDE_OTP.md](GUIDE_OTP.md) - Authentification OTP
2. [README.md](README.md) - Section "Utilisation"
3. [WORKFLOWS_OTP.md](WORKFLOWS_OTP.md) - Workflows visuels (si besoin)

**Total** : ~15 minutes

### 🔧 DevOps / Administrateur système
**Je veux déployer et monitorer**
1. [KERNEL-README.md](KERNEL-README.md) - Installation et lancement
2. [COMMANDES.md](COMMANDES.md) - Commandes de monitoring
3. [SYSTEM-STATUS.md](SYSTEM-STATUS.md) - Configuration des services

**Total** : ~20 minutes

### 🧪 Testeur / QA
**Je veux tester le système**
1. [GUIDE_OTP.md](GUIDE_OTP.md) - Fonctionnalités OTP
2. [test_otp.sh](test_otp.sh) - Scripts de test
3. [WORKFLOWS_OTP.md](WORKFLOWS_OTP.md) - Scénarios de test
4. [COMMANDES.md](COMMANDES.md) - Commandes de diagnostic

**Total** : ~25 minutes

---

## 🔍 Index thématique

### Authentification et sécurité
- **Guide utilisateur** : [GUIDE_OTP.md](GUIDE_OTP.md)
- **Workflows** : [WORKFLOWS_OTP.md](WORKFLOWS_OTP.md)
- **Implémentation** : [CORRECTIONS_OTP.md](CORRECTIONS_OTP.md)
- **Tests** : [test_otp.sh](test_otp.sh)

### Installation et démarrage
- **Installation** : [README.md](README.md#installation)
- **Lancement** : [KERNEL-README.md](KERNEL-README.md#démarrage-rapide)
- **Configuration** : [SYSTEM-STATUS.md](SYSTEM-STATUS.md#configuration)

### Monitoring et logs
- **Commandes** : [COMMANDES.md](COMMANDES.md#monitoring)
- **Dashboard** : [KERNEL-README.md](KERNEL-README.md#monitoring)
- **Logs** : [KERNEL-README.md](KERNEL-README.md#logs)

### Architecture technique
- **Vue d'ensemble** : [SYSTEM-STATUS.md](SYSTEM-STATUS.md#architecture)
- **Services** : [SYSTEM-STATUS.md](SYSTEM-STATUS.md#services)
- **Base de données** : [SYSTEM-STATUS.md](SYSTEM-STATUS.md#mongodb)

### Dépannage
- **OTP** : [GUIDE_OTP.md](GUIDE_OTP.md#dépannage)
- **Services** : [COMMANDES.md](COMMANDES.md#diagnostic)
- **Logs** : [KERNEL-README.md](KERNEL-README.md#logs)

---

## 📊 Statistiques de la documentation

| Type | Nombre | Taille totale |
|------|--------|--------------|
| 📘 Guides principaux | 3 | 21.3K |
| 🔐 Documentation OTP | 4 | 37.6K |
| ⚙️ Scripts et commandes | 2 | ~5K |
| **Total** | **9** | **~64K** |

---

## 🎯 Parcours recommandés

### 🚀 Démarrage rapide (10 min)
```
1. README.md (section "Installation")
   ↓
2. KERNEL-README.md (section "Démarrage rapide")
   ↓
3. GUIDE_OTP.md (section "Connexion avec OTP")
   ↓
✅ Prêt à utiliser !
```

### 🏗️ Développement complet (1h)
```
1. README.md
   ↓
2. SYSTEM-STATUS.md
   ↓
3. CORRECTIONS_OTP.md
   ↓
4. KERNEL-README.md
   ↓
5. COMMANDES.md
   ↓
✅ Maîtrise complète !
```

### 🔧 Administration système (30 min)
```
1. KERNEL-README.md
   ↓
2. COMMANDES.md
   ↓
3. SYSTEM-STATUS.md (section "Services")
   ↓
✅ Prêt à administrer !
```

---

## 🔗 Liens utiles

### Serveurs
- **Application** : http://172.20.31.35:8504
- **API Auth** : http://172.20.31.35:5000
- **Localhost** : http://localhost:8504

### Commandes rapides
```bash
# Statut
./setraf-kernel.sh status

# Monitoring
./setraf-kernel.sh monitor

# Logs
./setraf-kernel.sh logs all

# Test OTP
./test_otp.sh votre.email@example.com
```

### Fichiers de configuration
- `.env` - Variables d'environnement
- `node-auth/server.js` - Serveur d'authentification
- `ERTest.py` - Application Streamlit
- `auth_module.py` - Module d'authentification

---

## 📝 Conventions de la documentation

### Icônes utilisées
- 📘 Documentation générale
- 🔐 Sécurité et authentification
- ⚙️ Configuration et scripts
- 📊 Architecture et diagrammes
- 🧪 Tests et validation
- 🚀 Démarrage rapide
- 💡 Astuces et conseils
- ⚠️ Avertissements
- ✅ Validé / Opérationnel
- ❌ Erreur / Non fonctionnel
- 🔧 En développement

### Format des exemples de code
```bash
# Commandes shell
./setraf-kernel.sh start
```

```python
# Code Python
auth = AuthManager()
```

```javascript
// Code JavaScript
const otpCode = generateOTP();
```

---

## 🆕 Dernières mises à jour

### 8 novembre 2025
- ✅ Correction complète du système OTP
- ✅ Intégration des inputs OTP dans les formulaires
- ✅ Mode développement avec affichage du code
- ✅ Documentation complète (4 nouveaux fichiers)
- ✅ Script de test automatique
- ✅ Logs de débogage détaillés

### Prochaines versions
- ⏳ Intégration PyGIMLi pour inversions ERT
- ⏳ Rate limiting sur l'API OTP
- ⏳ Authentification 2FA complète
- ⏳ Interface d'administration

---

## 📞 Support et contribution

### Pour signaler un bug
1. Consulter [GUIDE_OTP.md](GUIDE_OTP.md#dépannage)
2. Vérifier les logs : `./setraf-kernel.sh logs all`
3. Tester l'API : `./test_otp.sh`

### Pour contribuer
1. Lire l'architecture : [SYSTEM-STATUS.md](SYSTEM-STATUS.md)
2. Comprendre les workflows : [WORKFLOWS_OTP.md](WORKFLOWS_OTP.md)
3. Suivre les conventions de code

---

**Maintenu par** : Équipe SETRAF  
**Licence** : AGPL v3  
**Version documentation** : 1.0
