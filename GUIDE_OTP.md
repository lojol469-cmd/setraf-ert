# 🔐 Guide d'utilisation de l'authentification OTP

## 📱 Qu'est-ce que l'OTP ?

L'OTP (One-Time Password) est un code à 6 chiffres envoyé par email qui expire après 10 minutes. C'est une méthode d'authentification plus sécurisée que le mot de passe classique.

## 🆕 Inscription avec OTP

### Option 1 : Email classique
1. Remplissez le formulaire d'inscription
2. Choisissez "📧 Email classique"
3. Cliquez sur "S'inscrire"
4. Vérifiez votre email pour activer votre compte

### Option 2 : Code OTP immédiat ⭐
1. Remplissez le formulaire d'inscription
2. Choisissez "🔐 Code OTP immédiat"
3. Cliquez sur "S'inscrire"
4. **Attendez quelques secondes** - Un code OTP vous sera envoyé
5. Entrez le code à 6 chiffres reçu par email dans le champ qui apparaît
6. Cliquez à nouveau sur "S'inscrire" pour valider
7. Votre compte est immédiatement activé ! ✅

## 🔑 Connexion avec OTP

### Option 1 : Connexion classique
1. Entrez votre email et mot de passe
2. Cliquez sur "Se connecter"

### Option 2 : Connexion OTP (plus sécurisé) ⭐
1. Cochez "🔐 Utiliser l'authentification OTP"
2. Entrez votre email
3. Cliquez sur "Envoyer le code OTP"
4. **Vérifiez votre boîte email** - Code à 6 chiffres
5. Entrez le code OTP dans le champ qui apparaît
6. Cliquez sur "✅ Vérifier et se connecter"
7. Vous êtes connecté ! ✅

## 📧 Exemple d'email OTP

Vous recevrez un email avec :
- **Sujet**: 🔐 Votre code OTP SETRAF-ERT
- **Code**: 6 chiffres en grand (ex: **123456**)
- **Validité**: 10 minutes
- **Design**: Dégradé violet/bleu professionnel

## 🔧 Mode Développement

En mode développement, le code OTP s'affiche également dans l'interface Streamlit pour faciliter les tests :
```
🔧 MODE DEV - Code OTP: 123456
```

## ⚠️ Sécurité

### ✅ Bonnes pratiques
- Ne partagez JAMAIS votre code OTP
- Utilisez le code dans les 10 minutes
- Demandez un nouveau code s'il est expiré
- Vérifiez que l'email provient bien de SETRAF-ERT

### ❌ Signes d'alerte
- Email demandant votre mot de passe (nous ne le demandons JAMAIS)
- Code OTP non sollicité (quelqu'un essaie peut-être d'accéder à votre compte)
- Email d'un expéditeur inconnu

## 🐛 Dépannage

### Le code n'arrive pas
1. **Vérifiez vos spams** - Regardez dans "Courrier indésirable"
2. **Attendez 2-3 minutes** - Les emails peuvent prendre du temps
3. **Vérifiez votre email** - Assurez-vous qu'il est correct
4. **Redemandez un code** - Cliquez sur "Annuler" puis refaites la demande

### Le code ne fonctionne pas
1. **Vérifiez les 6 chiffres** - Aucun espace, tous les chiffres
2. **Code expiré ?** - Valable 10 minutes seulement
3. **Redemandez un code** - Le nouveau remplacera l'ancien

### Erreur "Utilisateur non trouvé"
- Vérifiez que vous avez bien créé un compte avec cet email
- L'email est sensible à la casse : `Test@email.com` ≠ `test@email.com`

## 📊 Avantages de l'OTP

| Critère | Mot de passe | OTP |
|---------|-------------|-----|
| Sécurité | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Réutilisable | Oui (risque) | Non (usage unique) |
| Phishing | Vulnérable | Résistant |
| Validité | Permanente | 10 minutes |
| Vol de données | Risque élevé | Risque faible |

## 🔄 Processus technique

### Envoi de l'OTP
1. Backend génère un code aléatoire à 6 chiffres
2. Code stocké dans MongoDB avec timestamp d'expiration
3. Email envoyé avec nodemailer
4. Logs dans la console pour debug

### Vérification de l'OTP
1. Code comparé avec celui en base de données
2. Vérification de l'expiration (10 minutes)
3. Si valide : création de session + tokens JWT
4. Code supprimé de la base de données (usage unique)

## 📝 Logs de débogage

Les logs suivants apparaissent côté serveur :
```bash
🔐 OTP généré: 123456 pour user@example.com
✅ OTP sauvegardé dans la base de données
📧 Email OTP envoyé avec succès à: user@example.com
🔍 Vérification OTP pour: user@example.com
📝 OTP stocké: 123456 OTP reçu: 123456
✅ OTP valide, connexion de l'utilisateur
```

Pour voir les logs :
```bash
./setraf-kernel.sh logs node
```

## 🎯 Cas d'usage

### Première connexion
→ Utilisez **"Code OTP immédiat"** à l'inscription pour activer instantanément

### Connexion depuis un nouvel appareil
→ Utilisez **"Connexion OTP"** pour plus de sécurité

### Connexion habituelle
→ Utilisez **"Connexion classique"** avec mot de passe

### Mot de passe oublié
→ Utilisez **"Connexion OTP"** (pas besoin de mot de passe !)

## 💡 Astuces

1. **Inscription rapide** : Choisissez l'OTP immédiat pour sauter l'étape de vérification email
2. **Connexion sans mot de passe** : L'OTP permet de se connecter même si vous avez oublié votre mot de passe
3. **Sécurité maximale** : Utilisez toujours l'OTP depuis des réseaux publics
4. **Mode dev** : Le code s'affiche dans l'interface pour les tests

## 📞 Support

En cas de problème persistant :
1. Consultez les logs : `./setraf-kernel.sh logs all`
2. Vérifiez le statut : `./setraf-kernel.sh status`
3. Redémarrez : `./setraf-kernel.sh restart`

---

**Version** : 1.0  
**Date** : Novembre 2025  
**Plateforme** : SETRAF-ERT v2.0
