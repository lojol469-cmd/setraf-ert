# 🌊 SETRAF - Aide-Mémoire des Commandes

## 🚀 Démarrage/Arrêt

```bash
./start-setraf.sh              # Démarrer (méthode rapide)
./stop-setraf.sh               # Arrêter (méthode rapide)

./setraf-kernel.sh start       # Démarrer (méthode kernel)
./setraf-kernel.sh stop        # Arrêter (méthode kernel)
./setraf-kernel.sh restart     # Redémarrer
```

## 📊 Monitoring

```bash
./monitor-setraf.sh            # Dashboard complet (recommandé)
./setraf-kernel.sh monitor     # Dashboard via kernel
./setraf-kernel.sh status      # Statut simple
./setraf-kernel.sh stats       # Statistiques complètes
```

## 📝 Logs

```bash
# Logs en temps réel (tail -f)
./setraf-kernel.sh logs node        # Serveur Node.js
./setraf-kernel.sh logs streamlit   # Application Streamlit
./setraf-kernel.sh logs kernel      # Kernel système
./setraf-kernel.sh logs all         # Tous les logs

# Journal d'activité
./setraf-kernel.sh activity         # 50 dernières lignes
./setraf-kernel.sh activity 100     # 100 dernières lignes
./setraf-kernel.sh activity 200     # 200 dernières lignes
```

## 🔍 Diagnostic

```bash
# Vérifier les processus
ps aux | grep "node.exe\|streamlit"

# Vérifier les ports
netstat -an | grep -E ":(5000|8504)"

# Vérifier les PID
cat /tmp/setraf_node.pid
cat /tmp/setraf_streamlit.pid

# Voir les erreurs
./setraf-kernel.sh logs node | grep -i error
./setraf-kernel.sh logs streamlit | grep -i error
```

## 🌐 URLs d'Accès

```bash
# Application Streamlit
http://localhost:8504              # Localhost
http://172.20.31.35:8504          # IP WSL

# API Authentification
http://localhost:5000              # Localhost
http://192.168.1.66:5000          # IP WiFi (Windows)
http://172.20.31.35:5000          # IP WSL
```

## 🧪 Tests API

```bash
# Test de santé
curl http://192.168.1.66:5000/api/health

# Test d'inscription
curl -X POST http://192.168.1.66:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "Test1234",
    "fullName": "Test User"
  }'

# Envoyer OTP
curl -X POST http://192.168.1.66:5000/api/auth/send-otp \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'
```

## 🛠️ Maintenance

```bash
# Nettoyer les logs anciens
cd logs
ls -t kernel.log.* | tail -n +6 | xargs rm -f

# Redémarrer en cas de problème
./setraf-kernel.sh stop
pkill -f "node.exe\|streamlit"
rm /tmp/setraf_*.pid
./setraf-kernel.sh start

# Vérifier l'environnement Python
~/miniconda3/envs/gestmodo/bin/python --version
~/miniconda3/envs/gestmodo/bin/python -m pip list | grep streamlit
```

## 📈 Dashboard Monitoring

Le dashboard affiche en temps réel :

- **Services** : Status, PID, Uptime, CPU, Memory, Threads
- **Réseau** : Connexions actives par port
- **Logs** : 3 dernières requêtes/événements
- **Système** : Load Average, Memory, Disk
- **Stats** : Requêtes totales, Erreurs

Rafraîchissement automatique toutes les 3 secondes.

## 🔧 Variables d'Environnement

Fichier : `/home/belikan/KIbalione8/SETRAF/.env`

```env
# MongoDB
MONGO_URI=mongodb+srv://...

# JWT
JWT_SECRET=...
JWT_REFRESH_SECRET=...

# Email
EMAIL_USER=...
EMAIL_PASS=...

# Port
PORT=5000
```

## 📂 Structure des Fichiers

```
SETRAF/
├── setraf-kernel.sh           # Kernel principal
├── monitor-setraf.sh          # Dashboard monitoring
├── start-setraf.sh            # Démarrage rapide
├── stop-setraf.sh             # Arrêt rapide
├── ERTest.py                  # App Streamlit
├── auth_module.py             # Auth Python
├── node-auth/                 # Backend Node.js
└── logs/                      # Logs système
    ├── kernel.log
    ├── node-auth.log
    └── streamlit.log
```

## 🎯 Raccourcis Utiles

```bash
# Alias à ajouter dans ~/.bashrc
alias setraf-start='cd ~/KIbalione8/SETRAF && ./start-setraf.sh'
alias setraf-stop='cd ~/KIbalione8/SETRAF && ./stop-setraf.sh'
alias setraf-monitor='cd ~/KIbalione8/SETRAF && ./monitor-setraf.sh'
alias setraf-status='cd ~/KIbalione8/SETRAF && ./setraf-kernel.sh status'
alias setraf-logs='cd ~/KIbalione8/SETRAF && ./setraf-kernel.sh logs all'
```

Après ajout, recharger : `source ~/.bashrc`

## ⚡ Résolution Rapide de Problèmes

| Problème | Solution |
|----------|----------|
| Services ne démarrent pas | `./setraf-kernel.sh restart` |
| Port déjà utilisé | `pkill -f "node.exe\|streamlit"` puis redémarrer |
| MongoDB non connecté | Vérifier `MONGO_URI` dans `.env` |
| Erreur import Python | Vérifier environnement gestmodo |
| Logs trop volumineux | Ils sont archivés automatiquement |
| IP incorrecte | Le kernel détecte automatiquement au démarrage |

## 📞 Support

- **Logs** : `./setraf-kernel.sh logs all`
- **Stats** : `./setraf-kernel.sh stats`
- **Monitor** : `./monitor-setraf.sh`

---

**Version** : 1.0  
**Date** : 08 Novembre 2025  
**Auteur** : BelikanM
