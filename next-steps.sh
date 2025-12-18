#!/bin/bash

###############################################################################
# SETRAF - Guide d'activation Docker WSL2 + Build
###############################################################################

clear

cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🐋 SETRAF - PROCHAINES ÉTAPES                            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

📋 STATUT ACTUEL
═══════════════════════════════════════════════════════════════

✅ Dockerfile créé
✅ Scripts d'automatisation prêts
✅ Documentation complète
❌ Docker pas encore accessible dans WSL2

═══════════════════════════════════════════════════════════════
🔧 ÉTAPE 1: ACTIVER DOCKER DANS WSL2
═══════════════════════════════════════════════════════════════

1️⃣  Ouvrir Docker Desktop (Windows)
    - Cliquer sur l'icône Docker 🐋 dans la barre des tâches
    - Attendre que Docker démarre (baleine bleue)

2️⃣  Aller dans les Settings
    - Cliquer sur l'icône ⚙️ (Settings) en haut à droite
    - Ou: Menu → Settings

3️⃣  Activer WSL Integration
    - Aller dans: Resources → WSL Integration
    - Cocher: ☑ Enable integration with my default WSL distro
    - Cocher votre distribution (Ubuntu / autre)
    - Cliquer: "Apply & Restart"

4️⃣  Attendre le redémarrage de Docker Desktop
    - Docker va redémarrer (~30 secondes)

5️⃣  Revenir dans ce terminal WSL et vérifier:

EOF

echo -e "\033[1;33m    docker --version\033[0m"
echo ""
echo "    Si vous voyez une version (ex: Docker version 24.x.x), ✅ c'est bon !"
echo ""

cat << 'EOF'
═══════════════════════════════════════════════════════════════
🚀 ÉTAPE 2: BUILDER L'IMAGE DOCKER
═══════════════════════════════════════════════════════════════

Une fois Docker activé, lancez:

EOF

echo -e "\033[1;32m    cd /home/belikan/KIbalione8/SETRAF\033[0m"
echo -e "\033[1;32m    ./docker-build.sh\033[0m"
echo ""

cat << 'EOF'
Cette commande va:
   ⏱️  Prendre 10-15 minutes
   📥 Télécharger Python 3.10-slim (~150 MB)
   📦 Installer toutes les dépendances SETRAF
   🏷️  Créer les tags 1.0.0 et latest
   📊 Taille finale: ~800 MB

═══════════════════════════════════════════════════════════════
🧪 ÉTAPE 3: TESTER L'IMAGE
═══════════════════════════════════════════════════════════════

Après le build réussi:

EOF

echo -e "\033[1;32m    ./docker-test.sh\033[0m"
echo ""

cat << 'EOF'
Cela va:
   ✓ Lancer un container de test
   ✓ Vérifier que Streamlit démarre correctement
   ✓ Ouvrir http://localhost:8504
   ✓ Afficher les logs en temps réel

═══════════════════════════════════════════════════════════════
📤 ÉTAPE 4: PUSHER VERS DOCKER HUB
═══════════════════════════════════════════════════════════════

Si le test fonctionne:

EOF

echo -e "\033[1;32m    docker login\033[0m"
echo "    (Entrer: username = belikanm, password = votre_mot_de_passe)"
echo ""
echo -e "\033[1;32m    ./docker-push.sh\033[0m"
echo ""

cat << 'EOF'
Cela va:
   ✓ Authentifier sur Docker Hub
   ✓ Pusher belikanm/kibaertanalyste:1.0.0
   ✓ Pusher belikanm/kibaertanalyste:latest
   ✓ Image disponible publiquement sur hub.docker.com

═══════════════════════════════════════════════════════════════
💡 VÉRIFICATION RAPIDE
═══════════════════════════════════════════════════════════════

Pour vérifier si Docker est déjà activé:

EOF

echo -e "\033[1;33m    docker ps\033[0m"
echo ""
echo "Si ça affiche un tableau (même vide), Docker fonctionne ! ✅"
echo "Si erreur 'command not found', activez WSL Integration ⚠️"
echo ""

cat << 'EOF'
═══════════════════════════════════════════════════════════════
🆘 EN CAS DE PROBLÈME
═══════════════════════════════════════════════════════════════

❌ Docker Desktop ne démarre pas
   → Redémarrer Windows
   → Vérifier que la virtualisation est activée (BIOS)

❌ WSL Integration grisée
   → Mettre à jour Docker Desktop
   → Mettre à jour WSL: wsl --update

❌ "Cannot connect to Docker daemon"
   → Docker Desktop n'est pas démarré
   → Attendre ~30s après le lancement

❌ Build échoue
   → Vérifier connexion internet
   → Nettoyer: docker system prune -a

═══════════════════════════════════════════════════════════════
📞 BESOIN D'AIDE ?
═══════════════════════════════════════════════════════════════

Documentation complète:
   📄 DOCKER_SETUP_GUIDE.txt
   📄 DOCKER_README.md
   📄 DOCKER_COMPLETE_SUMMARY.txt

Email: nyundumathryme@gmail.com

═══════════════════════════════════════════════════════════════

🎯 RÉSUMÉ DES COMMANDES

1. Vérifier Docker:
EOF

echo -e "   \033[1;33mdocker --version\033[0m"
echo ""
echo "2. Builder:"
echo -e "   \033[1;32mcd /home/belikan/KIbalione8/SETRAF\033[0m"
echo -e "   \033[1;32m./docker-build.sh\033[0m"
echo ""
echo "3. Tester:"
echo -e "   \033[1;32m./docker-test.sh\033[0m"
echo ""
echo "4. Pusher:"
echo -e "   \033[1;32mdocker login\033[0m"
echo -e "   \033[1;32m./docker-push.sh\033[0m"
echo ""

cat << 'EOF'
═══════════════════════════════════════════════════════════════
EOF

echo ""
echo -e "\033[1;36m💡 COMMENCEZ PAR VÉRIFIER SI DOCKER EST ACCESSIBLE:\033[0m"
echo ""
echo -e "\033[1;33m    docker --version\033[0m"
echo ""
echo -e "\033[0;90mSi erreur, suivez les étapes d'activation ci-dessus ⬆️\033[0m"
echo ""
