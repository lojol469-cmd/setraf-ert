#!/bin/bash

###############################################################################
# SETRAF Mini Kernel OS - Gestionnaire de services
# Lance et supervise le serveur Node.js (authentification) et Streamlit
###############################################################################

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NODE_AUTH_DIR="$SCRIPT_DIR/node-auth"
STREAMLIT_APP="$SCRIPT_DIR/ERTest.py"
NODE_EXEC="/mnt/c/Program Files/nodejs/node.exe"
CONDA_BASE="/home/belikan/miniconda3"
CONDA_ENV="gestmodo"  # Environnement avec toutes les dépendances installées

# Fichiers PID
NODE_PID_FILE="/tmp/setraf_node.pid"
STREAMLIT_PID_FILE="/tmp/setraf_streamlit.pid"

# Logs
LOG_DIR="$SCRIPT_DIR/logs"
NODE_LOG="$LOG_DIR/node-auth.log"
STREAMLIT_LOG="$LOG_DIR/streamlit.log"
KERNEL_LOG="$LOG_DIR/kernel.log"

###############################################################################
# Fonctions utilitaires
###############################################################################

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$KERNEL_LOG"
}

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║          🌊 SETRAF Mini Kernel OS v1.0                       ║"
    echo "║          Subaquifère ERT Analysis Platform                    ║"
    echo "║                                                               ║"
    echo "║          Services: Node.js Auth + Streamlit App              ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_dependencies() {
    log "INFO" "Vérification des dépendances..."
    
    # Vérifier Node.js
    if [ ! -f "$NODE_EXEC" ]; then
        log "ERROR" "Node.js non trouvé: $NODE_EXEC"
        echo -e "${RED}❌ Node.js non installé${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Node.js trouvé${NC}"
    
    # Vérifier Python/Conda
    if [ ! -d "$CONDA_BASE" ]; then
        log "ERROR" "Miniconda non trouvé: $CONDA_BASE"
        echo -e "${RED}❌ Miniconda non installé${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Miniconda trouvé${NC}"
    
    # Vérifier l'environnement gestmodo
    if [ ! -d "$CONDA_BASE/envs/$CONDA_ENV" ]; then
        log "ERROR" "Environnement conda '$CONDA_ENV' non trouvé"
        echo -e "${RED}❌ Environnement '$CONDA_ENV' non trouvé${NC}"
        echo -e "${YELLOW}Créez-le avec: conda create -n $CONDA_ENV python=3.10${NC}"
        echo -e "${YELLOW}Puis installez: conda activate $CONDA_ENV && pip install -r requirements.txt${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Environnement conda '$CONDA_ENV' trouvé${NC}"
    
    # Vérifier les fichiers
    if [ ! -f "$STREAMLIT_APP" ]; then
        log "ERROR" "Application Streamlit non trouvée: $STREAMLIT_APP"
        echo -e "${RED}❌ ERTest.py non trouvé${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Application Streamlit trouvée${NC}"
    
    if [ ! -d "$NODE_AUTH_DIR" ]; then
        log "ERROR" "Dossier Node.js Auth non trouvé: $NODE_AUTH_DIR"
        echo -e "${RED}❌ node-auth/ non trouvé${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Serveur d'authentification trouvé${NC}"
}

setup_environment() {
    log "INFO" "Configuration de l'environnement..."
    
    # Créer le dossier de logs
    mkdir -p "$LOG_DIR"
    echo -e "${GREEN}✓ Dossier de logs créé${NC}"
    
    # Nettoyer les anciens logs (garder les 5 derniers)
    cd "$LOG_DIR"
    ls -t kernel.log.* 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true
    
    # Archiver le log actuel s'il existe
    if [ -f "$KERNEL_LOG" ]; then
        mv "$KERNEL_LOG" "$KERNEL_LOG.$(date +%Y%m%d_%H%M%S)"
    fi
}

start_node_server() {
    log "INFO" "Démarrage du serveur Node.js (Authentification)..."
    echo -e "${YELLOW}🚀 Lancement du serveur d'authentification...${NC}"
    
    cd "$NODE_AUTH_DIR"
    
    # Démarrer Node.js en arrière-plan
    nohup "$NODE_EXEC" server.js > "$NODE_LOG" 2>&1 &
    local pid=$!
    echo $pid > "$NODE_PID_FILE"
    
    # Attendre que le serveur démarre
    sleep 3
    
    # Vérifier si le processus tourne
    if ps -p $pid > /dev/null 2>&1; then
        log "INFO" "Serveur Node.js démarré (PID: $pid)"
        echo -e "${GREEN}✓ Serveur Node.js démarré sur http://172.20.31.35:5000${NC}"
        return 0
    else
        log "ERROR" "Échec du démarrage du serveur Node.js"
        echo -e "${RED}❌ Échec du démarrage du serveur Node.js${NC}"
        cat "$NODE_LOG"
        return 1
    fi
}

start_streamlit_server() {
    log "INFO" "Démarrage des serveurs Streamlit..."
    echo -e "${YELLOW}🚀 Lancement des applications Streamlit...${NC}"
    
    cd "$SCRIPT_DIR"
    
    # Définir les chemins de l'environnement gestmodo
    local GESTMODO_PYTHON="$CONDA_BASE/envs/$CONDA_ENV/bin/python"
    local GESTMODO_STREAMLIT="$CONDA_BASE/envs/$CONDA_ENV/bin/streamlit"
    
    # Vérifier que Python existe dans gestmodo
    if [ ! -f "$GESTMODO_PYTHON" ]; then
        log "ERROR" "Python non trouvé dans l'environnement $CONDA_ENV: $GESTMODO_PYTHON"
        echo -e "${RED}❌ Python non trouvé dans gestmodo${NC}"
        return 1
    fi
    
    # Vérifier que streamlit est installé
    if ! $GESTMODO_PYTHON -m streamlit --version &>/dev/null; then
        log "WARN" "Streamlit non trouvé, installation..."
        echo -e "${YELLOW}⚠️  Installation de Streamlit dans gestmodo...${NC}"
        $GESTMODO_PYTHON -m pip install streamlit -q
    fi
    
    # Arrêter les instances Streamlit existantes
    pkill -9 -f "streamlit run" 2>/dev/null || true
    sleep 2
    
    # === DÉMARRER ERTest.py (port 8504) ===
    echo -e "${CYAN}🌊 Démarrage d'ERTest.py (port 8504)...${NC}"
    nohup $GESTMODO_PYTHON -m streamlit run "$STREAMLIT_APP" --server.port=8504 --server.address=0.0.0.0 > "$STREAMLIT_LOG" 2>&1 &
    local ertest_pid=$!
    echo $ertest_pid > "$STREAMLIT_PID_FILE"
    
    # Attendre que le serveur démarre
    sleep 5
    
    # Vérifier si le processus tourne
    if ps -p $ertest_pid > /dev/null 2>&1; then
        log "INFO" "ERTest.py démarré (PID: $ertest_pid)"
        echo -e "${GREEN}✓ ERTest.py démarré sur http://172.20.31.35:8504${NC}"
    else
        log "ERROR" "Échec du démarrage d'ERTest.py"
        echo -e "${RED}❌ Échec du démarrage d'ERTest.py${NC}"
        tail -20 "$STREAMLIT_LOG"
        return 1
    fi
    
    # === DÉMARRER ERT.py (Kibali avec ERTest intégré, port 8506) ===
    echo -e "${CYAN}🗺️ Démarrage d'ERT.py - Kibali Analyst (port 8506)...${NC}"
    local ERT_APP="$SCRIPT_DIR/ERT.py"
    local ERT_LOG="$LOG_DIR/ert-kibali.log"
    local ERT_PID_FILE="/tmp/setraf_ert.pid"
    
    nohup $GESTMODO_PYTHON -m streamlit run "$ERT_APP" --server.port=8506 --server.address=0.0.0.0 > "$ERT_LOG" 2>&1 &
    local ert_pid=$!
    echo $ert_pid > "$ERT_PID_FILE"
    
    # Attendre que le serveur démarre
    sleep 5
    
    # Vérifier si le processus tourne
    if ps -p $ert_pid > /dev/null 2>&1; then
        log "INFO" "ERT.py (Kibali) démarré (PID: $ert_pid)"
        echo -e "${GREEN}✓ ERT.py (Kibali) démarré sur http://172.20.31.35:8506${NC}"
        return 0
    else
        log "ERROR" "Échec du démarrage d'ERT.py"
        echo -e "${RED}❌ Échec du démarrage d'ERT.py${NC}"
        tail -20 "$ERT_LOG"
        # Continuer même si ERT échoue (ERTest fonctionne toujours)
        return 0
    fi
}

stop_services() {
    log "INFO" "Arrêt des services..."
    echo -e "${YELLOW}🛑 Arrêt des services SETRAF...${NC}"
    
    # Arrêter Node.js
    if [ -f "$NODE_PID_FILE" ]; then
        local node_pid=$(cat "$NODE_PID_FILE")
        if ps -p $node_pid > /dev/null 2>&1; then
            kill $node_pid 2>/dev/null || true
            log "INFO" "Serveur Node.js arrêté (PID: $node_pid)"
            echo -e "${GREEN}✓ Serveur Node.js arrêté${NC}"
        fi
        rm -f "$NODE_PID_FILE"
    fi
    
    # Arrêter ERTest.py (Streamlit port 8504)
    if [ -f "$STREAMLIT_PID_FILE" ]; then
        local streamlit_pid=$(cat "$STREAMLIT_PID_FILE")
        if ps -p $streamlit_pid > /dev/null 2>&1; then
            kill $streamlit_pid 2>/dev/null || true
            log "INFO" "ERTest.py arrêté (PID: $streamlit_pid)"
            echo -e "${GREEN}✓ ERTest.py arrêté${NC}"
        fi
        rm -f "$STREAMLIT_PID_FILE"
    fi
    
    # Arrêter ERT.py (Kibali port 8506)
    local ERT_PID_FILE="/tmp/setraf_ert.pid"
    if [ -f "$ERT_PID_FILE" ]; then
        local ert_pid=$(cat "$ERT_PID_FILE")
        if ps -p $ert_pid > /dev/null 2>&1; then
            kill $ert_pid 2>/dev/null || true
            log "INFO" "ERT.py (Kibali) arrêté (PID: $ert_pid)"
            echo -e "${GREEN}✓ ERT.py (Kibali) arrêté${NC}"
        fi
        rm -f "$ERT_PID_FILE"
    fi
    
    # Tuer tous les processus restants
    pkill -f "node.exe server.js" 2>/dev/null || true
    pkill -f "streamlit run ERTest.py" 2>/dev/null || true
    pkill -f "streamlit run ERT.py" 2>/dev/null || true
}

status_services() {
    echo -e "${CYAN}📊 Statut des services SETRAF${NC}"
    echo ""
    
    # Statut Node.js
    if [ -f "$NODE_PID_FILE" ]; then
        local node_pid=$(cat "$NODE_PID_FILE")
        if ps -p $node_pid > /dev/null 2>&1; then
            echo -e "${GREEN}● Node.js Auth Server${NC}"
            echo -e "  Status: ${GREEN}Running${NC} (PID: $node_pid)"
            echo -e "  URL: http://172.20.31.35:5000"
            echo -e "  Log: $NODE_LOG"
        else
            echo -e "${RED}● Node.js Auth Server${NC}"
            echo -e "  Status: ${RED}Stopped${NC}"
        fi
    else
        echo -e "${RED}● Node.js Auth Server${NC}"
        echo -e "  Status: ${RED}Not started${NC}"
    fi
    
    echo ""
    
    # Statut ERTest.py (port 8504)
    if [ -f "$STREAMLIT_PID_FILE" ]; then
        local streamlit_pid=$(cat "$STREAMLIT_PID_FILE")
        if ps -p $streamlit_pid > /dev/null 2>&1; then
            echo -e "${GREEN}● ERTest.py (Standalone)${NC}"
            echo -e "  Status: ${GREEN}Running${NC} (PID: $streamlit_pid)"
            echo -e "  URL: http://172.20.31.35:8504"
            echo -e "  Log: $STREAMLIT_LOG"
        else
            echo -e "${RED}● ERTest.py${NC}"
            echo -e "  Status: ${RED}Stopped${NC}"
        fi
    else
        echo -e "${RED}● ERTest.py${NC}"
        echo -e "  Status: ${RED}Not started${NC}"
    fi
    
    echo ""
    
    # Statut ERT.py (Kibali, port 8506)
    local ERT_PID_FILE="/tmp/setraf_ert.pid"
    local ERT_LOG="$LOG_DIR/ert-kibali.log"
    if [ -f "$ERT_PID_FILE" ]; then
        local ert_pid=$(cat "$ERT_PID_FILE")
        if ps -p $ert_pid > /dev/null 2>&1; then
            echo -e "${GREEN}● ERT.py (Kibali Analyst)${NC}"
            echo -e "  Status: ${GREEN}Running${NC} (PID: $ert_pid)"
            echo -e "  URL: http://172.20.31.35:8506"
            echo -e "  Log: $ERT_LOG"
        else
            echo -e "${RED}● ERT.py (Kibali)${NC}"
            echo -e "  Status: ${RED}Stopped${NC}"
        fi
    else
        echo -e "${RED}● ERT.py (Kibali)${NC}"
        echo -e "  Status: ${RED}Not started${NC}"
    fi
}

restart_services() {
    log "INFO" "Redémarrage des services..."
    stop_services
    sleep 2
    start_services
}

start_services() {
    print_banner
    check_dependencies
    setup_environment
    
    echo ""
    log "INFO" "Démarrage du système SETRAF..."
    
    # Détecter l'adresse IP automatiquement
    log "INFO" "Détection de l'adresse IP..."
    local LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "127.0.0.1")
    if [ -z "$LOCAL_IP" ] || [ "$LOCAL_IP" = "127.0.0.1" ]; then
        # Fallback pour WSL/Windows
        LOCAL_IP=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K\S+' || echo "172.20.31.35")
    fi
    echo -e "${GREEN}✓ Adresse IP détectée: $LOCAL_IP${NC}"
    log "INFO" "IP détectée: $LOCAL_IP"
    
    # Démarrer Node.js
    if ! start_node_server; then
        log "ERROR" "Impossible de démarrer le serveur Node.js"
        exit 1
    fi
    
    echo ""
    
    # Démarrer Streamlit
    if ! start_streamlit_server; then
        log "ERROR" "Impossible de démarrer Streamlit"
        stop_services
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║  ✅ Système SETRAF démarré avec succès !                     ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║  🔐 Authentification: http://$LOCAL_IP:5000              ║${NC}"
    echo -e "${GREEN}║  🌊 ERTest (standalone): http://$LOCAL_IP:8504           ║${NC}"
    echo -e "${GREEN}║  🗺️ ERT Kibali (complet): http://$LOCAL_IP:8506         ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║  📝 Logs: $LOG_DIR                        ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}💡 Accès depuis le réseau local:${NC}"
    echo -e "   - Auth: http://$LOCAL_IP:5000"
    echo -e "   - ERTest: http://$LOCAL_IP:8504"
    echo -e "   - ERT Kibali (avec ERTest intégré): http://$LOCAL_IP:8506"
    echo -e "   - Localhost: http://localhost:8504 et http://localhost:8506"
    echo ""
    log "INFO" "Système SETRAF opérationnel sur $LOCAL_IP"
}

show_logs() {
    local service=$1
    case $service in
        node|auth)
            echo -e "${CYAN}📄 Logs Node.js Auth Server (Temps réel):${NC}"
            echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter${NC}"
            echo ""
            tail -f "$NODE_LOG"
            ;;
        streamlit|app)
            echo -e "${CYAN}📄 Logs Streamlit App (Temps réel):${NC}"
            echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter${NC}"
            echo ""
            tail -f "$STREAMLIT_LOG"
            ;;
        kernel|system)
            echo -e "${CYAN}📄 Logs Kernel (Temps réel):${NC}"
            echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter${NC}"
            echo ""
            tail -f "$KERNEL_LOG"
            ;;
        all)
            echo -e "${CYAN}📄 Logs de tous les services (Temps réel):${NC}"
            echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter${NC}"
            echo ""
            tail -f "$NODE_LOG" "$STREAMLIT_LOG" "$KERNEL_LOG"
            ;;
        *)
            echo -e "${RED}Service inconnu. Utilisez: node, streamlit, kernel, ou all${NC}"
            ;;
    esac
}

monitor_services() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          📊 SETRAF - Monitoring en Temps Réel               ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter le monitoring${NC}"
    echo ""
    
    while true; do
        clear
        echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║          📊 SETRAF - Monitoring en Temps Réel               ║${NC}"
        echo -e "${CYAN}║          $(date '+%Y-%m-%d %H:%M:%S')                                  ║${NC}"
        echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        # Statut des services
        echo -e "${MAGENTA}═══ SERVICES ═══${NC}"
        echo ""
        
        # Node.js
        if [ -f "$NODE_PID_FILE" ]; then
            local node_pid=$(cat "$NODE_PID_FILE")
            if ps -p $node_pid > /dev/null 2>&1; then
                local node_mem=$(ps -p $node_pid -o rss= 2>/dev/null | awk '{printf "%.1f MB", $1/1024}')
                local node_cpu=$(ps -p $node_pid -o %cpu= 2>/dev/null | xargs)
                local node_time=$(ps -p $node_pid -o etime= 2>/dev/null | xargs)
                echo -e "${GREEN}● Node.js Auth Server${NC}"
                echo -e "  PID:     ${node_pid}"
                echo -e "  Status:  ${GREEN}Running${NC}"
                echo -e "  Uptime:  ${node_time}"
                echo -e "  CPU:     ${node_cpu}%"
                echo -e "  Memory:  ${node_mem}"
                echo -e "  Port:    5000"
                
                # Dernière activité
                local last_request=$(tail -1 "$NODE_LOG" 2>/dev/null | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' | tail -1)
                if [ -n "$last_request" ]; then
                    echo -e "  Last:    ${last_request}"
                fi
            else
                echo -e "${RED}● Node.js Auth Server${NC}"
                echo -e "  Status:  ${RED}Stopped${NC}"
            fi
        else
            echo -e "${RED}● Node.js Auth Server${NC}"
            echo -e "  Status:  ${RED}Not started${NC}"
        fi
        
        echo ""
        
        # Streamlit
        if [ -f "$STREAMLIT_PID_FILE" ]; then
            local streamlit_pid=$(cat "$STREAMLIT_PID_FILE")
            if ps -p $streamlit_pid > /dev/null 2>&1; then
                local streamlit_mem=$(ps -p $streamlit_pid -o rss= 2>/dev/null | awk '{printf "%.1f MB", $1/1024}')
                local streamlit_cpu=$(ps -p $streamlit_pid -o %cpu= 2>/dev/null | xargs)
                local streamlit_time=$(ps -p $streamlit_pid -o etime= 2>/dev/null | xargs)
                echo -e "${GREEN}● Streamlit App${NC}"
                echo -e "  PID:     ${streamlit_pid}"
                echo -e "  Status:  ${GREEN}Running${NC}"
                echo -e "  Uptime:  ${streamlit_time}"
                echo -e "  CPU:     ${streamlit_cpu}%"
                echo -e "  Memory:  ${streamlit_mem}"
                echo -e "  Port:    8504"
            else
                echo -e "${RED}● Streamlit App${NC}"
                echo -e "  Status:  ${RED}Stopped${NC}"
            fi
        else
            echo -e "${RED}● Streamlit App${NC}"
            echo -e "  Status:  ${RED}Not started${NC}"
        fi
        
        echo ""
        echo -e "${MAGENTA}═══ ACTIVITÉ RÉCENTE ═══${NC}"
        echo ""
        
        # Dernières lignes des logs Node.js
        echo -e "${CYAN}🔐 Node.js (dernières 3 requêtes):${NC}"
        tail -3 "$NODE_LOG" 2>/dev/null | grep -E "GET|POST|PUT|DELETE" | tail -3 | sed 's/^/  /' || echo -e "  ${YELLOW}Aucune activité récente${NC}"
        echo ""
        
        # Dernières lignes des logs Streamlit
        echo -e "${CYAN}💧 Streamlit (derniers événements):${NC}"
        tail -5 "$STREAMLIT_LOG" 2>/dev/null | grep -v "^$" | tail -3 | sed 's/^/  /' || echo -e "  ${YELLOW}Aucune activité récente${NC}"
        echo ""
        
        # Statistiques système
        echo -e "${MAGENTA}═══ SYSTÈME ═══${NC}"
        echo ""
        
        # Charge système
        local load_avg=$(uptime | grep -oP 'load average: \K.*')
        echo -e "${CYAN}Load Average:${NC} ${load_avg}"
        
        # Mémoire
        local mem_info=$(free -h | grep "Mem:" | awk '{printf "Used: %s / Total: %s (%.0f%%)", $3, $2, ($3/$2)*100}')
        echo -e "${CYAN}Memory:${NC} ${mem_info}"
        
        # Disque
        local disk_info=$(df -h "$SCRIPT_DIR" | tail -1 | awk '{printf "Used: %s / Total: %s (%s)", $3, $2, $5}')
        echo -e "${CYAN}Disk:${NC} ${disk_info}"
        
        # Connexions réseau
        local connections=$(netstat -an 2>/dev/null | grep -E ":(5000|8504)" | grep ESTABLISHED | wc -l)
        echo -e "${CYAN}Active Connections:${NC} ${connections}"
        
        echo ""
        echo -e "${YELLOW}Rafraîchissement dans 5 secondes... (Ctrl+C pour quitter)${NC}"
        
        sleep 5
    done
}

activity_log() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          📈 SETRAF - Journal d'Activité                     ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    local lines=${1:-50}
    
    echo -e "${MAGENTA}═══ Activité Node.js (${lines} dernières) ═══${NC}"
    echo ""
    tail -${lines} "$NODE_LOG" 2>/dev/null | grep -E "POST|GET|PUT|DELETE|Connecté|Erreur" | nl
    
    echo ""
    echo -e "${MAGENTA}═══ Activité Streamlit (${lines} dernières) ═══${NC}"
    echo ""
    tail -${lines} "$STREAMLIT_LOG" 2>/dev/null | grep -v "^$" | tail -20 | nl
    
    echo ""
    echo -e "${MAGENTA}═══ Événements Kernel (${lines} derniers) ═══${NC}"
    echo ""
    tail -${lines} "$KERNEL_LOG" 2>/dev/null | grep -E "INFO|ERROR|WARN" | tail -20 | nl
}

stats_summary() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          📊 SETRAF - Statistiques                            ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Uptime des services
    if [ -f "$NODE_PID_FILE" ] && ps -p $(cat "$NODE_PID_FILE") > /dev/null 2>&1; then
        local node_uptime=$(ps -p $(cat "$NODE_PID_FILE") -o etime= | xargs)
        echo -e "${GREEN}Node.js Uptime:${NC} ${node_uptime}"
    fi
    
    if [ -f "$STREAMLIT_PID_FILE" ] && ps -p $(cat "$STREAMLIT_PID_FILE") > /dev/null 2>&1; then
        local streamlit_uptime=$(ps -p $(cat "$STREAMLIT_PID_FILE") -o etime= | xargs)
        echo -e "${GREEN}Streamlit Uptime:${NC} ${streamlit_uptime}"
    fi
    
    echo ""
    
    # Statistiques des logs
    echo -e "${MAGENTA}═══ Statistiques des Logs ═══${NC}"
    echo ""
    
    local node_lines=$(wc -l < "$NODE_LOG" 2>/dev/null || echo "0")
    local streamlit_lines=$(wc -l < "$STREAMLIT_LOG" 2>/dev/null || echo "0")
    local kernel_lines=$(wc -l < "$KERNEL_LOG" 2>/dev/null || echo "0")
    
    echo -e "${CYAN}Node.js logs:${NC} ${node_lines} lignes"
    echo -e "${CYAN}Streamlit logs:${NC} ${streamlit_lines} lignes"
    echo -e "${CYAN}Kernel logs:${NC} ${kernel_lines} lignes"
    
    echo ""
    
    # Requêtes API (Node.js)
    local total_requests=$(grep -c -E "GET|POST|PUT|DELETE" "$NODE_LOG" 2>/dev/null || echo "0")
    local get_requests=$(grep -c "GET" "$NODE_LOG" 2>/dev/null || echo "0")
    local post_requests=$(grep -c "POST" "$NODE_LOG" 2>/dev/null || echo "0")
    
    echo -e "${MAGENTA}═══ Requêtes API ═══${NC}"
    echo ""
    echo -e "${CYAN}Total:${NC} ${total_requests}"
    echo -e "${CYAN}GET:${NC} ${get_requests}"
    echo -e "${CYAN}POST:${NC} ${post_requests}"
    
    echo ""
    
    # Erreurs
    local node_errors=$(grep -c "ERROR\|Erreur" "$NODE_LOG" 2>/dev/null || echo "0")
    local streamlit_errors=$(grep -c "error\|Error\|ERROR" "$STREAMLIT_LOG" 2>/dev/null || echo "0")
    
    echo -e "${MAGENTA}═══ Erreurs ═══${NC}"
    echo ""
    if [ "$node_errors" -gt 0 ] || [ "$streamlit_errors" -gt 0 ]; then
        echo -e "${YELLOW}Node.js:${NC} ${node_errors} erreur(s)"
        echo -e "${YELLOW}Streamlit:${NC} ${streamlit_errors} erreur(s)"
    else
        echo -e "${GREEN}Aucune erreur détectée${NC}"
    fi
    
    echo ""
    
    # Taille des logs
    echo -e "${MAGENTA}═══ Taille des Logs ═══${NC}"
    echo ""
    
    local node_size=$(du -h "$NODE_LOG" 2>/dev/null | cut -f1)
    local streamlit_size=$(du -h "$STREAMLIT_LOG" 2>/dev/null | cut -f1)
    local kernel_size=$(du -h "$KERNEL_LOG" 2>/dev/null | cut -f1)
    
    echo -e "${CYAN}Node.js:${NC} ${node_size}"
    echo -e "${CYAN}Streamlit:${NC} ${streamlit_size}"
    echo -e "${CYAN}Kernel:${NC} ${kernel_size}"
}

###############################################################################
# Menu principal
###############################################################################

case "${1:-start}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        status_services
        ;;
    logs)
        show_logs "${2:-kernel}"
        ;;
    monitor|watch)
        monitor_services
        ;;
    activity)
        activity_log "${2:-50}"
        ;;
    stats)
        stats_summary
        ;;
    *)
        echo -e "${CYAN}Usage: $0 {start|stop|restart|status|logs|monitor|activity|stats}${NC}"
        echo ""
        echo -e "${YELLOW}Commandes disponibles:${NC}"
        echo -e "  ${GREEN}start${NC}              - Démarrer les services"
        echo -e "  ${GREEN}stop${NC}               - Arrêter les services"
        echo -e "  ${GREEN}restart${NC}            - Redémarrer les services"
        echo -e "  ${GREEN}status${NC}             - Voir le statut des services"
        echo -e "  ${GREEN}logs [service]${NC}     - Voir les logs (node|streamlit|kernel|all)"
        echo -e "  ${GREEN}monitor${NC}            - Monitoring en temps réel"
        echo -e "  ${GREEN}activity [n]${NC}       - Journal d'activité (n dernières lignes)"
        echo -e "  ${GREEN}stats${NC}              - Statistiques complètes"
        echo ""
        echo -e "${CYAN}Exemples:${NC}"
        echo -e "  $0 start"
        echo -e "  $0 logs node"
        echo -e "  $0 logs all"
        echo -e "  $0 monitor"
        echo -e "  $0 activity 100"
        exit 1
        ;;
esac
