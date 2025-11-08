#!/bin/bash

###############################################################################
# SETRAF Monitor - Monitoring en temps réel avancé
###############################################################################

cd "$(dirname "$0")"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
LOG_DIR="./logs"
NODE_LOG="$LOG_DIR/node-auth.log"
STREAMLIT_LOG="$LOG_DIR/streamlit.log"
NODE_PID_FILE="/tmp/setraf_node.pid"
STREAMLIT_PID_FILE="/tmp/setraf_streamlit.pid"

# Fonction d'affichage du header
print_header() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          📊 SETRAF - Monitoring Dashboard                     ║${NC}"
    echo -e "${CYAN}║          ${timestamp}                                ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
}

# Dashboard principal
dashboard() {
    while true; do
        clear
        print_header
        echo ""
        
        # Section Services
        echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${MAGENTA}║ SERVICES                                                       ║${NC}"
        echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        # Node.js Status
        if [ -f "$NODE_PID_FILE" ]; then
            local node_pid=$(cat "$NODE_PID_FILE")
            if ps -p $node_pid > /dev/null 2>&1; then
                local node_mem=$(ps -p $node_pid -o rss= 2>/dev/null | awk '{printf "%.1f", $1/1024}')
                local node_cpu=$(ps -p $node_pid -o %cpu= 2>/dev/null | xargs)
                local node_time=$(ps -p $node_pid -o etime= 2>/dev/null | xargs)
                local node_threads=$(ps -p $node_pid -o nlwp= 2>/dev/null | xargs)
                
                echo -e "  ${GREEN}●${NC} ${CYAN}Node.js Auth Server${NC}"
                echo -e "     Status:  ${GREEN}Running${NC}    PID: ${node_pid}    Uptime: ${node_time}"
                echo -e "     CPU:     ${node_cpu}%    Memory: ${node_mem} MB    Threads: ${node_threads}"
                echo -e "     Port:    ${BLUE}5000${NC}    URL: ${BLUE}http://192.168.1.66:5000${NC}"
            else
                echo -e "  ${RED}●${NC} ${CYAN}Node.js Auth Server${NC} - ${RED}Stopped${NC}"
            fi
        else
            echo -e "  ${YELLOW}○${NC} ${CYAN}Node.js Auth Server${NC} - ${YELLOW}Not Started${NC}"
        fi
        
        echo ""
        
        # Streamlit Status
        if [ -f "$STREAMLIT_PID_FILE" ]; then
            local streamlit_pid=$(cat "$STREAMLIT_PID_FILE")
            if ps -p $streamlit_pid > /dev/null 2>&1; then
                local streamlit_mem=$(ps -p $streamlit_pid -o rss= 2>/dev/null | awk '{printf "%.1f", $1/1024}')
                local streamlit_cpu=$(ps -p $streamlit_pid -o %cpu= 2>/dev/null | xargs)
                local streamlit_time=$(ps -p $streamlit_pid -o etime= 2>/dev/null | xargs)
                local streamlit_threads=$(ps -p $streamlit_pid -o nlwp= 2>/dev/null | xargs)
                
                echo -e "  ${GREEN}●${NC} ${CYAN}Streamlit Application${NC}"
                echo -e "     Status:  ${GREEN}Running${NC}    PID: ${streamlit_pid}    Uptime: ${streamlit_time}"
                echo -e "     CPU:     ${streamlit_cpu}%    Memory: ${streamlit_mem} MB    Threads: ${streamlit_threads}"
                echo -e "     Port:    ${BLUE}8504${NC}    URL: ${BLUE}http://localhost:8504${NC}"
            else
                echo -e "  ${RED}●${NC} ${CYAN}Streamlit Application${NC} - ${RED}Stopped${NC}"
            fi
        else
            echo -e "  ${YELLOW}○${NC} ${CYAN}Streamlit Application${NC} - ${YELLOW}Not Started${NC}"
        fi
        
        echo ""
        
        # Section Activité Réseau
        echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${MAGENTA}║ ACTIVITÉ RÉSEAU                                                ║${NC}"
        echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        local conn_5000=$(netstat -an 2>/dev/null | grep ":5000" | grep ESTABLISHED | wc -l)
        local conn_8504=$(netstat -an 2>/dev/null | grep ":8504" | grep ESTABLISHED | wc -l)
        local total_conn=$((conn_5000 + conn_8504))
        
        echo -e "  ${CYAN}Connexions actives:${NC}"
        echo -e "     Port 5000 (Auth):      ${GREEN}${conn_5000}${NC} connexion(s)"
        echo -e "     Port 8504 (App):       ${GREEN}${conn_8504}${NC} connexion(s)"
        echo -e "     ${CYAN}Total:${NC}                 ${GREEN}${total_conn}${NC} connexion(s)"
        
        echo ""
        
        # Section Logs Récents
        echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${MAGENTA}║ ACTIVITÉ RÉCENTE                                               ║${NC}"
        echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        echo -e "  ${CYAN}🔐 Node.js (3 dernières requêtes):${NC}"
        tail -100 "$NODE_LOG" 2>/dev/null | grep -E "GET|POST|PUT|DELETE" | tail -3 | sed 's/^/     /' | cut -c 1-70 || echo -e "     ${YELLOW}Aucune activité${NC}"
        
        echo ""
        echo -e "  ${CYAN}💧 Streamlit (derniers événements):${NC}"
        tail -50 "$STREAMLIT_LOG" 2>/dev/null | grep -v "^$" | grep -v "Duplicate" | tail -3 | sed 's/^/     /' | cut -c 1-70 || echo -e "     ${YELLOW}Aucune activité${NC}"
        
        echo ""
        
        # Section Ressources Système
        echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${MAGENTA}║ RESSOURCES SYSTÈME                                             ║${NC}"
        echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        # Load Average
        local load=$(uptime | grep -oP 'load average: \K[^,]+')
        echo -e "  ${CYAN}Load Average (1 min):${NC}  ${load}"
        
        # Mémoire
        local mem_used=$(free -h | grep "Mem:" | awk '{print $3}')
        local mem_total=$(free -h | grep "Mem:" | awk '{print $2}')
        local mem_percent=$(free | grep "Mem:" | awk '{printf "%.0f", ($3/$2)*100}')
        
        if [ "$mem_percent" -gt 80 ]; then
            echo -e "  ${CYAN}Memory:${NC}               ${RED}${mem_used}${NC} / ${mem_total} (${RED}${mem_percent}%${NC})"
        elif [ "$mem_percent" -gt 60 ]; then
            echo -e "  ${CYAN}Memory:${NC}               ${YELLOW}${mem_used}${NC} / ${mem_total} (${YELLOW}${mem_percent}%${NC})"
        else
            echo -e "  ${CYAN}Memory:${NC}               ${GREEN}${mem_used}${NC} / ${mem_total} (${GREEN}${mem_percent}%${NC})"
        fi
        
        # Disque
        local disk_used=$(df -h . | tail -1 | awk '{print $3}')
        local disk_total=$(df -h . | tail -1 | awk '{print $2}')
        local disk_percent=$(df -h . | tail -1 | awk '{print $5}' | tr -d '%')
        
        if [ "$disk_percent" -gt 80 ]; then
            echo -e "  ${CYAN}Disk:${NC}                 ${RED}${disk_used}${NC} / ${disk_total} (${RED}${disk_percent}%${NC})"
        elif [ "$disk_percent" -gt 60 ]; then
            echo -e "  ${CYAN}Disk:${NC}                 ${YELLOW}${disk_used}${NC} / ${disk_total} (${YELLOW}${disk_percent}%${NC})"
        else
            echo -e "  ${CYAN}Disk:${NC}                 ${GREEN}${disk_used}${NC} / ${disk_total} (${GREEN}${disk_percent}%${NC})"
        fi
        
        echo ""
        
        # Section Statistiques
        echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${MAGENTA}║ STATISTIQUES                                                   ║${NC}"
        echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        local total_requests=$(grep -c -E "GET|POST|PUT|DELETE" "$NODE_LOG" 2>/dev/null || echo "0")
        local node_errors=$(grep -c "ERROR\|Erreur" "$NODE_LOG" 2>/dev/null || echo "0")
        local streamlit_errors=$(grep -c "error\|Error" "$STREAMLIT_LOG" 2>/dev/null || echo "0")
        
        echo -e "  ${CYAN}Requêtes API totales:${NC}  ${total_requests}"
        echo -e "  ${CYAN}Erreurs Node.js:${NC}       ${node_errors}"
        echo -e "  ${CYAN}Erreurs Streamlit:${NC}     ${streamlit_errors}"
        
        echo ""
        echo -e "${YELLOW}Rafraîchissement automatique dans 3 secondes...${NC}"
        echo -e "${YELLOW}Appuyez sur Ctrl+C pour quitter${NC}"
        
        sleep 3
    done
}

# Lancer le dashboard
dashboard
