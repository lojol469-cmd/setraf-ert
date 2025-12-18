#!/bin/bash

# === Auto IP/Platform Detection Utility ===
# Détecte automatiquement l'IP et la plateforme pour configurer les services

detect_platform() {
    if grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
        echo "wsl"
    elif [ -f "/.dockerenv" ]; then
        echo "docker"
    elif uname -s | grep -qi "darwin"; then
        echo "macos"
    elif uname -s | grep -qi "linux"; then
        echo "linux"
    else
        echo "unknown"
    fi
}

get_local_ip() {
    local platform=$1
    
    # Essayer différentes méthodes selon la plateforme
    if [ "$platform" = "wsl" ]; then
        # WSL: Get Windows host IP
        ip route show | grep -i default | awk '{ print $3}'
    elif [ "$platform" = "docker" ]; then
        # Docker: Get container IP
        hostname -i | awk '{print $1}'
    else
        # Linux/macOS: Get primary interface IP
        ip -4 addr show $(ip route | grep default | awk '{print $5}' | head -n1) 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1 || \
        ifconfig $(route get default 2>/dev/null | grep interface | awk '{print $2}') 2>/dev/null | grep 'inet ' | awk '{print $2}' || \
        hostname -I | awk '{print $1}'
    fi
}

get_public_ip() {
    # Essayer plusieurs services
    curl -s ifconfig.me || \
    curl -s icanhazip.com || \
    curl -s ipinfo.io/ip || \
    echo "0.0.0.0"
}

get_all_ips() {
    local platform=$(detect_platform)
    local local_ip=$(get_local_ip "$platform")
    local public_ip=$(get_public_ip)
    
    echo "PLATFORM=$platform"
    echo "LOCAL_IP=$local_ip"
    echo "PUBLIC_IP=$public_ip"
    echo "LOCALHOST=127.0.0.1"
    
    # IPs spécifiques selon plateforme
    if [ "$platform" = "wsl" ]; then
        # WSL: Add Windows host IP
        local wsl_host=$(ip route show | grep -i default | awk '{ print $3}')
        echo "WSL_HOST=$wsl_host"
    fi
}

find_free_port() {
    local start_port=$1
    local max_attempts=${2:-10}
    
    for ((i=0; i<max_attempts; i++)); do
        local port=$((start_port + i))
        if ! lsof -i:$port &>/dev/null; then
            echo $port
            return 0
        fi
    done
    
    echo $start_port
    return 1
}

generate_env_file() {
    local output_file=$1
    local platform=$(detect_platform)
    local local_ip=$(get_local_ip "$platform")
    
    # Trouver des ports libres
    local backend_port=$(find_free_port 8505)
    local api_port=$(find_free_port 8000)
    local frontend_port=$(find_free_port 3000)
    
    cat > "$output_file" << EOF
# Auto-generated configuration
# Platform: $platform
# Generated: $(date)

# === Network Configuration ===
PLATFORM=$platform
LOCAL_IP=$local_ip
LOCALHOST=127.0.0.1

# === Service Ports ===
BACKEND_PORT=$backend_port
API_PORT=$api_port
FRONTEND_PORT=$frontend_port

# === Service URLs (Internal) ===
BACKEND_URL=http://${local_ip}:${backend_port}
API_URL=http://${local_ip}:${api_port}
FRONTEND_URL=http://${local_ip}:${frontend_port}

# === Service URLs (Localhost) ===
BACKEND_URL_LOCAL=http://localhost:${backend_port}
API_URL_LOCAL=http://localhost:${api_port}
FRONTEND_URL_LOCAL=http://localhost:${frontend_port}

# === MongoDB (from existing .env) ===
$(grep "^MONGO_URI=" .env 2>/dev/null || echo "MONGO_URI=mongodb://localhost:27017")
$(grep "^MONGO_DB_NAME=" .env 2>/dev/null || echo "MONGO_DB_NAME=kibalock")

# === FAISS Configuration ===
$(grep "^FAISS_" .env 2>/dev/null)

# === AI Models ===
WHISPER_MODEL=base
FACE_MODEL=Facenet512
PHI_MODEL=microsoft/Phi-3.5-mini-instruct
TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2

# === Security (from existing .env) ===
$(grep "^JWT_" .env 2>/dev/null)
$(grep "^API_KEY" .env 2>/dev/null || echo "PUBLIC_KEY=qazghazz
PRIVATE_KEY=264419a2-cd4e-471a-81b3-04c522669052")

# === Email Configuration (from existing .env) ===
$(grep "^EMAIL_" .env 2>/dev/null)
$(grep "^SMTP_" .env 2>/dev/null)

# === Thresholds ===
$(grep "THRESHOLD" .env 2>/dev/null || echo "VOICE_THRESHOLD=0.85
FACE_THRESHOLD=0.90")

# === Session ===
$(grep "SESSION_DURATION" .env 2>/dev/null || echo "SESSION_DURATION=24")

# === Directories ===
BASE_DIR=\${HOME}/kibalock
LIFEMODO_DIR=\${HOME}/lifemodo_api
EOF

    echo "$output_file"
}

# Export functions for use in other scripts
export -f detect_platform
export -f get_local_ip
export -f get_public_ip
export -f get_all_ips
export -f find_free_port
export -f generate_env_file

# Si exécuté directement
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    case "${1:-detect}" in
        detect)
            get_all_ips
            ;;
        platform)
            detect_platform
            ;;
        ip)
            get_local_ip $(detect_platform)
            ;;
        port)
            find_free_port ${2:-8000}
            ;;
        env)
            generate_env_file ${2:-.env.auto}
            ;;
        *)
            echo "Usage: $0 {detect|platform|ip|port [start]|env [output]}"
            exit 1
            ;;
    esac
fi
