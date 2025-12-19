# CONFIGURATION TEMPLATE KIBALI ULTRA-RAPIDE
# ============================================
#
# Ce fichier contient tous les paramètres de configuration
# pour personnaliser le comportement du template KIBALI.
#
# Utilisation:
#   from kibali_config import KIBALI_CONFIG
#   tokenizer, model = load_kibali_ultra_fast(**KIBALI_CONFIG)
#
# Ou directement dans vos scripts:
#   import kibali_config
#   config = kibali_config.get_config_for_production()

import os
from typing import Dict, Any

# =============================================================================
# CONFIGURATIONS PRÉDÉFINIES
# =============================================================================

# Configuration de PRODUCTION (recommandée)
PRODUCTION_CONFIG = {
    'model_path': '/home/belikan/kibali-finetune/kibali-final-merged-model',
    'device': 'auto',
    'use_4bit': True,
    'use_8bit': False,
    'force_no_quantization': False,
    'monitor_gpu': True
}

# Configuration de DÉVELOPPEMENT (plus rapide à charger)
DEVELOPMENT_CONFIG = {
    'model_path': '/home/belikan/kibali-finetune/kibali-final-merged-model',
    'device': 'auto',
    'use_4bit': False,
    'use_8bit': False,
    'force_no_quantization': True,  # Chargement plus rapide
    'monitor_gpu': True
}

# Configuration CPU (pour machines sans GPU)
CPU_CONFIG = {
    'model_path': '/home/belikan/kibali-finetune/kibali-final-merged-model',
    'device': 'cpu',
    'use_4bit': False,
    'use_8bit': False,
    'force_no_quantization': True,
    'monitor_gpu': False
}

# Configuration HAUTE PERFORMANCE (nécessite beaucoup de VRAM)
HIGH_PERFORMANCE_CONFIG = {
    'model_path': '/home/belikan/kibali-finetune/kibali-final-merged-model',
    'device': 'cuda',
    'use_4bit': False,
    'use_8bit': False,
    'force_no_quantization': True,  # Performance maximale
    'monitor_gpu': True
}

# Configuration ÉCONOMIQUE (faible utilisation mémoire)
ECONOMIC_CONFIG = {
    'model_path': '/home/belikan/kibali-finetune/kibali-final-merged-model',
    'device': 'auto',
    'use_4bit': True,
    'use_8bit': False,
    'force_no_quantization': False,
    'monitor_gpu': False
}

# =============================================================================
# CONFIGURATION PAR DÉFAUT
# =============================================================================

KIBALI_CONFIG = PRODUCTION_CONFIG

# =============================================================================
# FONCTIONS UTILITAIRES DE CONFIGURATION
# =============================================================================

def get_config_for_environment() -> Dict[str, Any]:
    """
    Retourne la configuration adaptée à l'environnement détecté

    Returns:
        Dict de configuration
    """
    # Détection GPU
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if has_gpu else 0
    except:
        has_gpu = False
        gpu_memory = 0

    # Variables d'environnement
    env = os.getenv('KIBALI_ENV', 'production').lower()
    device_override = os.getenv('KIBALI_DEVICE')
    quantization_override = os.getenv('KIBALI_QUANTIZATION')

    # Logique de sélection automatique
    if env == 'development' or not has_gpu:
        config = DEVELOPMENT_CONFIG.copy()
    elif env == 'cpu':
        config = CPU_CONFIG.copy()
    elif has_gpu and gpu_memory >= 16:  # GPUs avec 16GB+ VRAM
        config = HIGH_PERFORMANCE_CONFIG.copy()
    elif has_gpu and gpu_memory >= 8:   # GPUs standards
        config = PRODUCTION_CONFIG.copy()
    else:
        config = ECONOMIC_CONFIG.copy()

    # Appliquer les overrides d'environnement
    if device_override:
        config['device'] = device_override

    if quantization_override:
        if quantization_override == '4bit':
            config.update({'use_4bit': True, 'use_8bit': False, 'force_no_quantization': False})
        elif quantization_override == '8bit':
            config.update({'use_4bit': False, 'use_8bit': True, 'force_no_quantization': False})
        elif quantization_override == 'none':
            config.update({'force_no_quantization': True})

    return config

def get_config_for_production() -> Dict[str, Any]:
    """Configuration optimisée pour la production"""
    return PRODUCTION_CONFIG.copy()

def get_config_for_development() -> Dict[str, Any]:
    """Configuration optimisée pour le développement"""
    return DEVELOPMENT_CONFIG.copy()

def get_config_for_cpu() -> Dict[str, Any]:
    """Configuration pour machines CPU uniquement"""
    return CPU_CONFIG.copy()

def get_config_for_high_performance() -> Dict[str, Any]:
    """Configuration haute performance (nécessite beaucoup de VRAM)"""
    return HIGH_PERFORMANCE_CONFIG.copy()

def get_config_for_economic() -> Dict[str, Any]:
    """Configuration économique (faible utilisation mémoire)"""
    return ECONOMIC_CONFIG.copy()

def create_custom_config(
    model_path: str = None,
    device: str = None,
    quantization: str = '4bit',
    monitor_gpu: bool = True
) -> Dict[str, Any]:
    """
    Crée une configuration personnalisée

    Args:
        model_path: Chemin vers le modèle
        device: Device ('auto', 'cuda', 'cpu')
        quantization: Type de quantification ('4bit', '8bit', 'none')
        monitor_gpu: Activer monitoring GPU

    Returns:
        Dict de configuration personnalisée
    """
    # Configuration de base
    config = {
        'model_path': model_path or PRODUCTION_CONFIG['model_path'],
        'device': device or 'auto',
        'monitor_gpu': monitor_gpu
    }

    # Configuration quantification
    if quantization == '4bit':
        config.update({
            'use_4bit': True,
            'use_8bit': False,
            'force_no_quantization': False
        })
    elif quantization == '8bit':
        config.update({
            'use_4bit': False,
            'use_8bit': True,
            'force_no_quantization': False
        })
    else:  # none
        config.update({
            'use_4bit': False,
            'use_8bit': False,
            'force_no_quantization': True
        })

    return config

# =============================================================================
# EXEMPLES D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    print("🔧 CONFIGURATIONS TEMPLATE KIBALI")
    print("=" * 50)

    # Afficher les configurations disponibles
    configs = {
        'Production': get_config_for_production(),
        'Développement': get_config_for_development(),
        'CPU': get_config_for_cpu(),
        'Haute Performance': get_config_for_high_performance(),
        'Économique': get_config_for_economic(),
        'Auto-détectée': get_config_for_environment()
    }

    for name, config in configs.items():
        print(f"\n📋 {name}:")
        print("-" * 30)
        for key, value in config.items():
            print(f"  {key}: {value}")

    # Exemple de configuration personnalisée
    print("
🔧 Configuration personnalisée:"    print("-" * 30)
    custom = create_custom_config(
        model_path="/custom/path/model",
        device="cuda",
        quantization="none",
        monitor_gpu=False
    )
    for key, value in custom.items():
        print(f"  {key}: {value}")

    print("
✨ Configurations prêtes à l'emploi!"</content>
<parameter name="filePath">/home/belikan/kibali_config.py