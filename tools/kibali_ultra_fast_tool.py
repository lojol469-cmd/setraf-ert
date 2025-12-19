#!/usr/bin/env python3
"""
OUTIL KIBALI ULTRA-FAST - Outil d'IA géologique ultra-rapide
===========================================================

Cet outil fournit une interface unifiée pour utiliser le modèle KIBALI
avec des performances ultra-rapides et synchronisation GPU parfaite.

Intégration dans le système d'outils géologiques.
"""

import sys
import os
from typing import Dict, Any, Optional, Tuple
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import du template ultra-fast
try:
    # Essayer d'abord depuis le répertoire parent
    sys.path.append('/home/belikan')
    from template_kibali_ultra_fast import (
        load_kibali_ultra_fast,
        generate_ultra_fast,
        analyze_geological_data_ultra_fast,
        setup_ultra_fast_gpu
    )
    TEMPLATE_AVAILABLE = True
except ImportError:
    try:
        # Essayer depuis le dossier du modèle
        sys.path.append('/home/belikan/kibali-finetune')
        from template_kibali_ultra_fast import (
            load_kibali_ultra_fast,
            generate_ultra_fast,
            analyze_geological_data_ultra_fast,
            setup_ultra_fast_gpu
        )
        TEMPLATE_AVAILABLE = True
    except ImportError:
        TEMPLATE_AVAILABLE = False
        logger.error("❌ Template KIBALI ultra-fast non trouvé")

class KIBALIUltraFastTool:
    """
    Outil KIBALI Ultra-Fast pour analyses géologiques IA
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.name = "kibali_ultra_fast"
        self.description = "Modèle KIBALI ultra-rapide pour analyses géologiques IA"

    def initialize(self) -> bool:
        """
        Initialise l'outil KIBALI ultra-fast
        """
        if not TEMPLATE_AVAILABLE:
            logger.error("❌ Template KIBALI non disponible")
            return False

        try:
            logger.info("🚀 Initialisation outil KIBALI Ultra-Fast...")

            # Setup GPU ultra-fast
            setup_ultra_fast_gpu()

            # Charger le modèle
            self.tokenizer, self.model = load_kibali_ultra_fast(force_no_quantization=True)

            if self.tokenizer and self.model:
                self.is_loaded = True
                logger.info("✅ Outil KIBALI Ultra-Fast initialisé avec succès")
                return True
            else:
                logger.error("❌ Échec chargement modèle KIBALI")
                return False

        except Exception as e:
            logger.error(f"❌ Erreur initialisation outil KIBALI: {str(e)}")
            return False

    def is_ready(self) -> bool:
        """
        Vérifie si l'outil est prêt à être utilisé
        """
        return self.is_loaded and self.tokenizer is not None and self.model is not None

    def analyze_geological_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse des données géologiques avec IA ultra-rapide

        Args:
            data: Données géologiques (résistivité, mesures, etc.)

        Returns:
            Analyse IA détaillée
        """
        if not self.is_ready():
            return {"error": "Outil KIBALI non initialisé"}

        try:
            logger.info("🧠 Analyse IA en cours...")

            # Préparer les données pour l'analyse
            geo_data = {
                'n_measures': data.get('n_measures', 0),
                'rho_min': data.get('rho_min', data.get('resistivity_range', [0, 1000])[0]),
                'rho_max': data.get('rho_max', data.get('resistivity_range', [0, 1000])[1]),
                'rho_mean': data.get('rho_mean', data.get('mean', 500))
            }

            # Générer l'analyse
            analysis = analyze_geological_data_ultra_fast(
                self.tokenizer,
                self.model,
                geo_data,
                max_tokens=300
            )

            return {
                "success": True,
                "analysis": analysis,
                "model": "KIBALI Ultra-Fast",
                "performance": "GPU 100% synchronisé",
                "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else None
            }

        except Exception as e:
            logger.error(f"❌ Erreur analyse IA: {str(e)}")
            return {"error": f"Erreur analyse IA: {str(e)}"}

    def generate_geological_insights(self, context: str, question: str) -> Dict[str, Any]:
        """
        Génère des insights géologiques basés sur le contexte

        Args:
            context: Contexte géologique
            question: Question spécifique

        Returns:
            Insights générés par IA
        """
        if not self.is_ready():
            return {"error": "Outil KIBALI non initialisé"}

        try:
            # Construire le prompt optimisé
            prompt = f"""[INST] Expert géophysicien ERT. Analyse le contexte suivant et réponds précisément:

CONTEXTE: {context}

QUESTION: {question}

RÉPONDS en français, sois concis et technique. [/INST]"""

            # Générer la réponse
            response = generate_ultra_fast(
                self.tokenizer,
                self.model,
                prompt,
                max_new_tokens=200
            )

            return {
                "success": True,
                "insights": response,
                "model": "KIBALI Ultra-Fast",
                "context_used": len(context),
                "question": question
            }

        except Exception as e:
            logger.error(f"❌ Erreur génération insights: {str(e)}")
            return {"error": f"Erreur génération insights: {str(e)}"}

    def interpret_resistivity_anomaly(self, anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interprète une anomalie de résistivité

        Args:
            anomaly_data: Données de l'anomalie

        Returns:
            Interprétation de l'anomalie
        """
        if not self.is_ready():
            return {"error": "Outil KIBALI non initialisé"}

        try:
            # Extraire les données d'anomalie
            rho_anomaly = anomaly_data.get('rho_anomaly', 0)
            rho_background = anomaly_data.get('rho_background', 100)
            position = anomaly_data.get('position', 'inconnue')

            prompt = f"""[INST] Interprète cette anomalie de résistivité en ERT:

ANOMALIE: ρ = {rho_anomaly} Ω·m (fond = {rho_background} Ω·m)
POSITION: {position}

Quelle est la nature probable de cette anomalie?
- Roche/Structure géologique?
- Cause anthropique?
- Artefact de mesure?

Sois précis et argumenté. [/INST]"""

            interpretation = generate_ultra_fast(
                self.tokenizer,
                self.model,
                prompt,
                max_new_tokens=150
            )

            return {
                "success": True,
                "interpretation": interpretation,
                "anomaly_rho": rho_anomaly,
                "background_rho": rho_background,
                "position": position,
                "model": "KIBALI Ultra-Fast"
            }

        except Exception as e:
            logger.error(f"❌ Erreur interprétation anomalie: {str(e)}")
            return {"error": f"Erreur interprétation anomalie: {str(e)}"}

    def get_tool_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur l'outil
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": "1.0 Ultra-Fast",
            "capabilities": [
                "analyse_geological_data",
                "generate_geological_insights",
                "interpret_resistivity_anomaly"
            ],
            "performance": {
                "loading_time": "~3 secondes",
                "generation_speed": "~0.6 réponses/seconde",
                "gpu_usage": "100% synchronisé"
            },
            "status": "ready" if self.is_ready() else "not_initialized"
        }

# Instance globale de l'outil
kibali_tool = KIBALIUltraFastTool()

def get_kibali_tool() -> KIBALIUltraFastTool:
    """
    Fonction utilitaire pour obtenir l'instance de l'outil KIBALI
    """
    return kibali_tool

# Fonctions d'interface pour l'orchestrateur
def initialize_kibali_tool() -> bool:
    """
    Initialise l'outil KIBALI pour l'orchestrateur
    """
    return kibali_tool.initialize()

def kibali_analyze_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fonction wrapper pour analyse de données
    """
    return kibali_tool.analyze_geological_data(data)

def kibali_generate_insights(context: str, question: str) -> Dict[str, Any]:
    """
    Fonction wrapper pour génération d'insights
    """
    return kibali_tool.generate_geological_insights(context, question)

def kibali_interpret_anomaly(anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fonction wrapper pour interprétation d'anomalies
    """
    return kibali_tool.interpret_resistivity_anomaly(anomaly_data)

if __name__ == "__main__":
    print("🧪 TEST OUTIL KIBALI ULTRA-FAST")
    print("=" * 40)

    # Test d'initialisation
    if initialize_kibali_tool():
        print("✅ Outil initialisé")

        # Test d'analyse
        test_data = {
            'n_measures': 1000,
            'rho_min': 5,
            'rho_max': 800,
            'rho_mean': 200
        }

        result = kibali_analyze_data(test_data)
        if "success" in result:
            print("✅ Analyse réussie:")
            print(result["analysis"][:200] + "...")
        else:
            print(f"❌ Erreur: {result}")

    else:
        print("❌ Échec initialisation outil")