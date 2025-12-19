#!/usr/bin/env python3
"""
SCRIPT D'EXEMPLE - TEMPLATE KIBALI ULTRA-RAPIDE
===============================================

Ce script démontre l'utilisation complète du template KIBALI Ultra-Fast
dans différents scénarios pratiques.

UTILISATION:
    python example_kibali_usage.py --mode [chat|analysis|benchmark|api]

EXEMPLES:
    # Mode interactif (chat)
    python example_kibali_usage.py --mode chat

    # Analyse de données géologiques
    python example_kibali_usage.py --mode analysis

    # Benchmark de performance
    python example_kibali_usage.py --mode benchmark

    # Test API REST simulé
    python example_kibali_usage.py --mode api

AUTEUR: KIBALI AI Team
VERSION: 1.0
"""

import argparse
import time
import json
import sys
from typing import Dict, List, Optional
from pathlib import Path

# Import du template
try:
    from template_kibali_ultra_fast import (
        load_kibali_ultra_fast,
        generate_ultra_fast,
        analyze_geological_data_ultra_fast,
        setup_ultra_fast_gpu,
        monitor_gpu_usage,
        create_kibali_pipeline
    )
    TEMPLATE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Template non trouvé: {e}")
    TEMPLATE_AVAILABLE = False

# Données d'exemple pour tests
SAMPLE_GEOLOGICAL_DATA = [
    {
        'name': 'Site Argileux',
        'data': {
            'n_measures': 1200,
            'rho_min': 8,
            'rho_max': 150,
            'rho_mean': 45,
            'depth_max': 12,
            'location': 'Zone agricole'
        }
    },
    {
        'name': 'Site Rocheux',
        'data': {
            'n_measures': 800,
            'rho_min': 200,
            'rho_max': 5000,
            'rho_mean': 1200,
            'depth_max': 25,
            'location': 'Zone montagneuse'
        }
    },
    {
        'name': 'Site Aquifère',
        'data': {
            'n_measures': 2000,
            'rho_min': 15,
            'rho_max': 300,
            'rho_mean': 85,
            'depth_max': 18,
            'location': 'Zone de recherche d\'eau'
        }
    }
]

class KIBALIExampleRunner:
    """Classe pour exécuter les exemples d'utilisation"""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.loaded = False

    def load_model(self, **kwargs) -> bool:
        """Charge le modèle KIBALI"""
        if not TEMPLATE_AVAILABLE:
            print("❌ Template non disponible")
            return False

        print("🚀 Chargement du modèle KIBALI Ultra-Fast...")
        start_time = time.time()

        self.tokenizer, self.model = load_kibali_ultra_fast(**kwargs)

        if self.tokenizer and self.model:
            load_time = time.time() - start_time
            print(".1f"            print("✅ Modèle chargé avec succès!")

            # Créer pipeline
            self.pipeline = create_kibali_pipeline(self.tokenizer, self.model)
            self.loaded = True
            return True
        else:
            print("❌ Échec du chargement du modèle")
            return False

    def run_chat_mode(self):
        """Mode chat interactif"""
        print("\n" + "="*60)
        print("💬 MODE CHAT INTERACTIF - KIBALI")
        print("="*60)
        print("Tapez vos questions géologiques (ou 'quit' pour quitter)")
        print("Exemples:")
        print("- Qu'est-ce que la résistivité électrique?")
        print("- Comment interpréter une valeur de 100 Ω·m?")
        print("- Décris une formation argileuse")
        print()

        while True:
            try:
                user_input = input("🧑‍🔬 Votre question: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Au revoir!")
                    break

                if not user_input:
                    continue

                # Générer réponse
                prompt = f"[INST] Question géologique: {user_input}\nRéponds de façon experte, concise et pédagogique. [/INST]"

                print("🤔 KIBALI réfléchit...")
                start_time = time.time()

                response = generate_ultra_fast(
                    self.tokenizer, self.model,
                    prompt,
                    max_new_tokens=200,
                    temperature=0.1
                )

                response_time = time.time() - start_time

                print(f"\n🧠 KIBALI ({response_time:.2f}s):")
                print("-" * 40)
                print(response)
                print("-" * 40)
                print()

            except KeyboardInterrupt:
                print("\n👋 Interruption détectée. Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                continue

    def run_analysis_mode(self):
        """Mode analyse de données géologiques"""
        print("\n" + "="*60)
        print("📊 MODE ANALYSE GÉOLOGIQUE - KIBALI")
        print("="*60)

        for i, site in enumerate(SAMPLE_GEOLOGICAL_DATA, 1):
            print(f"\n🔍 ANALYSE {i}: {site['name']}")
            print("-" * 40)

            # Analyser les données
            analysis = analyze_geological_data_ultra_fast(
                self.tokenizer, self.model,
                site['data'],
                max_tokens=250
            )

            print(analysis)
            print()

            # Pause entre analyses
            if i < len(SAMPLE_GEOLOGICAL_DATA):
                input("⏯️  Appuyez sur Entrée pour l'analyse suivante...")

    def run_benchmark_mode(self):
        """Mode benchmark de performance"""
        print("\n" + "="*60)
        print("⚡ BENCHMARK PERFORMANCE - KIBALI")
        print("="*60)

        # Test de génération simple
        test_prompts = [
            "Explique la tomographie électrique en 2 phrases.",
            "Quels sont les facteurs influençant la résistivité des sols?",
            "Comment différencier argile et sable en géophysique?"
        ]

        print("🧪 Test de génération rapide:")
        print("-" * 40)

        total_time = 0
        total_tokens = 0

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt[:50]}...")

            start_time = time.time()
            response = generate_ultra_fast(
                self.tokenizer, self.model,
                prompt,
                max_new_tokens=100,
                temperature=0.0,
                monitor_gpu=True
            )
            end_time = time.time()

            response_time = end_time - start_time
            token_count = len(response.split())  # Approximation
            tokens_per_sec = token_count / response_time if response_time > 0 else 0

            print(".2f"            print(".1f"            print(f"📝 Réponse: {response[:100]}...")

            total_time += response_time
            total_tokens += token_count

        # Statistiques globales
        avg_time = total_time / len(test_prompts)
        avg_tokens_sec = total_tokens / total_time if total_time > 0 else 0

        print(f"\n📊 STATISTIQUES GLOBALES:")
        print("-" * 40)
        print(".2f"        print(".1f"        print(f"🔄 Tests exécutés: {len(test_prompts)}")

        # Test analyse géologique
        print(f"\n🪨 Test analyse géologique:")
        print("-" * 40)

        start_time = time.time()
        analysis = analyze_geological_data_ultra_fast(
            self.tokenizer, self.model,
            SAMPLE_GEOLOGICAL_DATA[0]['data']
        )
        analysis_time = time.time() - start_time

        print(".2f"        print(f"📊 Analyse: {analysis[:150]}...")

    def run_api_mode(self):
        """Mode simulation API REST"""
        print("\n" + "="*60)
        print("🌐 MODE API REST SIMULÉ - KIBALI")
        print("="*60)
        print("Simulation d'un serveur API pour analyses géologiques")
        print()

        # Simuler des requêtes API
        api_requests = [
            {
                'endpoint': '/analyze/resistivity',
                'method': 'POST',
                'data': SAMPLE_GEOLOGICAL_DATA[0]['data']
            },
            {
                'endpoint': '/chat/geological',
                'method': 'POST',
                'data': {'question': 'Comment interpréter une résistivité de 50 Ω·m?'}
            },
            {
                'endpoint': '/analyze/batch',
                'method': 'POST',
                'data': {'sites': [site['data'] for site in SAMPLE_GEOLOGICAL_DATA[:2]]}
            }
        ]

        for request in api_requests:
            print(f"📡 {request['method']} {request['endpoint']}")
            print("-" * 40)

            start_time = time.time()

            try:
                if 'resistivity' in request['endpoint']:
                    # Analyse de résistivité
                    response = analyze_geological_data_ultra_fast(
                        self.tokenizer, self.model,
                        request['data']
                    )
                    response_data = {'analysis': response, 'status': 'success'}

                elif 'chat' in request['endpoint']:
                    # Chat géologique
                    prompt = f"[INST] {request['data']['question']} [/INST]"
                    response = generate_ultra_fast(
                        self.tokenizer, self.model,
                        prompt,
                        max_new_tokens=150
                    )
                    response_data = {'response': response, 'status': 'success'}

                elif 'batch' in request['endpoint']:
                    # Analyse par lot
                    analyses = []
                    for site_data in request['data']['sites']:
                        analysis = analyze_geological_data_ultra_fast(
                            self.tokenizer, self.model,
                            site_data,
                            max_tokens=100
                        )
                        analyses.append(analysis)

                    response_data = {'analyses': analyses, 'count': len(analyses), 'status': 'success'}

                response_time = time.time() - start_time

                print(".2f"                print(f"📤 Réponse: {json.dumps(response_data, indent=2, ensure_ascii=False)[:300]}...")

            except Exception as e:
                response_time = time.time() - start_time
                print(".2f"                print(f"❌ Erreur: {str(e)[:100]}")

            print()

    def run_diagnostic_mode(self):
        """Mode diagnostic système"""
        print("\n" + "="*60)
        print("🔍 DIAGNOSTIC SYSTÈME - KIBALI")
        print("="*60)

        # Vérifier la disponibilité du template
        print("📦 Template KIBALI:"        print(f"   Disponible: {'✅' if TEMPLATE_AVAILABLE else '❌'}")

        if TEMPLATE_AVAILABLE:
            print("   Fonctions importées: ✅")
        else:
            print("   Fonctions importées: ❌")
            return

        # Vérifier GPU
        print("\n🖥️  GPU:"        try:
            import torch
            gpu_available = torch.cuda.is_available()
            print(f"   CUDA disponible: {'✅' if gpu_available else '❌'}")

            if gpu_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"   Nombre GPU: {gpu_count}")
                print(f"   Modèle: {gpu_name}")
                print(".1f"            else:
                print("   Mode: CPU")

        except Exception as e:
            print(f"   Erreur GPU: {e}")

        # Tester le chargement du modèle
        print("\n🚀 Test de chargement modèle:")
        success = self.load_model(force_no_quantization=True, monitor_gpu=False)

        if success:
            print("   Chargement: ✅")

            # Test de génération simple
            print("\n🧪 Test de génération:")
            try:
                test_response = generate_ultra_fast(
                    self.tokenizer, self.model,
                    "Test rapide",
                    max_new_tokens=10
                )
                print("   Génération: ✅"                print(f"   Réponse: {test_response}")
            except Exception as e:
                print(f"   Génération: ❌ ({e})")

        else:
            print("   Chargement: ❌")

        print("\n✨ Diagnostic terminé!")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Script d'exemple - Template KIBALI Ultra-Fast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLES D'UTILISATION:

  # Mode chat interactif
  python example_kibali_usage.py --mode chat

  # Analyse de données géologiques
  python example_kibali_usage.py --mode analysis

  # Benchmark de performance
  python example_kibali_usage.py --mode benchmark

  # Simulation API REST
  python example_kibali_usage.py --mode api

  # Diagnostic système
  python example_kibali_usage.py --mode diagnostic

OPTIONS AVANCÉES:

  # Avec quantification 4-bit (recommandé)
  python example_kibali_usage.py --mode chat --quantization 4bit

  # Mode CPU uniquement
  python example_kibali_usage.py --mode analysis --device cpu

  # Désactiver monitoring GPU
  python example_kibali_usage.py --mode benchmark --no-gpu-monitor
        """
    )

    parser.add_argument(
        '--mode',
        choices=['chat', 'analysis', 'benchmark', 'api', 'diagnostic'],
        default='diagnostic',
        help='Mode d\'exécution'
    )

    parser.add_argument(
        '--quantization',
        choices=['4bit', '8bit', 'none'],
        default='4bit',
        help='Type de quantification'
    )

    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device à utiliser'
    )

    parser.add_argument(
        '--no-gpu-monitor',
        action='store_true',
        help='Désactiver le monitoring GPU'
    )

    args = parser.parse_args()

    # Vérifier disponibilité
    if not TEMPLATE_AVAILABLE:
        print("❌ ERREUR: Template KIBALI non trouvé!")
        print("   Assurez-vous que template_kibali_ultra_fast.py est dans le PYTHONPATH")
        sys.exit(1)

    # Configuration du chargement
    load_config = {
        'device': args.device,
        'monitor_gpu': not args.no_gpu_monitor
    }

    if args.quantization == '4bit':
        load_config.update({'use_4bit': True, 'use_8bit': False, 'force_no_quantization': False})
    elif args.quantization == '8bit':
        load_config.update({'use_4bit': False, 'use_8bit': True, 'force_no_quantization': False})
    else:  # none
        load_config.update({'force_no_quantization': True})

    # Initialiser le runner
    runner = KIBALIExampleRunner()

    # Pour les modes nécessitant le modèle, le charger
    if args.mode in ['chat', 'analysis', 'benchmark', 'api']:
        if not runner.load_model(**load_config):
            print("❌ Impossible de charger le modèle. Arrêt.")
            sys.exit(1)

    # Exécuter le mode demandé
    try:
        if args.mode == 'chat':
            runner.run_chat_mode()
        elif args.mode == 'analysis':
            runner.run_analysis_mode()
        elif args.mode == 'benchmark':
            runner.run_benchmark_mode()
        elif args.mode == 'api':
            runner.run_api_mode()
        elif args.mode == 'diagnostic':
            runner.run_diagnostic_mode()

    except KeyboardInterrupt:
        print("\n👋 Interruption détectée. Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Bannière
    print("🚀 TEMPLATE KIBALI ULTRA-RAPIDE - SCRIPT D'EXEMPLE")
    print("=" * 60)
    print("IA Géologique Ultra-Optimisée pour Analyses ERT")
    print("KIBALI AI Team - 2025")
    print("=" * 60)

    # Lancer le script
    main()

    print("\n✨ Script terminé avec succès!")</content>
<parameter name="filePath">/home/belikan/example_kibali_usage.py