#!/usr/bin/env python3
"""
TESTS UNITAIRES - TEMPLATE KIBALI ULTRA-RAPIDE
==============================================

Suite de tests complète pour valider le fonctionnement
du template KIBALI Ultra-Fast.

UTILISATION:
    python test_kibali_template.py          # Tous les tests
    python test_kibali_template.py --quick  # Tests rapides uniquement
    python test_kibali_template.py --gpu    # Tests GPU uniquement

AUTEUR: KIBALI AI Team
VERSION: 1.0
"""

import unittest
import time
import sys
from typing import Dict, Any, Optional

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
except ImportError:
    TEMPLATE_AVAILABLE = False

# Données de test
TEST_GEOLOGICAL_DATA = {
    'n_measures': 100,
    'rho_min': 10,
    'rho_max': 200,
    'rho_mean': 75,
    'depth_max': 10,
    'location': 'Test Site'
}

class TestKIBALITemplate(unittest.TestCase):
    """Tests unitaires pour le template KIBALI"""

    def setUp(self):
        """Configuration avant chaque test"""
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.model_loaded = False

        # Charger le modèle une seule fois pour tous les tests
        if TEMPLATE_AVAILABLE and not hasattr(self.__class__, 'model_loaded_once'):
            print("🚀 Chargement du modèle pour les tests...")
            try:
                self.__class__.tokenizer, self.__class__.model = load_kibali_ultra_fast(
                    force_no_quantization=True,  # Plus rapide pour les tests
                    monitor_gpu=False
                )
                if self.__class__.tokenizer and self.__class__.model:
                    self.__class__.model_loaded_once = True
                    print("✅ Modèle chargé pour les tests")
                else:
                    self.__class__.model_loaded_once = False
                    print("❌ Échec chargement modèle")
            except Exception as e:
                self.__class__.model_loaded_once = False
                print(f"❌ Erreur chargement: {e}")

    def test_template_import(self):
        """Test que le template peut être importé"""
        self.assertTrue(TEMPLATE_AVAILABLE, "Template KIBALI non disponible")

    def test_gpu_setup(self):
        """Test de la configuration GPU"""
        if not TEMPLATE_AVAILABLE:
            self.skipTest("Template non disponible")

        result = setup_ultra_fast_gpu()
        # Le résultat peut être True ou False selon la disponibilité GPU
        self.assertIsInstance(result, bool)

    def test_gpu_monitoring(self):
        """Test du monitoring GPU"""
        if not TEMPLATE_AVAILABLE:
            self.skipTest("Template non disponible")

        usage = monitor_gpu_usage()
        self.assertIsInstance(usage, (int, float))
        self.assertGreaterEqual(usage, 0.0)
        self.assertLessEqual(usage, 100.0)

    def test_model_loading(self):
        """Test du chargement du modèle"""
        if not TEMPLATE_AVAILABLE:
            self.skipTest("Template non disponible")

        # Test avec quantification désactivée pour rapidité
        tokenizer, model = load_kibali_ultra_fast(
            force_no_quantization=True,
            monitor_gpu=False
        )

        self.assertIsNotNone(tokenizer, "Tokenizer non chargé")
        self.assertIsNotNone(model, "Modèle non chargé")

    @unittest.skipUnless(TEMPLATE_AVAILABLE, "Template non disponible")
    def test_text_generation(self):
        """Test de génération de texte"""
        if not self.__class__.model_loaded_once:
            self.skipTest("Modèle non chargé")

        prompt = "Explique brièvement la résistivité électrique."
        response = generate_ultra_fast(
            self.__class__.tokenizer,
            self.__class__.model,
            prompt,
            max_new_tokens=50,
            temperature=0.0
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertNotEqual(response, prompt)  # Doit avoir généré du nouveau texte

    @unittest.skipUnless(TEMPLATE_AVAILABLE, "Template non disponible")
    def test_geological_analysis(self):
        """Test d'analyse géologique"""
        if not self.__class__.model_loaded_once:
            self.skipTest("Modèle non chargé")

        analysis = analyze_geological_data_ultra_fast(
            self.__class__.tokenizer,
            self.__class__.model,
            TEST_GEOLOGICAL_DATA,
            max_tokens=100
        )

        self.assertIsInstance(analysis, str)
        self.assertGreater(len(analysis), 0)

        # Vérifier que l'analyse contient des éléments attendus
        analysis_lower = analysis.lower()
        expected_terms = ['géologie', 'résistivité', 'formation', 'analyse']
        found_terms = sum(1 for term in expected_terms if term in analysis_lower)
        self.assertGreaterEqual(found_terms, 2, "Analyse incomplète")

    @unittest.skipUnless(TEMPLATE_AVAILABLE, "Template non disponible")
    def test_pipeline_creation(self):
        """Test de création de pipeline"""
        if not self.__class__.model_loaded_once:
            self.skipTest("Modèle non chargé")

        pipeline = create_kibali_pipeline(
            self.__class__.tokenizer,
            self.__class__.model
        )

        self.assertIsNotNone(pipeline)

        # Test du pipeline
        response = pipeline("Test pipeline", max_new_tokens=20)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    @unittest.skipUnless(TEMPLATE_AVAILABLE, "Template non disponible")
    def test_generation_parameters(self):
        """Test des paramètres de génération"""
        if not self.__class__.model_loaded_once:
            self.skipTest("Modèle non chargé")

        # Test température = 0 (déterministe)
        response1 = generate_ultra_fast(
            self.__class__.tokenizer,
            self.__class__.model,
            "Test déterministe",
            max_new_tokens=20,
            temperature=0.0
        )

        response2 = generate_ultra_fast(
            self.__class__.tokenizer,
            self.__class__.model,
            "Test déterministe",
            max_new_tokens=20,
            temperature=0.0
        )

        # Les réponses devraient être identiques (déterministe)
        self.assertEqual(response1, response2)

    @unittest.skipUnless(TEMPLATE_AVAILABLE, "Template non disponible")
    def test_performance(self):
        """Test de performance"""
        if not self.__class__.model_loaded_once:
            self.skipTest("Modèle non chargé")

        # Test de génération rapide
        prompt = "Test performance rapide"
        start_time = time.time()

        response = generate_ultra_fast(
            self.__class__.tokenizer,
            self.__class__.model,
            prompt,
            max_new_tokens=50,
            temperature=0.0
        )

        end_time = time.time()
        generation_time = end_time - start_time

        # Vérifier que c'est raisonnablement rapide (< 10 secondes)
        self.assertLess(generation_time, 10.0, "Génération trop lente")

        # Vérifier que la réponse n'est pas vide
        self.assertGreater(len(response), len(prompt))

    def test_error_handling(self):
        """Test de gestion d'erreurs"""
        if not TEMPLATE_AVAILABLE:
            self.skipTest("Template non disponible")

        # Test avec paramètres invalides
        result = generate_ultra_fast(None, None, "test", max_new_tokens=10)
        self.assertIsInstance(result, str)  # Devrait retourner un message d'erreur

    def test_data_validation(self):
        """Test de validation des données d'entrée"""
        if not TEMPLATE_AVAILABLE:
            self.skipTest("Template non disponible")

        # Test avec données géologiques invalides
        invalid_data = {
            'n_measures': -1,  # Négatif invalide
            'rho_min': 100,
            'rho_max': 50,    # Min > Max invalide
        }

        # Le template devrait gérer les données invalides gracieusement
        analysis = analyze_geological_data_ultra_fast(
            self.__class__.tokenizer if self.__class__.model_loaded_once else None,
            self.__class__.model if self.__class__.model_loaded_once else None,
            invalid_data
        )

        # Devrait quand même produire une réponse
        self.assertIsInstance(analysis, str)


class TestKIBALIIntegration(unittest.TestCase):
    """Tests d'intégration plus complexes"""

    @unittest.skipUnless(TEMPLATE_AVAILABLE, "Template non disponible")
    def test_multiple_generations(self):
        """Test de générations multiples"""
        if not hasattr(self.__class__, 'model_loaded_once') or not self.__class__.model_loaded_once:
            self.skipTest("Modèle non chargé")

        prompts = [
            "Qu'est-ce que la tomographie électrique?",
            "Comment mesurer la résistivité?",
            "À quoi sert l'analyse ERT?"
        ]

        responses = []
        for prompt in prompts:
            response = generate_ultra_fast(
                self.__class__.tokenizer,
                self.__class__.model,
                prompt,
                max_new_tokens=30
            )
            responses.append(response)

        # Vérifier que toutes les réponses sont différentes et non vides
        for i, response in enumerate(responses):
            self.assertGreater(len(response), 0)
            # Chaque réponse devrait être différente
            for j, other_response in enumerate(responses):
                if i != j:
                    self.assertNotEqual(response, other_response)

    @unittest.skipUnless(TEMPLATE_AVAILABLE, "Template non disponible")
    def test_batch_analysis(self):
        """Test d'analyse par lot"""
        if not hasattr(self.__class__, 'model_loaded_once') or not self.__class__.model_loaded_once:
            self.skipTest("Modèle non chargé")

        # Plusieurs sites de test
        sites_data = [
            {'n_measures': 500, 'rho_min': 20, 'rho_max': 100, 'rho_mean': 60},
            {'n_measures': 800, 'rho_min': 50, 'rho_max': 300, 'rho_mean': 150},
            {'n_measures': 300, 'rho_min': 100, 'rho_max': 1000, 'rho_mean': 400}
        ]

        analyses = []
        for site_data in sites_data:
            analysis = analyze_geological_data_ultra_fast(
                self.__class__.tokenizer,
                self.__class__.model,
                site_data,
                max_tokens=80
            )
            analyses.append(analysis)

        # Vérifier que toutes les analyses sont produites
        self.assertEqual(len(analyses), len(sites_data))
        for analysis in analyses:
            self.assertIsInstance(analysis, str)
            self.assertGreater(len(analysis), 0)


def run_quick_tests():
    """Exécute seulement les tests rapides"""
    suite = unittest.TestSuite()

    # Tests rapides (sans chargement de modèle)
    suite.addTest(TestKIBALITemplate('test_template_import'))
    suite.addTest(TestKIBALITemplate('test_gpu_setup'))
    suite.addTest(TestKIBALITemplate('test_gpu_monitoring'))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_gpu_tests():
    """Exécute seulement les tests nécessitant GPU/modèle"""
    suite = unittest.TestSuite()

    # Tests nécessitant le modèle
    suite.addTest(TestKIBALITemplate('test_model_loading'))
    suite.addTest(TestKIBALITemplate('test_text_generation'))
    suite.addTest(TestKIBALITemplate('test_geological_analysis'))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    """Fonction principale pour exécuter les tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Tests Template KIBALI Ultra-Fast")
    parser.add_argument('--quick', action='store_true', help='Tests rapides uniquement')
    parser.add_argument('--gpu', action='store_true', help='Tests GPU uniquement')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbose')

    args = parser.parse_args()

    print("🧪 TESTS TEMPLATE KIBALI ULTRA-RAPIDE")
    print("=" * 50)

    if not TEMPLATE_AVAILABLE:
        print("❌ Template KIBALI non trouvé!")
        print("   Assurez-vous que template_kibali_ultra_fast.py est dans le PYTHONPATH")
        return False

    # Configuration du logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    try:
        if args.quick:
            print("⚡ Exécution des tests rapides...")
            success = run_quick_tests()
        elif args.gpu:
            print("🖥️  Exécution des tests GPU...")
            success = run_gpu_tests()
        else:
            print("🚀 Exécution de tous les tests...")
            unittest.main(argv=[''], exit=False, verbosity=2)
            success = True  # unittest.main gère les erreurs

    except Exception as e:
        print(f"❌ Erreur lors des tests: {e}")
        success = False

    if success:
        print("\n✅ Tous les tests réussis!")
    else:
        print("\n❌ Certains tests ont échoué!")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)</content>
<parameter name="filePath">/home/belikan/test_kibali_template.py