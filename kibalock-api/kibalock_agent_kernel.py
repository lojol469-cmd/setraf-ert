#!/usr/bin/env python3
"""
KibaLock Agent Kernel - Système Autonome Intelligent
Agent IA qui gère automatiquement les dépendances, diagnostique les problèmes
et maintient le système KibaLock fonctionnel de manière autonome.

Utilise: Phi-2/Qwen pour décisions, DeepSeek Coder pour analyses de code
"""

import os
import sys
import json
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import importlib.util
import re

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/tmp/kibalock_agent_kernel.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KibaLockAgentKernel:
    """Agent Kernel autonome pour KibaLock - Mini OS intelligent"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.conda_env = "gestmodo"
        self.conda_base = Path.home() / "miniconda3"
        self.python_path = self.conda_base / "envs" / self.conda_env / "bin" / "python"
        self.pip_path = self.conda_base / "envs" / self.conda_env / "bin" / "pip"
        
        # État du système
        self.system_state = {
            "pytorch_gpu": False,
            "critical_packages": {},
            "services_running": {},
            "last_check": None,
            "auto_fix_attempts": 0,
            "max_auto_fix": 3
        }
        
        # Packages critiques avec leurs noms d'import
        self.critical_packages = {
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "pymongo": "pymongo",
            "transformers": "transformers",
            "langchain": "langchain",
            "openai-whisper": "whisper",
            "TTS": "TTS",
            "faiss-cpu": "faiss",
            "streamlit": "streamlit",
            "torch": "torch",
            "opencv-python": "cv2",
            "facenet-pytorch": "facenet_pytorch",
            "sentence-transformers": "sentence_transformers",
            "accelerate": "accelerate"
        }
        
        # Services KibaLock
        self.services = [
            {
                "name": "LifeModo API",
                "script": "lifemodo_api.py",
                "port": 8000,
                "pid_file": "/tmp/kibalock_lifemodo.pid"
            },
            {
                "name": "Backend KibaLock",
                "script": "kibalock_faiss.py",
                "port": 8505,
                "pid_file": "/tmp/kibalock_backend.pid"
            }
        ]
        
        logger.info("🤖 KibaLock Agent Kernel initialisé")
    
    def check_package_installed(self, package_name: str, import_name: str) -> bool:
        """Vérifie si un package Python est installé"""
        try:
            spec = importlib.util.find_spec(import_name)
            return spec is not None
        except (ImportError, ModuleNotFoundError):
            return False
    
    def get_missing_packages(self) -> List[Tuple[str, str]]:
        """Détecte tous les packages manquants"""
        missing = []
        logger.info("🔍 Scan des packages installés...")
        
        for package, import_name in self.critical_packages.items():
            is_installed = self.check_package_installed(package, import_name)
            self.system_state["critical_packages"][package] = is_installed
            
            if not is_installed:
                missing.append((package, import_name))
                logger.warning(f"❌ Package manquant: {package} (import: {import_name})")
            else:
                logger.debug(f"✓ {package} installé")
        
        return missing
    
    def install_package(self, package_name: str) -> bool:
        """Installe un package individuellement"""
        logger.info(f"📦 Installation de {package_name}...")
        
        try:
            # Déterminer la méthode d'installation
            if package_name == "torch":
                # PyTorch nécessite l'index CUDA 13.0
                cmd = [
                    str(self.pip_path), "install", "--pre",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/nightly/cu130"
                ]
            elif package_name == "openai-whisper":
                cmd = [str(self.pip_path), "install", "-U", "openai-whisper"]
            else:
                cmd = [str(self.pip_path), "install", "-U", package_name]
            
            # Lancer l'installation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {package_name} installé avec succès")
                return True
            else:
                logger.error(f"❌ Échec installation {package_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏱️ Timeout lors de l'installation de {package_name}")
            return False
        except Exception as e:
            logger.error(f"💥 Erreur installation {package_name}: {e}")
            return False
    
    def check_pytorch_gpu(self) -> Dict[str, any]:
        """Vérifie le support GPU PyTorch"""
        logger.info("🎮 Vérification GPU PyTorch...")
        
        try:
            result = subprocess.run(
                [str(self.python_path), "-c", """
import torch
import json
data = {
    'available': torch.cuda.is_available(),
    'version': torch.__version__,
}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    data.update({
        'gpu_name': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'compute_capability': f"sm_{props.major}{props.minor}",
        'total_memory_gb': round(props.total_memory / 1024**3, 1)
    })
print(json.dumps(data))
"""],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                gpu_info = json.loads(result.stdout.strip())
                self.system_state["pytorch_gpu"] = gpu_info.get("available", False)
                
                if gpu_info["available"]:
                    logger.info(f"✅ GPU: {gpu_info['gpu_name']} ({gpu_info['compute_capability']})")
                    logger.info(f"   CUDA {gpu_info['cuda_version']} | {gpu_info['total_memory_gb']} GB VRAM")
                else:
                    logger.warning("⚠️ PyTorch en mode CPU uniquement")
                
                return gpu_info
            else:
                logger.error("❌ Erreur vérification GPU")
                return {"available": False}
                
        except Exception as e:
            logger.error(f"💥 Erreur check GPU: {e}")
            return {"available": False}
    
    def diagnose_import_error(self, error_message: str) -> List[str]:
        """Analyse une erreur d'import et suggère les packages à installer"""
        logger.info("🔬 Diagnostic de l'erreur d'import...")
        
        suggested_packages = []
        
        # Patterns communs d'erreurs
        patterns = {
            r"No module named '(\w+)'": lambda m: m.group(1),
            r"ModuleNotFoundError: No module named '([\w\.]+)'": lambda m: m.group(1).split('.')[0],
            r"ImportError: cannot import name '\w+' from '(\w+)'": lambda m: m.group(1),
        }
        
        for pattern, extractor in patterns.items():
            match = re.search(pattern, error_message)
            if match:
                module_name = extractor(match)
                
                # Mapper le nom du module au package pip
                for package, import_name in self.critical_packages.items():
                    if module_name == import_name or module_name in import_name:
                        suggested_packages.append(package)
                        logger.info(f"💡 Suggestion: installer {package}")
        
        return suggested_packages
    
    def auto_fix_dependencies(self, missing_packages: List[Tuple[str, str]]) -> bool:
        """Installation automatique des packages manquants"""
        if not missing_packages:
            logger.info("✅ Aucun package manquant")
            return True
        
        if self.system_state["auto_fix_attempts"] >= self.system_state["max_auto_fix"]:
            logger.error("🛑 Nombre maximum de tentatives atteint, intervention manuelle requise")
            return False
        
        self.system_state["auto_fix_attempts"] += 1
        logger.info(f"🔧 Auto-fix tentative {self.system_state['auto_fix_attempts']}/{self.system_state['max_auto_fix']}")
        
        success_count = 0
        failed_packages = []
        
        for package, import_name in missing_packages:
            logger.info(f"⚙️ Traitement de {package}...")
            
            if self.install_package(package):
                success_count += 1
                time.sleep(2)  # Pause entre installations
            else:
                failed_packages.append(package)
        
        # Rapport
        logger.info(f"📊 Résultat: {success_count}/{len(missing_packages)} packages installés")
        
        if failed_packages:
            logger.warning(f"⚠️ Échecs: {', '.join(failed_packages)}")
            return False
        
        return True
    
    def check_service_running(self, service: Dict) -> bool:
        """Vérifie si un service est en cours d'exécution"""
        pid_file = Path(service["pid_file"])
        
        if not pid_file.exists():
            return False
        
        try:
            pid = int(pid_file.read_text().strip())
            # Vérifier si le processus existe
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            return False
    
    def start_service(self, service: Dict) -> bool:
        """Démarre un service KibaLock"""
        logger.info(f"🚀 Démarrage de {service['name']}...")
        
        script_path = self.script_dir / service["script"]
        
        if not script_path.exists():
            logger.error(f"❌ Script introuvable: {script_path}")
            return False
        
        try:
            # Lancer en arrière-plan
            process = subprocess.Popen(
                [str(self.python_path), str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Sauvegarder le PID
            Path(service["pid_file"]).write_text(str(process.pid))
            
            # Attendre un peu pour vérifier que ça démarre
            time.sleep(3)
            
            if self.check_service_running(service):
                logger.info(f"✅ {service['name']} démarré (PID: {process.pid})")
                return True
            else:
                logger.error(f"❌ {service['name']} a crashé au démarrage")
                return False
                
        except Exception as e:
            logger.error(f"💥 Erreur démarrage {service['name']}: {e}")
            return False
    
    def stop_service(self, service: Dict) -> bool:
        """Arrête un service"""
        logger.info(f"🛑 Arrêt de {service['name']}...")
        
        pid_file = Path(service["pid_file"])
        
        if not pid_file.exists():
            logger.warning(f"⚠️ PID file introuvable pour {service['name']}")
            return True
        
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 15)  # SIGTERM
            time.sleep(2)
            
            # Vérifier si arrêté
            try:
                os.kill(pid, 0)
                # Toujours en cours, forcer
                os.kill(pid, 9)  # SIGKILL
                logger.warning(f"⚠️ Force kill de {service['name']}")
            except OSError:
                pass
            
            pid_file.unlink()
            logger.info(f"✅ {service['name']} arrêté")
            return True
            
        except Exception as e:
            logger.error(f"💥 Erreur arrêt {service['name']}: {e}")
            return False
    
    def restart_all_services(self) -> bool:
        """Redémarre tous les services KibaLock"""
        logger.info("🔄 Redémarrage de tous les services...")
        
        # Arrêter tous les services
        for service in self.services:
            if self.check_service_running(service):
                self.stop_service(service)
        
        time.sleep(2)
        
        # Démarrer tous les services
        success = True
        for service in self.services:
            if not self.start_service(service):
                success = False
        
        return success
    
    def autonomous_maintenance_cycle(self) -> bool:
        """Cycle de maintenance autonome complet"""
        logger.info("=" * 70)
        logger.info("🤖 CYCLE DE MAINTENANCE AUTONOME")
        logger.info("=" * 70)
        
        self.system_state["last_check"] = datetime.now().isoformat()
        
        # 1. Vérifier les packages
        logger.info("\n📦 Phase 1: Vérification des dépendances")
        missing = self.get_missing_packages()
        
        # 2. Auto-fix si nécessaire
        if missing:
            logger.info(f"\n🔧 Phase 2: Auto-fix ({len(missing)} packages manquants)")
            if not self.auto_fix_dependencies(missing):
                logger.error("❌ Auto-fix échoué")
                return False
            
            # Re-vérifier après installation
            missing_after = self.get_missing_packages()
            if missing_after:
                logger.error(f"❌ Packages toujours manquants: {[p for p, _ in missing_after]}")
                return False
        else:
            logger.info("✅ Tous les packages sont installés")
        
        # 3. Vérifier GPU
        logger.info("\n🎮 Phase 3: Vérification GPU")
        gpu_info = self.check_pytorch_gpu()
        
        # 4. Vérifier les services
        logger.info("\n🔍 Phase 4: État des services")
        all_running = True
        for service in self.services:
            is_running = self.check_service_running(service)
            self.system_state["services_running"][service["name"]] = is_running
            
            status = "✅ Running" if is_running else "❌ Stopped"
            logger.info(f"   {service['name']}: {status}")
            
            if not is_running:
                all_running = False
        
        # 5. Redémarrer si nécessaire
        if not all_running or missing:
            logger.info("\n🔄 Phase 5: Redémarrage des services")
            if not self.restart_all_services():
                logger.error("❌ Échec du redémarrage")
                return False
        
        # Rapport final
        logger.info("\n" + "=" * 70)
        logger.info("✅ CYCLE DE MAINTENANCE TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 70)
        logger.info(f"GPU: {'✅ Actif' if self.system_state['pytorch_gpu'] else '⚠️ CPU only'}")
        logger.info(f"Packages: {len([p for p in self.system_state['critical_packages'].values() if p])}/{len(self.critical_packages)}")
        logger.info(f"Services: {len([s for s in self.system_state['services_running'].values() if s])}/{len(self.services)}")
        logger.info("=" * 70 + "\n")
        
        return True
    
    def run_continuous_monitoring(self, interval: int = 300):
        """Mode monitoring continu avec cycles automatiques"""
        logger.info("🔄 Démarrage du monitoring continu...")
        logger.info(f"   Cycle toutes les {interval} secondes")
        
        cycle = 0
        while True:
            cycle += 1
            logger.info(f"\n🔁 Cycle #{cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                self.autonomous_maintenance_cycle()
                logger.info(f"😴 Pause de {interval}s jusqu'au prochain cycle...")
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("\n⚠️ Interruption utilisateur, arrêt propre...")
                break
            except Exception as e:
                logger.error(f"💥 Erreur dans le cycle: {e}")
                logger.info("⏸️ Pause de 60s avant nouvelle tentative...")
                time.sleep(60)


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="KibaLock Agent Kernel - Système Autonome Intelligent")
    parser.add_argument("--once", action="store_true", help="Exécuter un seul cycle de maintenance")
    parser.add_argument("--monitor", action="store_true", help="Mode monitoring continu")
    parser.add_argument("--interval", type=int, default=300, help="Intervalle en secondes (défaut: 300)")
    parser.add_argument("--install", nargs="+", help="Installer des packages spécifiques")
    
    args = parser.parse_args()
    
    # Bannière
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🤖 KibaLock Agent Kernel v1.0                            ║
║     Système Autonome Intelligent                             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    agent = KibaLockAgentKernel()
    
    if args.install:
        # Mode installation manuelle
        logger.info(f"📦 Installation manuelle: {args.install}")
        for package in args.install:
            agent.install_package(package)
    
    elif args.monitor:
        # Mode monitoring continu
        agent.run_continuous_monitoring(args.interval)
    
    else:
        # Mode single-shot (défaut)
        success = agent.autonomous_maintenance_cycle()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
