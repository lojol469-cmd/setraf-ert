#!/usr/bin/env python3
"""KibaLock Kernel Agent - Installation Rapide des Dépendances"""
import subprocess
import sys
from pathlib import Path

#!/usr/bin/env python3
"""KibaLock Kernel Agent - Installation Rapide des Dépendances"""
import subprocess
import sys
from pathlib import Path

def check_package(pkg_name):
    """Vérifie si un package est installé"""
    try:
        __import__(pkg_name.replace("-", "_"))
        return True
    except:
        return False

def install_package(pkg_name):
    """Installe un package rapidement"""
    print(f"⚙️  Installation: {pkg_name}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg_name, "-q"], 
                      timeout=60, check=True)
        print(f"✅ {pkg_name}")
        return True
    except:
        print(f"❌ {pkg_name}")
        return False

def main():
    critical = ["fastapi", "uvicorn", "pymongo", "streamlit"]
    
    print("🔍 Vérification packages critiques...")
    missing = [p for p in critical if not check_package(p)]
    
    if not missing:
        print("✅ Tous les packages sont installés")
        return 0
    
    print(f"⚠️  Packages manquants: {len(missing)}")
    for pkg in missing:
        install_package(pkg)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

    """Agent intelligent pour gestion autonome des dépendances système"""
    
    def __init__(self, conda_env: str = "gestmodo"):
        self.conda_env = conda_env
        self.conda_base = Path.home() / "miniconda3"
        self.python_path = self.conda_base / "envs" / conda_env / "bin" / "python"
        self.pip_path = self.conda_base / "envs" / conda_env / "bin" / "pip"
        self.log_file = Path("/tmp/kibalock_kernel_agent.log")
        
        # Historique des tentatives d'installation
        self.installation_history = {}
        
        # Méthodes d'installation par priorité
        self.installation_methods = [
            "pip",              # Méthode standard
            "pip-no-cache",     # Sans cache (corruption)
            "pip-binary",       # Binaire pré-compilé (rapide)
            "pip-no-deps",      # Sans dépendances (léger)
            "conda",            # Alternative conda
        ]
        
        # Packages avec méthodes spéciales
        self.special_packages = {
            "torch": {
                "method": "custom",
                "command": "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130"
            },
            "faiss-cpu": {
                "alternatives": ["faiss", "faiss-gpu"],
                "prefer": "conda"
            },
            "pyaudio": {
                "prefer": "conda",
                "alternatives": ["sounddevice"]
            }
        }
        
        self.log("=" * 60)
        self.log("🤖 KibaLock Kernel Agent - Démarrage")
        self.log(f"📦 Environnement: {conda_env}")
        self.log(f"🐍 Python: {self.python_path}")
        self.log("=" * 60)
    
    def log(self, message: str, level: str = "INFO"):
        """Log avec timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} [{level}] {message}"
        print(log_line)
        
        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")
    
    def check_package(self, package: str) -> bool:
        """Vérifie si un package est installé"""
        # Nettoyer le nom du package (enlever version, extras)
        clean_name = re.split(r'[=<>!\[]', package)[0].strip()
        import_name = clean_name.replace("-", "_")
        
        try:
            result = subprocess.run(
                [str(self.python_path), "-c", f"import {import_name}"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def get_package_size(self, package: str) -> Optional[int]:
        """Estime la taille d'un package (en MB)"""
        try:
            result = subprocess.run(
                [str(self.pip_path), "download", "--no-deps", "--dry-run", package],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Parser la sortie pour extraire la taille
            match = re.search(r'(\d+\.?\d*)\s*(MB|GB|KB)', result.stdout)
            if match:
                size = float(match.group(1))
                unit = match.group(2)
                if unit == "GB":
                    return int(size * 1024)
                elif unit == "MB":
                    return int(size)
                elif unit == "KB":
                    return int(size / 1024)
            return None
        except:
            return None
    
    def install_package(self, package: str, method: str = "pip") -> Dict[str, Any]:
        """Installe un package Python manquant avec méthodes alternatives"""
        try:
            self.log(f"🔧 Installation de {package} (méthode: {method})...")
            
            # Vérifier si package spécial
            clean_name = re.split(r'[=<>!\[]', package)[0].strip()
            if clean_name in self.special_packages:
                special = self.special_packages[clean_name]
                if special.get("method") == "custom":
                    self.log(f"⚙️  Package spécial détecté: {clean_name}")
                    result = subprocess.run(
                        special["command"],
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    if result.returncode == 0:
                        return {
                            "success": True,
                            "package": package,
                            "method": "custom",
                            "message": f"Installation personnalisée réussie: {package}"
                        }
            
            # Estimer la taille
            size = self.get_package_size(package)
            if size and size > 500:
                self.log(f"⚠️  Package volumineux détecté: ~{size} MB")
            
            # Sélection de la commande selon la méthode
            if method == "pip":
                cmd = [str(self.pip_path), "install", package, "--upgrade"]
                timeout = 300
            
            elif method == "pip-no-deps":
                self.log(f"🔄 Installation sans dépendances automatiques...")
                cmd = [str(self.pip_path), "install", package, "--no-deps"]
                timeout = 180
            
            elif method == "pip-no-cache":
                self.log(f"🔄 Installation sans cache...")
                cmd = [str(self.pip_path), "install", package, "--no-cache-dir", "--upgrade"]
                timeout = 300
            
            elif method == "pip-binary":
                self.log(f"🔄 Installation binaire pré-compilé...")
                cmd = [str(self.pip_path), "install", package, "--only-binary", ":all:", "--upgrade"]
                timeout = 240
            
            elif method == "pip-user":
                self.log(f"🔄 Installation en mode utilisateur...")
                cmd = [str(self.pip_path), "install", package, "--user", "--upgrade"]
                timeout = 240
            
            elif method == "conda":
                self.log(f"🔄 Installation via conda...")
                conda_pkg = package.replace("_", "-").replace("==", "=")
                cmd = ["conda", "install", "-n", self.conda_env, "-y", conda_pkg]
                timeout = 300
            
            elif method == "git":
                # Pour packages depuis GitHub
                self.log(f"🔄 Installation depuis git...")
                if "github.com" in package or package.startswith("git+"):
                    cmd = [str(self.pip_path), "install", package]
                    timeout = 400
                else:
                    return {
                        "success": False,
                        "package": package,
                        "method": method,
                        "error": "URL git invalide"
                    }
            
            else:
                return {
                    "success": False,
                    "package": package,
                    "error": f"Méthode inconnue: {method}"
                }
            
            # Exécution avec monitoring de progression
            self.log(f"⏳ Exécution: {' '.join(cmd)}")
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            self.log(f"⏱️  Durée: {elapsed:.1f}s")
            
            if result.returncode == 0:
                self.log(f"✅ {package} installé avec succès ({method})")
                
                # Enregistrer dans l'historique
                self.installation_history[package] = {
                    "method": method,
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "duration": elapsed
                }
                
                return {
                    "success": True,
                    "package": package,
                    "method": method,
                    "duration": elapsed,
                    "message": f"Installation réussie: {package}",
                    "output": result.stdout[-500:] if result.stdout else ""
                }
            else:
                error_msg = result.stderr[-500:] if result.stderr else "Erreur inconnue"
                self.log(f"❌ Échec installation {package} ({method})", "ERROR")
                self.log(f"   Erreur: {error_msg}", "ERROR")
                
                return {
                    "success": False,
                    "package": package,
                    "method": method,
                    "error": error_msg,
                    "suggestion": "Essayer une méthode alternative"
                }
                
        except subprocess.TimeoutExpired:
            self.log(f"⏱️ TIMEOUT lors de l'installation de {package} ({method})", "WARN")
            self.log(f"   Temps limite: {timeout}s dépassé", "WARN")
            
            return {
                "success": False,
                "package": package,
                "method": method,
                "error": "TIMEOUT",
                "timeout": timeout,
                "suggestion": "Téléchargement trop lent ou RAM insuffisante, essayer méthode alternative"
            }
            
        except Exception as e:
            self.log(f"❌ EXCEPTION installation {package} ({method}): {e}", "ERROR")
            return {
                "success": False,
                "package": package,
                "method": method,
                "error": str(e),
                "suggestion": "Erreur système, vérifier les logs"
            }
    
    def install_with_fallback(self, package: str) -> Dict[str, Any]:
        """Installe un package avec stratégie de fallback intelligente"""
        self.log(f"🎯 Installation intelligente de: {package}")
        
        # Vérifier si déjà installé
        if self.check_package(package):
            self.log(f"✓ {package} déjà installé")
            return {
                "success": True,
                "package": package,
                "message": "Déjà installé",
                "skipped": True
            }
        
        # Vérifier si package spécial avec préférence
        clean_name = re.split(r'[=<>!\[]', package)[0].strip()
        if clean_name in self.special_packages:
            special = self.special_packages[clean_name]
            if special.get("prefer"):
                preferred_method = special["prefer"]
                self.log(f"🌟 Méthode préférée pour {clean_name}: {preferred_method}")
                methods = [preferred_method] + [m for m in self.installation_methods if m != preferred_method]
            else:
                methods = self.installation_methods
        else:
            methods = self.installation_methods
        
        # Tentatives avec chaque méthode
        for i, method in enumerate(methods, 1):
            self.log(f"📥 Tentative {i}/{len(methods)}: {method}")
            
            result = self.install_package(package, method)
            
            if result["success"]:
                self.log(f"🎉 SUCCÈS avec méthode: {method}")
                return result
            else:
                self.log(f"⚠️  Échec avec {method}: {result.get('error', 'Unknown')}", "WARN")
                
                # Si timeout, essayer méthode plus légère
                if "TIMEOUT" in result.get("error", "").upper() or "timeout" in result.get("error", "").lower():
                    self.log(f"💡 Timeout détecté, priorisation des méthodes légères", "WARN")
                    # Forcer méthode sans dépendances
                    if "pip-no-deps" in methods and method != "pip-no-deps":
                        self.log(f"🔄 Tentative immédiate avec pip-no-deps")
                        result = self.install_package(package, "pip-no-deps")
                        if result["success"]:
                            return result
                
                # Attendre un peu avant retry
                if i < len(methods):
                    time.sleep(2)
        
        # Si toutes les méthodes ont échoué, suggérer alternative
        self.log(f"❌ ÉCHEC TOTAL pour {package} après {len(methods)} tentatives", "ERROR")
        
        alternatives = []
        if clean_name in self.special_packages:
            alternatives = self.special_packages[clean_name].get("alternatives", [])
        
        return {
            "success": False,
            "package": package,
            "error": "Toutes les méthodes ont échoué",
            "tried_methods": methods,
            "alternatives": alternatives,
            "suggestion": f"Essayer manuellement ou alternatives: {alternatives}" if alternatives else "Installation manuelle requise"
        }
    
    def scan_and_install_missing(self, requirements_file: str = "requirements.txt") -> Dict[str, Any]:
        """Scanne requirements.txt et installe tous les packages manquants"""
        self.log("🔍 Scan des dépendances manquantes...")
        
        req_path = Path(requirements_file)
        if not req_path.exists():
            self.log(f"❌ Fichier non trouvé: {requirements_file}", "ERROR")
            return {
                "success": False,
                "error": f"Fichier {requirements_file} introuvable"
            }
        
        # Lire requirements.txt
        packages = []
        with open(req_path) as f:
            for line in f:
                line = line.strip()
                # Ignorer commentaires et lignes vides
                if line and not line.startswith("#"):
                    # Ignorer les directives spéciales
                    if not line.startswith("-") and not line.startswith("git+"):
                        packages.append(line)
        
        self.log(f"📦 {len(packages)} packages à vérifier")
        
        missing = []
        installed = []
        failed = []
        
        # Vérifier chaque package
        for package in packages:
            clean_name = re.split(r'[=<>!\[]', package)[0].strip()
            
            if not self.check_package(clean_name):
                self.log(f"❌ Manquant: {package}")
                missing.append(package)
            else:
                self.log(f"✓ Installé: {clean_name}")
                installed.append(clean_name)
        
        if not missing:
            self.log("🎉 Tous les packages sont déjà installés!")
            return {
                "success": True,
                "installed": installed,
                "missing": [],
                "failed": [],
                "message": "Tous les packages sont installés"
            }
        
        self.log(f"⚠️  {len(missing)} packages manquants détectés")
        
        # Installer chaque package manquant
        for package in missing:
            self.log(f"\n{'='*60}")
            result = self.install_with_fallback(package)
            
            if result["success"] and not result.get("skipped"):
                installed.append(package)
            elif not result["success"]:
                failed.append({
                    "package": package,
                    "error": result.get("error"),
                    "suggestion": result.get("suggestion")
                })
        
        # Résumé
        self.log(f"\n{'='*60}")
        self.log("📊 RÉSUMÉ DE L'INSTALLATION")
        self.log(f"✅ Installés: {len(installed)}")
        self.log(f"❌ Échecs: {len(failed)}")
        
        if failed:
            self.log("\n⚠️  Packages en échec:")
            for fail in failed:
                self.log(f"   - {fail['package']}: {fail['error']}")
        
        return {
            "success": len(failed) == 0,
            "installed": installed,
            "missing": missing,
            "failed": failed,
            "total_packages": len(packages),
            "message": f"Installation terminée: {len(installed)} succès, {len(failed)} échecs"
        }
    
    def save_report(self, results: Dict[str, Any], output_file: str = "/tmp/kibalock_install_report.json"):
        """Sauvegarde un rapport JSON des installations"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.conda_env,
            "results": results,
            "history": self.installation_history
        }
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        self.log(f"📄 Rapport sauvegardé: {output_file}")

def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="KibaLock Kernel Agent - Gestion Intelligente des Dépendances")
    parser.add_argument("--env", default="gestmodo", help="Environnement conda (défaut: gestmodo)")
    parser.add_argument("--requirements", default="requirements.txt", help="Fichier requirements.txt")
    parser.add_argument("--package", help="Installer un package spécifique")
    parser.add_argument("--method", default="auto", help="Méthode d'installation (auto, pip, conda, etc.)")
    parser.add_argument("--report", default="/tmp/kibalock_install_report.json", help="Fichier rapport JSON")
    
    args = parser.parse_args()
    
    # Créer l'agent
    agent = KibaLockKernelAgent(conda_env=args.env)
    
    if args.package:
        # Installer un package spécifique
        if args.method == "auto":
            result = agent.install_with_fallback(args.package)
        else:
            result = agent.install_package(args.package, args.method)
        
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)
    else:
        # Scanner et installer depuis requirements.txt
        results = agent.scan_and_install_missing(args.requirements)
        agent.save_report(results, args.report)
        
        sys.exit(0 if results["success"] else 1)

if __name__ == "__main__":
    main()
