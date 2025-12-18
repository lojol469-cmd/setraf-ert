#!/usr/bin/env python3
"""
Script d'initialisation pour copier les modèles locaux vers HF Spaces
"""
import os
import shutil
from pathlib import Path

def copy_model_to_cache(local_model_path, cache_model_name):
    """Copie un modèle local vers le cache HF"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    target_dir = cache_dir / f"models--{cache_model_name.replace('/', '--')}"

    if target_dir.exists():
        print(f"✅ {cache_model_name} déjà présent dans le cache")
        return True

    source_dir = Path(local_model_path)
    if not source_dir.exists():
        print(f"❌ Source {local_model_path} introuvable")
        return False

    print(f"📋 Copie {cache_model_name} vers le cache...")
    try:
        shutil.copytree(source_dir, target_dir)
        print(f"✅ {cache_model_name} copié avec succès")
        return True
    except Exception as e:
        print(f"❌ Erreur copie {cache_model_name}: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration des modèles locaux pour HF Spaces...")

    # Copier Phi-3-mini
    phi_success = copy_model_to_cache(
        "/home/belikan/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-mini-4k-instruct"
    )

    # Copier CLIP
    clip_success = copy_model_to_cache(
        "/home/belikan/.cache/huggingface/hub/models--openai--clip-vit-base-patch32",
        "openai/clip-vit-base-patch32"
    )

    if phi_success and clip_success:
        print("🎉 Tous les modèles configurés avec succès!")
    else:
        print("⚠️ Certains modèles n'ont pas pu être copiés")