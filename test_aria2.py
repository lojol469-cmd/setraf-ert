#!/usr/bin/env python3
"""
Test rapide avec aria2 pour téléchargement ultra-rapide
"""

import os
import subprocess
from huggingface_hub import snapshot_download

# Configuration
model_id = "BelikanM/kibali-final-merged"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
local_dir = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")

print("🚀 Téléchargement ultra-rapide avec aria2")
print("=" * 50)
print(f"📁 Modèle: {model_id}")
print(f"💾 Cache: {cache_dir}")

# Vérifier si déjà en cache
if os.path.exists(local_dir):
    print("✅ Modèle déjà en cache local")
    print("🎯 Pas besoin de télécharger")
else:
    print("📥 Téléchargement avec aria2...")

    # Utiliser huggingface_hub avec aria2
    try:
        # Télécharger avec aria2 activé
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # Active aria2

        downloaded_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4  # Parallélisation
        )

        print(f"✅ Téléchargé vers: {downloaded_path}")

    except Exception as e:
        print(f"❌ Erreur: {e}")

# Test rapide du modèle
print("\n🧪 Test du modèle...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Charger depuis le cache local
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,  # Utiliser le chemin local
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Modèle chargé depuis cache")

    # Test rapide
    prompt = "[INST] Test ERT [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("[/INST]")[-1].strip()

    print("💡 Test réussi!")
    print(f"📝 Réponse: {response[:100]}...")

except Exception as e:
    print(f"❌ Erreur test: {e}")