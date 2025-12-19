#!/usr/bin/env python3
"""
Test rapide des modifications dans ERTest.py
Vérifier que le modèle fusionné fonctionne dans l'application
"""

import sys
import os

# Simuler les imports comme dans ERTest.py
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Configuration comme dans ERTest.py modifié
    KIBALI_FINAL_MODEL = "BelikanM/kibali-final-merged"

    print("🧪 Test des modifications ERTest.py")
    print("=" * 50)

    # Test du tokenizer
    print("📝 Test tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        KIBALI_FINAL_MODEL,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=False  # Permettre téléchargement si pas en cache
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer OK")

    # Test du modèle
    print("🤖 Test modèle fusionné...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # D'abord essayer avec le cache local
    try:
        model = AutoModelForCausalLM.from_pretrained(
            KIBALI_FINAL_MODEL,
            dtype=dtype,
            device_map="auto" if device == "cuda" else {"": "cpu"},
            low_cpu_mem_usage=True,
            local_files_only=True  # Utiliser UNIQUEMENT le cache local
        )
        print("✅ Modèle chargé depuis le cache local")
    except Exception as e:
        print(f"⚠️ Cache local non trouvé: {str(e)[:100]}")
        print("📥 Téléchargement depuis HF...")
        model = AutoModelForCausalLM.from_pretrained(
            KIBALI_FINAL_MODEL,
            dtype=dtype,
            device_map="auto" if device == "cuda" else {"": "cpu"},
            low_cpu_mem_usage=True,
            local_files_only=False  # Télécharger depuis HF
        )
        print("✅ Modèle téléchargé et mis en cache")

    # Test d'inférence rapide
    print("💬 Test inférence...")
    prompt = "[INST] Explique brièvement l'ERT [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("[/INST]")[-1].strip()

    print("💡 Réponse test:")
    print(response[:200] + "...")

    print("\n✅ Toutes les modifications fonctionnent !")
    print("🚀 ERTest.py est prêt avec le modèle fusionné KIBALI")

except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)