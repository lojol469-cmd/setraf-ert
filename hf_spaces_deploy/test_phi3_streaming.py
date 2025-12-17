#!/usr/bin/env python3
"""Test Phi-3-mini en local avec streaming"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

print("🧪 TEST PHI-3-MINI STREAMING")
print("=" * 70)

# Configuration
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Use environment variable

print(f"\n📥 Chargement du modèle: {MODEL_NAME}")

# Charger tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True
)

# Charger modèle
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    token=HF_TOKEN
).to('cpu')

model.config.use_cache = False
print("✅ Modèle chargé")

# Test de génération
print("\n🧪 Test de génération...")
prompt = """Analysez ces données de résistivité électrique:
- Résistivité: 0.5 Ω·m
- Profondeur: 2-5 mètres
Que signifient ces valeurs?"""

print(f"\n📝 Prompt:\n{prompt}")

# Tokenization
inputs = tokenizer(prompt, return_tensors="pt")
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

print("\n⏳ Génération en cours...")

# Génération
outputs = model.generate(
    inputs['input_ids'],
    attention_mask=inputs.get('attention_mask'),
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Décoder
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.replace(prompt, "").strip()

print(f"\n✅ RÉPONSE GÉNÉRÉE:")
print("=" * 70)
print(response)
print("=" * 70)

print("\n✅ TEST RÉUSSI - Phi-3-mini fonctionne!")
