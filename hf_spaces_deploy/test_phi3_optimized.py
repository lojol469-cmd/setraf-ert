#!/usr/bin/env python3
"""Test optimisé de Phi-3-mini avec les nouveaux paramètres de vitesse"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("🔄 Chargement de Phi-3-mini...")
start = time.time()

model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.eval()
model.config.use_cache = False

load_time = time.time() - start
print(f"✅ Modèle chargé en {load_time:.1f}s")

# Prompt ultra-court optimisé
rho_mean = 45.3
rho_min = 12.5
rho_max = 156.8
geo_type = "argiles/marnes saturées"

context = f"""Données: {rho_mean:.0f} Ω·m (min:{rho_min:.0f}, max:{rho_max:.0f}). Type: {geo_type}. Interprétation:"""

print(f"\n📝 Prompt ({len(context)} caractères):")
print(context)
print("\n🤖 Génération optimisée (50 tokens, greedy search)...")

start_gen = time.time()

with torch.inference_mode():
    inputs = tokenizer(context, return_tensors="pt")
    
    # Paramètres optimisés pour vitesse maximale
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
        max_new_tokens=50,
        do_sample=False,          # Greedy = plus rapide
        num_beams=1,              # Pas de beam search
        early_stopping=True,      # Arrêt dès que possible
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(context, "").strip()

gen_time = time.time() - start_gen
total_time = time.time() - start

print(f"\n✅ Génération en {gen_time:.1f}s")
print(f"⏱️ Temps total: {total_time:.1f}s")
print(f"\n💬 Réponse:\n{response}")
