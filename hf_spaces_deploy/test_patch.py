#!/usr/bin/env python3
"""
Test SOLUTION ULTIME: Désactiver complètement past_key_values
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("🔧 Test SOLUTION ULTIME")
print("=" * 60)

PHI3_MODEL_PATH = "microsoft/Phi-3-mini-4k-instruct"

print("\n1️⃣ Chargement...")
tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    PHI3_MODEL_PATH,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to('cpu')

# PATCH: Modifier la méthode prepare_inputs_for_generation
original_prepare = model.prepare_inputs_for_generation

def patched_prepare_inputs_for_generation(input_ids, past_key_values=None, **kwargs):
    # Forcer past_key_values à None pour éviter seen_tokens
    return original_prepare(input_ids, past_key_values=None, **kwargs)

model.prepare_inputs_for_generation = patched_prepare_inputs_for_generation

model.config.use_cache = False
model.eval()

print("✅ Modèle patché")

print("\n2️⃣ Test génération...")
test_prompt = "[INST] Résistivité 0.2-0.4 Ω·m. Interprétation? [/INST]"

with torch.inference_mode():
    inputs = tokenizer(test_prompt, return_tensors="pt").to('cpu')
    
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n✅ SUCCÈS!")
        print(f"📝 Réponse: {response}")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")

print("\n" + "=" * 60)
