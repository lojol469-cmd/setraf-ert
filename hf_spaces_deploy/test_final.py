#!/usr/bin/env python3
"""Test final avec transformers 4.38.0"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("🔧 Test FINAL Phi-3-mini avec transformers 4.38.0")
print("=" * 70)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
print(f"✓ Transformers version: {transformers.__version__}")

PHI3_MODEL_PATH = "microsoft/Phi-3-mini-4k-instruct"

print("\n1️⃣ Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    PHI3_MODEL_PATH,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to('cpu')

model.config.use_cache = False
model.eval()
print("✅ Modèle chargé et configuré")

print("\n2️⃣ TEST 1: Génération simple")
test1 = "[INST] Bonjour! [/INST]"
with torch.inference_mode():
    inputs = tokenizer(test1, return_tensors="pt").to('cpu')
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    response1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✅ Réponse: {response1[len(test1):]}")

print("\n3️⃣ TEST 2: Analyse géophysique")
test2 = "[INST] Analyse géophysique: 350 mesures, résistivité moyenne 0.4 Ω·m (0.2-0.4). Interprétation en 2 phrases? [/INST]"
with torch.inference_mode():
    inputs = tokenizer(test2, return_tensors="pt").to('cpu')
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response2 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✅ Réponse:")
    print(response2.split('[/INST]')[-1].strip()[:200])

print("\n" + "=" * 70)
print("✅ TOUS LES TESTS RÉUSSIS - Prêt pour déploiement!")
