#!/usr/bin/env python3
"""
Test SOLUTION: Forcer past_key_values=None
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("🔧 Test SOLUTION - past_key_values=None")
print("=" * 60)

PHI3_MODEL_PATH = "microsoft/Phi-3-mini-4k-instruct"

print("\n1️⃣ Chargement tokenizer + modèle...")
tokenizer = AutoTokenizer.from_pretrained(
    PHI3_MODEL_PATH,
    trust_remote_code=True,
    use_fast=True
)

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

print("✅ Modèle prêt")

print("\n2️⃣ Test génération AVEC past_key_values=None...")
test_prompt = "[INST] Analyse géophysique: 350 mesures, résistivité 0.2-0.4 Ω·m. Quelle est l'interprétation? [/INST]"

with torch.inference_mode():
    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    
    try:
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
            past_key_values=None,  # SOLUTION: Forcer None
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n✅ SUCCÈS!")
        print(f"\n📝 Réponse complète:")
        print(response)
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
