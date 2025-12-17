#!/usr/bin/env python3
"""
Test local Phi-3-mini pour identifier l'erreur DynamicCache
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("🔧 Test Phi-3-mini local - Debug DynamicCache")
print("=" * 60)

PHI3_MODEL_PATH = "microsoft/Phi-3-mini-4k-instruct"

print("\n1️⃣ Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    PHI3_MODEL_PATH,
    trust_remote_code=True,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Tokenizer chargé")

print("\n2️⃣ Chargement du modèle...")
model = AutoModelForCausalLM.from_pretrained(
    PHI3_MODEL_PATH,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print("✅ Modèle chargé")

print("\n3️⃣ Déplacement sur CPU...")
model = model.to('cpu')
print(f"✅ Modèle sur device: {next(model.parameters()).device}")

print("\n4️⃣ Configuration du cache...")
print(f"   Avant: model.config.use_cache = {model.config.use_cache}")
model.config.use_cache = False
print(f"   Après: model.config.use_cache = {model.config.use_cache}")

print("\n5️⃣ Mode évaluation...")
model.eval()
print("✅ Model en mode eval")

print("\n6️⃣ Test de génération...")
test_prompt = "[INST] Bonjour, comment vas-tu? [/INST]"

with torch.inference_mode():
    inputs = tokenizer(test_prompt, return_tensors="pt")
    print(f"   Input shape: {inputs.input_ids.shape}")
    
    # Déplacer inputs sur CPU
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    
    print("\n   🔄 Génération en cours...")
    try:
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n   ✅ Génération réussie!")
        print(f"   📝 Réponse: {response[:100]}...")
        
    except Exception as e:
        print(f"\n   ❌ ERREUR: {type(e).__name__}: {e}")
        import traceback
        print("\n   📋 Traceback complet:")
        traceback.print_exc()

print("\n" + "=" * 60)
print("Test terminé")
