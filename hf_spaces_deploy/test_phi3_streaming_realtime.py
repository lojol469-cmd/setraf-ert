#!/usr/bin/env python3
"""Test du streaming de tokens avec Phi-3-mini"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

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
print(f"✅ Modèle chargé en {load_time:.1f}s\n")

# Prompt ultra-court
context = "Données: 45 Ω·m (min:12, max:157). Type: argiles/marnes saturées. Interprétation:"
print(f"📝 Prompt: {context}\n")
print("🤖 Génération avec STREAMING:")
print("-" * 60)

# Préparer inputs
inputs = tokenizer(context, return_tensors="pt")

# Créer le streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Paramètres de génération
generation_kwargs = {
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs.get('attention_mask'),
    'max_new_tokens': 50,
    'do_sample': False,
    'num_beams': 1,
    'pad_token_id': tokenizer.eos_token_id,
    'streamer': streamer
}

# Lancer génération dans thread
start_gen = time.time()
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Afficher tokens en temps réel
response_text = ""
token_count = 0
for new_text in streamer:
    response_text += new_text
    token_count += 1
    print(f"{new_text}", end="", flush=True)

thread.join()
gen_time = time.time() - start_gen

print("\n" + "-" * 60)
print(f"\n✅ Génération en {gen_time:.1f}s ({token_count} tokens)")
print(f"⚡ Vitesse: {token_count/gen_time:.1f} tokens/sec")
print(f"\n💬 Réponse complète:\n{response_text}")
