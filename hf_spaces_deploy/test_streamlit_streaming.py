#!/usr/bin/env python3
"""Test du streaming dans Streamlit"""

import streamlit as st
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

st.set_page_config(page_title="Test Streaming", layout="wide")

st.title("🤖 Test Streaming Phi-3 dans Streamlit")

# Initialiser le modèle en session_state
@st.cache_resource
def load_model():
    st.info("🔄 Chargement de Phi-3-mini... (60-90s)")
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
    return model, tokenizer

if st.button("🚀 Lancer Génération avec Streaming"):
    model, tokenizer = load_model()
    st.success("✅ Modèle chargé!")
    
    # Prompt de test
    context = "Données: 45 Ω·m (min:12, max:157). Type: argiles/marnes saturées. Interprétation:"
    
    st.markdown(f"**Prompt:** `{context}`")
    st.markdown("---")
    
    # Créer placeholder pour streaming
    streaming_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Préparer inputs
    inputs = tokenizer(context, return_tensors="pt")
    
    # Créer streamer
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
    start_time = time.time()
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Afficher tokens en temps réel
    streaming_placeholder.markdown("🤖 **Génération démarrée...**")
    
    response_text = ""
    token_count = 0
    
    for new_text in streamer:
        response_text += new_text
        token_count += 1
        elapsed = time.time() - start_time
        
        # Mettre à jour l'affichage
        streaming_placeholder.markdown(f"### 🤖 Streaming en cours:\n\n{response_text}▌")
        stats_placeholder.info(f"⏱️ {elapsed:.1f}s | 📊 {token_count} tokens | ⚡ {token_count/elapsed:.2f} tokens/s")
    
    thread.join()
    total_time = time.time() - start_time
    
    # Afficher résultat final
    streaming_placeholder.markdown(f"### ✅ Génération terminée:\n\n{response_text}")
    stats_placeholder.success(f"✅ {token_count} tokens en {total_time:.1f}s ({token_count/total_time:.2f} tokens/s)")
    
    st.markdown("---")
    st.markdown("**Conclusion:** Le streaming fonctionne! Chaque token apparaît en temps réel.")
