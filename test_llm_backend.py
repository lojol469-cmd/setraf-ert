#!/usr/bin/env python3
"""
Test Backend de la Génération d'Interprétation LLM
Test direct sans Streamlit pour vérifier les performances
"""

import os
import sys
import time
import numpy as np

# Configuration des chemins
SETRAF_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MISTRAL_MODEL_PATH = os.path.join(SETRAF_BASE_PATH, "models/mistral-7b")

print("=" * 70)
print("🧪 TEST BACKEND - Génération d'Interprétation LLM")
print("=" * 70)
print()

# Vérifier que le modèle existe
if not os.path.exists(MISTRAL_MODEL_PATH):
    print(f"❌ ERREUR : Modèle Mistral introuvable dans : {MISTRAL_MODEL_PATH}")
    print("📁 Contenu du dossier SETRAF :")
    for item in os.listdir(SETRAF_BASE_PATH):
        print(f"   - {item}")
    sys.exit(1)

print(f"✅ Modèle trouvé : {MISTRAL_MODEL_PATH}")
print()

# Étape 1 : Charger le LLM
print("🤖 ÉTAPE 1 : Chargement du LLM Mistral...")
print("-" * 70)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    
    start_load = time.time()
    
    # Charger le tokenizer
    print("📝 Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MISTRAL_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer chargé")
    
    # Optimisations CPU
    print("⚙️  Configuration CPU...")
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    print(f"   ✅ Threads CPU : {torch.get_num_threads()}")
    
    # Charger le modèle
    print("🔄 Chargement du modèle (cela peut prendre 30-60s)...")
    model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    model = model.to('cpu')
    model.eval()
    print("   ✅ Modèle chargé et prêt")
    
    # Créer le pipeline
    print("🔗 Création du pipeline...")
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        framework="pt",
        batch_size=1
    )
    
    elapsed_load = time.time() - start_load
    print(f"   ✅ Pipeline créé en {elapsed_load:.1f}s")
    print()
    
except Exception as e:
    print(f"❌ ERREUR lors du chargement : {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Étape 2 : Préparer les données de test
print("📊 ÉTAPE 2 : Préparation des données de test")
print("-" * 70)

# Données géophysiques simulées
geophysical_data = {
    'n_spectra': 150000,
    'rho_min': 12.5,
    'rho_max': 850.0,
    'rho_mean': 125.7,
    'rho_std': 85.3,
    'n_imputed': 2500,
    'imputation_method': 'KNN',
    'n_cells': 48000,
    'convergence': 'Optimal (5 iterations)',
    'n_trajectories': 15,
    'avg_ransac_score': 0.87
}

print("Données de test :")
for key, value in geophysical_data.items():
    print(f"   • {key}: {value}")
print()

# Étape 3 : Générer l'interprétation
print("🧠 ÉTAPE 3 : Génération de l'interprétation")
print("-" * 70)

# Préparer le contexte
n_spectra_display = f"{geophysical_data['n_spectra']/1000:.1f}K"
rho_min = geophysical_data['rho_min']
rho_max = geophysical_data['rho_max']
rho_mean = geophysical_data['rho_mean']

if rho_mean < 100:
    geo_type = "argiles/marnes saturées"
elif rho_mean < 300:
    geo_type = "sols mixtes argilo-sableux"
elif rho_mean < 600:
    geo_type = "sables/graviers semi-saturés"
else:
    geo_type = "roches consolidées/substratum"

context = f"""[INST] Géophysicien ERT. Analyse EXPRESS en 150 mots max:

DATA: {n_spectra_display} mesures, ρ={rho_min:.0f}-{rho_max:.0f} Ω·m (moy:{rho_mean:.0f}), {geo_type}, {geophysical_data.get('n_trajectories', 0)} structures

RÉPONDS EN 3 SECTIONS COURTES:
1. GÉOLOGIE (2 phrases): Nature sous-sol?
2. ACTIONS (2 points): Que faire?
3. IMAGE (1 phrase): Description coupe géologique

Sois BREF et PRÉCIS. [/INST]"""

print("📝 Prompt préparé :")
print(context)
print()

# Générer avec timeout
print("⏱️  Lancement de la génération (timeout: 45s)...")
print("    (Attendez 15-30 secondes pour une génération rapide)")
print()

from concurrent.futures import ThreadPoolExecutor, TimeoutError

def run_inference():
    with torch.inference_mode():
        return llm_pipeline(
            context, 
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            num_return_sequences=1,
            pad_token_id=llm_pipeline.tokenizer.eos_token_id,
            repetition_penalty=1.15
        )

start_gen = time.time()
response = None

try:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_inference)
        
        # Afficher la progression
        for i in range(45):
            try:
                response = future.result(timeout=1.0)
                break
            except TimeoutError:
                elapsed = time.time() - start_gen
                print(f"\r    ⏳ Génération en cours... {elapsed:.0f}s", end='', flush=True)
        
        if response is None:
            print(f"\n    ⏱️  TIMEOUT après 45 secondes")
            response = None
            
except Exception as e:
    print(f"\n    ❌ Erreur : {e}")
    response = None

elapsed_gen = time.time() - start_gen
print()

# Étape 4 : Analyser le résultat
print()
print("📊 ÉTAPE 4 : Analyse du résultat")
print("-" * 70)

if response and len(response) > 0:
    print(f"✅ SUCCÈS ! Génération terminée en {elapsed_gen:.1f}s")
    print()
    
    generated_text = response[0]['generated_text']
    
    # Extraire la réponse
    if '[/INST]' in generated_text:
        generated_text = generated_text.split('[/INST]')[-1].strip()
    
    print("🎯 INTERPRÉTATION GÉNÉRÉE :")
    print("=" * 70)
    print(generated_text)
    print("=" * 70)
    print()
    
    # Parser les sections
    lines = generated_text.split('\n')
    interpretation = ""
    recommendations = ""
    image_prompt = ""
    current_section = None
    
    for line in lines:
        line_upper = line.upper()
        if 'GÉOLOGIE' in line_upper or 'GEOLOGIE' in line_upper or '1.' in line:
            current_section = 'interp'
        elif 'ACTIONS' in line_upper or 'RECOMMANDATION' in line_upper or '2.' in line:
            current_section = 'reco'
        elif 'PROMPT' in line_upper or 'IMAGE' in line_upper or '3.' in line:
            current_section = 'prompt'
        elif line.strip() and current_section:
            if current_section == 'interp':
                interpretation += line.strip() + " "
            elif current_section == 'reco':
                recommendations += line.strip() + " "
            elif current_section == 'prompt':
                image_prompt += line.strip() + " "
    
    print("📌 Sections extraites :")
    print(f"   • Interprétation : {len(interpretation)} caractères")
    print(f"   • Recommandations : {len(recommendations)} caractères")
    print(f"   • Prompt image : {len(image_prompt)} caractères")
    print()
    
    if interpretation:
        print("🔬 INTERPRÉTATION :")
        print(interpretation.strip())
        print()
    
    if recommendations:
        print("🎯 RECOMMANDATIONS :")
        print(recommendations.strip())
        print()
    
    if image_prompt:
        print("🖼️  PROMPT IMAGE :")
        print(image_prompt.strip())
        print()
    
else:
    print(f"⚠️  ÉCHEC ou TIMEOUT après {elapsed_gen:.1f}s")
    print()
    print("📋 Génération d'un fallback automatique...")
    
    fallback_interp = f"""Analyse géologique automatique basée sur {n_spectra_display} mesures:
    
• Plage de résistivité : {rho_min:.1f} - {rho_max:.1f} Ω·m (moyenne: {rho_mean:.1f} Ω·m)
• {geophysical_data['n_trajectories']} structures géologiques détectées
• Modèle 3D construit avec {geophysical_data['n_cells']} cellules

Interprétation simplifiée :
{"- Formations argileuses dominantes" if rho_mean < 100 else "- Formations sablo-graveleuses" if rho_mean < 500 else "- Substrat rocheux consolidé"}
"""
    
    print("🔬 FALLBACK GÉNÉRÉ :")
    print(fallback_interp)
    print()

# Résumé final
print()
print("=" * 70)
print("📈 RÉSUMÉ DU TEST")
print("=" * 70)
print(f"⏱️  Temps chargement LLM : {elapsed_load:.1f}s")
print(f"⏱️  Temps génération : {elapsed_gen:.1f}s")
print(f"✅ Statut : {'SUCCÈS' if response else 'TIMEOUT/FALLBACK'}")
print()

if elapsed_gen < 30:
    print("🎉 EXCELLENT ! Génération très rapide (< 30s)")
elif elapsed_gen < 45:
    print("✅ BON ! Génération acceptable (< 45s)")
else:
    print("⚠️  LENT ! Génération > 45s (timeout activé)")

print()
print("=" * 70)
print("✅ Test terminé !")
print("=" * 70)
