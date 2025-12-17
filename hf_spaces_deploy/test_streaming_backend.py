#!/usr/bin/env python3
"""
Test backend Flask avec streaming SSE (Server-Sent Events)
Pour voir le streaming token par token en temps réel
"""

from flask import Flask, Response, stream_with_context
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

app = Flask(__name__)

# Variables globales pour le modèle
model = None
tokenizer = None

def load_model():
    """Charge Phi-3-mini au démarrage"""
    global model, tokenizer
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
    
    print(f"✅ Modèle chargé en {time.time() - start:.1f}s")

@app.route('/stream')
def stream():
    """Endpoint de streaming SSE"""
    
    def generate():
        # Prompt de test
        context = "Données: 45 Ω·m (min:12, max:157). Type: argiles/marnes saturées. Interprétation:"
        
        # Signal de début
        yield f"data: [START] Génération...\n\n"
        
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
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Streamer les tokens
        token_count = 0
        start_gen = time.time()
        for new_text in streamer:
            token_count += 1
            yield f"data: {new_text}\n\n"
        
        thread.join()
        gen_time = time.time() - start_gen
        
        # Signal de fin avec stats
        yield f"data: [END] {token_count} tokens en {gen_time:.1f}s ({token_count/gen_time:.2f} tokens/s)\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/')
def index():
    """Page HTML simple pour tester le streaming"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Streaming Phi-3</title>
        <style>
            body { font-family: monospace; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
            #output { background: #252526; padding: 15px; border-radius: 5px; min-height: 200px; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
            .token { display: inline; }
            .cursor { animation: blink 1s infinite; }
            @keyframes blink { 50% { opacity: 0; } }
        </style>
    </head>
    <body>
        <h1>🤖 Test Streaming Phi-3-mini</h1>
        <button onclick="startStream()">▶️ Lancer Génération</button>
        <button onclick="clearOutput()">🗑️ Effacer</button>
        <h3>Sortie en temps réel:</h3>
        <div id="output"><span class="cursor">▌</span></div>
        
        <script>
            const output = document.getElementById('output');
            
            function startStream() {
                output.innerHTML = '<span style="color: #4EC9B0;">⏳ Génération en cours...</span><span class="cursor">▌</span>';
                
                const eventSource = new EventSource('/stream');
                
                eventSource.onmessage = function(e) {
                    const data = e.data;
                    
                    if (data.startsWith('[START]')) {
                        output.innerHTML = '';
                    } else if (data.startsWith('[END]')) {
                        output.innerHTML += '<br><br><span style="color: #6A9955;">✅ ' + data.substring(6) + '</span>';
                        eventSource.close();
                    } else {
                        // Afficher le token avec cursor
                        const cursor = output.querySelector('.cursor');
                        if (cursor) cursor.remove();
                        output.innerHTML += '<span class="token">' + data + '</span><span class="cursor">▌</span>';
                        window.scrollTo(0, document.body.scrollHeight);
                    }
                };
                
                eventSource.onerror = function(e) {
                    output.innerHTML += '<br><span style="color: #f48771;">❌ Erreur de connexion</span>';
                    eventSource.close();
                };
            }
            
            function clearOutput() {
                output.innerHTML = '<span class="cursor">▌</span>';
            }
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    load_model()
    print("\n🌐 Serveur Flask démarré!")
    print("📍 Ouvrez: http://localhost:5001")
    print("🎯 Cliquez sur 'Lancer Génération' pour voir le streaming\n")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
