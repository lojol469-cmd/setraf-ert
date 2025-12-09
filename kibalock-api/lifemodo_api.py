#!/usr/bin/env python3
"""
LifeModo API - Real-time Training Service for KibaLock
Entraînement continu des modèles vocaux et faciaux
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
from TTS.api import TTS
import whisper
from deepface import DeepFace
import cv2

# Configuration
BASE_DIR = Path.home() / "lifemodo_api"
MODELS_DIR = BASE_DIR / "models"
TRAINING_DATA_DIR = BASE_DIR / "training_data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

for dir_path in [MODELS_DIR, TRAINING_DATA_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="LifeModo API",
    description="Real-time Training Service for KibaLock Biometric Authentication",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Models
class ModelManager:
    def __init__(self):
        self.whisper_model = None
        self.tts_model = None
        self.phi_model = None
        self.phi_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def load_models(self):
        """Load all AI models"""
        try:
            logger.info("Loading AI models...")
            
            # Whisper for voice embeddings
            logger.info("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base", device=self.device)
            
            # TTS for voice cloning
            logger.info("Loading TTS model (XTTS-v2)...")
            self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            
            # Phi-3.5 for conversational AI
            logger.info("Loading Phi-3.5-mini-instruct...")
            self.phi_tokenizer = AutoProcessor.from_pretrained(
                "microsoft/Phi-3.5-mini-instruct",
                trust_remote_code=True
            )
            self.phi_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3.5-mini-instruct",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info(f"✅ All models loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

model_manager = ModelManager()

# Pydantic Models
class TrainingRequest(BaseModel):
    user_id: str
    training_type: str  # "voice", "face", "both"
    epochs: int = 10
    learning_rate: float = 0.001

class ChatRequest(BaseModel):
    user_id: str
    message: str
    use_voice: bool = False

class VoiceCloneRequest(BaseModel):
    user_id: str
    text: str

# ============ API ENDPOINTS ============

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        await model_manager.load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")

@app.get("/")
async def root():
    return {
        "service": "LifeModo API",
        "status": "online",
        "version": "1.0.0",
        "models_loaded": {
            "whisper": model_manager.whisper_model is not None,
            "tts": model_manager.tts_model is not None,
            "phi": model_manager.phi_model is not None,
        },
        "device": model_manager.device,
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available(),
    }

@app.post("/api/train/voice")
async def train_voice_model(
    user_id: str,
    voice_samples: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Train personalized voice model from samples
    """
    try:
        logger.info(f"Starting voice training for user {user_id}")
        
        # Save voice samples
        user_dir = TRAINING_DATA_DIR / user_id / "voice"
        user_dir.mkdir(parents=True, exist_ok=True)
        
        sample_paths = []
        for idx, sample in enumerate(voice_samples):
            file_path = user_dir / f"sample_{idx}_{datetime.now().timestamp()}.wav"
            with open(file_path, "wb") as f:
                content = await sample.read()
                f.write(content)
            sample_paths.append(str(file_path))
        
        # Extract embeddings with Whisper
        embeddings = []
        transcriptions = []
        
        for path in sample_paths:
            result = model_manager.whisper_model.transcribe(path)
            transcriptions.append(result["text"])
            
            audio = whisper.load_audio(path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model_manager.device)
            
            with torch.no_grad():
                embedding = model_manager.whisper_model.encode(mel.unsqueeze(0))
                embeddings.append(embedding.cpu().numpy())
        
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Save checkpoint
        checkpoint_path = CHECKPOINTS_DIR / f"{user_id}_voice.npy"
        np.save(checkpoint_path, avg_embedding)
        
        logger.info(f"✅ Voice training completed for {user_id}")
        
        return {
            "success": True,
            "user_id": user_id,
            "samples_processed": len(sample_paths),
            "transcriptions": transcriptions,
            "embedding_shape": avg_embedding.shape,
            "checkpoint": str(checkpoint_path),
        }
        
    except Exception as e:
        logger.error(f"Voice training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train/face")
async def train_face_model(
    user_id: str,
    face_images: List[UploadFile] = File(...)
):
    """
    Train personalized face model from images
    """
    try:
        logger.info(f"Starting face training for user {user_id}")
        
        # Save face images
        user_dir = TRAINING_DATA_DIR / user_id / "face"
        user_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        for idx, image in enumerate(face_images):
            file_path = user_dir / f"image_{idx}_{datetime.now().timestamp()}.jpg"
            with open(file_path, "wb") as f:
                content = await image.read()
                f.write(content)
            image_paths.append(str(file_path))
        
        # Extract embeddings with DeepFace
        embeddings = []
        
        for path in image_paths:
            embedding_objs = DeepFace.represent(
                img_path=path,
                model_name="Facenet512",
                enforce_detection=True,
                detector_backend='opencv'
            )
            
            if embedding_objs:
                embedding = np.array(embedding_objs[0]['embedding'])
                embeddings.append(embedding)
        
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        # Save checkpoint
        checkpoint_path = CHECKPOINTS_DIR / f"{user_id}_face.npy"
        np.save(checkpoint_path, avg_embedding)
        
        logger.info(f"✅ Face training completed for {user_id}")
        
        return {
            "success": True,
            "user_id": user_id,
            "images_processed": len(image_paths),
            "embedding_shape": avg_embedding.shape,
            "checkpoint": str(checkpoint_path),
        }
        
    except Exception as e:
        logger.error(f"Face training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """
    Chat with Phi-3.5 AI
    """
    try:
        prompt = f"<|user|>\n{request.message}<|end|>\n<|assistant|>\n"
        
        inputs = model_manager.phi_tokenizer(
            prompt, 
            return_tensors="pt"
        ).to(model_manager.device)
        
        with torch.no_grad():
            outputs = model_manager.phi_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        response = model_manager.phi_tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # Extract only the assistant's response
        assistant_response = response.split("<|assistant|>")[-1].strip()
        
        return {
            "success": True,
            "user_id": request.user_id,
            "message": request.message,
            "response": assistant_response,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/clone")
async def clone_voice(request: VoiceCloneRequest):
    """
    Generate speech using user's cloned voice
    """
    try:
        # Load user's voice checkpoint
        checkpoint_path = CHECKPOINTS_DIR / f"{request.user_id}_voice.npy"
        
        if not checkpoint_path.exists():
            raise HTTPException(404, "Voice model not found. Please train first.")
        
        # Generate speech with TTS
        output_path = TRAINING_DATA_DIR / f"{request.user_id}_speech_{datetime.now().timestamp()}.wav"
        
        model_manager.tts_model.tts_to_file(
            text=request.text,
            file_path=str(output_path),
            speaker_wav=str(TRAINING_DATA_DIR / request.user_id / "voice" / "sample_0_*.wav"),
            language="fr"
        )
        
        return {
            "success": True,
            "user_id": request.user_id,
            "text": request.text,
            "audio_file": str(output_path),
        }
        
    except Exception as e:
        logger.error(f"Voice cloning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update/embedding")
async def update_embedding(
    user_id: str,
    embedding_type: str,  # "voice" or "face"
    new_sample: UploadFile = File(...)
):
    """
    Update user embedding incrementally (online learning)
    """
    try:
        # Load existing checkpoint
        checkpoint_path = CHECKPOINTS_DIR / f"{user_id}_{embedding_type}.npy"
        
        if not checkpoint_path.exists():
            raise HTTPException(404, f"{embedding_type} model not found")
        
        existing_embedding = np.load(checkpoint_path)
        
        # Process new sample
        temp_path = TRAINING_DATA_DIR / f"temp_{datetime.now().timestamp()}"
        with open(temp_path, "wb") as f:
            content = await new_sample.read()
            f.write(content)
        
        # Extract new embedding
        if embedding_type == "voice":
            audio = whisper.load_audio(str(temp_path))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model_manager.device)
            
            with torch.no_grad():
                new_embedding = model_manager.whisper_model.encode(mel.unsqueeze(0))
                new_embedding = new_embedding.cpu().numpy()
        else:  # face
            embedding_objs = DeepFace.represent(
                img_path=str(temp_path),
                model_name="Facenet512",
                enforce_detection=True
            )
            new_embedding = np.array(embedding_objs[0]['embedding'])
        
        # Update with weighted average (90% old, 10% new)
        updated_embedding = 0.9 * existing_embedding + 0.1 * new_embedding
        
        # Normalize
        updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
        
        # Save updated checkpoint
        np.save(checkpoint_path, updated_embedding)
        
        # Cleanup
        os.remove(temp_path)
        
        logger.info(f"✅ Updated {embedding_type} embedding for {user_id}")
        
        return {
            "success": True,
            "user_id": user_id,
            "embedding_type": embedding_type,
            "update_method": "weighted_average",
            "weights": {"old": 0.9, "new": 0.1},
        }
        
    except Exception as e:
        logger.error(f"Embedding update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/{user_id}/stats")
async def get_user_training_stats(user_id: str):
    """
    Get training statistics for a user
    """
    try:
        voice_checkpoint = CHECKPOINTS_DIR / f"{user_id}_voice.npy"
        face_checkpoint = CHECKPOINTS_DIR / f"{user_id}_face.npy"
        
        voice_dir = TRAINING_DATA_DIR / user_id / "voice"
        face_dir = TRAINING_DATA_DIR / user_id / "face"
        
        stats = {
            "user_id": user_id,
            "voice_model": {
                "trained": voice_checkpoint.exists(),
                "samples_count": len(list(voice_dir.glob("*.wav"))) if voice_dir.exists() else 0,
                "checkpoint_size": voice_checkpoint.stat().st_size if voice_checkpoint.exists() else 0,
            },
            "face_model": {
                "trained": face_checkpoint.exists(),
                "images_count": len(list(face_dir.glob("*.jpg"))) if face_dir.exists() else 0,
                "checkpoint_size": face_checkpoint.stat().st_size if face_checkpoint.exists() else 0,
            },
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LifeModo API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║            🧠 LifeModo API - Real-time Training               ║
║                    for KibaLock Biometrics                     ║
╚════════════════════════════════════════════════════════════════╝

🚀 Starting server...
📍 Host: {args.host}
🔌 Port: {args.port}
🔄 Reload: {args.reload}
💾 Base directory: {BASE_DIR}
🤖 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}

Models:
  • Whisper (base) - Voice embeddings
  • XTTS-v2 - Voice cloning
  • Phi-3.5-mini-instruct - Conversational AI
  • FaceNet512 - Face embeddings

API Docs: http://{args.host}:{args.port}/docs
    """)
    
    uvicorn.run(
        "lifemodo_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
