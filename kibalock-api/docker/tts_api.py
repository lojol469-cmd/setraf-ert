"""
TTS Service API - Text-to-Speech microservice isolé
NumPy 1.22.0 compatible
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from TTS.api import TTS
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KibaLock TTS Service",
    description="Text-to-Speech microservice with Coqui TTS",
    version="1.0.0"
)

# Initialisation TTS
tts_model = None

@app.on_event("startup")
async def startup_event():
    global tts_model
    try:
        logger.info("Loading TTS model...")
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        logger.info("TTS model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        tts_model = None

class TTSRequest(BaseModel):
    text: str
    language: str = "fr"
    speaker: str = "default"

@app.get("/health")
async def health():
    """Health check endpoint"""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    return {"status": "healthy", "model": "xtts_v2"}

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech from text"""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    try:
        # Génération audio
        audio_buffer = io.BytesIO()
        tts_model.tts_to_file(
            text=request.text,
            language=request.language,
            file_path=audio_buffer
        )
        
        audio_buffer.seek(0)
        
        return Response(
            content=audio_buffer.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service": "KibaLock TTS",
        "status": "running",
        "endpoints": ["/health", "/synthesize", "/docs"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
