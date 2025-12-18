"""
SETRAF API - Subaquifère ERT Analysis Tool
FastAPI Backend pour analyse ERT programmatique
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import io
import os
import tempfile
import json
from datetime import datetime
import hashlib

# =====================================================
# Configuration
# =====================================================

app = FastAPI(
    title="SETRAF API",
    description="💧 API pour l'analyse géophysique ERT (Electrical Resistivity Tomography)",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Modèles de données
# =====================================================

class AnalysisRequest(BaseModel):
    survey_points: List[float]
    depths: List[float]
    resistivities: List[float]
    project_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    status: str
    analysis_id: str
    timestamp: str
    statistics: Dict
    classifications: Dict
    message: str

class StatusResponse(BaseModel):
    status: str
    version: str
    uptime: str
    endpoints: int

# =====================================================
# Fonctions d'analyse (reprises d'ERTest.py)
# =====================================================

def classify_material_by_resistivity(rho: float) -> tuple:
    """Classifier un matériau selon sa résistivité"""
    if rho < 1:
        return '💎 Minéraux métalliques', 0.001, 1, '#FFD700'
    elif rho < 10:
        return '💧 Eaux salées + Argiles', 1, 10, '#FF4500'
    elif rho < 50:
        return '🧱 Argiles compactes', 10, 50, '#8B4513'
    elif rho < 200:
        return '💧 Eaux douces + Sols', 50, 200, '#90EE90'
    elif rho < 1000:
        return '🏖️ Sables + Graviers', 200, 1000, '#F4A460'
    elif rho < 5000:
        return '🏔️ Roches sédimentaires', 1000, 5000, '#87CEEB'
    elif rho < 100000:
        return '🌋 Roches ignées', 5000, 100000, '#FFB6C1'
    else:
        return '💎 Quartzite', 100000, 1000000, '#E0E0E0'

def analyze_ert_data(df: pd.DataFrame) -> dict:
    """Analyser les données ERT"""
    
    # Statistiques de base
    stats = {
        "total_measurements": len(df),
        "survey_points": int(df['survey-point'].nunique()),
        "depth_range": {
            "min": float(df['depth'].min()),
            "max": float(df['depth'].max()),
            "mean": float(df['depth'].mean())
        },
        "resistivity_range": {
            "min": float(df['data'].min()),
            "max": float(df['data'].max()),
            "mean": float(df['data'].mean()),
            "median": float(df['data'].median())
        }
    }
    
    # Classification des matériaux
    classifications = {}
    for rho in df['data']:
        material, rho_min, rho_max, color = classify_material_by_resistivity(rho)
        if material not in classifications:
            classifications[material] = {
                "count": 0,
                "resistivity_range": f"{rho_min}-{rho_max} Ω·m",
                "color": color,
                "percentage": 0.0
            }
        classifications[material]["count"] += 1
    
    # Calculer les pourcentages
    total = len(df)
    for material in classifications:
        classifications[material]["percentage"] = round(
            (classifications[material]["count"] / total) * 100, 2
        )
    
    return {
        "statistics": stats,
        "classifications": classifications
    }

# =====================================================
# Endpoints API
# =====================================================

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "💧 SETRAF API - Subaquifère ERT Analysis Tool",
        "version": "1.0.0",
        "docs": "/api/docs",
        "status": "operational",
        "endpoints": [
            "GET /api/status",
            "POST /api/upload",
            "POST /api/analyze",
            "GET /api/results/{analysis_id}",
            "POST /api/generate-pdf"
        ]
    }

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Obtenir le statut de l'API"""
    return StatusResponse(
        status="operational",
        version="1.0.0",
        uptime="Active",
        endpoints=5
    )

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload un fichier .dat pour analyse"""
    
    # Vérifier l'extension
    if not file.filename.endswith('.dat'):
        raise HTTPException(
            status_code=400,
            detail="Format de fichier non supporté. Utilisez un fichier .dat"
        )
    
    try:
        # Lire le contenu
        content = await file.read()
        
        # Décoder avec détection d'encodage
        try:
            df = pd.read_csv(
                io.BytesIO(content),
                delim_whitespace=True,
                names=['survey-point', 'depth', 'data', 'project']
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Erreur de lecture du fichier: {str(e)}"
            )
        
        # Générer un ID d'analyse
        analysis_id = hashlib.md5(content).hexdigest()[:16]
        
        # Analyser les données
        results = analyze_ert_data(df)
        
        return {
            "status": "success",
            "analysis_id": analysis_id,
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "preview": {
                "rows": len(df),
                "columns": list(df.columns),
                "sample": df.head(5).to_dict('records')
            },
            "analysis": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur serveur: {str(e)}"
        )

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """Analyser des données ERT fournies directement"""
    
    try:
        # Créer un DataFrame
        df = pd.DataFrame({
            'survey-point': request.survey_points,
            'depth': request.depths,
            'data': request.resistivities
        })
        
        # Générer un ID
        data_str = f"{request.survey_points}{request.depths}{request.resistivities}"
        analysis_id = hashlib.md5(data_str.encode()).hexdigest()[:16]
        
        # Analyser
        results = analyze_ert_data(df)
        
        return AnalysisResponse(
            status="success",
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            statistics=results["statistics"],
            classifications=results["classifications"],
            message="Analyse complétée avec succès"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur d'analyse: {str(e)}"
        )

@app.get("/api/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Récupérer les résultats d'une analyse"""
    # Note: Dans une vraie implémentation, stocker les résultats en base de données
    return {
        "status": "success",
        "analysis_id": analysis_id,
        "message": "Les résultats sont disponibles. Implémentation complète à venir."
    }

@app.post("/api/generate-pdf")
async def generate_pdf():
    """Générer un rapport PDF"""
    raise HTTPException(
        status_code=501,
        detail="Génération PDF à implémenter. Utilisez l'interface Streamlit pour le moment."
    )

# =====================================================
# Lancement
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_setraf:app",
        host="0.0.0.0",
        port=8505,
        reload=True,
        log_level="info"
    )
