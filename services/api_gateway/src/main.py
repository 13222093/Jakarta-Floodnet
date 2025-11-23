import sys
import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional

# --- FIX PATH UNTUK LOCALHOST ---
# Tambahkan current working directory ke path agar bisa import src.ml_core
sys.path.append(os.getcwd())

# Imports dari Shared Library
try:
    from src.ml_core.lstm_model import FloodLevelLSTM
    from src.ml_core.yolo_model import FloodVisualVerifier
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("⚠️ Pastikan kamu menjalankan perintah dari ROOT FOLDER (Jakarta-Floodnet/)")
    sys.exit(1)

app = FastAPI(
    title="Jakarta FloodNet API (Local)",
    description="AI-Powered Early Warning System",
    version="2.0.Local"
)

# ===== GLOBAL STATE =====
lstm_service: Optional[FloodLevelLSTM] = None
yolo_service: Optional[FloodVisualVerifier] = None

# ===== CONFIG PATHS (Local Default) =====
# Menggunakan folder lokal ./models
MODEL_DIR = os.getenv("MODEL_PATH", "models")
LSTM_PATH = os.path.join(MODEL_DIR, "best_model_modular.h5")
YOLO_PATH = os.getenv("YOLO_PATH", "models/yolov8n.pt")

@app.on_event("startup")
async def load_models():
    """Load models saat server nyala"""
    global lstm_service, yolo_service
    print(f"🚀 API Starting on Localhost... Model Dir: {MODEL_DIR}")

    # 1. Load LSTM
    try:
        lstm_service = FloodLevelLSTM()
        if os.path.exists(LSTM_PATH):
            success = lstm_service.load_model(LSTM_PATH)
            if success:
                print(f"✅ LSTM Model loaded: {LSTM_PATH}")
            else:
                print("⚠️ LSTM load failed internally.")
        else:
            print(f"⚠️ LSTM Model not found at {LSTM_PATH}. Run training first!")
    except Exception as e:
        print(f"❌ LSTM Error: {e}")

    # 2. Load YOLO
    try:
        # Pastikan folder models ada untuk download yolo
        os.makedirs(MODEL_DIR, exist_ok=True)
        yolo_service = FloodVisualVerifier(model_path=YOLO_PATH)
        yolo_service.load_model()
        print(f"✅ YOLO Service ready")
    except Exception as e:
        print(f"❌ YOLO Error: {e}")

# ===== DATA MODELS & ENDPOINTS =====

class FloodPredictionInput(BaseModel):
    rainfall_mm: float
    water_level_cm: float
    
class PredictionResponse(BaseModel):
    status: str
    prediction_cm: float
    risk_level: str
    alert_message: str

@app.get("/")
def root():
    return {"message": "Jakarta FloodNet API (Localhost) 🟢"}

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "models": {
            "lstm_ready": lstm_service.is_trained if lstm_service else False,
            "yolo_ready": yolo_service.is_loaded if yolo_service else False
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_flood(data: FloodPredictionInput):
    if not lstm_service or not lstm_service.is_trained:
        # Fallback Logic
        dummy_pred = data.water_level_cm + (data.rainfall_mm * 0.5)
        return {
            "status": "fallback_mode",
            "prediction_cm": round(dummy_pred, 1),
            "risk_level": "UNKNOWN",
            "alert_message": "Model belum siap/dilatih."
        }

    try:
        features = np.array([[data.rainfall_mm, data.rainfall_mm, data.water_level_cm]])
        pred_array = lstm_service.predict(features)
        pred_cm = float(pred_array[0])
        
        risk = "AMAN"
        msg = "Kondisi normal."
        if pred_cm > 150:
            risk = "BAHAYA"
            msg = "⚠️ POTENSI BANJIR! Segera evakuasi."
        elif pred_cm > 100:
            risk = "SIAGA"
            msg = "Waspada kenaikan muka air."
            
        return {
            "status": "success",
            "prediction_cm": round(pred_cm, 2),
            "risk_level": risk,
            "alert_message": msg
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-visual")
async def verify_visual(file: UploadFile = File(...)):
    if not yolo_service:
        raise HTTPException(status_code=503, detail="Visual service not available")
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        return yolo_service.detect_flood_features(img)
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)