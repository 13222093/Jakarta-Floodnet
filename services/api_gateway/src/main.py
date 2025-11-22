import sys
import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.ml_core.lstm_model import FloodLevelLSTM
from src.ml_core.yolo_model import FloodVisualVerifier

app = FastAPI(
    title="Jakarta FloodNet API",
    description="AI-Powered Early Warning System",
    version="2.0.0" # Version up!
)

# ===== GLOBAL STATE =====
lstm_service: Optional[FloodLevelLSTM] = None
yolo_service: Optional[FloodVisualVerifier] = None

# Config Paths (Dari Docker Env)
MODEL_DIR = os.getenv("MODEL_PATH", "/app/models")
LSTM_PATH = os.path.join(MODEL_DIR, "best_model_modular.h5")
YOLO_PATH = os.getenv("YOLO_PATH", "yolov8n.pt")

@app.on_event("startup")
async def load_models():
    """Load models saat server nyala (Cold Start)"""
    global lstm_service, yolo_service
    print("🚀 API Starting... Loading Models...")

    # 1. Load LSTM
    try:
        lstm_service = FloodLevelLSTM()
        # Cek apakah file model ada (Mungkin belum ditraining)
        if os.path.exists(LSTM_PATH):
            success = lstm_service.load_model(LSTM_PATH)
            if success:
                print(f"✅ LSTM Model loaded: {LSTM_PATH}")
            else:
                print("⚠️ LSTM load failed internally.")
        else:
            print(f"⚠️ LSTM Model not found at {LSTM_PATH}. Waiting for training worker...")
    except Exception as e:
        print(f"❌ LSTM Error: {e}")

    # 2. Load YOLO
    try:
        yolo_service = FloodVisualVerifier(model_path=YOLO_PATH)
        yolo_service.load_model() # Auto fallback ke yolov8n kalau custom gak ada
        print(f"✅ YOLO Service ready (Hybrid Mode)")
    except Exception as e:
        print(f"❌ YOLO Error: {e}")

# ===== DATA MODELS =====

class FloodPredictionInput(BaseModel):
    rainfall_mm: float      # Curah hujan rata-rata
    water_level_cm: float   # TMA saat ini
    
class PredictionResponse(BaseModel):
    status: str
    prediction_cm: float
    risk_level: str         # AMAN / SIAGA / BAHAYA
    alert_message: str

# ===== ENDPOINTS =====

@app.get("/")
def root():
    return {"message": "Jakarta FloodNet API is Online 🟢"}

@app.get("/health")
def health_check():
    """Cek status model untuk Dashboard"""
    return {
        "status": "active",
        "models": {
            "lstm_ready": lstm_service.is_trained if lstm_service else False,
            "yolo_ready": yolo_service.is_loaded if yolo_service else False
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_flood(data: FloodPredictionInput):
    """
    Endpoint Prediksi Banjir (LSTM).
    Menerima data sensor -> Output prediksi TMA 1 jam ke depan.
    """
    # Default Fallback (Jika model belum ready)
    if not lstm_service or not lstm_service.is_trained:
        # Simple heuristic buat demo kalau model mati
        dummy_pred = data.water_level_cm + (data.rainfall_mm * 0.5)
        return {
            "status": "fallback_mode",
            "prediction_cm": round(dummy_pred, 1),
            "risk_level": "UNKNOWN",
            "alert_message": "Model belum siap. Menggunakan estimasi kasar."
        }

    try:
        # Prepare Input [hujan_bogor, hujan_jakarta, tma_manggarai]
        # Simplifikasi: Kita asumsi input rainfall_mm mewakili kedua wilayah
        # Shape (1, 3) karena model mengharapkan 3 fitur
        features = np.array([[data.rainfall_mm, data.rainfall_mm, data.water_level_cm]])
        
        # PREDICT (Class LSTM akan handle scaling & sequence creation otomatis!)
        pred_array = lstm_service.predict(features)
        pred_cm = float(pred_array[0])
        
        # Risk Logic
        risk = "AMAN"
        msg = "Kondisi normal."
        
        if pred_cm > 150: # Threshold Siaga 1
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
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/verify-visual")
async def verify_visual(file: UploadFile = File(...)):
    """
    Endpoint Verifikasi Visual (YOLO).
    Menerima Gambar -> Output Probabilitas Banjir.
    """
    if not yolo_service:
        raise HTTPException(status_code=503, detail="Visual service not available")
        
    try:
        # Read Image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # DETECT (Panggil Shared Library)
        result = yolo_service.detect_flood_features(img)
        
        return result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)