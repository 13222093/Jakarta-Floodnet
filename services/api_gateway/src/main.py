import sys
import os
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List
try:
    from scenarios import get_scenario, list_scenarios
except ImportError:
    from services.api_gateway.src.scenarios import get_scenario, list_scenarios

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JakartaFloodNet")

# --- PATH HACKS (Hackathon Standard Procedure) ---
# Ensure we can import from src/ml_core regardless of where we run this
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.getcwd()) # Extra safety for local run

# --- APP INITIALIZATION ---
app = FastAPI(title="Jakarta FloodNet API Gateway", version="1.0.0 (FINAL)")

# --- GLOBAL STATE (GOD MODE) ---
class DemoState:
    active: bool = False
    scenario: str = "CRITICAL"

demo_state = DemoState()

# --- MODEL LOADING (REAL AI) ---
lstm_model = None

# CONFIG: Updated Paths based on your fix
MODEL_DIR = os.getenv("MODEL_PATH", "models")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_flood_forecaster.h5")
LSTM_SCALER_BASE_PATH = os.path.join(MODEL_DIR, "lstm") # Will resolve to _scaler_X.pkl internally
YOLO_PATH = os.getenv("YOLO_PATH", "models/yolov8n.pt")

# Load YOLO Service (Preserving existing logic)
yolo_service = None

try:
    # Attempt to load the Real AI
    from src.ml_core.lstm_model import FloodLevelLSTM
    from src.ml_core.yolo_model import FloodVisualVerifier
    
    logger.info(f"Loading LSTM from {LSTM_MODEL_PATH}...")
    lstm_model = FloodLevelLSTM()
    if os.path.exists(LSTM_MODEL_PATH):
        success = lstm_model.load_model(LSTM_MODEL_PATH, LSTM_SCALER_BASE_PATH)
        if success:
            logger.info("✅ REAL AI LOADED: LSTM Neural Network Active.")
        else:
            logger.error("❌ LSTM Load returned False.")
            lstm_model = None
    else:
        logger.warning(f"⚠️ LSTM Model file not found at {LSTM_MODEL_PATH}")

    # Load YOLO
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        yolo_service = FloodVisualVerifier(model_path=YOLO_PATH)
        yolo_service.load_model()
        logger.info("✅ YOLO Service ready")
    except Exception as e:
        logger.error(f"❌ YOLO Error: {e}")

except Exception as e:
    logger.error(f"⚠️ MODEL LOAD FAILED: {e}")
    logger.warning("System running in API-Only mode (Real AI unavailable).")


# --- DATA MODELS ---
class PredictionRequest(BaseModel):
    water_level_cm: float
    rainfall_mm: float

# --- ENDPOINTS ---

@app.get("/")
def root():
    return {"message": "Jakarta FloodNet API (Final Production) 🟢"}

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "mode": "DEMO" if demo_state.active else "LIVE",
        "models": {
            "lstm_ready": lstm_model is not None and lstm_model.is_trained,
            "vision_ready": yolo_service is not None
        }
    }

# 1. GOD MODE TOGGLE
@app.post("/admin/set-demo-mode")
async def set_demo_mode(request: dict):
    """
    MASTER SWITCH for Pitch Demo.
    Accepts {"enable": true, "scenario": "CRITICAL"}
    """
    enable = request.get("enable", False)
    scenario = request.get("scenario", "CRITICAL")
    
    demo_state.active = enable
    demo_state.scenario = scenario
    logger.info(f"🚨 DEMO MODE CHANGED: {enable} ({scenario})")
    return {
        "status": "success", 
        "mode": "DEMO" if enable else "LIVE", 
        "scenario": scenario
    }

@app.get("/admin/demo-status")
async def get_demo_status():
    return {
        "demo_mode_enabled": demo_state.active,
        "scenario": demo_state.scenario
    }

# 2. HYBRID PREDICTION ENGINE (Physics/AI + God Mode)
@app.post("/predict")
async def predict_flood_risk(data: PredictionRequest):
    # A. INTERCEPT: GOD MODE
    if demo_state.active:
        logger.info("⚡ GOD MODE TRIGGERED: Returning CRITICAL prediction.")
        return {
            "status": "demo_mode_active",
            "prediction_cm": 250.0,
            "risk_level": "CRITICAL",
            "alert_message": "🚨 SIAGA 1 - EVAKUASI SEGERA! Ketinggian air mencapai level kritis.",
            "timestamp": "2025-11-25T10:00:00Z"
        }

    # B. REAL AI LOGIC
    if lstm_model and lstm_model.is_trained:
        try:
            # Prediction logic - Adapted for FloodLevelLSTM
            # Input shape: (1, 3) -> [rain, rain, water]
            features = np.array([[data.rainfall_mm, data.rainfall_mm, data.water_level_cm]])
            prediction_array = lstm_model.predict(features)
            prediction = float(prediction_array[0])
            
            # Risk Logic
            risk = "AMAN"
            if prediction > 150: risk = "BAHAYA" # >150 is Danger
            elif prediction > 100: risk = "SIAGA" # >100 is Warning
            
            msg = f"Status: {risk}. Ketinggian diprediksi {round(prediction, 1)} cm."
            if risk == "BAHAYA":
                msg = "⚠️ POTENSI BANJIR! Segera evakuasi."
            
            return {
                "status": "success",
                "prediction_cm": round(prediction, 2),
                "risk_level": risk,
                "alert_message": msg
            }
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            # Fallback if Real AI crashes mid-request
            return {"status": "error", "message": "Computation Error", "prediction_cm": data.water_level_cm}
    
    # Fallback if model not loaded
    dummy_pred = data.water_level_cm + (data.rainfall_mm * 0.5)
    return {
        "status": "fallback_mode",
        "prediction_cm": round(dummy_pred, 1),
        "risk_level": "UNKNOWN",
        "alert_message": "Model belum siap/dilatih."
    }

# 3. VISUAL VERIFICATION (YOLO + God Mode)
@app.post("/verify-visual")
async def verify_visual(file: UploadFile = File(...)):
    # A. INTERCEPT: GOD MODE
    if demo_state.active:
        logger.info("⚡ GOD MODE TRIGGERED: Returning SUSTAINABILITY proof.")
        return {
            "is_flooded": True,
            "flood_probability": 0.99,
            "objects_detected": ["flood", "water"], 
            "timestamp": "2025-11-25T10:00:00Z"
        }

    # B. REAL LOGIC
    if not yolo_service:
         return {"status": "error", "message": "Visual service not available"}
         
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        return yolo_service.detect_flood_features(img)
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 4. SCENARIO SYSTEM (One-Click Demo)
@app.get("/scenarios")
def get_available_scenarios():
    """List all scenarios for Frontend Dropdown"""
    return list_scenarios()

@app.post("/predict/scenario/{scenario_id}")
async def predict_scenario(scenario_id: str):
    """
    Run a specific demo scenario.
    - If God Mode is enabled in scenario -> Force CRITICAL
    - If Normal -> Feed scenario data into Real AI
    """
    scenario = get_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    
    # A. SCENARIO GOD MODE
    if scenario["god_mode_enabled"]:
        logger.info(f"⚡ SCENARIO GOD MODE: {scenario['name']}")
        return {
            "status": "demo_mode_active",
            "prediction_cm": 250.0,
            "risk_level": "CRITICAL",
            "alert_message": "🚨 SIAGA 1 - EVAKUASI SEGERA! Ketinggian air mencapai level kritis.",
            "timestamp": "2025-11-25T10:00:00Z"
        }

    # B. REAL AI WITH SCENARIO DATA
    data = scenario["data"]
    rainfall = data["rainfall_mm"]
    water_level = data["water_level_cm"]
    
    if lstm_model and lstm_model.is_trained:
        try:
            # Input shape: (1, 3) -> [rain, rain, water]
            features = np.array([[rainfall, rainfall, water_level]])
            prediction_array = lstm_model.predict(features)
            prediction = float(prediction_array[0])
            
            # Risk Logic
            risk = "AMAN"
            if prediction > 150: risk = "BAHAYA"
            elif prediction > 100: risk = "SIAGA"
            
            msg = f"Status: {risk}. Ketinggian diprediksi {round(prediction, 1)} cm."
            if risk == "BAHAYA":
                msg = "⚠️ POTENSI BANJIR! Segera evakuasi."
            
            return {
                "status": "success",
                "prediction_cm": round(prediction, 2),
                "risk_level": risk,
                "alert_message": msg,
                "scenario_used": scenario["name"]
            }
        except Exception as e:
            logger.error(f"Scenario Inference Error: {e}")
            return {"status": "error", "message": "Computation Error"}

    # Fallback
    dummy_pred = water_level + (rainfall * 0.5)
    return {
        "status": "fallback_mode",
        "prediction_cm": round(dummy_pred, 1),
        "risk_level": "UNKNOWN",
        "alert_message": "Model belum siap/dilatih.",
        "scenario_used": scenario["name"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
