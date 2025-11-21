from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import numpy as np
import cv2
from ultralytics import YOLO
import os

app = FastAPI(
    title="Jakarta FloodNet API",
    description="Real-time flood early warning system",
    version="0.1.0"
)

# ===== GLOBAL MODEL LOADING =====
# Load model at startup to avoid latency per request
model = None
MODEL_PATH_BEST = "models/best.pt"
MODEL_PATH_FALLBACK = "yolov8n.pt" # Standard model

try:
    if os.path.exists(MODEL_PATH_BEST):
        print(f"🚀 Loading custom YOLOv8 model: {MODEL_PATH_BEST}")
        model = YOLO(MODEL_PATH_BEST)
    else:
        print(f"⚠️ Custom model not found. Loading fallback: {MODEL_PATH_FALLBACK}")
        model = YOLO(MODEL_PATH_FALLBACK)
    print("✅ YOLOv8 Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLOv8 model: {e}")

# ===== PYDANTIC MODELS (Data Validation) =====

class FloodInput(BaseModel):
    """Input untuk prediksi banjir - SIMPLIFIED VERSION"""
    rainfall_mm: float  # Curah hujan dalam mm
    water_level_cm: float  # Ketinggian Muka Air dalam cm
    location_id: str  # ID lokasi (ex: MANGGARAI_01)
    class Config:
        schema_extra = {
            "example": {
                "rainfall_mm": 15.5,
                "water_level_cm": 150,
                "location_id": "MANGGARAI_01"
            }
        }

class DetectInput(BaseModel):
    """Input untuk deteksi area banjir"""
    image_url: str
    confidence_threshold: Optional[float] = 0.5
    class Config:
        schema_extra = {
            "example": {
                "image_url": "https://example.com/cctv_manggarai.jpg",
                "confidence_threshold": 0.5
            }
        }

class PredictResponse(BaseModel):
    """Response dari endpoint /predict"""
    statusCode: int
    flooding_probability: float
    risk_level: str
    confidence: float
    recommendation: str
    timestamp: str

class DetectResponse(BaseModel):
    """Response dari endpoint /detect"""
    statusCode: int
    detections: List[dict]
    total_flooded_area_m2: float
    detection_count: int
    timestamp: str

class VisualVerifyResponse(BaseModel):
    """Response dari endpoint /verify-visual"""
    is_flood_verified: bool
    confidence: float
    detected_objects: List[str]
    object_count: int

# ===== ENDPOINTS =====

@app.get("/health")
async def health():
    """Check if API is running"""
    return {
        "status": "ok",
        "message": "Jakarta FloodNet API is running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(data: FloodInput):
    """
    Predict flooding probability (24-hour forecast)
    ### Input:
    - `rainfall_mm` (float): Curah hujan dalam mm
    - `water_level_cm` (float): Ketinggian Muka Air dalam cm
    - `location_id` (str): ID lokasi monitoring
    ### Output:
    - `flooding_probability`: Probabilitas banjir (0-1)
    - `risk_level`: HIGH / MEDIUM / LOW
    - `recommendation`: Rekomendasi aksi
    """
    # Dummy logic: semakin tinggi water_level + rainfall → higher probability
    probability = min((data.water_level_cm / 200 + data.rainfall_mm / 100) / 2, 1.0)
    confidence = 0.88
    # Determine risk level
    if probability > 0.70:
        risk_level = "HIGH"
        recommendation = "Initiate evacuation protocol"
    elif probability > 0.40:
        risk_level = "MEDIUM"
        recommendation = "Prepare evacuation readiness"
    else:
        risk_level = "LOW"
        recommendation = "Continue monitoring"
    return {
        "statusCode": 200,
        "flooding_probability": round(probability, 3),
        "risk_level": risk_level,
        "confidence": round(confidence, 3),
        "recommendation": recommendation,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect", response_model=DetectResponse)
async def detect(data: DetectInput):
    """
    Detect flooded areas from satellite/CCTV image
    ### Input:
    - `image_url` (str): URL of the image to analyze
    - `confidence_threshold` (float): Minimum confidence score (default: 0.5)
    ### Output:
    - `detections`: List of detected flooded areas
    - `total_flooded_area_m2`: Total area of flooding
    """
    # Dummy detection
    detections = [
        {
            "id": 1,
            "bbox": [100, 150, 200, 250],
            "confidence": 0.82,
            "class": "flooded_area",
            "area_m2": 245
        }
    ]
    total_area = sum([d['area_m2'] for d in detections])
    return {
        "statusCode": 200,
        "detections": detections,
        "total_flooded_area_m2": total_area,
        "detection_count": len(detections),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/verify-visual", response_model=VisualVerifyResponse)
async def verify_visual(file: UploadFile = File(...)):
    """
    Verify flood visually using YOLOv8
    ### Input:
    - `file`: Image file (UploadFile)
    ### Output:
    - `is_flood_verified`: Boolean
    - `confidence`: Confidence score
    - `detected_objects`: List of detected object classes
    """
    if model is None:
        return {
            "is_flood_verified": False,
            "confidence": 0.0,
            "detected_objects": ["Model not loaded"],
            "object_count": 0
        }

    try:
        # 1. Read image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Inference
        results = model(img)
        
        # 3. Process results
        detected_objects = []
        max_conf = 0.0
        
        # Check for flood-related classes (or just any objects for now)
        # If using standard YOLOv8n, we might see 'person', 'car', 'bus', etc.
        # If using custom model, we look for 'flood', 'water', etc.
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                
                detected_objects.append(class_name)
                if conf > max_conf:
                    max_conf = conf

        # Logic: If we detect ANY object, we assume verification (mock logic)
        # In real scenario: if 'flood' in detected_objects
        is_verified = len(detected_objects) > 0
        
        return {
            "is_flood_verified": is_verified,
            "confidence": round(max_conf, 3),
            "detected_objects": list(set(detected_objects)), # Unique classes
            "object_count": len(detected_objects)
        }

    except Exception as e:
        print(f"❌ Error in visual verification: {e}")
        return {
            "is_flood_verified": False,
            "confidence": 0.0,
            "detected_objects": [f"Error: {str(e)}"],
            "object_count": 0
        }

@app.get("/metrics")
async def metrics():
    """Return model performance metrics"""
    return {
        "statusCode": 200,
        "models": {
            "classification": {
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.89,
                "f1_score": 0.87
            },
            "detection": {
                "precision": 0.82,
                "recall": 0.78,
                "map50": 0.80
            }
        },
        "system": {
            "uptime_seconds": 0,
            "requests_processed": 0,
            "average_response_time_ms": 0
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Jakarta FloodNet API",
        "docs": "http://localhost:8000/docs",
        "redoc": "http://localhost:8000/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Jakarta FloodNet API...")
    print("📍 Server running at http://localhost:8000")
    print("📚 API Documentation (Swagger UI): http://localhost:8000/docs")
    print("📚 Alternative Docs (ReDoc): http://localhost:8000/redoc")
    uvicorn.run(app, host="0.0.0.0", port=8000)
