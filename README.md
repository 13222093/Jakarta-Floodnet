# 🌊 Jakarta FloodNet - Backend API

Real-time flood early warning system for BPBD Jakarta.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation
```bash
pip install -r requirements.txt
```

### Run API
```bash
uvicorn main:app --reload
```
API will be available at: `http://localhost:8000`

### API Documentation
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 📚 Endpoints

### 1. GET /health
Health check endpoint.
**Response:**
```json
{
  "status": "ok",
  "message": "Jakarta FloodNet API is running",
  "timestamp": "2025-11-21T22:00:00"
}
```
### 2. POST /predict
Predict flooding probability (24-hour forecast).
**Input Data (SIMPLIFIED - MVP):**
- `rainfall_mm`: Curah hujan dalam mm (dari BMKG/sensor)
- `water_level_cm`: Ketinggian Muka Air dalam cm (dari TMA sensor)
- `location_id`: ID lokasi monitoring (ex: MANGGARAI_01, KRUKUT_02)
**Request:**
```json
{
  "rainfall_mm": 15.5,
  "water_level_cm": 150,
  "location_id": "MANGGARAI_01"
}
```
**Response:**
```json
{
  "statusCode": 200,
  "flooding_probability": 0.78,
  "risk_level": "HIGH",
  "confidence": 0.92,
  "recommendation": "Initiate evacuation protocol",
  "timestamp": "2025-11-21T22:00:00"
}
```
**Risk Levels:**
- `HIGH` (> 70%): Segera lakukan evakuasi
- `MEDIUM` (40-70%): Siapkan protokol evakuasi
- `LOW` (< 40%): Lanjutkan monitoring

### 3. POST /detect
Detect flooded areas from satellite/CCTV image.
**Request:**
```json
{
  "image_url": "https://example.com/cctv_manggarai.jpg",
  "confidence_threshold": 0.5
}
```
**Response:**
```json
{
  "statusCode": 200,
  "detections": [
    {
      "id": 1,
      "bbox": [100, 150, 200, 250],
      "confidence": 0.82,
      "class": "flooded_area",
      "area_m2": 245
    }
  ],
  "total_flooded_area_m2": 245,
  "detection_count": 1,
  "timestamp": "2025-11-21T22:00:00"
}
```
### 4. GET /metrics
Get model performance metrics.
**Response:**
```json
{
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
  "timestamp": "2025-11-21T22:00:00"
}
```

## 🔄 Data Sources

### Current MVP (Simple)
- **Rainfall**: BMKG API atau sensor lokal
- **Water Level (TMA)**: Sensor TMA Manggarai, Krukut, dll
- **Location ID**: Predefined monitoring points

### Future (Production)
- Satellite imagery integration
- Multiple sensor networks
- Real-time data streaming

## 🔗 Integration Notes
- **LSTM Model (Classification):** Will be loaded in `/predict` endpoint (Hari 2-3)
  - Input: rainfall + water level
  - Output: flooding probability
- **YOLOv8 Model (Detection):** Will be loaded in `/detect` endpoint (Hari 2-3)
  - Input: satellite/CCTV image
  - Output: detected flooded areas
- **Dashboard:** Will consume these endpoints (STI team)

## 📝 Development Timeline
- **Hari 1:** API skeleton ✅ (DONE)
- **Hari 2-3:** Model integration
- **Hari 4:** Testing & optimization
- **Hari 5:** Demo & presentation

## 👥 Team
| Role | Person | Responsibility |
|------|--------|-----------------|
| API Backend | Ari Azis (EE) | `/predict`, `/detect`, integration |
| ML Models | Informatika | LSTM, YOLOv8 training |
| Frontend | STI | Dashboard, visualization |
| Strategy | SBM | Pitch, BPBD relations |

## 📋 Required Libraries
See `requirements.txt` for complete list.
Key dependencies:
- FastAPI: Web framework
- Uvicorn: ASGI server
- Pydantic: Data validation
- TensorFlow: LSTM model loading (Hari 2-3)
- Ultralytics: YOLOv8 model loading (Hari 2-3)

---
**Last Updated:** Nov 21, 2025