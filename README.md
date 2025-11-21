# Jakarta-Floodnet
Real-time flood prediction using LSTM + YOLOv8 untuk Jakarta

## 🎯 Problem
- BPBD butuh 15-30 menit untuk verifikasi visual sebelum evakuasi
- Kami mempercepat decision loop dengan fusion data TMA + visual detection

## 💡 Solution
- **LSTM Prediction:** Prediksi ketinggian air 1-3 jam ke depan
- **YOLOv8 Detection:** Real-time flood area detection dari CCTV
- **API Integration:** `/predict` & `/detect` endpoints
- **Dashboard:** Streamlit visualization

## 📊 Data Structure

### `floodgauges_dummy.json`
```json
{
  "result": [
    {
      "nama": "Kali Sunter - Manggarai",
      "readings": [
        {
          "timestamp": "2025-11-21T10:30:00",
          "tma_tinggi": 0.85,
          "status": "normal"
        }
      ]
    }
  ]
}
```

## 👥 Team
1. **Ari Azis** (EE) - Strategy & API
2. **SBM Student** - Pitch & Stakeholder
3. **STI Student** - Frontend/Dashboard
4. **Informatika Student** - ML/CV

## 📅 Timeline
- **Hari 1-2:** Data prep + LSTM training
- **Hari 3-4:** YOLOv8 + API integration
- **Hari 5:** Dashboard + presentation

## 🚀 Quick Start
```bash
# Generate dummy data
python scripts/generate_dummy_data.py

# Start API
python api/app.py

# Start dashboard (later)
streamlit run dashboard.py
```

## 📝 License
MIT