import requests
import json

URL = "http://localhost:8000/predict"

def test(rainfall, water_level):
    print(f"Testing with Rainfall={rainfall}, WaterLevel={water_level}")
    payload = {
        "rainfall_mm": rainfall,
        "water_level_cm": water_level,
        "location_id": "MANGGARAI_01"
    }
    try:
        response = requests.post(URL, json=payload)
        
        # Cek kalau server error (bukan 200 OK)
        if response.status_code != 200:
            print(f"  -> Server Error: {response.text}")
            return None

        res = response.json()
        
        # --- PERBAIKAN DI SINI ---
        # Sesuaikan key dengan output di main.py
        pred = res.get('prediction_cm')      # Dulu: predicted_water_level_cm
        rec = res.get('alert_message')       # Dulu: recommendation
        status = res.get('status')
        risk = res.get('risk_level')
        
        print(f"  -> Status: {status}")
        print(f"  -> Predicted Level: {pred} cm")
        print(f"  -> Risk: {risk}")
        print(f"  -> Message: {rec}")
        
        return pred
        
        return pred
    except Exception as e:
        print(f"  -> Error: {e}")
        return None

if __name__ == "__main__":
    # Test 1: Normal level
    p1 = test(10.0, 500.0)

