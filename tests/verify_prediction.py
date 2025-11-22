import requests
import json

URL = "http://localhost:8001/predict"

def test(rainfall, water_level):
    print(f"Testing with Rainfall={rainfall}, WaterLevel={water_level}")
    payload = {
        "rainfall_mm": rainfall,
        "water_level_cm": water_level,
        "location_id": "MANGGARAI_01"
    }
    try:
        res = requests.post(URL, json=payload).json()
        pred = res.get('predicted_water_level_cm')
        rec = res.get('recommendation')
        print(f"  -> Predicted: {pred}")
        print(f"  -> Recommendation: {rec}")
        return pred
    except Exception as e:
        print(f"  -> Error: {e}")
        return None

if __name__ == "__main__":
    # Test 1: Normal level
    p1 = test(10.0, 500.0)

