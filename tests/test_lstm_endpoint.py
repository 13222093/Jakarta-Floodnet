import requests
import json

URL = "http://localhost:8001/predict"

def test_predict():
    print("🧪 Testing /predict endpoint (LSTM)...")
    
    # Test case: High rainfall and high water level
    payload = {
        "rainfall_mm": 50.0,
        "water_level_cm": 800.0,
        "location_id": "MANGGARAI_01"
    }
    
    print(f"   Sending payload: {payload}")
    
    try:
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Request successful!")
            print(json.dumps(data, indent=2))
            
            if "predicted_water_level_cm" in data:
                print(f"🌊 Predicted Water Level: {data['predicted_water_level_cm']} cm")
            else:
                print("⚠️ 'predicted_water_level_cm' not found in response.")
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the server running?")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_predict()
