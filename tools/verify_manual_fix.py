import requests
import json

BASE_URL = "http://localhost:8000"

def test_manual_endpoint():
    print("🧪 Testing Balanced Physics Engine")
    print("==================================")
    
    url = f"{BASE_URL}/predict"

    # Test Case 1: Low Risk (Physics Correctness)
    # Rain 100mm (Bogor), 100mm (Jkt), TMA 100cm.
    # Expected: 100 + (100*0.8 + 100*0.5) = 100 + 130 = 230 cm. (Status: AMAN < 400)
    payload_low = {
        "rainfall_jakarta": 100.0,
        "rainfall_bogor": 100.0,
        "rainfall_mm": 100.0, # Legacy
        "water_level_cm": 100.0
    }
    
    print(f"\n🚀 Sending Request 1 (Low Risk): {payload_low}")
    try:
        response = requests.post(url, json=payload_low)
        print(f"Response Status: {response.status_code}")
        # print(f"Response Body: {response.text}")
        
        data = response.json()
        if data.get("status") == "success":
            pred = data["prediction_cm"]
            risk = data["risk_level"]
            print(f"Prediction: {pred}cm, Risk: {risk}")
            
            # Allow small float variance
            if 225 <= pred <= 235 and risk == "AMAN":
                print("✅ PASS: Low Risk Scenario Correct (Physics Logic OK)")
            else:
                print(f"❌ FAIL: Low Risk Scenario Failed (Expected ~230cm AMAN, Got {pred}cm {risk})")
        else:
            print(f"❌ FAIL: API Error - {data.get('message')}")
            
    except Exception as e:
        print(f"❌ CRASH: {e}")

    # Test Case 2: High Risk (Demo Scenario)
    # Rain 100mm (Bogor), 100mm (Jkt), TMA 700cm.
    # Expected: 700 + 130 = 830 cm. (Status: BAHAYA >= 700)
    payload_high = {
        "rainfall_jakarta": 100.0,
        "rainfall_bogor": 100.0,
        "rainfall_mm": 100.0, # Legacy
        "water_level_cm": 700.0
    }
    
    print(f"\n🚀 Sending Request 2 (High Risk): {payload_high}")
    try:
        response = requests.post(url, json=payload_high)
        print(f"Response Status: {response.status_code}")
        # print(f"Response Body: {response.text}")
        
        data = response.json()
        if data.get("status") == "success":
            pred = data["prediction_cm"]
            risk = data["risk_level"]
            print(f"Prediction: {pred}cm, Risk: {risk}")
            
            # Allow small float variance
            if 825 <= pred <= 835 and (risk == "BAHAYA" or risk == "CRITICAL"):
                print("✅ PASS: High Risk Scenario Correct (Demo Logic OK)")
            else:
                print(f"❌ FAIL: High Risk Scenario Failed (Expected ~830cm BAHAYA, Got {pred}cm {risk})")
        else:
            print(f"❌ FAIL: API Error - {data.get('message')}")
            
    except Exception as e:
        print(f"❌ CRASH: {e}")

if __name__ == "__main__":
    test_manual_endpoint()
