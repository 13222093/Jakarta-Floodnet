# Add root to path
import sys
import os
# Ensure we are at the root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.api_service.main import app

client = TestClient(app)

def test_lstm_scenarios():
    print("\n🚀 Starting LSTM Logic Verification...")
    
    scenarios = [
        {
            "name": "Scenario A (BADAI/FLOOD)",
            "input": {"hujan_bogor": 150.0, "hujan_jakarta": 80.0, "tma_saat_ini": 800.0},
            "expected_desc": "Prediction > 800 (Water rising)"
        },
        {
            "name": "Scenario B (NORMAL)",
            "input": {"hujan_bogor": 10.0, "hujan_jakarta": 5.0, "tma_saat_ini": 400.0},
            "expected_desc": "Prediction ~400 (+/- 20cm)"
        },
        {
            "name": "Scenario C (KERING/DRY)",
            "input": {"hujan_bogor": 0.0, "hujan_jakarta": 0.0, "tma_saat_ini": 300.0},
            "expected_desc": "Prediction <= 300 (Water recedes)"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🔹 Testing: {scenario['name']}")
        print(f"   Input: {scenario['input']}")
        
        try:
            response = client.post("/predict", json=scenario['input'])
            
            if response.status_code == 200:
                data = response.json()
                pred_cm = data.get("predicted_water_level_cm")
                risk = data.get("risk_level")
                debug = data.get("debug_info")
                
                print(f"   ✅ Response: {response.status_code}")
                print(f"   🌊 Predicted TMA: {pred_cm} cm")
                print(f"   ⚠️ Risk Level: {risk}")
                
                # Simple logic check
                input_tma = scenario['input']['tma_saat_ini']
                status = "STABLE"
                if pred_cm > input_tma + 5: status = "RISING 📈"
                elif pred_cm < input_tma - 5: status = "FALLING 📉"
                
                print(f"   📊 Status: {status}")
                if debug:
                    print(f"   ℹ️ Debug: {debug}")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   Message: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_lstm_scenarios()
