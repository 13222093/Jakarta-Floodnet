"""
Test Physics Engine in /predict endpoint
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
import json

# Try to import app
try:
    from src.api_service.main import app
    client = TestClient(app)
    API_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Could not load FastAPI app: {e}")
    print("This test requires the API to be importable.")
    API_AVAILABLE = False
    exit(1)

print("=" * 60)
print("🧪 Testing Advanced Physics Engine")
print("=" * 60)

# Test Scenarios
scenarios = [
    {
        "name": "Scenario A: Heavy Rain + High Water (800cm)",
        "input": {
            "hujan_bogor": 150.0,
            "hujan_jakarta": 80.0,
            "tma_saat_ini": 800.0
        },
        "expected": "Prediction >= 800cm (water cannot drop during heavy rain)"
    },
    {
        "name": "Scenario B: Moderate Rain + Normal Water (400cm)",
        "input": {
            "hujan_bogor": 25.0,
            "hujan_jakarta": 20.0,
            "tma_saat_ini": 400.0
        },
        "expected": "Prediction >= 400cm (water rises with rain)"
    },
    {
        "name": "Scenario C: No Rain + Low Water (300cm)",
        "input": {
            "hujan_bogor": 0.0,
            "hujan_jakarta": 0.0,
            "tma_saat_ini": 300.0
        },
        "expected": "Prediction ~300cm (stable or slight drop)"
    },
    {
        "name": "Scenario D: Extreme Rain + Medium Water (500cm)",
        "input": {
            "hujan_bogor": 100.0,
            "hujan_jakarta": 60.0,
            "tma_saat_ini": 500.0
        },
        "expected": "Prediction >= 500cm (heavy rain prevents drop)"
    }
]

print("\n🚀 Running Tests...\n")

for i, scenario in enumerate(scenarios, 1):
    print(f"Test {i}: {scenario['name']}")
    print(f"  Input: Bogor={scenario['input']['hujan_bogor']}mm, Jakarta={scenario['input']['hujan_jakarta']}mm, TMA={scenario['input']['tma_saat_ini']}cm")
    
    try:
        response = client.post("/predict", json=scenario['input'])
        
        if response.status_code == 200:
            data = response.json()
            pred_tma = data.get("predicted_water_level_cm")
            risk = data.get("risk_level")
            debug = data.get("debug_info")
            
            # Analyze result
            input_tma = scenario['input']['tma_saat_ini']
            delta = pred_tma - input_tma
            
            # Check physics constraints
            heavy_rain = scenario['input']['hujan_bogor'] > 20 or scenario['input']['hujan_jakarta'] > 20
            
            if heavy_rain and pred_tma < input_tma:
                status = "❌ FAILED - Water dropped during heavy rain!"
            elif heavy_rain and pred_tma >= input_tma:
                status = "✅ PASSED - Water maintained/rose during rain"
            elif not heavy_rain and abs(delta) <= 50:
                status = "✅ PASSED - Reasonable change without rain"
            else:
                status = "⚠️ CHECK - Unusual behavior"
            
            print(f"  Result: {pred_tma:.2f} cm (Δ {delta:+.2f} cm)")
            print(f"  Risk: {risk}")
            print(f"  {status}")
            print(f"  Expected: {scenario['expected']}")
            
            if debug:
                print(f"  Debug: {debug}")
                
        else:
            print(f"  ❌ API Error: {response.status_code}")
            print(f"  {response.text}")
            
    except Exception as e:
        print(f"  ❌ Test Error: {e}")
    
    print()

print("=" * 60)
print("✅ Physics Engine Test Complete")
print("=" * 60)
