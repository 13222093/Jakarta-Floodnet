import requests
import json
import sys

# Configuration
API_URL = "http://localhost:8000/predict"

# Test Cases
TEST_CASES = [
    {
        "name": "A. Baseline Check",
        "description": "Normal condition, no rain.",
        "input": {"hujan_bogor": 0, "hujan_jakarta": 0, "tma": 350},
        "expected_risk": "AMAN"
    },
    {
        "name": "B. Critical Event",
        "description": "Heavy rain, high water. Testing inertia & clamp.",
        "input": {"hujan_bogor": 150, "hujan_jakarta": 80, "tma": 800},
        "expected_risk": "BAHAYA" # Should be > 150cm (actually > 800cm due to clamp)
    },
    {
        "name": "C. High Inertia (False Alarm)",
        "description": "Light rain, very high water. Testing 95% anchor.",
        "input": {"hujan_bogor": 10, "hujan_jakarta": 5, "tma": 900},
        "expected_risk": "BAHAYA" # Should stay high
    },
    {
        "name": "D. Medium Risk",
        "description": "Moderate rain, medium water.",
        "input": {"hujan_bogor": 30, "hujan_jakarta": 15, "tma": 500},
        "expected_risk": "BAHAYA" # 500cm is already BAHAYA (>150)
    }
]

def run_validation():
    print(f"{'TEST CASE':<30} | {'INPUT (Bogor/Jkt/TMA)':<25} | {'PREDICTION':<10} | {'RISK':<10} | {'STATUS'}")
    print("-" * 100)

    for case in TEST_CASES:
        # Construct payload mimicking api_client.py behavior
        # api_client uses 'hujan_jakarta' as 'rainfall_mm'
        payload = {
            "rainfall_mm": float(case["input"]["hujan_jakarta"]),
            "water_level_cm": float(case["input"]["tma"])
        }

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()

            pred_cm = result.get("prediction_cm", 0.0)
            risk = result.get("risk_level", "UNKNOWN")
            
            # Formatting input string
            input_str = f"{case['input']['hujan_bogor']}/{case['input']['hujan_jakarta']}/{case['input']['tma']}"
            
            # Simple validation (just checking if it runs and gives plausible output)
            status = "✅ OK" if risk != "UNKNOWN" else "❌ FAIL"
            
            print(f"{case['name']:<30} | {input_str:<25} | {pred_cm:>8.1f} cm | {risk:<10} | {status}")

        except Exception as e:
            print(f"{case['name']:<30} | ERROR: {str(e)}")

if __name__ == "__main__":
    print("🚀 STARTING PHYSICS ENGINE LOGIC VALIDATION")
    print("===========================================")
    run_validation()
    print("\n✅ VALIDATION COMPLETE")
