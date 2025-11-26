import requests
import sys
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(name, method, url, payload=None):
    print(f"Testing {name}...", end=" ")
    try:
        if method == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("✅ OK")
            return response.json()
        else:
            print(f"❌ FAILED ({response.status_code})")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def run_final_check():
    print("🚀 FINAL SYSTEM INTEGRITY CHECK")
    print("==============================")
    
    # 1. Health Check
    health = test_endpoint("Health Check", "GET", f"{BASE_URL}/health")
    if health:
        print(f"   Status: {health.get('status')}")
        print(f"   Mode: {health.get('mode')}")
        print(f"   Models: {health.get('models')}")

    # 2. Scenarios List
    scenarios = test_endpoint("Scenarios List", "GET", f"{BASE_URL}/scenarios")
    if scenarios:
        print(f"   Count: {len(scenarios)}")
        
    # 3. Prediction (Normal)
    payload = {"rainfall_mm": 0, "water_level_cm": 350}
    pred = test_endpoint("Prediction (Normal)", "POST", f"{BASE_URL}/predict", payload)
    if pred:
        print(f"   Prediction: {pred.get('prediction_cm')} cm")
        print(f"   Risk: {pred.get('risk_level')}")
        if pred.get('risk_level') == "AMAN":
             print("   ✅ Logic Verification: PASS (350cm is AMAN)")
        else:
             print("   ❌ Logic Verification: FAIL (350cm should be AMAN)")

    # 4. God Mode Trigger
    print("\n⚡ Testing God Mode Trigger...")
    requests.post(f"{BASE_URL}/admin/set-demo-mode", json={"enable": True, "scenario": "CRITICAL"})
    
    god_pred = test_endpoint("Prediction (God Mode)", "POST", f"{BASE_URL}/predict", payload)
    if god_pred:
        print(f"   Risk: {god_pred.get('risk_level')}")
        if god_pred.get('risk_level') == "CRITICAL":
            print("   ✅ God Mode Verification: PASS")
        else:
            print("   ❌ God Mode Verification: FAIL")
            
    # Reset God Mode
    requests.post(f"{BASE_URL}/admin/set-demo-mode", json={"enable": False})
    print("\n✅ System Check Complete")

if __name__ == "__main__":
    run_final_check()
