import requests
import os
import sys
import json

# Configuration
BASE_URL = "http://localhost:8000"
IMAGE_FILENAME = "demo_image.jpg"

def print_result(test_name, status, payload=None):
    symbol = "✅" if status == "PASS" else "❌"
    print(f"\n{symbol} {status}: {test_name}")
    if payload:
        print(json.dumps(payload, indent=2))

def run_tests():
    print(f"🚀 Starting Integration Tests against {BASE_URL}...")

    # Setup: Check image
    if not os.path.exists(IMAGE_FILENAME):
        print(f"❌ CRITICAL: {IMAGE_FILENAME} not found in current directory.")
        print("   Please place a 'demo_image.jpg' file here before running.")
        sys.exit(1)
    else:
        print(f"✅ Setup: Found {IMAGE_FILENAME}")

    # Test 0: Health Check (Sanity)
    try:
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code == 200:
            print_result("Health Check", "PASS", resp.json())
        else:
            print_result("Health Check", "FAIL", {"status_code": resp.status_code, "text": resp.text})
    except Exception as e:
        print(f"❌ CRITICAL: Cannot connect to {BASE_URL}. Is the server running?")
        print(f"   Error: {e}")
        sys.exit(1)

    # Test 1: Visual Verification
    print("\n--- Test 1: Visual Verification (POST /verify-visual) ---")
    try:
        with open(IMAGE_FILENAME, "rb") as f:
            files = {"file": (IMAGE_FILENAME, f, "image/jpeg")}
            resp = requests.post(f"{BASE_URL}/verify-visual", files=files)
            
            if resp.status_code == 200:
                print_result("Visual Verification", "PASS", resp.json())
            else:
                print_result("Visual Verification", "FAIL", {"status_code": resp.status_code, "text": resp.text})
    except Exception as e:
        print_result("Visual Verification", "FAIL", {"error": str(e)})

    # Test 2: God Mode Toggle
    print("\n--- Test 2: God Mode Toggle (POST /admin/set-demo-mode) ---")
    try:
        # Note: This endpoint might not exist in the current version, handling 404 gracefully
        resp = requests.post(f"{BASE_URL}/admin/set-demo-mode", params={"enable": "true"})
        
        if resp.status_code == 200:
            print_result("Enable Demo Mode", "PASS", resp.json())
        elif resp.status_code == 404:
            print_result("Enable Demo Mode", "FAIL", {"error": "Endpoint not found (404). Feature might be missing."})
        else:
            print_result("Enable Demo Mode", "FAIL", {"status_code": resp.status_code, "text": resp.text})
    except Exception as e:
        print_result("Enable Demo Mode", "FAIL", {"error": str(e)})

    # Test 3: Panic Button Check
    print("\n--- Test 3: Panic Button Check (POST /predict-lstm) ---")
    # Note: User requested /predict-lstm, but codebase shows /predict. Trying both.
    endpoint = "/predict-lstm"
    
    # Payload for a "Critical" scenario (Heavy rain)
    # Note: API Gateway uses 'rainfall_mm' and 'water_level_cm'
    payload = {
        "rainfall_mm": 100.0,
        "water_level_cm": 200.0
    }

    try:
        resp = requests.post(f"{BASE_URL}{endpoint}", json=payload)
        
        # If /predict-lstm fails (404), try /predict
        if resp.status_code == 404:
            print(f"⚠️ Endpoint {endpoint} not found. Retrying with /predict...")
            endpoint = "/predict"
            resp = requests.post(f"{BASE_URL}{endpoint}", json=payload)

        if resp.status_code == 200:
            data = resp.json()
            # Check for CRITICAL status or HIGH risk
            risk_level = data.get("risk_level", "UNKNOWN")
            
            if risk_level in ["HIGH", "CRITICAL"]:
                print_result("Panic Button Check", "PASS", data)
            else:
                print_result("Panic Button Check", "FAIL", {
                    "message": f"Expected HIGH/CRITICAL risk, got {risk_level}",
                    "response": data
                })
        else:
            print_result("Panic Button Check", "FAIL", {"status_code": resp.status_code, "text": resp.text})

    except Exception as e:
        print_result("Panic Button Check", "FAIL", {"error": str(e)})

if __name__ == "__main__":
    run_tests()
