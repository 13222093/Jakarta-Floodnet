import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_list_scenarios():
    print("\nTesting GET /scenarios...")
    try:
        response = requests.get(f"{BASE_URL}/scenarios")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 5
        print("✅ List Scenarios Passed")
    except Exception as e:
        print(f"❌ List Scenarios Failed: {e}")
        sys.exit(1)

def test_scenario_historical_flood():
    print("\nTesting Scenario 3 (Historical Flood)...")
    try:
        response = requests.post(f"{BASE_URL}/predict/scenario/scenario_3")
        assert response.status_code == 200
        data = response.json()
        print(f"Response: {data}")
        
        assert data["status"] == "success"
        assert data["scenario_used"] == "Banjir Historis (Historical)"
        assert data["prediction_cm"] > 150
        assert data["risk_level"] in ["BAHAYA", "SIAGA"]
        print("✅ Historical Flood Scenario Passed")
    except Exception as e:
        print(f"❌ Historical Flood Failed: {e}")
        sys.exit(1)

def test_scenario_clear():
    print("\nTesting Scenario 1 (Clear)...")
    try:
        response = requests.post(f"{BASE_URL}/predict/scenario/scenario_1")
        assert response.status_code == 200
        data = response.json()
        print(f"Response: {data}")
        
        assert data["status"] == "success"
        assert data["scenario_used"] == "Cerah Berawan (Clear)"
        assert data["prediction_cm"] < 100
        assert data["risk_level"] == "AMAN"
        print("✅ Clear Scenario Passed")
    except Exception as e:
        print(f"❌ Clear Scenario Failed: {e}")
        sys.exit(1)

def test_scenario_god_mode():
    print("\nTesting Scenario 5 (God Mode)...")
    try:
        response = requests.post(f"{BASE_URL}/predict/scenario/scenario_5")
        assert response.status_code == 200
        data = response.json()
        print(f"Response: {data}")
        
        assert data["status"] == "demo_mode_active"
        assert data["risk_level"] == "CRITICAL"
        assert data["prediction_cm"] == 250.0
        print("✅ God Mode Scenario Passed")
    except Exception as e:
        print(f"❌ God Mode Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Wait for server to be ready
    print("Waiting for server...")
    for _ in range(10):
        try:
            requests.get(f"{BASE_URL}/health")
            break
        except:
            time.sleep(1)
    
    test_list_scenarios()
    test_scenario_historical_flood()
    test_scenario_clear()
    test_scenario_god_mode()
