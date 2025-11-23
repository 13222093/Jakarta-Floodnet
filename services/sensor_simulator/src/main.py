import time
import pandas as pd
import requests
import os
import random
import logging
import sys

# --- FIX PATH UNTUK LOCALHOST ---
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SIMULATOR] - %(message)s')
logger = logging.getLogger("SensorSimulator")

# --- CONFIG LOKAL ---
API_URL = os.getenv("API_URL", "http://localhost:8000")
# Path data relatif terhadap root folder
DATA_PATH = os.getenv("DATA_PATH", "data/DATASET_FINAL_TRAINING.csv")
INTERVAL = 2

def load_dataset():
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            logger.info(f"✅ Loaded dataset: {DATA_PATH}")
            return df
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
    else:
        logger.warning(f"⚠️ Dataset not found at {DATA_PATH}. Using Random Mode.")
    return None

def generate_payload(df, index):
    if df is not None:
        row = df.iloc[index]
        avg_rain = (row['hujan_bogor'] + row['hujan_jakarta']) / 2
        return {
            "rainfall_mm": round(avg_rain, 1),
            "water_level_cm": round(row['tma_manggarai'], 1),
            "location_id": "MANGGARAI_01"
        }
    return {
        "rainfall_mm": round(random.uniform(0, 50), 1),
        "water_level_cm": round(random.uniform(50, 900), 1),
        "location_id": "MANGGARAI_01"
    }

def run_simulation():
    logger.info(f"🚀 Starting Simulator -> Target: {API_URL}")
    df = load_dataset()
    idx = 0
    
    while True:
        try:
            payload = generate_payload(df, idx)
            if df is not None: idx = (idx + 1) % len(df)

            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=1)
            if resp.status_code == 200:
                res = resp.json()
                logger.info(f"📡 Sent: TMA={payload['water_level_cm']} | Pred: {res['prediction_cm']}")
            else:
                logger.warning(f"⚠️ API Error: {resp.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error("❌ API Offline. Retrying...")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error: {e}")
        
        time.sleep(INTERVAL)

if __name__ == "__main__":
    run_simulation()