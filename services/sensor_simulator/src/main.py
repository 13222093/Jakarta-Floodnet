import time
import pandas as pd
import requests
import os
import random
import logging
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SIMULATOR] - %(message)s')
logger = logging.getLogger("SensorSimulator")

# Configuration
API_URL = os.getenv("API_URL", "http://api_gateway:8000")
DATA_PATH = os.getenv("DATA_PATH", "/app/data/DATASET_FINAL_TRAINING.csv")
INTERVAL = int(os.getenv("INTERVAL", "2")) # Kirim data tiap 2 detik

def load_dataset():
    """Load dataset for replay simulation"""
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            # Pastikan kolom yang dibutuhkan ada
            required = ['hujan_bogor', 'hujan_jakarta', 'tma_manggarai']
            if all(col in df.columns for col in required):
                logger.info(f"✅ Loaded dataset from {DATA_PATH} ({len(df)} rows)")
                return df
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
    
    logger.warning(f"⚠️ Dataset not found at {DATA_PATH}. Using RANDOM GENERATOR mode.")
    return None

def generate_random_data():
    """Fallback if CSV missing"""
    return {
        "rainfall_mm": round(random.uniform(0, 50), 1),
        "water_level_cm": round(random.uniform(50, 900), 1),
        "location_id": "MANGGARAI_01"
    }

def get_csv_data(df, index):
    """Get specific row from dataframe"""
    row = df.iloc[index]
    # Logic simplifikasi: rainfall ambil rata-rata bogor & jakarta
    avg_rain = (row['hujan_bogor'] + row['hujan_jakarta']) / 2
    return {
        "rainfall_mm": round(avg_rain, 1),
        "water_level_cm": round(row['tma_manggarai'], 1),
        "location_id": "MANGGARAI_01"
    }

def run_simulation():
    logger.info(f"🚀 Starting Sensor Simulator... Target: {API_URL}")
    
    # 1. Prepare Data
    df = load_dataset()
    current_idx = 0
    
    # Tunggu sebentar biar API nyala duluan
    time.sleep(5) 

    while True:
        try:
            # 2. Get Data Packet
            if df is not None:
                payload = get_csv_data(df, current_idx)
                # Increment index & loop if needed
                current_idx = (current_idx + 1) % len(df)
            else:
                payload = generate_random_data()

            # 3. Send to API
            endpoint = f"{API_URL}/predict"
            response = requests.post(endpoint, json=payload, timeout=2)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"📡 Sent: TMA={payload['water_level_cm']}cm | 🔙 API: {result['risk_level']} (Pred: {result['prediction_cm']}cm)")
            else:
                logger.warning(f"⚠️ API Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            logger.error("❌ API Unreachable. Retrying in 5s...")
            time.sleep(3)
        except Exception as e:
            logger.error(f"❌ Unexpected Error: {e}")
        
        # 4. Wait for next tick
        time.sleep(INTERVAL)

if __name__ == "__main__":
    run_simulation()