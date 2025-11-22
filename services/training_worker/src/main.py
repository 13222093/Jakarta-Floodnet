import sys
import os
import time
import logging

# Add /app to python path to allow imports from src.ml_core
sys.path.append('/app')

from src.pipeline import FloodTrainingPipeline

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [WORKER] - %(message)s')
logger = logging.getLogger("WorkerMain")

def main():
    logger.info("🤖 Training Worker Started")
    
    # Mode: One-off training or Scheduled?
    # Untuk Hackathon/Demo: Jalan sekali saat container up, lalu sleep (biar container gak mati)
    
    try:
        pipeline = FloodTrainingPipeline()
        pipeline.run()
        logger.info("✅ Job Done. Sleeping...")
        
    except Exception as e:
        logger.error(f"❌ Job Failed: {e}")
        import traceback
        traceback.print_exc()

    # Keep container alive (optional, biar bisa debug kalau masuk ke container)
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()