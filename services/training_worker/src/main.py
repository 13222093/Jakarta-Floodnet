import sys
import os
import logging
import time

# --- FIX PATH UNTUK LOCALHOST ---
# Menambahkan current working directory (Root Project) ke sys.path
# Ini wajib agar Python bisa menemukan modul 'src' dan 'services'
sys.path.append(os.getcwd())

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [WORKER_MAIN] - %(message)s')
logger = logging.getLogger("WorkerMain")

def main():
    logger.info("🤖 Training Worker Started (Local Mode)")
    logger.info(f"📂 Working Directory: {os.getcwd()}")

    try:
        # Import Pipeline secara dinamis setelah path diperbaiki
        # Kita mengimpor class FloodTrainingPipeline dari file pipeline.py
        from services.training_worker.src.pipeline import FloodTrainingPipeline
        
        logger.info("🚀 Initializing Pipeline...")
        pipeline = FloodTrainingPipeline()
        
        logger.info("▶️  Starting Training Run...")
        pipeline.run()
        
        logger.info("✅ Training Job Completed. Model ready in 'models/' folder.")
        
    except ImportError as e:
        logger.error(f"❌ Import Error: {e}")
        logger.error("💡 TIPS: Pastikan kamu menjalankan script ini dari ROOT FOLDER project!")
        logger.error("   Contoh: python services/training_worker/src/main.py")
        
    except Exception as e:
        logger.error(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()