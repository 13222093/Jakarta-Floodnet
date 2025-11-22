"""
Training Pipeline Module
=======================
Orchestrates the entire training workflow:
Data Loading -> Preprocessing -> LSTM Training -> Evaluation -> Model Saving
"""

import os
import logging
import pandas as pd
from datetime import datetime

# Import Shared Libraries (Otot & Otak yang sudah kita refactor)
from src.ml_core.preprocessing import FloodDataPreprocessor, PreprocessingConfig
from src.ml_core.lstm_model import FloodLevelLSTM, LSTMConfig
from src.ml_core.metrics import calculate_regression_metrics
from src.ml_core.yolo_model import FloodVisualVerifier

# Import Data Utils
from src.data_acquisition.download_data2020 import FILE_OUTPUT as DATASET_PATH

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainingPipeline")

class FloodTrainingPipeline:
    def __init__(self):
        # 1. Setup Paths (Docker Friendly)
        self.base_data_dir = os.getenv('DATA_PATH', '/app/data')
        self.base_model_dir = os.getenv('SAVE_PATH', '/app/models')
        
        # Pastikan folder ada
        os.makedirs(self.base_data_dir, exist_ok=True)
        os.makedirs(self.base_model_dir, exist_ok=True)

        self.dataset_path = os.path.join(self.base_data_dir, 'DATASET_FINAL_TRAINING.csv')
        self.model_save_path = os.path.join(self.base_model_dir, 'best_model_modular.h5')
        
    def run(self):
        """Run the complete training pipeline."""
        logger.info("🚀 Starting FloodNet Training Pipeline...")
        
        # STEP 1: CHECK / DOWNLOAD DATA
        if not os.path.exists(self.dataset_path):
            logger.warning(f"Dataset not found at {self.dataset_path}. Triggering downloader...")
            # Kita bisa panggil script download di sini atau raise error
            # Untuk MVP, kita asumsikan data sudah di-mount via Volume
            raise FileNotFoundError(f"❌ Dataset missing: {self.dataset_path}. Please mount data volume.")

        # STEP 2: PREPROCESSING
        logger.info("🛠️ Phase 1: Preprocessing...")
        # Config Preprocessor
        prep_config = PreprocessingConfig(
            target_col='tma_manggarai',
            rainfall_cols=['hujan_bogor', 'hujan_jakarta'],
            lag_hours=[1, 3, 6, 12, 24],
            rolling_windows=[3, 6, 12, 24]
        )
        preprocessor = FloodDataPreprocessor(prep_config)
        
        # Load & Transform
        df_raw = preprocessor.load_data(self.dataset_path)
        preprocessor.fit(df_raw) # Learn stats
        df_processed = preprocessor.transform(df_raw, training_mode=True)
        logger.info(f"✅ Data Processed. Shape: {df_processed.shape}")

        # STEP 3: PREPARE DATA FOR LSTM
        logger.info("🧠 Phase 2: Preparing Tensors...")
        # Split Data
        train_size = int(len(df_processed) * 0.8)
        train_df = df_processed.iloc[:train_size]
        test_df = df_processed.iloc[train_size:]

        # Separate Features (X) and Target (y)
        target_col = 'tma_manggarai'
        feature_cols = [c for c in df_processed.columns if c != target_col and c != 'timestamp']
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        # STEP 4: TRAIN LSTM
        logger.info("🏋️ Phase 3: Training LSTM Model...")
        lstm_config = LSTMConfig(
            sequence_length=24,
            epochs=50, # Bisa dinaikkan
            batch_size=32,
            lstm_units=[64, 32],
            learning_rate=0.001
        )
        model = FloodLevelLSTM(lstm_config)
        
        # Fit Model (Internal scaling included)
        history = model.fit(X_train, y_train, validation_split=0.2)
        logger.info("✅ Training Finished.")

        # STEP 5: EVALUATION
        logger.info("📊 Phase 4: Evaluation...")
        y_pred = model.predict(X_test)
        
        # Metrics
        metrics = calculate_regression_metrics(y_test, y_pred)
        logger.info(f"🏆 Test Results: {metrics}")

        # STEP 6: SAVE ARTIFACTS
        logger.info("💾 Phase 5: Saving Model...")
        model.save_model(self.model_save_path)
        
        # Optional: Save Config/Metrics to JSON for Dashboard
        import json
        metrics_path = os.path.join(self.base_model_dir, 'latest_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
            
        logger.info(f"✨ Pipeline Completed Successfully! Model saved to {self.model_save_path}")

if __name__ == "__main__":
    pipeline = FloodTrainingPipeline()
    pipeline.run()