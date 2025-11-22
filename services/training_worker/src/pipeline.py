import os
import logging
import sys

# --- FIX PATH UNTUK LOCALHOST ---
# Menambahkan root folder ke sys.path agar imports berhasil
sys.path.append(os.getcwd())

import pandas as pd
from src.ml_core.preprocessing import FloodDataPreprocessor, PreprocessingConfig
from src.ml_core.lstm_model import FloodLevelLSTM, LSTMConfig
from src.ml_core.metrics import calculate_regression_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [TRAINER] - %(message)s')
logger = logging.getLogger("TrainingPipeline")

class FloodTrainingPipeline:
    def __init__(self):
        # --- CONFIG PATH LOKAL ---
        # Menggunakan folder ./data dan ./models relatif terhadap root
        self.base_data_dir = os.getenv('DATA_PATH', 'data')
        self.base_model_dir = os.getenv('SAVE_PATH', 'models')
        
        os.makedirs(self.base_data_dir, exist_ok=True)
        os.makedirs(self.base_model_dir, exist_ok=True)

        self.dataset_path = os.path.join(self.base_data_dir, 'DATASET_FINAL_TRAINING.csv')
        self.model_save_path = os.path.join(self.base_model_dir, 'best_model_modular.h5')
        
    def run(self):
        logger.info("🚀 Starting Training Pipeline (Local)...")
        
        # 1. Cek Data
        if not os.path.exists(self.dataset_path):
            logger.error(f"❌ Dataset missing at: {self.dataset_path}")
            logger.info("💡 Tips: Pastikan file CSV ada di folder 'data/' di root project.")
            return

        # 2. Preprocessing
        logger.info("🛠️ Preprocessing...")
        prep_config = PreprocessingConfig(
            target_col='tma_manggarai',
            rainfall_cols=['hujan_bogor', 'hujan_jakarta']
        )
        preprocessor = FloodDataPreprocessor(prep_config)
        df_raw = preprocessor.load_data(self.dataset_path)
        preprocessor.fit(df_raw)
        df_processed = preprocessor.transform(df_raw, training_mode=True)

        # 3. Prepare Data
        train_size = int(len(df_processed) * 0.8)
        train_df = df_processed.iloc[:train_size]
        test_df = df_processed.iloc[train_size:]
        
        feature_cols = [c for c in df_processed.columns if c != 'tma_manggarai' and c != 'timestamp']
        X_train, y_train = train_df[feature_cols].values, train_df['tma_manggarai'].values
        X_test, y_test = test_df[feature_cols].values, test_df['tma_manggarai'].values

        # 4. Train
        logger.info("🏋️ Training LSTM...")
        model = FloodLevelLSTM(LSTMConfig(epochs=20)) # Epoch dikit biar cepet demo
        model.fit(X_train, y_train)

        # 5. Save
        logger.info(f"💾 Saving to {self.model_save_path}...")
        model.save_model(self.model_save_path)
        logger.info("✨ Pipeline Finished Successfully!")

if __name__ == "__main__":
    pipeline = FloodTrainingPipeline()
    pipeline.run()