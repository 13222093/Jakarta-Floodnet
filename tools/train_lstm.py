import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Constants
DATA_PATH = 'data/DATASET_FINAL_TRAINING.csv'
MODEL_PATH = 'models/lstm_model.h5'
SCALER_PATH = 'models/scaler.pkl'
LOOKBACK = 1  # Use 1 hour lookback for simplicity with current API
EPOCHS = 20
BATCH_SIZE = 32

def train_model():
    print("🚀 Starting LSTM Training...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df)} rows.")
    
    # 2. Preprocess
    # Features: Hujan Bogor, Hujan Jakarta, TMA Manggarai
    # Target: TMA Manggarai (Next hour? Or same hour classification? Let's predict next hour TMA)
    # For simplicity in this MVP, let's predict *current* risk or just next step TMA based on current.
    # Let's try to predict TMA Manggarai (t+1) based on t.
    
    features = ['hujan_bogor', 'hujan_jakarta', 'tma_manggarai']
    data = df[features].values
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK:i]) # Previous `lookback` steps
        y.append(scaled_data[i, 2])         # Target: TMA Manggarai at current step (index 2)
        
    X, y = np.array(X), np.array(y)
    
    print(f"   Training data shape: X={X.shape}, y={y.shape}")
    
    # 3. Build Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(LOOKBACK, len(features))))
    model.add(Dense(1)) # Predict scaled TMA
    
    model.compile(optimizer='adam', loss='mse')
    
    # 4. Train
    print("   Training...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
    
    # 5. Save
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Scaler saved to {SCALER_PATH}")

if __name__ == "__main__":
    train_model()
