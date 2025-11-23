"""
Lightweight LSTM Training Script
=================================
Trains a simple LSTM model with ONLY 3 features to match API architecture.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

print("=" * 60)
print("🚀 Lightweight LSTM Training (3 Features, 12 Hour Lookback)")
print("=" * 60)

# Configuration
DATA_PATH = "data/DATASET_FINAL_TRAINING.csv"
MODEL_PATH = "models/lstm_flood_forecaster.h5"
SCALER_X_PATH = "models/lstm_scaler_X.pkl"
SCALER_Y_PATH = "models/lstm_scaler_y.pkl"

LOOKBACK = 12  # 12 hours
BATCH_SIZE = 16  # Smaller batch for better gradient updates
EPOCHS = 50  # More epochs to learn patterns

# 1. Load Data
print(f"\n📥 Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} rows")
print(f"   Columns: {list(df.columns)}")

# 2. Feature Selection - ONLY 3 features
FEATURE_COLS = ['hujan_bogor', 'hujan_jakarta', 'tma_manggarai']
TARGET_COL = 'tma_manggarai'

print(f"\n🎯 Selecting features: {FEATURE_COLS}")
print(f"   Target: {TARGET_COL}")

# Check if columns exist
missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    print(f"   Available columns: {list(df.columns)}")
    exit(1)

# Extract features and target
X_raw = df[FEATURE_COLS].values
y_raw = df[TARGET_COL].values

print(f"   X shape: {X_raw.shape}")
print(f"   y shape: {y_raw.shape}")

# 3. Scale Data (0-1 range for LSTM)
print(f"\n⚙️ Scaling data with MinMaxScaler(0, 1)...")
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

print(f"✅ Scaling complete")
print(f"   X scaled range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
print(f"   y scaled range: [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")

# 4. Create Sequences (Windowing)
print(f"\n🪟 Creating sequences with lookback={LOOKBACK}...")

def create_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:(i + lookback)])
        y_seq.append(y[i + lookback])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOKBACK)

print(f"✅ Sequences created")
print(f"   X_seq shape: {X_seq.shape}  # (Samples, Timesteps, Features)")
print(f"   y_seq shape: {y_seq.shape}")

# 5. Train/Val Split
split_idx = int(len(X_seq) * 0.8)
X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

print(f"\n📊 Train/Val Split:")
print(f"   Train: {len(X_train)} samples")
print(f"   Val:   {len(X_val)} samples")

# 6. Build Model - STACKED LSTM for better pattern recognition
print(f"\n🏗️ Building STACKED LSTM model...")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 3)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

# Use Huber loss - better for outliers/floods
model.compile(
    optimizer='adam',
    loss='huber',  # More robust to outliers
    metrics=['mae']
)

print(f"✅ Model built")
model.summary()

# 7. Train
print(f"\n🎓 Training for {EPOCHS} epochs with batch_size={BATCH_SIZE}...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,  # More patience for 50 epochs
        restore_best_weights=True,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print(f"\n✅ Training complete!")
print(f"   Final train loss: {history.history['loss'][-1]:.4f}")
print(f"   Final val loss: {history.history['val_loss'][-1]:.4f}")

# 8. Save Model and Scalers
print(f"\n💾 Saving model and scalers...")

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

joblib.dump(scaler_X, SCALER_X_PATH)
print(f"✅ Scaler X saved to {SCALER_X_PATH}")

joblib.dump(scaler_y, SCALER_Y_PATH)
print(f"✅ Scaler y saved to {SCALER_Y_PATH}")

# 9. Quick Validation
print(f"\n🧪 Quick validation test...")

# Test on last validation sample
test_input = X_val[-1:] # Shape: (1, 12, 3)
test_actual = y_val[-1]

pred_scaled = model.predict(test_input, verbose=0)
pred_cm = scaler_y.inverse_transform(pred_scaled)[0][0]
actual_cm = scaler_y.inverse_transform([[test_actual]])[0][0]

print(f"   Predicted: {pred_cm:.2f} cm")
print(f"   Actual:    {actual_cm:.2f} cm")
print(f"   Error:     {abs(pred_cm - actual_cm):.2f} cm")

print(f"\n" + "=" * 60)
print(f"✅ Training Complete!")
print(f"=" * 60)
print(f"\nModel Info:")
print(f"  - Input shape: (12, 3)  # 12 timesteps, 3 features")
print(f"  - Features: {FEATURE_COLS}")
print(f"  - Output: TMA prediction (cm)")
print(f"\nNext step: Run tests/test_lstm_logic_direct.py to verify")
