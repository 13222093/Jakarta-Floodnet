import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def evaluate_model():
    print("🚀 Starting Model Evaluation...")
    
    model_path = 'models/lstm_flood_forecaster.h5'
    data_path = 'data/DATASET_FINAL_TRAINING.csv'
    scaler_X_path = 'models/lstm_scaler_X.pkl'
    scaler_y_path = 'models/lstm_scaler_y.pkl'
    
    # 1. Load Resources
    try:
        model = load_model(model_path, compile=False)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        df = pd.read_csv(data_path)
        print("✅ Resources loaded successfully")
    except Exception as e:
        print(f"❌ Error loading resources: {e}")
        return

    # 2. Prepare Data
    # Assuming the same feature engineering as training
    # For simplicity, we'll just take the raw columns if they match what the scaler expects
    # The scaler expects 3 features based on previous audit: [hujan_bogor, hujan_jakarta, tma_manggarai] (likely)
    
    feature_cols = ['hujan_bogor', 'hujan_jakarta', 'tma_manggarai']
    target_col = 'tma_manggarai' # We are predicting TMA
    
    # Check if columns exist
    if not all(col in df.columns for col in feature_cols):
        print(f"❌ Missing columns in dataset. Available: {df.columns}")
        return
        
    X_raw = df[feature_cols].values
    y_raw = df[target_col].values
    
    # Scale
    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()
    
    # Create Sequences
    sequence_length = model.input_shape[1]
    print(f"ℹ️ Model Sequence Length: {sequence_length}")
    
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y_scaled[i+sequence_length])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"ℹ️ Test Data Shape: {X_seq.shape}")
    
    # 3. Predict
    print("🔮 Running Prediction...")
    y_pred_scaled = model.predict(X_seq, verbose=0)
    
    # Inverse Transform
    y_pred_cm = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true_cm = scaler_y.inverse_transform(y_seq.reshape(-1, 1)).flatten()
    
    # 4. Metrics
    mae = mean_absolute_error(y_true_cm, y_pred_cm)
    rmse = np.sqrt(mean_squared_error(y_true_cm, y_pred_cm))
    
    print(f"📊 Metrics:")
    print(f"   MAE:  {mae:.2f} cm")
    print(f"   RMSE: {rmse:.2f} cm")
    
    # 5. Plot
    print("📈 Generating Plot...")
    plt.figure(figsize=(12, 6))
    
    # Plot a subset to make it readable (e.g., last 200 points)
    subset_size = 200
    plt.plot(y_true_cm[-subset_size:], label='Actual Water Level (cm)', color='blue', alpha=0.7)
    plt.plot(y_pred_cm[-subset_size:], label='Predicted (LSTM)', color='red', linestyle='--', alpha=0.8)
    
    plt.title(f'Flood Model Evaluation (MAE: {mae:.2f}cm)')
    plt.xlabel('Time Steps (Hours)')
    plt.ylabel('Water Level (cm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = 'models/training_accuracy_plot.png'
    plt.savefig(output_path)
    print(f"✅ Plot saved to {output_path}")

if __name__ == "__main__":
    evaluate_model()
