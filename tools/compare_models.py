import os
import glob
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def compare_models():
    with open('comparison_results.txt', 'w', encoding='utf-8') as f:
        f.write("🚀 Starting Model Comparison...\n")
        
        # 1. Load Scalers
        try:
            scaler_X = joblib.load('models/lstm_scaler_X.pkl')
            scaler_y = joblib.load('models/lstm_scaler_y.pkl')
            f.write("✅ Scalers loaded successfully\n")
        except Exception as e:
            f.write(f"❌ Error loading scalers: {e}\n")
            return

        # 2. Define Test Input (Heavy Rain + High Water)
        raw_input = [150.0, 85.0, 800.0] # Bogor, Jkt, TMA
        
        n_features_expected = scaler_X.n_features_in_
        f.write(f"ℹ️ Scaler expects {n_features_expected} features\n")
        
        if len(raw_input) < n_features_expected:
            f.write(f"⚠️ Input has {len(raw_input)} features, padding to {n_features_expected}\n")
            raw_input = raw_input + [0.0] * (n_features_expected - len(raw_input))
        
        # Scale Input
        input_scaled = scaler_X.transform([raw_input]) # Shape (1, n_features)
        
        # 3. Find all .h5 files
        model_files = glob.glob("models/*.h5")
        f.write(f"🔎 Found {len(model_files)} models: {model_files}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"{'Filename':<35} | {'Prediction (cm)':<15} | {'Status'}\n")
        f.write("="*80 + "\n")
        
        for model_path in model_files:
            filename = os.path.basename(model_path)
            try:
                model = load_model(model_path, compile=False)
                
                # Check input shape
                input_shape = model.input_shape
                
                # Prepare input based on shape
                if len(input_shape) == 3:
                    # (batch, timesteps, features)
                    timesteps = input_shape[1]
                    n_features = input_shape[2]
                    
                    if n_features != n_features_expected:
                         f.write(f"{filename:<35} | {'ERROR':<15} | ❌ Feature mismatch (Model: {n_features}, Scaler: {n_features_expected})\n")
                         continue
                    
                    # Repeat input to match timesteps
                    X_test = np.repeat(input_scaled[:, np.newaxis, :], timesteps, axis=1)
                    
                else:
                    # Maybe (batch, features) ?
                    X_test = input_scaled
                    
                # Predict
                pred_scaled = model.predict(X_test, verbose=0)
                
                # Inverse Transform
                pred_cm = scaler_y.inverse_transform(pred_scaled)[0][0]
                
                # Evaluate
                status = ""
                if pred_cm > 700:
                    status = "✅ PASS (Smart)"
                elif pred_cm < 500:
                    status = "⚠️ FAIL (Lazy/Mean Reversion)"
                else:
                    status = "❓ INDETERMINATE"
                    
                f.write(f"{filename:<35} | {pred_cm:<15.2f} | {status}\n")
                
            except Exception as e:
                f.write(f"{filename:<35} | {'ERROR':<15} | ❌ {str(e)[:30]}...\n")

        f.write("="*80 + "\n")

if __name__ == "__main__":
    compare_models()
