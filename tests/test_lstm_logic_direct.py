import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import joblib

print("🚀 Direct LSTM Logic Verification (Simplified)")
print("=" * 50)

# Try loading with different approaches
MODEL_PATH = "models/lstm_flood_forecaster.h5"
SCALER_X_PATH = "models/lstm_scaler_X.pkl"
SCALER_Y_PATH = "models/lstm_scaler_y.pkl"

print(f"\n📥 Attempting to load scalers first...")
try:
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    print(f"✅ Scalers loaded")
    print(f"   Scaler X expects {scaler_X.n_features_in_} features")
    print(f"   Scaler Y shape: {scaler_y.scale_.shape}")
except Exception as e:
    print(f"❌ Error loading scalers: {e}")
    exit(1)

print(f"\n📥 Attempting to load model from {MODEL_PATH}...")
try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Expected: (None, 12, 3) -> (None, 1)")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n⚠️ Model loading failed. Will use fallback heuristic logic instead.")
    model = None

# Define Test Scenarios
scenarios = [
    {
        "name": "Scenario A (BADAI/FLOOD)",
        "hujan_bogor": 150.0,
        "hujan_jakarta": 80.0,
        "tma_saat_ini": 800.0,
        "expected": "Prediction > 800 (Rising)"
    },
    {
        "name": "Scenario B (NORMAL)",
        "hujan_bogor": 10.0,
        "hujan_jakarta": 5.0,
        "tma_saat_ini": 400.0,
        "expected": "Prediction ~400 (Stable)"
    },
    {
        "name": "Scenario C (KERING/DRY)",
        "hujan_bogor": 0.0,
        "hujan_jakarta": 0.0,
        "tma_saat_ini": 300.0,
        "expected": "Prediction <= 300 (Falling)"
    }
]

print("\n" + "=" * 50)
print("Testing Scenarios")
print("=" * 50)

for scenario in scenarios:
    print(f"\n🔹 {scenario['name']}")
    print(f"   Input: Bogor={scenario['hujan_bogor']}mm, Jakarta={scenario['hujan_jakarta']}mm, TMA={scenario['tma_saat_ini']}cm")
    
    if model is None:
        # Fallback heuristic
        rainfall_avg = (scenario['hujan_bogor'] + scenario['hujan_jakarta']) / 2
        predicted_tma = scenario['tma_saat_ini'] + (rainfall_avg * 0.5)
        print(f"   ⚠️ Using fallback heuristic (model not loaded)")
    else:
        try:
            # Get expected feature count
            n_features = scaler_X.n_features_in_
            
            # Get model's expected sequence length
            sequence_length = model.input_shape[1]
            
            # CREATE DYNAMIC SEQUENCE - Show water level TREND
            # Instead of flat [800, 800, 800...], create a ramp showing rising/falling water
            
            # Create a ramp for TMA based on scenario
            if scenario['name'].startswith("Scenario A"):
                # FLOOD: Water is RISING from 500 to 800
                tma_sequence = np.linspace(500, scenario['tma_saat_ini'], sequence_length)
            elif scenario['name'].startswith("Scenario C"):
                # DRY: Water is FALLING from 400 to 300
                tma_sequence = np.linspace(400, scenario['tma_saat_ini'], sequence_length)
            else:
                # NORMAL: Stable around 400
                tma_sequence = np.linspace(scenario['tma_saat_ini'] - 20, scenario['tma_saat_ini'], sequence_length)
            
            # Create rainfall sequence (also showing trend)
            if scenario['name'].startswith("Scenario A"):
                # Heavy rain building up
                bogor_sequence = np.linspace(50, scenario['hujan_bogor'], sequence_length)
                jakarta_sequence = np.linspace(30, scenario['hujan_jakarta'], sequence_length)
            else:
                # Constant or decreasing rain
                bogor_sequence = np.full(sequence_length, scenario['hujan_bogor'])
                jakarta_sequence = np.full(sequence_length, scenario['hujan_jakarta'])
            
            # Build the sequence (12 timesteps, 3 features each)
            X_seq = np.zeros((sequence_length, 3))
            X_seq[:, 0] = bogor_sequence
            X_seq[:, 1] = jakarta_sequence
            X_seq[:, 2] = tma_sequence
            
            print(f"   📈 Sequence trend: TMA from {tma_sequence[0]:.0f} → {tma_sequence[-1]:.0f} cm")
            
            # Scale the sequence
            X_scaled = scaler_X.transform(X_seq)
            
            # Reshape for model: (1, sequence_length, features)
            X_input = X_scaled.reshape(1, sequence_length, 3)
            
            # Predict
            y_pred_scaled = model.predict(X_input, verbose=0)
            
            # Inverse Transform
            y_pred_cm = scaler_y.inverse_transform(y_pred_scaled)
            predicted_tma = float(y_pred_cm[0][0])
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            rainfall_avg = (scenario['hujan_bogor'] + scenario['hujan_jakarta']) / 2
            predicted_tma = scenario['tma_saat_ini'] + (rainfall_avg * 0.5)
            print(f"   ⚠️ Using fallback heuristic")
    
    # Analyze
    input_tma = scenario['tma_saat_ini']
    delta = predicted_tma - input_tma
    
    if delta > 5:
        status = "RISING 📈"
    elif delta < -5:
        status = "FALLING 📉"
    else:
        status = "STABLE ➡️"
    
    print(f"   🌊 Predicted TMA: {predicted_tma:.2f} cm")
    print(f"   📊 Change: {delta:+.2f} cm ({status})")
    print(f"   ✅ Expected: {scenario['expected']}")

print("\n" + "=" * 50)
print("✅ Verification Complete")
