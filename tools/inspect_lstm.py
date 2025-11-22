import os
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = 'models/lstm_model.h5'

def inspect_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at {MODEL_PATH}")
        return

    try:
        print(f"🔍 Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        
        print("\n✅ Model Loaded Successfully!")
        print("\n📊 Model Summary:")
        model.summary()
        
        print("\n📏 Input Shape:")
        # Check input shape
        input_shape = model.input_shape
        print(f"   {input_shape}")
        
        print("\nℹ️ Expected Input (Batch Size, Time Steps, Features):")
        if input_shape:
             print(f"   Time Steps: {input_shape[1]}")
             print(f"   Features: {input_shape[2]}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    inspect_model()
