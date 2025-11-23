import os
import sys
import dotenv
import requests
import json

# Add src to path
sys.path.append(os.getcwd())

from src.ml_core.yolo_model import FloodVisualVerifier

# Setup: Load environment variables
dotenv.load_dotenv()

def test_real_image():
    image_path = "test_banjir.jpg"
    
    # Validation
    if not os.path.exists(image_path):
        print(f"❌ Error: {image_path} not found in root directory.")
        sys.exit(1)
        
    print(f"✅ Found {image_path}")
    print("🚀 Initializing FloodVisualVerifier...")
    
    # Execution
    verifier = FloodVisualVerifier()
    
    # Note: The current implementation of FloodVisualVerifier might still have the hardcoded key 
    # or might need to be updated to use os.getenv('ROBOFLOW_API_KEY').
    # For this test, we assume the class handles authentication internally (as per previous step).
    
    print("📸 Running detection...")
    result = verifier.detect_flood_features(image_path)
    
    # Output
    print("\n⬇️ RAW JSON RESPONSE ⬇️")
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    test_real_image()
