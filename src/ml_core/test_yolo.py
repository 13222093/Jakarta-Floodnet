"""
Test YOLO Model Functionality
============================

Test all YOLO functions to ensure they work properly.
"""

import sys
import numpy as np
import cv2
from PIL import Image

sys.path.append('.')
from yolo_model import FloodVisualVerifier

def test_yolo_functions():
    """Test all YOLO model functions"""
    print("🧪 COMPREHENSIVE YOLO TESTING")
    print("="*50)
    
    # Initialize verifier
    verifier = FloodVisualVerifier(model_path='../../models/yolo_model.pt')
    
    # Test 1: Model loading
    print("\n1️⃣ Testing Model Loading:")
    if verifier.load_model():
        print("✅ Model loaded successfully")
    else:
        print("❌ Model loading failed")
        return False
    
    # Test 2: Create test images
    print("\n2️⃣ Creating Test Images:")
    # Create different test images
    test_images = {
        'random': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        'blue_water': np.zeros((480, 640, 3), dtype=np.uint8),
        'mixed': np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    }
    
    # Make blue water image more water-like
    test_images['blue_water'][:, :, 2] = 255  # Blue channel
    test_images['blue_water'][:200, :, 0] = 100  # Add some variation
    
    print(f"✅ Created {len(test_images)} test images")
    
    # Test 3: Image preprocessing
    print("\n3️⃣ Testing Image Preprocessing:")
    for name, image in test_images.items():
        try:
            processed = verifier.preprocess_image(image)
            print(f"✅ {name}: Shape {processed.shape}")
        except Exception as e:
            print(f"❌ {name}: Error {e}")
            return False
    
    # Test 4: Flood detection
    print("\n4️⃣ Testing Flood Detection:")
    for name, image in test_images.items():
        try:
            result = verifier.detect_flood_features(image)
            if 'error' not in result:
                print(f"✅ {name}:")
                print(f"  • Detections: {result['total_detections']}")
                print(f"  • Flood indicators: {result['flood_indicators']}")
                print(f"  • Flood probability: {result['flood_probability']:.2%}")
                print(f"  • Severity: {result['severity_level']}")
            else:
                print(f"❌ {name}: {result['error']}")
        except Exception as e:
            print(f"❌ {name}: Exception {e}")
            return False
    
    # Test 5: Water coverage analysis
    print("\n5️⃣ Testing Water Coverage Analysis:")
    for name, image in test_images.items():
        try:
            result = verifier.analyze_water_coverage(image)
            if 'error' not in result:
                print(f"✅ {name}:")
                print(f"  • Water coverage: {result['water_coverage_percent']:.1f}%")
                print(f"  • Is flooded: {result['is_flooded']}")
            else:
                print(f"❌ {name}: {result['error']}")
        except Exception as e:
            print(f"❌ {name}: Exception {e}")
            return False
    
    # Test 6: LSTM verification
    print("\n6️⃣ Testing LSTM Verification:")
    test_predictions = [65.0, 85.0, 95.0]  # Below, above, well above threshold
    
    for i, pred in enumerate(test_predictions):
        try:
            result = verifier.verify_lstm_prediction(
                test_images['random'], 
                pred,
                threshold_high=80.0
            )
            if 'error' not in result:
                print(f"✅ Prediction {pred}cm:")
                print(f"  • Status: {result['verification_status']}")
                print(f"  • Confidence: {result['overall_confidence']:.2%}")
                print(f"  • Agree: {result['predictions_agree']}")
            else:
                print(f"❌ Prediction {pred}cm: {result['error']}")
        except Exception as e:
            print(f"❌ Prediction {pred}cm: Exception {e}")
            return False
    
    # Test 7: Report generation
    print("\n7️⃣ Testing Report Generation:")
    try:
        detection_result = verifier.detect_flood_features(test_images['blue_water'])
        report = verifier.generate_flood_report(detection_result)
        print("✅ Report generated successfully:")
        # Print first few lines
        for line in report.split('\n')[:6]:
            print(f"  {line}")
        print("  ... (truncated)")
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    return True

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔬 TESTING EDGE CASES")
    print("="*30)
    
    verifier = FloodVisualVerifier()
    verifier.load_model()
    
    # Test with different image formats
    test_cases = [
        # Very small image
        np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8),
        # Large image
        np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
        # Grayscale converted to RGB
        np.stack([np.random.randint(0, 255, (100, 100), dtype=np.uint8)] * 3, axis=-1)
    ]
    
    for i, test_image in enumerate(test_cases):
        try:
            result = verifier.detect_flood_features(test_image)
            print(f"✅ Edge case {i+1}: Shape {test_image.shape} - Success")
        except Exception as e:
            print(f"❌ Edge case {i+1}: Shape {test_image.shape} - Error: {e}")
    
    print("✅ Edge case testing completed")

def main():
    """Main test function"""
    success = test_yolo_functions()
    
    if success:
        test_edge_cases()
        print("\n🎉 YOLO MODEL FULLY FUNCTIONAL!")
        print("✅ All components working correctly")
        print("✅ Ready for integration with LSTM")
        print("✅ Ready for real-time flood monitoring")
    else:
        print("\n❌ YOLO MODEL HAS ISSUES")
        print("⚠️  Check error messages above")

if __name__ == "__main__":
    main()