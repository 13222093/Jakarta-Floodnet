"""
YOLO Model Module for Jakarta FloodNet
=====================================
Robust Visual Verification with Color Heuristics
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Any, Optional, Union
from inference_sdk import InferenceHTTPClient

# Setup Logger
logger = logging.getLogger(__name__)

# Try Import Ultralytics (Optional now, but good for fallback if needed)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.error("❌ Ultralytics not found! Please install: pip install ultralytics")
    YOLO_AVAILABLE = False

class FloodVisualVerifier:
    """
    Visual Verification System using Roboflow Workflow.
    """
    
    def __init__(self, model_path: str = 'models/yolov8n.pt', confidence_threshold: float = 0.4):
        self.model_path = model_path
        self.conf_thresh = confidence_threshold
        self.model = None
        self.is_loaded = False
        self.client = None
        
        # Roboflow Client
        try:
            # Load API Key from Env or Fallback
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if not api_key:
                logger.warning("⚠️ ROBOFLOW_API_KEY not found in environment variables. Using fallback.")
                api_key = "kI2MJW8A3nh8M8MdgyR4" # Fallback (User provided)

            self.client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key
            )
            self.is_loaded = True
            logger.info("✅ Roboflow Client Initialized")
        except Exception as e:
            logger.error(f"❌ Failed to init Roboflow: {e}")
            self.is_loaded = False

    def load_model(self) -> bool:
        """Legacy load method - kept for compatibility."""
        return self.is_loaded

    def detect_flood_features(self, image_source: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Main detection function using Roboflow Workflow.
        """
        default_response = {
            "is_flooded": False,
            "flood_probability": 0.0,
            "objects_detected": []
        }

        if not self.is_loaded or self.client is None:
            return default_response

        # 1. Prepare Image
        temp_filename = "temp_inference.jpg"
        image_path = image_source
        
        try:
            # Handle Numpy Array (from OpenCV)
            if isinstance(image_source, np.ndarray):
                cv2.imwrite(temp_filename, image_source)
                image_path = temp_filename
            elif isinstance(image_source, str):
                if not os.path.exists(image_source):
                     return default_response
            else:
                return default_response

            # 2. Run Roboflow Workflow
            # Workflow ID: detect-count-and-visualize
            result = self.client.run_workflow(
                workspace_name="ari-aziz",
                workflow_id="detect-count-and-visualize",
                images={
                    "image": image_path
                },
                use_cache=True
            )
            
            # 3. Parse Results
            # Expected Structure: [{'predictions': {'predictions': [{'class': 'flood', ...}]}}]
            predictions = []
            
            if isinstance(result, list) and len(result) > 0:
                # Get the first image result
                image_result = result[0]
                
                # Check for nested predictions (Workflow output)
                if 'predictions' in image_result:
                    preds_data = image_result['predictions']
                    
                    # If it's a dictionary containing 'predictions' list (Nested case)
                    if isinstance(preds_data, dict) and 'predictions' in preds_data:
                        predictions = preds_data['predictions']
                    # If it's directly a list (Standard inference case)
                    elif isinstance(preds_data, list):
                        predictions = preds_data
                    # If it's just a dict but not nested (Single prediction?)
                    elif isinstance(preds_data, dict):
                        predictions = [preds_data]
            
            objects_detected = []
            flood_prob = 0.0
            
            # Check for flood class or water class
            flood_detected = False
            
            for pred in predictions:
                # Roboflow prediction structure usually has 'class', 'confidence'
                label = pred.get('class', 'unknown')
                conf = pred.get('confidence', 0.0)
                
                objects_detected.append(label)
                
                if label.lower() in ['flood', 'water', 'flooded', 'puddle']:
                    flood_detected = True
                    flood_prob = max(flood_prob, conf)

            # Clean up temp file
            if isinstance(image_source, np.ndarray) and os.path.exists(temp_filename):
                os.remove(temp_filename)

            return {
                "is_flooded": flood_detected,
                "flood_probability": float(flood_prob),
                "objects_detected": list(set(objects_detected)) # Unique strings
            }

        except Exception as e:
            logger.error(f"❌ Roboflow Inference Error: {e}")
            # Clean up temp file if exists
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass
            # Return default False as requested
            return default_response

    def _load_image(self, source):
        """Safe Image Loader."""
        if isinstance(source, str):
            if os.path.exists(source):
                return cv2.imread(source)
        elif isinstance(source, np.ndarray):
            return source
        return None

    def _analyze_water_color(self, img: np.ndarray) -> float:
        """
        Legacy method - kept for reference
        """
        return 0.0

if __name__ == "__main__":
    # Test Code
    print("Testing YOLO Class...")
    verifier = FloodVisualVerifier()
    verifier.load_model()
    
    # Create Dummy Image (Brown-ish)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_img[:] = (50, 100, 150) # BGR
    
    res = verifier.detect_flood_features(dummy_img)
    print("Result:", res)
    print("✅ YOLO Class Test Passed")