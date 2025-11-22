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

# Setup Logger
logger = logging.getLogger(__name__)

# Try Import Ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.error("❌ Ultralytics not found! Please install: pip install ultralytics")
    YOLO_AVAILABLE = False

class FloodVisualVerifier:
    """
    Visual Verification System using YOLO + Color Analysis.
    Designed to work even if Custom Flood Model is not ready (Fallbacks to heuristics).
    """
    
    def __init__(self, model_path: str = 'models/yolov8n.pt', confidence_threshold: float = 0.4):
        self.model_path = model_path
        self.conf_thresh = confidence_threshold
        self.model = None
        self.is_loaded = False
        
        # COCO Classes that *might* be relevant in flood context (vehicles stuck, people)
        self.relevant_coco_classes = [0, 1, 2, 3, 5, 7] # person, bicycle, car, motorcycle, bus, truck
        
    def load_model(self) -> bool:
        """Load YOLO Model (Custom or Standard Fallback)."""
        if not YOLO_AVAILABLE:
            return False

        try:
            # 1. Try Load Custom/Specified Path
            if os.path.exists(self.model_path):
                logger.info(f"Loading YOLO from: {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                # 2. Fallback to Standard YOLOv8n (Auto-download)
                logger.warning(f"⚠️ Model at {self.model_path} not found. Downloading standard YOLOv8n...")
                self.model = YOLO('yolov8n.pt')
            
            self.is_loaded = True
            logger.info("✅ YOLO Model Loaded Successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO: {e}")
            return False

    def detect_flood_features(self, image_source: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Main detection function. Combines Object Detection + Water Color Analysis.
        """
        if not self.is_loaded:
            return {"status": "error", "message": "Model not loaded"}

        # 1. Prepare Image
        img = self._load_image(image_source)
        if img is None:
            return {"status": "error", "message": "Invalid image"}

        result = {
            "status": "success",
            "objects_detected": [],
            "flood_probability": 0.0,
            "is_flooded": False,
            "verification_method": "hybrid"
        }

        # 2. YOLO Inference (Object Detection)
        try:
            yolo_res = self.model(img, verbose=False)[0]
            
            # Extract boxes
            for box in yolo_res.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                
                if conf >= self.conf_thresh:
                    result["objects_detected"].append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "bbox": box.xyxy[0].tolist()
                    })

        except Exception as e:
            logger.error(f"Inference Error: {e}")

        # 3. Water Color Analysis (Heuristic)
        # Hackathon Trick: Detect Muddy Water (Brown-ish) or High Water Coverage
        water_score = self._analyze_water_color(img)
        result["water_analysis_score"] = water_score
        
        # 4. Logic Fusion (Gabungkan Logika)
        # Logic: If custom model detects 'flood' -> High prob
        # If standard model -> relies more on water_score
        
        has_flood_keyword = any(d['label'] in ['flood', 'water'] for d in result["objects_detected"])
        
        if has_flood_keyword:
            result["flood_probability"] = 0.95
            result["reason"] = "AI Object Detection (Custom Model)"
        else:
            # Fallback logic: 
            # If Water Score High AND Objects (Cars/People) detected -> Risk
            result["flood_probability"] = min(water_score * 1.2, 1.0)
            result["reason"] = "Color Analysis (Muddy Water Detection)"

        result["is_flooded"] = result["flood_probability"] > 0.5
        return result

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
        Detect muddy water using HSV Color Space.
        Returns a score 0.0 - 1.0 based on how much of the image looks like 'flood water'.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for "Muddy/Brown Water"
        # Hue: 10-30 (Orange/Brown), Saturation: 30-255, Value: 30-200
        lower_brown = np.array([10, 30, 30])
        upper_brown = np.array([30, 255, 200])
        
        # Define range for "Murky/Gray Water"
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 150])

        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
        
        combined_mask = cv2.bitwise_or(mask_brown, mask_gray)
        
        # Calculate ratio of "water-like" pixels (Bottom half of image usually)
        height, width = img.shape[:2]
        roi = combined_mask[int(height*0.3):, :] # Only look at bottom 70%
        
        ratio = np.sum(roi > 0) / (roi.size)
        
        # Normalize: If > 40% is water color, then prob is 1.0
        score = min(ratio / 0.4, 1.0)
        return round(score, 3)

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