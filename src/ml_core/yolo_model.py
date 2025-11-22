"""
YOLO Model Module for Jakarta FloodNet Visual Verification
=========================================================

This module contains the YOLO model class for visual flood verification
from satellite images, drone footage, or CCTV feeds.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Any
import logging

# Import YOLO with proper error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ Ultralytics YOLO not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

from PIL import Image

class FloodVisualVerifier:
    """
    YOLO-based visual flood verification system for Jakarta FloodNet.
    
    This class handles:
    - Loading pre-trained YOLO models
    - Detecting flood-related objects (water bodies, flooded areas, vehicles in water)
    - Analyzing flood severity from visual data
    - Providing confidence scores for flood detection
    """
    
    def __init__(
        self, 
        model_path: str = 'models/yolo_model.pt',
        confidence_threshold: float = 0.5,
        device: str = 'auto'
    ):
        """
        Initialize YOLO flood verification model.
        
        Args:
            model_path: Path to trained YOLO model weights
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.is_loaded = False
        
        # Flood-related class names (customize based on your training data)
        self.flood_classes = {
            'flooded_road': 0,
            'water_body': 1,
            'submerged_vehicle': 2,
            'flooded_building': 3,
            'debris_in_water': 4,
            'emergency_vehicle': 5,
            'person_in_water': 6,
            'flood_barrier': 7
        }
        
        # Severity levels based on detections
        self.severity_levels = {
            'no_flood': 0,
            'minor_flood': 1,
            'moderate_flood': 2,
            'major_flood': 3,
            'severe_flood': 4
        }
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the YOLO model"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> bool:
        """
        Load the YOLO model with proper error handling.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available - install ultralytics package")
            return False
            
        try:
            # Check if custom model exists
            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 0:
                self.logger.info(f"Loading custom model from {self.model_path}")
                self.model = YOLO(self.model_path)
                self.logger.info(f"✅ Loaded custom YOLO model from {self.model_path}")
            else:
                # Use pre-trained YOLOv8 as fallback
                self.logger.warning(f"Custom model not found or empty, using YOLOv8n as fallback")
                
                # Try to load YOLOv8n (will download if needed)
                try:
                    self.model = YOLO('yolov8n.pt')
                    self.logger.info("✅ Loaded YOLOv8n fallback model")
                except Exception as e:
                    self.logger.error(f"Failed to load YOLOv8n: {e}")
                    return False
                
            # Test the model with a dummy prediction
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            test_results = self.model(test_image, verbose=False)
            
            self.is_loaded = True
            self.logger.info("✅ Model loaded and tested successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading YOLO model: {str(e)}")
            self.logger.error(f"Model path: {self.model_path}")
            self.logger.error(f"Path exists: {os.path.exists(self.model_path)}")
            if os.path.exists(self.model_path):
                self.logger.error(f"File size: {os.path.getsize(self.model_path)} bytes")
            return False
    
    def preprocess_image(self, image_source: Any) -> Optional[np.ndarray]:
        """
        Preprocess image for YOLO inference with better error handling.
        
        Args:
            image_source: Image path, numpy array, or PIL Image
            
        Returns:
            Preprocessed image as numpy array, or None if failed
        """
        try:
            if isinstance(image_source, str):
                # Load from file path
                if not os.path.exists(image_source):
                    self.logger.error(f"Image file not found: {image_source}")
                    return None
                    
                image = cv2.imread(image_source)
                if image is None:
                    self.logger.error(f"Failed to load image: {image_source}")
                    return None
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_source, np.ndarray):
                # Already numpy array
                image = image_source.copy()
                
                # Ensure it's 3-channel RGB
                if len(image.shape) == 2:
                    # Grayscale to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    # RGBA to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    
            elif isinstance(image_source, Image.Image):
                # PIL Image
                image = np.array(image_source)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    
            else:
                self.logger.error(f"Unsupported image format: {type(image_source)}")
                return None
            
            # Validate image
            if image.shape[2] != 3:
                self.logger.error(f"Image must have 3 channels, got {image.shape[2]}")
                return None
                
            return image
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return None
    
    def detect_flood_features(self, image_source: Any) -> Dict[str, Any]:
        """
        Detect flood-related features in an image.
        
        Args:
            image_source: Image to analyze
            
        Returns:
            Dictionary with detection results
        """
        if not YOLO_AVAILABLE:
            return {'error': 'YOLO not available - install ultralytics package'}
            
        if not self.is_loaded:
            if not self.load_model():
                return {'error': 'Failed to load YOLO model'}
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_source)
            if image is None:
                return {'error': 'Failed to preprocess image'}
            
            # Run YOLO inference with error handling
            try:
                results = self.model(image, conf=self.confidence_threshold, verbose=False)
            except Exception as e:
                self.logger.error(f"YOLO inference failed: {e}")
                return {'error': f'YOLO inference failed: {str(e)}'}
            
            # Parse results safely
            detections = []
            total_confidence = 0
            flood_indicators = 0
            
            try:
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        
                        # Check if boxes have data
                        if len(boxes) == 0:
                            continue
                            
                        for i in range(len(boxes.cls)):
                            # Extract detection data safely
                            class_id = int(boxes.cls[i])
                            confidence = float(boxes.conf[i])
                            bbox = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                            
                            # Get class name safely
                            class_name = "unknown"
                            if hasattr(result, 'names') and class_id in result.names:
                                class_name = result.names[class_id]
                            else:
                                class_name = f"class_{class_id}"
                            
                            detection = {
                                'class_id': class_id,
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': bbox,
                                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            }
                            
                            detections.append(detection)
                            
                            # Count flood indicators
                            if self._is_flood_related(class_name):
                                flood_indicators += 1
                                total_confidence += confidence
                                
            except Exception as e:
                self.logger.error(f"Error parsing YOLO results: {e}")
                # Continue with empty detections rather than failing
                pass
            
            # Calculate flood probability
            if flood_indicators > 0 and len(detections) > 0:
                avg_confidence = total_confidence / flood_indicators
                flood_probability = min(avg_confidence * (flood_indicators / max(len(detections), 1)), 1.0)
            else:
                flood_probability = 0.0
            
            return {
                'detections': detections,
                'flood_indicators': flood_indicators,
                'total_detections': len(detections),
                'flood_probability': flood_probability,
                'severity_level': self._calculate_severity(detections, flood_indicators),
                'image_shape': image.shape
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in flood detection: {str(e)}")
            return {'error': f'Flood detection failed: {str(e)}'}
    
    def _is_flood_related(self, class_name: str) -> bool:
        """
        Check if detected class is flood-related.
        
        Args:
            class_name: Name of detected class
            
        Returns:
            True if flood-related, False otherwise
        """
        flood_keywords = [
            'water', 'flood', 'submerged', 'debris', 'emergency',
            'rescue', 'boat', 'raft', 'inundated', 'overflow'
        ]
        
        class_lower = class_name.lower()
        return any(keyword in class_lower for keyword in flood_keywords)
    
    def _calculate_severity(self, detections: List[Dict], flood_indicators: int) -> str:
        """
        Calculate flood severity based on detections.
        
        Args:
            detections: List of all detections
            flood_indicators: Number of flood-related detections
            
        Returns:
            Severity level string
        """
        if flood_indicators == 0:
            return 'no_flood'
        elif flood_indicators <= 2:
            return 'minor_flood'
        elif flood_indicators <= 5:
            return 'moderate_flood'
        elif flood_indicators <= 10:
            return 'major_flood'
        else:
            return 'severe_flood'
    
    def analyze_water_coverage(self, image_source: Any) -> Dict[str, float]:
        """
        Analyze water coverage percentage in image.
        
        Args:
            image_source: Image to analyze
            
        Returns:
            Dictionary with water coverage analysis
        """
        try:
            image = self.preprocess_image(image_source)
            if image is None:
                return {'error': 'Failed to preprocess image for water coverage analysis'}
            
            # Convert to HSV for better water detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Define range for blue/water colors
            lower_water = np.array([100, 50, 50])
            upper_water = np.array([130, 255, 255])
            
            # Create mask for water areas
            water_mask = cv2.inRange(hsv, lower_water, upper_water)
            
            # Calculate coverage
            total_pixels = image.shape[0] * image.shape[1]
            water_pixels = np.sum(water_mask > 0)
            water_coverage = water_pixels / total_pixels
            
            return {
                'water_coverage_percent': water_coverage * 100,
                'water_pixels': int(water_pixels),
                'total_pixels': int(total_pixels),
                'is_flooded': water_coverage > 0.15  # 15% threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error in water coverage analysis: {str(e)}")
            return {'error': str(e)}
    
    def verify_lstm_prediction(
        self, 
        lstm_prediction: Dict[str, Any], 
        image_source: Any,
        threshold_high: float = 150.0  # Water level threshold in cm
    ) -> Dict[str, Any]:
        """
        Verify LSTM flood prediction using visual evidence.
        
        Args:
            lstm_prediction: Dictionary with LSTM prediction results
            image_source: Current image for verification
            threshold_high: High water level threshold in cm
            
        Returns:
            Verification results
        """
        try:
            # Extract LSTM prediction value
            pred_value = lstm_prediction.get('prediction', 0.0)
            
            # Get visual analysis
            flood_detection = self.detect_flood_features(image_source)
            water_analysis = self.analyze_water_coverage(image_source)
            
            # Check for errors in visual analysis
            if 'error' in flood_detection:
                return {
                    'verification_status': "ERROR",
                    'error': f"Flood detection failed: {flood_detection['error']}"
                }
            
            if 'error' in water_analysis:
                return {
                    'verification_status': "ERROR", 
                    'error': f"Water analysis failed: {water_analysis['error']}"
                }
            
            # LSTM prediction analysis
            lstm_flood_predicted = pred_value >= threshold_high
            
            # Visual verification
            visual_flood_detected = (
                flood_detection['flood_probability'] > 0.3 or 
                water_analysis['water_coverage_percent'] > 20
            )
            
            # Agreement analysis
            predictions_agree = lstm_flood_predicted == visual_flood_detected
            
            # Confidence calculation
            visual_confidence = (flood_detection['flood_probability'] + 
                               (water_analysis['water_coverage_percent'] / 100)) / 2
            
            # Overall assessment
            if predictions_agree:
                if lstm_flood_predicted and visual_flood_detected:
                    verification_status = "FLOOD_CONFIRMED"
                    confidence = 0.8 + (visual_confidence * 0.2)
                else:
                    verification_status = "NO_FLOOD_CONFIRMED"
                    confidence = 0.7 + ((1 - visual_confidence) * 0.3)
            else:
                if lstm_flood_predicted and not visual_flood_detected:
                    verification_status = "LSTM_FALSE_POSITIVE"
                    confidence = 0.3 + ((1 - visual_confidence) * 0.4)
                else:
                    verification_status = "POTENTIAL_VISUAL_MISS"
                    confidence = 0.4 + (visual_confidence * 0.3)
            
            return {
                'verification_status': verification_status,
                'visual_confirmation': visual_flood_detected,
                'consensus': predictions_agree,
                'reliability_score': confidence,
                'lstm_prediction': pred_value,
                'lstm_flood_predicted': lstm_flood_predicted,
                'visual_flood_detected': visual_flood_detected,
                'visual_confidence': visual_confidence,
                'flood_severity': flood_detection['severity_level'],
                'water_coverage': water_analysis['water_coverage_percent'],
                'flood_detections': flood_detection['flood_indicators']
            }
            
        except Exception as e:
            self.logger.error(f"Error in LSTM verification: {str(e)}")
            return {
                'verification_status': "ERROR",
                'error': f"Verification failed: {str(e)}"
            }
    
    def process_video_stream(
        self, 
        video_source: Any,
        frame_skip: int = 5,
        max_frames: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Process video stream for flood detection.
        
        Args:
            video_source: Video file path or camera index
            frame_skip: Process every nth frame
            max_frames: Maximum frames to process
            
        Returns:
            List of detection results per frame
        """
        if not self.is_loaded:
            if not self.load_model():
                return [{'error': 'Failed to load model'}]
        
        try:
            # Open video
            if isinstance(video_source, int):
                cap = cv2.VideoCapture(video_source)  # Camera
            else:
                cap = cv2.VideoCapture(video_source)  # Video file
                
            results = []
            frame_count = 0
            processed_frames = 0
            
            while cap.isOpened() and processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for efficiency
                if frame_count % frame_skip == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect flood features
                    detection_result = self.detect_flood_features(frame_rgb)
                    detection_result['frame_number'] = frame_count
                    detection_result['timestamp'] = frame_count / 30.0  # Assume 30 FPS
                    
                    results.append(detection_result)
                    processed_frames += 1
                
                frame_count += 1
            
            cap.release()
            return results
            
        except Exception as e:
            self.logger.error(f"Error in video processing: {str(e)}")
            return [{'error': str(e)}]
    
    def generate_flood_report(self, detection_results: Dict[str, Any]) -> str:
        """
        Generate human-readable flood report from detection results.
        
        Args:
            detection_results: Results from detect_flood_features
            
        Returns:
            Formatted report string
        """
        if 'error' in detection_results:
            return f"❌ Analysis Error: {detection_results['error']}"
        
        report = []
        report.append("🌊 FLOOD VISUAL VERIFICATION REPORT")
        report.append("=" * 40)
        
        # Basic statistics
        report.append(f"📊 Detection Summary:")
        report.append(f"  • Total objects detected: {detection_results['total_detections']}")
        report.append(f"  • Flood indicators found: {detection_results['flood_indicators']}")
        report.append(f"  • Flood probability: {detection_results['flood_probability']:.2%}")
        report.append(f"  • Severity level: {detection_results['severity_level'].upper()}")
        
        # Detailed detections
        if detection_results['detections']:
            report.append(f"\n🔍 Detailed Detections:")
            for i, det in enumerate(detection_results['detections'][:10]):  # Limit to 10
                report.append(f"  {i+1}. {det['class_name']} (confidence: {det['confidence']:.2f})")
        
        # Recommendations
        report.append(f"\n💡 Recommendations:")
        severity = detection_results['severity_level']
        if severity == 'no_flood':
            report.append("  ✅ No immediate flood threat detected")
        elif severity == 'minor_flood':
            report.append("  ⚠️  Minor flooding detected - monitor situation")
        elif severity in ['moderate_flood', 'major_flood']:
            report.append("  🚨 Significant flooding detected - take precautions")
        else:
            report.append("  🆘 Severe flooding detected - immediate action required")
        
        return "\n".join(report)

def create_yolo_verifier(model_path: str = 'models/yolo_model.pt') -> FloodVisualVerifier:
    """
    Factory function to create YOLO flood verifier.
    
    Args:
        model_path: Path to YOLO model weights
        
    Returns:
        Configured FloodVisualVerifier instance
    """
    return FloodVisualVerifier(model_path=model_path)

# Demo function for testing
def demo_yolo_verification():
    """Demo function to test YOLO flood verification"""
    print("🧪 YOLO Flood Verification Demo")
    print("="*40)
    
    try:
        verifier = create_yolo_verifier()
        
        # Try to load model
        if verifier.load_model():
            print("✅ YOLO model loaded successfully")
            print(f"📋 Flood classes: {list(verifier.flood_classes.keys())}")
            print(f"📊 Severity levels: {list(verifier.severity_levels.keys())}")
            print("🎯 Ready for visual flood verification!")
        else:
            print("❌ Failed to load YOLO model")
            
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")

if __name__ == "__main__":
    demo_yolo_verification()
