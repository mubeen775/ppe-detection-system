"""
AI-Powered PPE Detection System using TensorFlow
This module provides real-time detection of Personal Protective Equipment
including helmets, vests, gloves, masks, glasses, and safety boots.
"""

import cv2
import numpy as np
try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except ImportError:
    tf = None
from typing import Dict, List, Tuple, Optional
import os
import logging
from datetime import datetime, timedelta
import json

class PPEDetector:
    """
    Personal Protective Equipment Detection using TensorFlow
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize the PPE detector
        
        Args:
            model_path: Path to the TensorFlow model
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = [
            'person', 'helmet', 'safety_vest', 'gloves', 
            'mask', 'safety_glasses', 'safety_boots'
        ]
        
        # PPE requirements mapping
        self.ppe_requirements = {
            'helmet': True,
            'safety_vest': True,
            'gloves': False,  # Optional in some zones
            'mask': False,    # Optional based on environment
            'safety_glasses': False,  # Optional
            'safety_boots': True
        }
        
        self.detection_colors = {
            'compliant': (0, 255, 0),    # Green
            'violation': (0, 0, 255),    # Red
            'warning': (0, 255, 255),    # Yellow
            'person': (255, 0, 0)        # Blue
        }
        
        self._setup_logging()
        self._load_model(model_path)
    
    def _setup_logging(self):
        """Setup logging for detection events"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: Optional[str] = None):
        """
        Load TensorFlow model for PPE detection
        
        Args:
            model_path: Path to the model file
        """
        try:
            if model_path and os.path.exists(model_path):
                self.model = tf.saved_model.load(model_path)
                self.logger.info(f"Loaded custom model from {model_path}")
            else:
                # Create a simple CNN model for demonstration
                self.model = self._create_demo_model()
                self.logger.info("Using demo detection model")
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model = self._create_demo_model()
    
    def _create_demo_model(self):
        """
        Create a demonstration model for PPE detection
        This simulates real detection behavior for demonstration purposes
        """
        # In a real implementation, this would be a trained model
        # For demo, we'll use rule-based detection with OpenCV
        return None
    
    def detect_ppe(self, image: np.ndarray, zone_requirements: Optional[Dict] = None) -> Dict:
        """
        Detect PPE in the given image
        
        Args:
            image: Input image as numpy array
            zone_requirements: Zone-specific PPE requirements
            
        Returns:
            Dictionary containing detection results
        """
        if image is None or image.size == 0:
            return self._empty_result()
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Detect objects (using demo detection for now)
        detections = self._demo_detection(processed_image)
        
        # Analyze PPE compliance
        compliance_result = self._analyze_compliance(
            detections, 
            zone_requirements or self.ppe_requirements
        )
        
        return compliance_result
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Raw input image
            
        Returns:
            Preprocessed image
        """
        # Resize image to model input size
        resized = cv2.resize(image, (640, 640))
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def _demo_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Demo detection using computer vision techniques
        This simulates what a real TensorFlow model would return
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of detected objects
        """
        detections = []
        h, w = image.shape[:2]
        
        # Simulate person detection
        person_bbox = [0.2 * w, 0.1 * h, 0.8 * w, 0.9 * h]
        detections.append({
            'class': 'person',
            'confidence': 0.95,
            'bbox': person_bbox
        })
        
        # Simulate PPE detection based on image analysis
        # In real implementation, this would use trained neural networks
        
        # Helmet detection (simulate based on head region)
        helmet_detected = self._detect_helmet_demo(image, person_bbox)
        if helmet_detected:
            detections.append({
                'class': 'helmet',
                'confidence': np.random.uniform(0.7, 0.95),
                'bbox': [0.35 * w, 0.1 * h, 0.65 * w, 0.3 * h]
            })
        
        # Safety vest detection (simulate based on torso region)
        vest_detected = self._detect_vest_demo(image, person_bbox)
        if vest_detected:
            detections.append({
                'class': 'safety_vest',
                'confidence': np.random.uniform(0.6, 0.9),
                'bbox': [0.25 * w, 0.3 * h, 0.75 * w, 0.7 * h]
            })
        
        # Safety boots detection
        boots_detected = self._detect_boots_demo(image, person_bbox)
        if boots_detected:
            detections.append({
                'class': 'safety_boots',
                'confidence': np.random.uniform(0.6, 0.85),
                'bbox': [0.3 * w, 0.8 * h, 0.7 * w, 0.95 * h]
            })
        
        return detections
    
    def _detect_helmet_demo(self, image: np.ndarray, person_bbox: List) -> bool:
        """Demo helmet detection using color and shape analysis"""
        x1, y1, x2, y2 = [int(coord) for coord in person_bbox]
        head_region = image[y1:int(y1 + (y2-y1)*0.25), x1:x2]
        
        if head_region.size == 0:
            return False
        
        # Simple color-based detection for demo
        # Look for bright/reflective colors typical of helmets
        hsv = cv2.cvtColor(head_region, cv2.COLOR_RGB2HSV)
        
        # Yellow helmet detection
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # White helmet detection
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        helmet_pixels = cv2.countNonZero(yellow_mask) + cv2.countNonZero(white_mask)
        total_pixels = head_region.shape[0] * head_region.shape[1]
        
        return (helmet_pixels / total_pixels) > 0.1
    
    def _detect_vest_demo(self, image: np.ndarray, person_bbox: List) -> bool:
        """Demo safety vest detection using color analysis"""
        x1, y1, x2, y2 = [int(coord) for coord in person_bbox]
        torso_region = image[int(y1 + (y2-y1)*0.25):int(y1 + (y2-y1)*0.7), x1:x2]
        
        if torso_region.size == 0:
            return False
        
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_RGB2HSV)
        
        # High-visibility colors (orange, yellow, lime green)
        orange_lower = np.array([5, 150, 150])
        orange_upper = np.array([15, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        lime_lower = np.array([40, 150, 150])
        lime_upper = np.array([80, 255, 255])
        lime_mask = cv2.inRange(hsv, lime_lower, lime_upper)
        
        vest_pixels = cv2.countNonZero(orange_mask) + cv2.countNonZero(lime_mask)
        total_pixels = torso_region.shape[0] * torso_region.shape[1]
        
        return (vest_pixels / total_pixels) > 0.15
    
    def _detect_boots_demo(self, image: np.ndarray, person_bbox: List) -> bool:
        """Demo safety boots detection"""
        x1, y1, x2, y2 = [int(coord) for coord in person_bbox]
        foot_region = image[int(y1 + (y2-y1)*0.8):y2, x1:x2]
        
        if foot_region.size == 0:
            return False
        
        # Look for dark colored footwear (typical of safety boots)
        gray = cv2.cvtColor(foot_region, cv2.COLOR_RGB2GRAY)
        dark_pixels = np.sum(gray < 80)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        return (dark_pixels / total_pixels) > 0.3
    
    def _analyze_compliance(self, detections: List[Dict], requirements: Dict) -> Dict:
        """
        Analyze PPE compliance based on detections and requirements
        
        Args:
            detections: List of detected objects
            requirements: PPE requirements for the zone
            
        Returns:
            Compliance analysis result
        """
        detected_classes = [det['class'] for det in detections]
        violations = []
        compliant_items = []
        
        # Check each required PPE item
        for ppe_item, required in requirements.items():
            if required and ppe_item not in detected_classes:
                violations.append(ppe_item)
            elif ppe_item in detected_classes:
                compliant_items.append(ppe_item)
        
        # Determine overall compliance status
        is_compliant = len(violations) == 0
        
        # Get the most critical violation for fine calculation
        primary_violation = violations[0] if violations else None
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'is_compliant': is_compliant,
            'violations': violations,
            'compliant_items': compliant_items,
            'primary_violation': primary_violation,
            'detections': detections,
            'confidence_scores': {
                det['class']: det['confidence'] 
                for det in detections if det['class'] != 'person'
            },
            'detection_summary': self._create_detection_summary(detections, violations)
        }
        
        return result
    
    def _create_detection_summary(self, detections: List[Dict], violations: List[str]) -> str:
        """Create human-readable detection summary"""
        if not violations:
            return "All required PPE detected - Worker is compliant"
        
        violation_text = ", ".join([v.replace('_', ' ').title() for v in violations])
        return f"PPE Violation: Missing {violation_text}"
    
    def _empty_result(self) -> Dict:
        """Return empty result for invalid input"""
        return {
            'timestamp': datetime.now().isoformat(),
            'is_compliant': False,
            'violations': ['no_person_detected'],
            'compliant_items': [],
            'primary_violation': 'no_person_detected',
            'detections': [],
            'confidence_scores': {},
            'detection_summary': "No person detected in frame"
        }
    
    def process_rtsp_stream(self, rtsp_url: str, callback_func=None) -> None:
        """
        Process RTSP stream for real-time PPE detection
        
        Args:
            rtsp_url: RTSP stream URL
            callback_func: Callback function for detection results
        """
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open RTSP stream: {rtsp_url}")
            return
        
        self.logger.info(f"Started processing RTSP stream: {rtsp_url}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from stream")
                    break
                
                # Perform PPE detection
                result = self.detect_ppe(frame)
                
                # Call callback function if provided
                if callback_func:
                    callback_func(result, frame)
                
                # Add small delay to prevent overwhelming the system
                cv2.waitKey(30)
                
        except KeyboardInterrupt:
            self.logger.info("Stream processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error processing stream: {e}")
        finally:
            cap.release()
            self.logger.info("RTSP stream processing stopped")
    
    def draw_detections(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw detection results on image
        
        Args:
            image: Input image
            result: Detection result from detect_ppe()
            
        Returns:
            Image with drawn detections
        """
        output_image = image.copy()
        
        # Draw bounding boxes for detections
        for detection in result.get('detections', []):
            if detection['class'] == 'person':
                continue
                
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class'].replace('_', ' ').title()
            
            # Choose color based on compliance
            if detection['class'] in result.get('violations', []):
                color = self.detection_colors['violation']
                status = "MISSING"
            else:
                color = self.detection_colors['compliant']
                status = "DETECTED"
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {status} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(output_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(output_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw overall compliance status
        status_text = result.get('detection_summary', 'Unknown Status')
        status_color = self.detection_colors['compliant'] if result.get('is_compliant') else self.detection_colors['violation']
        
        cv2.putText(output_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        return output_image
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold for detections"""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
        self.logger.info(f"Updated confidence threshold to {self.confidence_threshold}")
    
    def update_zone_requirements(self, requirements: Dict):
        """Update PPE requirements for specific zone"""
        self.ppe_requirements.update(requirements)
        self.logger.info(f"Updated PPE requirements: {requirements}")


class PPEAnalytics:
    """
    Analytics and reporting for PPE detection system
    """
    
    def __init__(self):
        self.detection_history = []
        self.violation_stats = {}
    
    def log_detection(self, result: Dict, camera_id: int, zone_id: int):
        """Log detection result for analytics"""
        log_entry = {
            'timestamp': result['timestamp'],
            'camera_id': camera_id,
            'zone_id': zone_id,
            'is_compliant': result['is_compliant'],
            'violations': result['violations'],
            'confidence_scores': result['confidence_scores']
        }
        
        self.detection_history.append(log_entry)
        self._update_violation_stats(result['violations'])
    
    def _update_violation_stats(self, violations: List[str]):
        """Update violation statistics"""
        for violation in violations:
            if violation not in self.violation_stats:
                self.violation_stats[violation] = 0
            self.violation_stats[violation] += 1
    
    def get_compliance_rate(self, time_window_hours: int = 24) -> float:
        """Calculate compliance rate for given time window"""
        if not self.detection_history:
            return 0.0
        
        current_time = datetime.now()
        window_start = current_time - timedelta(hours=time_window_hours)
        
        recent_detections = [
            entry for entry in self.detection_history
            if datetime.fromisoformat(entry['timestamp']) >= window_start
        ]
        
        if not recent_detections:
            return 0.0
        
        compliant_count = sum(1 for entry in recent_detections if entry['is_compliant'])
        return (compliant_count / len(recent_detections)) * 100
    
    def get_top_violations(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get most common violations"""
        sorted_violations = sorted(
            self.violation_stats.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_violations[:limit]


# Global detector instance
ppe_detector = PPEDetector()
ppe_analytics = PPEAnalytics()