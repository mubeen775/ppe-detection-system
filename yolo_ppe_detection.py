#!/usr/bin/env python3
"""
YOLOv8 PPE Detection Integration
Manual YOLO integration for PPE detection with fallback support
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOPPEDetector:
    """
    YOLOv8-based PPE detection with manual integration
    Falls back to computer vision simulation if YOLO unavailable
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25):
        """
        Initialize YOLO PPE detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.yolo_available = False
        
        # PPE class mappings for YOLO
        self.ppe_classes = {
            0: 'person',
            1: 'helmet', 
            2: 'vest',
            3: 'gloves',
            4: 'mask',
            5: 'glasses',
            6: 'boots'
        }
        
        # Try to load YOLO model
        self._load_yolo_model(model_path)
        
        # Setup fallback detection
        if not self.yolo_available:
            logger.warning("YOLO not available, using enhanced simulation mode")
            self._setup_fallback_detection()
    
    def _load_yolo_model(self, model_path: Optional[str] = None):
        """
        Attempt to load YOLO model with multiple methods
        """
        try:
            # Method 1: Try ultralytics YOLO
            from ultralytics import YOLO
            
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                logger.info(f"YOLO model loaded from {model_path}")
            else:
                # Try to download YOLOv8 nano model
                self.model = YOLO("yolov8n.pt")
                logger.info("YOLOv8n model downloaded and loaded")
            
            self.yolo_available = True
            logger.info("✅ Real YOLO model successfully loaded")
            
        except ImportError:
            logger.warning("ultralytics not available, trying manual YOLO loading...")
            self._try_manual_yolo_load(model_path)
            
        except Exception as e:
            logger.warning(f"YOLO loading failed: {e}")
            self._try_manual_yolo_load(model_path)
    
    def _try_manual_yolo_load(self, model_path: Optional[str] = None):
        """
        Try manual YOLO loading methods
        """
        try:
            # Method 2: Try OpenCV DNN with YOLO
            if model_path:
                weights_path = model_path
                config_path = model_path.replace('.weights', '.cfg')
                
                if os.path.exists(weights_path) and os.path.exists(config_path):
                    self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
                    self.yolo_available = True
                    self.yolo_type = "opencv_dnn"
                    logger.info("✅ YOLO loaded via OpenCV DNN")
                    return
            
            # Method 3: Try ONNX YOLO model
            try:
                import onnxruntime as ort
                onnx_path = "yolov8n.onnx"
                if os.path.exists(onnx_path):
                    self.model = ort.InferenceSession(onnx_path)
                    self.yolo_available = True
                    self.yolo_type = "onnx"
                    logger.info("✅ YOLO loaded via ONNX Runtime")
                    return
            except ImportError:
                pass
            
            # Method 4: Try TensorFlow Lite
            try:
                import tensorflow as tf
                tflite_path = "yolov8n.tflite"
                if os.path.exists(tflite_path):
                    self.model = tf.lite.Interpreter(model_path=tflite_path)
                    self.model.allocate_tensors()
                    self.yolo_available = True
                    self.yolo_type = "tflite"
                    logger.info("✅ YOLO loaded via TensorFlow Lite")
                    return
            except ImportError:
                pass
                
        except Exception as e:
            logger.warning(f"Manual YOLO loading failed: {e}")
    
    def _setup_fallback_detection(self):
        """
        Setup enhanced computer vision fallback when YOLO unavailable
        """
        logger.info("Setting up enhanced CV fallback detection")
        
        # Load OpenCV classifiers for fallback
        try:
            # Face cascade for person detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            # Upper body cascade for person detection
            self.body_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            )
            logger.info("OpenCV cascades loaded for fallback detection")
        except Exception as e:
            logger.warning(f"Cascade loading failed: {e}")
            self.face_cascade = None
            self.body_cascade = None
    
    def detect_ppe(self, image: np.ndarray, zone_requirements: Optional[Dict] = None) -> Dict:
        """
        Detect PPE using YOLO or fallback method
        
        Args:
            image: Input image as numpy array
            zone_requirements: Zone-specific PPE requirements
            
        Returns:
            Detection results dictionary
        """
        if image is None or image.size == 0:
            return self._empty_result()
        
        start_time = time.time()
        
        try:
            if self.yolo_available:
                result = self._yolo_detection(image, zone_requirements)
            else:
                result = self._fallback_detection(image, zone_requirements)
                
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['fps'] = 1.0 / processing_time if processing_time > 0 else 0
            
            return result
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return self._empty_result()
    
    def _yolo_detection(self, image: np.ndarray, zone_requirements: Optional[Dict] = None) -> Dict:
        """
        Real YOLO detection method
        """
        try:
            # Run YOLO inference
            if hasattr(self.model, 'predict'):
                # Ultralytics YOLO
                results = self.model.predict(
                    image, 
                    conf=self.confidence_threshold,
                    verbose=False
                )
                detections = self._parse_ultralytics_results(results)
                
            elif hasattr(self.model, 'run'):
                # ONNX Runtime
                detections = self._run_onnx_inference(image)
                
            elif hasattr(self.model, 'invoke'):
                # TensorFlow Lite
                detections = self._run_tflite_inference(image)
                
            else:
                # OpenCV DNN
                detections = self._run_opencv_dnn_inference(image)
            
            # Analyze compliance
            compliance = self._analyze_compliance_yolo(detections, zone_requirements or {})
            
            return {
                'method': 'YOLOv8 Real Detection',
                'detections': detections,
                'compliance': compliance,
                'violations': compliance.get('violations', []),
                'summary': self._create_detection_summary_yolo(detections, compliance),
                'confidence_level': 'HIGH',
                'model_type': 'YOLO'
            }
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return self._fallback_detection(image, zone_requirements)
    
    def _parse_ultralytics_results(self, results) -> List[Dict]:
        """Parse Ultralytics YOLO results"""
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detection = {
                            'type': self.ppe_classes.get(class_id, f'class_{class_id}'),
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            'center': [int((x1+x2)/2), int((y1+y2)/2)],
                            'area': int((x2-x1) * (y2-y1))
                        }
                        detections.append(detection)
        
        return detections
    
    def _run_onnx_inference(self, image: np.ndarray) -> List[Dict]:
        """Run ONNX model inference"""
        # Preprocess image for ONNX
        input_image = cv2.resize(image, (640, 640))
        input_image = input_image.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        
        # Run inference
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_image})
        
        # Parse outputs (simplified)
        detections = []
        output = outputs[0][0]  # Get first output
        
        for detection in output:
            confidence = detection[4]
            if confidence >= self.confidence_threshold:
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                
                if class_scores[class_id] >= self.confidence_threshold:
                    x, y, w, h = detection[:4]
                    detection_dict = {
                        'type': self.ppe_classes.get(class_id, f'class_{class_id}'),
                        'confidence': float(confidence * class_scores[class_id]),
                        'bbox': [int(x-w/2), int(y-h/2), int(w), int(h)],
                        'center': [int(x), int(y)],
                        'area': int(w * h)
                    }
                    detections.append(detection_dict)
        
        return detections
    
    def _run_tflite_inference(self, image: np.ndarray) -> List[Dict]:
        """Run TensorFlow Lite inference"""
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        
        # Preprocess
        input_shape = input_details[0]['shape']
        input_image = cv2.resize(image, (input_shape[1], input_shape[2]))
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0
        
        # Run inference
        self.model.set_tensor(input_details[0]['index'], input_image)
        self.model.invoke()
        
        # Get outputs
        output_data = self.model.get_tensor(output_details[0]['index'])
        
        # Parse outputs (simplified)
        detections = []
        # Add parsing logic based on your TFLite model output format
        
        return detections
    
    def _run_opencv_dnn_inference(self, image: np.ndarray) -> List[Dict]:
        """Run OpenCV DNN inference"""
        height, width = image.shape[:2]
        
        # Create blob
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        
        # Run inference
        outputs = self.model.forward()
        
        # Parse outputs
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence >= self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    detection_dict = {
                        'type': self.ppe_classes.get(class_id, f'class_{class_id}'),
                        'confidence': float(confidence),
                        'bbox': [x, y, w, h],
                        'center': [center_x, center_y],
                        'area': w * h
                    }
                    detections.append(detection_dict)
        
        return detections
    
    def _fallback_detection(self, image: np.ndarray, zone_requirements: Optional[Dict] = None) -> Dict:
        """
        Enhanced fallback detection using computer vision
        """
        detections = []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        
        # Detect persons using cascades
        persons = []
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                persons.append([x, y-50, w, h+100])  # Extend to include body
        
        if self.body_cascade is not None:
            bodies = self.body_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in bodies:
                persons.append([x, y, w, h])
        
        # If no persons detected, create default detection area
        if not persons:
            persons = [[width//4, height//4, width//2, height//2]]
        
        # Analyze each person for PPE
        for person_bbox in persons:
            x, y, w, h = person_bbox
            person_roi = image[max(0, y):min(height, y+h), max(0, x):min(width, x+w)]
            
            if person_roi.size > 0:
                # Detect helmet (look for hard, bright objects in upper region)
                helmet_detected = self._detect_helmet_fallback(person_roi, person_bbox)
                if helmet_detected:
                    detections.append({
                        'type': 'helmet',
                        'confidence': 0.75,
                        'bbox': [x, y, w//3, h//4],
                        'center': [x + w//6, y + h//8],
                        'area': (w//3) * (h//4)
                    })
                
                # Detect vest (look for bright colors in torso region)
                vest_detected = self._detect_vest_fallback(person_roi, person_bbox)
                if vest_detected:
                    detections.append({
                        'type': 'vest',
                        'confidence': 0.70,
                        'bbox': [x, y + h//4, w, h//2],
                        'center': [x + w//2, y + h//2],
                        'area': w * (h//2)
                    })
                
                # Add person detection
                detections.append({
                    'type': 'person',
                    'confidence': 0.80,
                    'bbox': person_bbox,
                    'center': [x + w//2, y + h//2],
                    'area': w * h
                })
        
        # Analyze compliance
        compliance = self._analyze_compliance_yolo(detections, zone_requirements or {})
        
        return {
            'method': 'Enhanced CV Fallback (YOLO-style)',
            'detections': detections,
            'compliance': compliance,
            'violations': compliance.get('violations', []),
            'summary': self._create_detection_summary_yolo(detections, compliance),
            'confidence_level': 'MEDIUM',
            'model_type': 'Fallback'
        }
    
    def _detect_helmet_fallback(self, person_roi: np.ndarray, person_bbox: List) -> bool:
        """Detect helmet using color and shape analysis"""
        upper_region = person_roi[:person_roi.shape[0]//3, :]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
        
        # Define ranges for common helmet colors (white, yellow, red)
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        # Combine masks
        helmet_mask = cv2.bitwise_or(cv2.bitwise_or(white_mask, yellow_mask), red_mask)
        
        # Check if significant helmet-colored area exists
        helmet_area = cv2.countNonZero(helmet_mask)
        total_area = upper_region.shape[0] * upper_region.shape[1]
        
        return helmet_area > (total_area * 0.15)  # 15% threshold
    
    def _detect_vest_fallback(self, person_roi: np.ndarray, person_bbox: List) -> bool:
        """Detect safety vest using bright color analysis"""
        # Focus on torso region
        h = person_roi.shape[0]
        torso_region = person_roi[h//4:3*h//4, :]
        
        # Convert to HSV
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        
        # Define ranges for high-visibility colors
        # Bright yellow/lime green (high-vis)
        hiviz_lower = np.array([40, 100, 100])
        hiviz_upper = np.array([80, 255, 255])
        hiviz_mask = cv2.inRange(hsv, hiviz_lower, hiviz_upper)
        
        # Bright orange
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([20, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # Combine masks
        vest_mask = cv2.bitwise_or(hiviz_mask, orange_mask)
        
        # Check for significant high-vis area
        vest_area = cv2.countNonZero(vest_mask)
        total_area = torso_region.shape[0] * torso_region.shape[1]
        
        return vest_area > (total_area * 0.20)  # 20% threshold
    
    def _analyze_compliance_yolo(self, detections: List[Dict], requirements: Dict) -> Dict:
        """Analyze PPE compliance from YOLO detections"""
        detected_ppe = [d['type'] for d in detections if d['type'] != 'person']
        detected_persons = len([d for d in detections if d['type'] == 'person'])
        
        # Default requirements if none specified
        if not requirements:
            requirements = {
                'helmet': True,
                'vest': True,
                'gloves': False,
                'mask': False,
                'glasses': False,
                'boots': False
            }
        
        violations = []
        compliance_score = 0
        total_requirements = sum(requirements.values())
        
        if detected_persons > 0:
            for ppe_type, required in requirements.items():
                if required:
                    if ppe_type not in detected_ppe:
                        violations.append(f"Missing {ppe_type}")
                    else:
                        compliance_score += 1
        
        compliance_percentage = (compliance_score / max(1, total_requirements)) * 100
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'compliance_score': compliance_score,
            'total_requirements': total_requirements,
            'compliance_percentage': compliance_percentage,
            'persons_detected': detected_persons,
            'ppe_detected': detected_ppe
        }
    
    def _create_detection_summary_yolo(self, detections: List[Dict], compliance: Dict) -> str:
        """Create human-readable detection summary"""
        persons = compliance.get('persons_detected', 0)
        ppe_items = compliance.get('ppe_detected', [])
        violations = compliance.get('violations', [])
        
        summary = f"Detected {persons} person(s)"
        
        if ppe_items:
            ppe_summary = ", ".join(set(ppe_items))
            summary += f" with PPE: {ppe_summary}"
        
        if violations:
            summary += f". Violations: {', '.join(violations)}"
        else:
            summary += ". Full compliance achieved"
        
        return summary
    
    def _empty_result(self) -> Dict:
        """Return empty result for invalid input"""
        return {
            'method': 'YOLO PPE Detection',
            'detections': [],
            'compliance': {'compliant': False, 'violations': [], 'compliance_percentage': 0},
            'violations': [],
            'summary': 'No valid input provided',
            'confidence_level': 'LOW',
            'processing_time': 0,
            'fps': 0
        }
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return {
            'yolo_available': self.yolo_available,
            'model_type': 'YOLOv8' if self.yolo_available else 'Fallback CV',
            'confidence_threshold': self.confidence_threshold,
            'supported_classes': list(self.ppe_classes.values()),
            'real_time_capable': self.yolo_available
        }
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
        logger.info(f"Confidence threshold updated to {self.confidence_threshold}")

def create_yolo_detector(model_path: Optional[str] = None) -> YOLOPPEDetector:
    """
    Factory function to create YOLO PPE detector
    
    Args:
        model_path: Optional path to custom YOLO model
        
    Returns:
        YOLOPPEDetector instance
    """
    return YOLOPPEDetector(model_path)

# Example usage and testing
if __name__ == "__main__":
    # Create detector
    detector = create_yolo_detector()
    
    # Show model info
    info = detector.get_model_info()
    print("YOLO PPE Detector Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'✅' if info['yolo_available'] else '⚠️'} YOLO Status: {'Available' if info['yolo_available'] else 'Using Fallback'}")