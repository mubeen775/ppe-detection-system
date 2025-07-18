#!/usr/bin/env python3
"""
Dual Detection System: TensorFlow + SSD Working Together
Combines TensorFlow and SSD for enhanced PPE detection reliability
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualPPEDetector:
    """
    Dual PPE Detection System combining TensorFlow and SSD
    Provides enhanced reliability and performance options
    """
    
    def __init__(self, mode: str = 'dual'):
        """
        Initialize dual detection system
        
        Args:
            mode: 'dual', 'tensorflow', 'ssd', or 'auto'
        """
        self.mode = mode
        self.tf_detector = None
        self.ssd_detector = None
        
        self._initialize_detectors()
        logger.info(f"Dual PPE Detector initialized in {mode} mode")
    
    def _initialize_detectors(self):
        """Initialize both detection systems"""
        try:
            from ai_detection import PPEDetector
            self.tf_detector = PPEDetector()
            logger.info("TensorFlow detector loaded successfully")
        except Exception as e:
            logger.warning(f"TensorFlow detector failed to load: {e}")
        
        try:
            from yolo_like_alternatives import SSDPPEDetector
            self.ssd_detector = SSDPPEDetector()
            logger.info("SSD detector loaded successfully")
        except Exception as e:
            logger.warning(f"SSD detector failed to load: {e}")
    
    def detect_ppe(self, image: np.ndarray, zone_requirements: Optional[Dict] = None) -> Dict:
        """
        Detect PPE using selected detection strategy
        
        Args:
            image: Input image as numpy array
            zone_requirements: Zone-specific PPE requirements
            
        Returns:
            Detection results with enhanced confidence
        """
        if self.mode == 'dual':
            return self._dual_detection(image, zone_requirements)
        elif self.mode == 'tensorflow':
            return self._tensorflow_detection(image, zone_requirements)
        elif self.mode == 'ssd':
            return self._ssd_detection(image, zone_requirements)
        elif self.mode == 'auto':
            return self._auto_detection(image, zone_requirements)
        else:
            raise ValueError(f"Unknown detection mode: {self.mode}")
    
    def _dual_detection(self, image: np.ndarray, zone_requirements: Optional[Dict] = None) -> Dict:
        """
        Run both detectors and combine results
        """
        results = {}
        
        # Run TensorFlow detection
        tf_result = self._tensorflow_detection(image, zone_requirements)
        results['tensorflow'] = tf_result
        
        # Run SSD detection
        ssd_result = self._ssd_detection(image, zone_requirements)
        results['ssd'] = ssd_result
        
        # Combine results
        combined_result = self._combine_results(tf_result, ssd_result)
        results['combined'] = combined_result
        
        return {
            'method': 'Dual Detection (TensorFlow + SSD)',
            'individual_results': results,
            'final_result': combined_result,
            'confidence_level': self._calculate_confidence_level(tf_result, ssd_result),
            'timestamp': time.time()
        }
    
    def _tensorflow_detection(self, image: np.ndarray, zone_requirements: Optional[Dict] = None) -> Dict:
        """Run TensorFlow detection"""
        if self.tf_detector is None:
            return {'error': 'TensorFlow detector not available'}
        
        try:
            return self.tf_detector.detect_ppe(image, zone_requirements)
        except Exception as e:
            logger.error(f"TensorFlow detection failed: {e}")
            return {'error': str(e)}
    
    def _ssd_detection(self, image: np.ndarray, zone_requirements: Optional[Dict] = None) -> Dict:
        """Run SSD detection"""
        if self.ssd_detector is None:
            return {'error': 'SSD detector not available'}
        
        try:
            return self.ssd_detector.detect_ppe(image, zone_requirements)
        except Exception as e:
            logger.error(f"SSD detection failed: {e}")
            return {'error': str(e)}
    
    def _auto_detection(self, image: np.ndarray, zone_requirements: Optional[Dict] = None) -> Dict:
        """
        Automatically choose best detector based on performance
        """
        # Quick performance test
        if self.tf_detector and self.ssd_detector:
            # Test both with small image
            test_img = np.random.randint(100, 200, (240, 320, 3), dtype=np.uint8)
            
            tf_start = time.time()
            self.tf_detector.detect_ppe(test_img)
            tf_time = time.time() - tf_start
            
            ssd_start = time.time()
            self.ssd_detector.detect_ppe(test_img)
            ssd_time = time.time() - ssd_start
            
            # Choose faster detector
            if ssd_time < tf_time:
                logger.info(f"Auto-selected SSD (faster: {ssd_time:.3f}s vs {tf_time:.3f}s)")
                return self._ssd_detection(image, zone_requirements)
            else:
                logger.info(f"Auto-selected TensorFlow (faster: {tf_time:.3f}s vs {ssd_time:.3f}s)")
                return self._tensorflow_detection(image, zone_requirements)
        
        # Fallback to available detector
        if self.ssd_detector:
            return self._ssd_detection(image, zone_requirements)
        elif self.tf_detector:
            return self._tensorflow_detection(image, zone_requirements)
        else:
            return {'error': 'No detectors available'}
    
    def _combine_results(self, tf_result: Dict, ssd_result: Dict) -> Dict:
        """
        Combine results from both detectors
        """
        combined = {
            'detections': [],
            'confidence_boost': False,
            'agreement_score': 0.0
        }
        
        # Get detections from both systems
        tf_detections = tf_result.get('detections', [])
        ssd_detections = ssd_result.get('detections', [])
        
        # Combine unique detections
        all_detections = []
        
        # Add TensorFlow detections
        for detection in tf_detections:
            detection['source'] = 'tensorflow'
            all_detections.append(detection)
        
        # Add SSD detections
        for detection in ssd_detections:
            detection['source'] = 'ssd'
            all_detections.append(detection)
        
        # Calculate agreement
        tf_count = len(tf_detections)
        ssd_count = len(ssd_detections)
        
        if tf_count > 0 and ssd_count > 0:
            combined['confidence_boost'] = True
            combined['agreement_score'] = min(tf_count, ssd_count) / max(tf_count, ssd_count)
        
        combined['detections'] = all_detections
        return combined
    
    def _calculate_confidence_level(self, tf_result: Dict, ssd_result: Dict) -> str:
        """Calculate overall confidence level"""
        tf_detections = len(tf_result.get('detections', []))
        ssd_detections = len(ssd_result.get('detections', []))
        
        if tf_detections > 0 and ssd_detections > 0:
            return 'HIGH (Both systems detected PPE)'
        elif tf_detections > 0 or ssd_detections > 0:
            return 'MEDIUM (One system detected PPE)'
        else:
            return 'LOW (No PPE detected by either system)'
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for both detectors"""
        stats = {
            'tensorflow_available': self.tf_detector is not None,
            'ssd_available': self.ssd_detector is not None,
            'current_mode': self.mode
        }
        
        if self.tf_detector and self.ssd_detector:
            # Quick performance test
            test_img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
            
            tf_start = time.time()
            self.tf_detector.detect_ppe(test_img)
            tf_time = time.time() - tf_start
            
            ssd_start = time.time()
            self.ssd_detector.detect_ppe(test_img)
            ssd_time = time.time() - ssd_start
            
            stats['tensorflow_speed'] = f"{tf_time:.3f}s"
            stats['ssd_speed'] = f"{ssd_time:.3f}s"
            stats['faster_detector'] = 'SSD' if ssd_time < tf_time else 'TensorFlow'
        
        return stats
    
    def switch_mode(self, new_mode: str):
        """Switch detection mode"""
        valid_modes = ['dual', 'tensorflow', 'ssd', 'auto']
        if new_mode not in valid_modes:
            raise ValueError(f"Invalid mode. Choose from: {valid_modes}")
        
        self.mode = new_mode
        logger.info(f"Switched to {new_mode} mode")


def test_dual_system():
    """Test the dual detection system"""
    print("=== DUAL DETECTION SYSTEM TEST ===")
    print()
    
    # Initialize dual detector
    dual_detector = DualPPEDetector(mode='dual')
    
    # Create test image
    test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    test_image[80:140, 280:340] = [30, 215, 255]  # Yellow helmet
    test_image[200:300, 260:380] = [0, 165, 255]  # Orange vest
    
    # Test dual detection
    print("1. Running dual detection...")
    result = dual_detector.detect_ppe(test_image)
    
    print(f"   Method: {result['method']}")
    print(f"   Confidence Level: {result['confidence_level']}")
    
    # Show individual results
    individual = result['individual_results']
    print(f"   TensorFlow detections: {len(individual['tensorflow'].get('detections', []))}")
    print(f"   SSD detections: {len(individual['ssd'].get('detections', []))}")
    
    # Test performance stats
    print()
    print("2. Performance Statistics:")
    stats = dual_detector.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print()
    print("3. Testing mode switching...")
    for mode in ['auto', 'tensorflow', 'ssd']:
        dual_detector.switch_mode(mode)
        quick_result = dual_detector.detect_ppe(test_image)
        print(f"   {mode} mode: {quick_result.get('method', 'N/A')}")
    
    print()
    print("✅ Dual detection system test completed!")


if __name__ == "__main__":
    test_dual_system()