#!/usr/bin/env python3
"""
NVIDIA NIM Florence 2 PPE Detection Integration
High-accuracy PPE detection using Florence 2 vision model
"""

import os
import cv2
import numpy as np
import requests
import json
import base64
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

class Florence2PPEDetector:
    """
    High-accuracy PPE detection using NVIDIA NIM Florence 2 model
    """
    
    def __init__(self, api_key: Optional[str] = None, confidence_threshold: float = 0.7):
        """
        Initialize Florence 2 PPE detector
        
        Args:
            api_key: NVIDIA NIM API key
            confidence_threshold: Minimum confidence for detections
        """
        self.api_key = api_key or os.environ.get('NVIDIA_NIM_API_KEY')
        self.confidence_threshold = confidence_threshold
        self.base_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/florence-2-large"
        
        # PPE detection prompts optimized for Florence 2
        self.ppe_prompts = {
            'detection': '<OD>Detect safety equipment: hard hat, safety helmet, high visibility vest, safety vest, work gloves, protective gloves, face mask, safety mask, safety glasses, protective eyewear, safety boots, steel toe boots',
            'detailed': '<DETAILED_CAPTION>Describe personal protective equipment worn by workers including helmets, vests, gloves, masks, safety glasses, and boots with colors and conditions',
            'region': '<REGION_PROPOSAL>Find regions containing safety equipment and protective gear'
        }
        
        # PPE categories mapping
        self.ppe_categories = {
            'helmet': ['hard hat', 'safety helmet', 'helmet', 'construction helmet'],
            'vest': ['high visibility vest', 'safety vest', 'reflective vest', 'hi-vis vest'],
            'gloves': ['work gloves', 'protective gloves', 'safety gloves', 'construction gloves'],
            'mask': ['face mask', 'safety mask', 'protective mask', 'respirator'],
            'glasses': ['safety glasses', 'protective eyewear', 'safety goggles', 'protective glasses'],
            'boots': ['safety boots', 'steel toe boots', 'work boots', 'protective footwear']
        }
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for detection events"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _try_load_local_model(self):
        """Try to load local Florence 2 model for real-time detection"""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
            
            # Try to load Florence 2 model locally
            model_name = "microsoft/Florence-2-base"
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            logger.info("Local Florence 2 model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Local Florence 2 model not available: {e}")
            self.local_model = None
            self.processor = None
        
    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 for API request
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Base64 encoded image string
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode image
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
        
    def _make_api_request(self, image_base64: str, prompt: str) -> Dict:
        """
        Make API request to Florence 2 model
        
        Args:
            image_base64: Base64 encoded image
            prompt: Detection prompt
            
        Returns:
            API response dictionary
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'<img src="data:image/jpeg;base64,{image_base64}" />{prompt}'
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return {"error": str(e)}
            
    def _parse_detection_response(self, response: Dict) -> List[Dict]:
        """
        Parse Florence 2 detection response
        
        Args:
            response: API response
            
        Returns:
            List of detected PPE items
        """
        detections = []
        
        if "error" in response:
            self.logger.error(f"Detection error: {response['error']}")
            return detections
            
        try:
            # Extract detection content
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Parse detection results (Florence 2 specific parsing)
            if '<loc_' in content:
                # Parse bounding box format
                detections = self._parse_bounding_boxes(content)
            else:
                # Parse text description
                detections = self._parse_text_description(content)
                
        except Exception as e:
            self.logger.error(f"Response parsing error: {e}")
            
        return detections
        
    def _parse_bounding_boxes(self, content: str) -> List[Dict]:
        """Parse bounding box detection format"""
        detections = []
        
        # Florence 2 bounding box parsing logic
        import re
        
        # Extract location and object pairs
        pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>([^<]+)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            x1, y1, x2, y2, obj_name = match
            
            # Normalize coordinates (Florence 2 uses 1000-scale)
            bbox = [
                int(x1) / 1000.0,
                int(y1) / 1000.0, 
                int(x2) / 1000.0,
                int(y2) / 1000.0
            ]
            
            # Classify PPE type
            ppe_type = self._classify_ppe_type(obj_name.strip())
            
            if ppe_type:
                detections.append({
                    'type': ppe_type,
                    'confidence': 0.9,  # Florence 2 high confidence
                    'bbox': bbox,
                    'description': obj_name.strip()
                })
                
        return detections
        
    def _parse_text_description(self, content: str) -> List[Dict]:
        """Parse text description for PPE items"""
        detections = []
        
        content_lower = content.lower()
        
        # Check for each PPE category
        for ppe_type, keywords in self.ppe_categories.items():
            for keyword in keywords:
                if keyword in content_lower:
                    detections.append({
                        'type': ppe_type,
                        'confidence': 0.8,
                        'bbox': [0.0, 0.0, 1.0, 1.0],  # Full image
                        'description': f"Detected {keyword} in image"
                    })
                    break  # One detection per type
                    
        return detections
        
    def _classify_ppe_type(self, obj_name: str) -> Optional[str]:
        """Classify detected object as PPE type"""
        obj_lower = obj_name.lower()
        
        for ppe_type, keywords in self.ppe_categories.items():
            for keyword in keywords:
                if keyword in obj_lower:
                    return ppe_type
                    
        return None
        
    def detect_ppe_florence2(self, image: np.ndarray, zone_requirements: Optional[Dict] = None) -> Dict:
        """
        Detect PPE using Florence 2 model
        
        Args:
            image: Input image as numpy array
            zone_requirements: Zone-specific PPE requirements
            
        Returns:
            Detection results dictionary
        """
        if self.api_key is None:
            return self._fallback_detection(image, zone_requirements)
            
        try:
            # Encode image
            image_base64 = self._encode_image(image)
            
            # Make detection request
            response = self._make_api_request(image_base64, self.ppe_prompts['detection'])
            
            # Parse results
            detections = self._parse_detection_response(response)
            
            # Filter by confidence
            filtered_detections = [
                d for d in detections 
                if d['confidence'] >= self.confidence_threshold
            ]
            
            # Analyze compliance
            compliance_result = self._analyze_compliance_florence2(
                filtered_detections, 
                zone_requirements or {}
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'model': 'florence-2-large',
                'detections': filtered_detections,
                'compliance': compliance_result,
                'summary': self._create_detection_summary_florence2(filtered_detections, compliance_result)
            }
            
        except Exception as e:
            self.logger.error(f"Florence 2 detection failed: {e}")
            return self._fallback_detection(image, zone_requirements)
            
    def _analyze_compliance_florence2(self, detections: List[Dict], requirements: Dict) -> Dict:
        """Analyze PPE compliance using Florence 2 detections"""
        detected_types = set(d['type'] for d in detections)
        required_types = set(requirements.get('required_ppe', []))
        
        violations = []
        for required in required_types:
            if required not in detected_types:
                violations.append(f"Missing {required}")
                
        compliance_rate = 1.0 - (len(violations) / max(len(required_types), 1))
        
        return {
            'compliant': len(violations) == 0,
            'compliance_rate': compliance_rate,
            'violations': violations,
            'detected_ppe': list(detected_types),
            'required_ppe': list(required_types)
        }
        
    def _create_detection_summary_florence2(self, detections: List[Dict], compliance: Dict) -> str:
        """Create human-readable detection summary"""
        if not detections:
            return "No PPE detected in image"
            
        detected_items = [f"{d['type']} (confidence: {d['confidence']:.2f})" for d in detections]
        summary = f"Detected PPE: {', '.join(detected_items)}"
        
        if compliance['violations']:
            summary += f" | Violations: {', '.join(compliance['violations'])}"
        else:
            summary += " | Fully compliant"
            
        return summary
        
    def _fallback_detection(self, image: np.ndarray, zone_requirements: Optional[Dict] = None) -> Dict:
        """Fallback to simple detection when Florence 2 unavailable"""
        self.logger.warning("Using fallback detection - Florence 2 unavailable")
        
        # Use existing simple detection as fallback
        from simple_ai_detection import SimplePPEDetector
        fallback_detector = SimplePPEDetector()
        return fallback_detector.detect_ppe(image, zone_requirements)
        
    def get_detailed_analysis(self, image: np.ndarray) -> Dict:
        """Get detailed PPE analysis using Florence 2"""
        if self.api_key is None:
            return {"error": "NVIDIA NIM API key required"}
            
        try:
            image_base64 = self._encode_image(image)
            response = self._make_api_request(image_base64, self.ppe_prompts['detailed'])
            
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            return {
                'detailed_analysis': content,
                'model': 'florence-2-large',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Detailed analysis failed: {e}"}
            
    def batch_detect(self, images: List[np.ndarray], zone_requirements: Optional[Dict] = None) -> List[Dict]:
        """Process multiple images for PPE detection"""
        results = []
        
        for i, image in enumerate(images):
            self.logger.info(f"Processing image {i+1}/{len(images)}")
            result = self.detect_ppe_florence2(image, zone_requirements)
            results.append(result)
            
            # Rate limiting
            time.sleep(0.1)
            
        return results

# Integration function for existing system
def create_florence2_detector() -> Florence2PPEDetector:
    """Create Florence 2 detector instance"""
    return Florence2PPEDetector()

if __name__ == "__main__":
    # Test Florence 2 integration
    detector = Florence2PPEDetector()
    
    print("NVIDIA NIM Florence 2 PPE Detection System")
    print("=" * 50)
    print(f"API Key configured: {'Yes' if detector.api_key else 'No'}")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    print("PPE categories:", list(detector.ppe_categories.keys()))
    print("Ready for high-accuracy PPE detection")