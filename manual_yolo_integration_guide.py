#!/usr/bin/env python3
"""
Manual YOLO Integration Guide for PPE Detection Software
Complete step-by-step guide to manually add real YOLO models
"""

def show_manual_yolo_integration():
    """Show how to manually integrate YOLO into existing PPE detection system"""
    
    print("=== MANUAL YOLO INTEGRATION GUIDE ===")
    print()
    
    methods = [
        {
            "method": "Method 1: Download Pre-trained YOLO Weights",
            "difficulty": "Easy",
            "time": "5-10 minutes",
            "steps": [
                "Download YOLOv8 weights from GitHub or Ultralytics",
                "Place model files in your project directory",
                "Update yolo_ppe_detection.py to load your weights",
                "Test with your construction site images"
            ],
            "files_needed": [
                "yolov8n.pt (nano - 6MB)",
                "yolov8s.pt (small - 22MB)", 
                "yolov8m.pt (medium - 52MB)",
                "yolov8l.pt (large - 88MB)"
            ],
            "code": '''
# Add to your yolo_ppe_detection.py
def _load_custom_weights(self, weights_path):
    """Load custom YOLO weights"""
    try:
        if os.path.exists(weights_path):
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.yolo_available = True
            logger.info(f"✅ Custom YOLO weights loaded: {weights_path}")
            return True
    except Exception as e:
        logger.error(f"Custom weights loading failed: {e}")
    return False

# Usage in __init__:
if self._load_custom_weights("yolov8n.pt"):
    self.yolo_available = True
'''
        },
        
        {
            "method": "Method 2: Use ONNX Runtime (Cross-platform)",
            "difficulty": "Medium",
            "time": "15-20 minutes",
            "steps": [
                "Install ONNX Runtime: pip install onnxruntime",
                "Download YOLO ONNX model or convert existing weights",
                "Update detection code to use ONNX inference",
                "Optimize for faster inference"
            ],
            "files_needed": [
                "yolov8n.onnx (optimized model)",
                "yolov8s.onnx (better accuracy)"
            ],
            "code": '''
# ONNX integration code (already in yolo_ppe_detection.py)
import onnxruntime as ort

def _load_onnx_model(self, onnx_path):
    """Load ONNX YOLO model"""
    try:
        if os.path.exists(onnx_path):
            self.model = ort.InferenceSession(onnx_path)
            self.yolo_available = True
            self.yolo_type = "onnx"
            logger.info(f"✅ ONNX YOLO model loaded: {onnx_path}")
            return True
    except Exception as e:
        logger.error(f"ONNX loading failed: {e}")
    return False
'''
        },
        
        {
            "method": "Method 3: OpenCV DNN Backend",
            "difficulty": "Medium",
            "time": "20-30 minutes",
            "steps": [
                "Get YOLO weights (.weights) and config (.cfg) files",
                "Use OpenCV DNN module (already available in your system)",
                "Implement preprocessing and postprocessing",
                "Test detection accuracy"
            ],
            "files_needed": [
                "yolov4.weights (YOLO v4 weights)",
                "yolov4.cfg (YOLO v4 config)",
                "coco.names (class names file)"
            ],
            "code": '''
# OpenCV DNN integration (already in yolo_ppe_detection.py)
import cv2

def _load_opencv_yolo(self, weights_path, config_path):
    """Load YOLO via OpenCV DNN"""
    try:
        if os.path.exists(weights_path) and os.path.exists(config_path):
            self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            self.yolo_available = True
            self.yolo_type = "opencv_dnn"
            logger.info("✅ YOLO loaded via OpenCV DNN")
            return True
    except Exception as e:
        logger.error(f"OpenCV DNN loading failed: {e}")
    return False
'''
        },
        
        {
            "method": "Method 4: Custom YOLO Training",
            "difficulty": "Advanced",
            "time": "2-4 hours",
            "steps": [
                "Collect PPE-specific dataset (or use Roboflow)",
                "Train custom YOLO model on Google Colab",
                "Export trained weights",
                "Integrate custom model into your system"
            ],
            "files_needed": [
                "custom_ppe_dataset/",
                "best.pt (trained weights)",
                "dataset.yaml (training config)"
            ],
            "code": '''
# Custom YOLO training on Google Colab
from ultralytics import YOLO
import wandb

# Initialize tracking
wandb.init(project="ppe-detection")

# Load base model
model = YOLO("yolov8n.pt")

# Train on PPE dataset
results = model.train(
    data="ppe_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)

# Export for deployment
model.export(format="onnx")
model.export(format="tensorrt")  # For GPU optimization
'''
        }
    ]
    
    for i, method in enumerate(methods, 1):
        print(f"{i}. {method['method']}")
        print(f"   Difficulty: {method['difficulty']}")
        print(f"   Time Required: {method['time']}")
        print("   Steps:")
        for step in method['steps']:
            print(f"     • {step}")
        print("   Files Needed:")
        for file in method['files_needed']:
            print(f"     - {file}")
        print(f"   Code Example:")
        print(method['code'])
        print()

def show_file_locations():
    """Show where to place YOLO model files"""
    
    print("=== WHERE TO PLACE YOLO MODEL FILES ===")
    print()
    
    locations = {
        "Project Root": {
            "path": "./",
            "files": ["yolov8n.pt", "yolov8s.pt", "best.pt"],
            "description": "Main project directory (recommended)"
        },
        "Models Directory": {
            "path": "./models/",
            "files": ["yolo/yolov8n.pt", "onnx/yolov8n.onnx", "custom/best.pt"],
            "description": "Organized model storage"
        },
        "Current Directory": {
            "path": "Same as yolo_ppe_detection.py",
            "files": ["Any YOLO weight files"],
            "description": "Simplest approach for testing"
        }
    }
    
    for location, details in locations.items():
        print(f"{location}:")
        print(f"  Path: {details['path']}")
        print(f"  Description: {details['description']}")
        print("  Example files:")
        for file in details['files']:
            print(f"    - {file}")
        print()

def show_download_links():
    """Show official download links for YOLO models"""
    
    print("=== OFFICIAL YOLO MODEL DOWNLOAD LINKS ===")
    print()
    
    models = [
        {
            "name": "YOLOv8 Nano (6MB)",
            "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "description": "Fastest, good for real-time applications"
        },
        {
            "name": "YOLOv8 Small (22MB)", 
            "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
            "description": "Good balance of speed and accuracy"
        },
        {
            "name": "YOLOv8 Medium (52MB)",
            "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt", 
            "description": "Better accuracy, moderate speed"
        },
        {
            "name": "YOLOv8 Large (88MB)",
            "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
            "description": "Highest accuracy, slower inference"
        },
        {
            "name": "PPE-Specific Model (Roboflow)",
            "link": "https://universe.roboflow.com/ai-project-yolo/ppe-detection-q897z",
            "description": "Pre-trained specifically for PPE detection"
        }
    ]
    
    for model in models:
        print(f"📦 {model['name']}")
        print(f"   Link: {model['link']}")
        print(f"   Use: {model['description']}")
        print()

def show_integration_steps():
    """Show step-by-step integration process"""
    
    print("=== STEP-BY-STEP INTEGRATION PROCESS ===")
    print()
    
    steps = [
        {
            "step": "Step 1: Download YOLO Model",
            "action": "Download yolov8n.pt to your project directory",
            "command": "wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "verification": "File should be ~6MB in size"
        },
        {
            "step": "Step 2: Test Model Loading",
            "action": "Run your yolo_ppe_detection.py to test model loading",
            "command": "python3 yolo_ppe_detection.py",
            "verification": "Should show '✅ Real YOLO model successfully loaded'"
        },
        {
            "step": "Step 3: Update Flask Integration",
            "action": "Your utils.py already prioritizes YOLO detection",
            "command": "No changes needed - already integrated",
            "verification": "Web app will automatically use YOLO when available"
        },
        {
            "step": "Step 4: Test Web Application",
            "action": "Upload images through your web interface",
            "command": "Access your running Flask app",
            "verification": "Detection results should show 'YOLO' method"
        },
        {
            "step": "Step 5: Optimize Performance",
            "action": "Export model to ONNX for faster inference",
            "command": "model.export(format='onnx') in Python",
            "verification": "Should create yolov8n.onnx file"
        }
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step['step']}")
        print(f"   Action: {step['action']}")
        print(f"   Command: {step['command']}")
        print(f"   Verification: {step['verification']}")
        print()

def show_troubleshooting():
    """Show common issues and solutions"""
    
    print("=== TROUBLESHOOTING COMMON ISSUES ===")
    print()
    
    issues = [
        {
            "problem": "ImportError: No module named 'ultralytics'",
            "solution": "Install ultralytics: pip install ultralytics",
            "alternative": "Use ONNX or OpenCV DNN methods instead"
        },
        {
            "problem": "CUDA/GPU not available",
            "solution": "YOLO will automatically use CPU",
            "alternative": "Use lighter models (yolov8n.pt) for better CPU performance"
        },
        {
            "problem": "Model file not found",
            "solution": "Check file path and download model again",
            "alternative": "Use absolute paths: /full/path/to/model.pt"
        },
        {
            "problem": "Slow inference speed",
            "solution": "Use ONNX export or smaller model (yolov8n.pt)",
            "alternative": "Reduce image resolution in preprocessing"
        },
        {
            "problem": "Poor detection accuracy",
            "solution": "Use larger model (yolov8l.pt) or train custom model",
            "alternative": "Fine-tune confidence threshold settings"
        }
    ]
    
    for issue in issues:
        print(f"❌ Problem: {issue['problem']}")
        print(f"✅ Solution: {issue['solution']}")
        print(f"🔄 Alternative: {issue['alternative']}")
        print()

def show_current_status():
    """Show current integration status"""
    
    print("=== CURRENT INTEGRATION STATUS ===")
    print()
    
    print("✅ ALREADY IMPLEMENTED:")
    print("• YOLO detection class (YOLOPPEDetector)")
    print("• Multi-format model loading (PT, ONNX, TFLite, OpenCV)")
    print("• Enhanced fallback detection")
    print("• Flask web app integration") 
    print("• Priority system (YOLO → Dual → SSD → TensorFlow → Simple)")
    print()
    
    print("⚠️ CURRENTLY MISSING:")
    print("• Actual YOLO model files (need to download)")
    print("• Ultralytics package (optional - fallback works)")
    print()
    
    print("🎯 TO ACTIVATE REAL YOLO:")
    print("1. Download yolov8n.pt to project directory")
    print("2. Install: pip install ultralytics (if possible)")
    print("3. Restart your application")
    print("4. Test with construction site images")
    print()
    
    print("📊 EXPECTED RESULTS WITH REAL YOLO:")
    print("• Detection method: 'YOLOv8 Real Detection'")
    print("• Confidence level: 'HIGH'") 
    print("• Higher accuracy on PPE detection")
    print("• Faster processing (especially with GPU)")

if __name__ == "__main__":
    print("🤖 MANUAL YOLO INTEGRATION GUIDE FOR PPE DETECTION")
    print("=" * 60)
    print()
    
    show_manual_yolo_integration()
    show_file_locations()
    show_download_links()
    show_integration_steps()
    show_troubleshooting()
    show_current_status()
    
    print()
    print("🚀 QUICK START COMMAND:")
    print("wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
    print("python3 yolo_ppe_detection.py")
    print()
    print("Your PPE detection system is ready for YOLO integration!")