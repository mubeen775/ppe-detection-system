#!/usr/bin/env python3
"""
YOLO Model Downloader for PPE Detection System
Simple script to download YOLO models for manual integration
"""

import os
import urllib.request
import sys
from pathlib import Path

def download_file(url, filename, description):
    """Download a file with progress indication"""
    
    print(f"Downloading {description}...")
    print(f"URL: {url}")
    print(f"Saving as: {filename}")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\rProgress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
            else:
                mb_downloaded = downloaded / (1024 * 1024)
                print(f"\rDownloaded: {mb_downloaded:.1f} MB", end="")
        
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ Successfully downloaded {filename}")
        
        # Verify file size
        size = os.path.getsize(filename) / (1024 * 1024)
        print(f"File size: {size:.1f} MB")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def download_yolo_models():
    """Download YOLO models for PPE detection"""
    
    print("=== YOLO MODEL DOWNLOADER FOR PPE DETECTION ===")
    print()
    
    # Model definitions
    models = [
        {
            "name": "YOLOv8 Nano (Recommended for CPU)",
            "filename": "yolov8n.pt",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt",
            "size": "6 MB",
            "description": "Fastest model, good for real-time detection"
        },
        {
            "name": "YOLOv8 Small (Balanced)",
            "filename": "yolov8s.pt", 
            "url": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8s.pt",
            "size": "22 MB",
            "description": "Good balance of speed and accuracy"
        },
        {
            "name": "YOLOv8 Medium (Higher Accuracy)",
            "filename": "yolov8m.pt",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8m.pt", 
            "size": "52 MB",
            "description": "Better accuracy, moderate speed"
        }
    ]
    
    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']} ({model['size']})")
        print(f"   {model['description']}")
    
    print("\nNote: YOLOv8 Nano is recommended for CPU-only environments")
    print()
    
    # Get user choice
    while True:
        try:
            choice = input("Enter model number to download (1-3) or 'all' for all models: ").strip().lower()
            
            if choice == 'all':
                selected_models = models
                break
            elif choice in ['1', '2', '3']:
                selected_models = [models[int(choice) - 1]]
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 'all'")
                continue
        except KeyboardInterrupt:
            print("\nDownload cancelled.")
            return False
    
    print()
    
    # Download selected models
    success_count = 0
    for model in selected_models:
        print(f"Processing {model['name']}...")
        
        # Check if file already exists
        if os.path.exists(model['filename']):
            print(f"File {model['filename']} already exists. Skipping...")
            success_count += 1
            continue
        
        # Download the model
        if download_file(model['url'], model['filename'], model['name']):
            success_count += 1
        
        print()
    
    print("=" * 50)
    print(f"Download Summary: {success_count}/{len(selected_models)} models downloaded successfully")
    
    if success_count > 0:
        print()
        print("✅ YOLO models ready for integration!")
        print("Next steps:")
        print("1. Test model loading: python3 yolo_ppe_detection.py") 
        print("2. Run your web application to use YOLO detection")
        print("3. Upload images to test improved accuracy")
        
        return True
    else:
        print("\n❌ No models downloaded successfully")
        return False

def download_alternative_formats():
    """Download alternative model formats"""
    
    print("\n=== ALTERNATIVE MODEL FORMATS ===")
    print()
    
    alternatives = [
        {
            "name": "YOLOv8 Nano ONNX (Cross-platform)",
            "filename": "yolov8n.onnx",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.onnx",
            "description": "Optimized for CPU inference"
        },
        {
            "name": "YOLOv4 Weights (OpenCV DNN)",
            "filename": "yolov4.weights",
            "url": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
            "description": "Works with OpenCV DNN backend"
        }
    ]
    
    print("Alternative formats available:")
    for i, alt in enumerate(alternatives, 1):
        print(f"{i}. {alt['name']}")
        print(f"   {alt['description']}")
    
    choice = input("\nDownload alternative formats? (y/n): ").strip().lower()
    
    if choice == 'y':
        for alt in alternatives:
            if not os.path.exists(alt['filename']):
                download_file(alt['url'], alt['filename'], alt['name'])
                print()

def verify_downloads():
    """Verify downloaded models work"""
    
    print("=== VERIFYING DOWNLOADED MODELS ===")
    print()
    
    # Check for downloaded files
    model_files = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8n.onnx']
    found_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)
            found_models.append((model_file, size))
            print(f"✅ {model_file} ({size:.1f} MB)")
        else:
            print(f"❌ {model_file} (not found)")
    
    if found_models:
        print(f"\n{len(found_models)} model(s) ready for use")
        
        # Test YOLO integration
        print("\nTesting YOLO integration...")
        try:
            from yolo_ppe_detection import YOLOPPEDetector
            detector = YOLOPPEDetector()
            info = detector.get_model_info()
            
            if info['yolo_available']:
                print("✅ YOLO successfully integrated!")
                print(f"Model type: {info['model_type']}")
            else:
                print("⚠️ YOLO not detected, using fallback")
                
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
    else:
        print("\n❌ No models found. Please download models first.")

def show_download_commands():
    """Show command-line download options"""
    
    print("\n=== COMMAND-LINE DOWNLOAD OPTIONS ===")
    print()
    
    print("Using wget (Linux/Mac):")
    print("wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt")
    print()
    
    print("Using curl (Linux/Mac):")
    print("curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt")
    print()
    
    print("Using PowerShell (Windows):")
    print("Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt' -OutFile 'yolov8n.pt'")
    print()
    
    print("Manual download:")
    print("1. Open: https://github.com/ultralytics/assets/releases/tag/v8.0.0")
    print("2. Click on yolov8n.pt to download")
    print("3. Place file in your project directory")

if __name__ == "__main__":
    print("🤖 YOLO Model Downloader")
    print("This script will download YOLO models for your PPE detection system")
    print()
    
    try:
        # Main download process
        if download_yolo_models():
            
            # Offer alternative formats
            download_alternative_formats()
            
            # Verify everything works
            verify_downloads()
            
        # Show manual options
        show_download_commands()
        
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("\nTrying alternative download methods...")
        show_download_commands()