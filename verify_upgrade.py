import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from detector import HumanDetector
    print("SUCCESS: HumanDetector imported")
except ImportError as e:
    print(f"FAILED: HumanDetector import - {e}")
    sys.exit(1)

def verify_detector():
    try:
        # Initialize detector (will try to download yolo11n.pt if not present)
        detector = HumanDetector(weights_path="yolo11n.pt", device="cpu")
        print("SUCCESS: Detector initialized")
        
        # Create a dummy frame (black image)
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run detection
        results = detector.detect_and_track(frame, persist=False)
        print(f"SUCCESS: Detection run completed. Results: {results}")
        
    except Exception as e:
        print(f"FAILED: Detector verification - {e}")

if __name__ == "__main__":
    verify_detector()
