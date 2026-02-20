"""
src/detector.py
---------------
Human detection and tracking module based on YOLO11 (Ultralytics).

Responsibilities:
- Loading the YOLO11 model
- Running inference and tracking on frames
- Filtering results to the 'person' class
- Handling Region of Interest (ROI)
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO


class HumanDetector:
    """
    YOLO11-based detector and tracker specialized for human detection.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        weights_path: str = "yolo11n.pt",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = "cpu"
    ) -> None:
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.model = YOLO(weights_path)
        
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.roi: Optional[List[int]] = None # [x1, y1, x2, y2]

    def set_roi(self, roi: Optional[List[int]]) -> None:
        """Set Region of Interest [x1, y1, x2, y2]."""
        self.roi = roi

    def detect_and_track(self, frame_bgr: np.ndarray, persist: bool = True) -> List[Dict]:
        """
        Run detection and tracking on a single frame.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # Process ROI if set
        input_frame = frame_bgr
        offset_x, offset_y = 0, 0
        if self.roi:
            x1, y1, x2, y2 = self.roi
            # Ensure ROI is within frame boundaries
            h, w = frame_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            input_frame = frame_bgr[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        # Use model.track for native tracking (ByteTrack by default)
        # Note: 'persist=True' is crucial for maintaining IDs across frames
        results = self.model.track(
            source=input_frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            persist=persist,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
            tracker="bytetrack.yaml"
        )

        detections: List[Dict] = []
        result = results[0]

        if result.boxes is None:
            return detections
            
        print(f"[DEBUG] Found {len(result.boxes)} potential boxes")
        
        boxes = result.boxes
        has_id = boxes.id is not None
        
        for i in range(len(boxes)):
            # Bounding box in xyxy format
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            
            # Map back to original frame coordinates if ROI was used
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y

            track_id = int(boxes.id[i].item()) if has_id else (i + 1) # Fallback to index if no ID
            
            det = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(boxes.conf[i].item()),
                "track_id": track_id
            }
            detections.append(det)

        print(f"[DEBUG] Final detections count: {len(detections)}")
        return detections

    def detect_pose(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Optional: Pose estimation for action recognition."""
        # This would require a pose model (e.g., yolo11n-pose.pt)
        # Placeholder for future expansion
        return []