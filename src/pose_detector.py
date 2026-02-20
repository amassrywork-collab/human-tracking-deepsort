"""
src/pose_detector.py
--------------------
Pose estimation module based on YOLO11-pose.

Responsibilities:
- Loading the YOLO11 pose model
- Extracting keypoints for tracked persons
- Basic action recognition (e.g., Falling, Running)
"""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
from ultralytics import YOLO

class PoseDetector:
    """
    YOLO11-pose based detector for action recognition.
    """

    def __init__(
        self,
        weights_path: str = "yolo11n-pose.pt",
        conf_thres: float = 0.25,
        device: str = "cpu"
    ) -> None:
        self.model = YOLO(weights_path)
        self.device = device
        self.model.to(self.device)

    def estimate_pose(self, frame_bgr: np.ndarray, bboxes: List[List[int]]) -> List[Dict]:
        """
        Estimate poses for specific bounding boxes (tracked persons).
        """
        if not bboxes:
            return []

        # YOLO11-pose can also run in bulk on the whole frame
        results = self.model.predict(
            source=frame_bgr,
            conf=0.25,
            device=self.device,
            verbose=False
        )

        pose_results = []
        result = results[0]
        
        if result.keypoints is not None:
            for i, kpts in enumerate(result.keypoints.data):
                # kpts is [17, 3] (x, y, conf)
                keypoints = kpts.cpu().numpy()
                pose_results.append({
                    "keypoints": keypoints,
                    "action": self._classify_action(keypoints)
                })

        return pose_results

    def _classify_action(self, keypoints: np.ndarray) -> str:
        """
        Simple heuristic-based action recognition.
        """
        # Keypoints indices (COCO): 
        # 0: nose, 5: l_shoulder, 6: r_shoulder, 11: l_hip, 12: r_hip, 15: l_ankle, 16: r_ankle
        try:
            nose = keypoints[0]
            l_ankle = keypoints[15]
            r_ankle = keypoints[16]
            
            # Simple "Fall" detection: if nose is lower than hips or ankles have high horizontal distance?
            # Very basic logic: if vertical distance between nose and ankles is small
            if abs(nose[1] - (l_ankle[1] + r_ankle[1])/2) < 50: # Threshold in pixels
                return "Falling/Lying"
            
            # Simple "Running" detection could be based on bbox speed, but here we only have one frame
            return "Standing/Walking"
        except:
            return "Unknown"
