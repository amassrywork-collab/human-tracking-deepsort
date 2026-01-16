"""
src/detector.py
---------------
Human detection module based on YOLOv8 (Ultralytics).

This module is responsible ONLY for:
- Loading the YOLOv8 model
- Running inference on a single frame
- Filtering detections to the 'person' class
- Returning detections in a clean, unified format

Design goals:
- Clear separation of concerns (detection only, no tracking)
- Minimal assumptions about downstream components
- Stable interface for main.py
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np
import torch
from ultralytics import YOLO


class YOLOv8PersonDetector:
    """
    YOLOv8-based detector specialized for human (person) detection.

    Public API:
    -----------
    detector = YOLOv8PersonDetector(...)
    detections = detector.detect(frame_bgr)

    Each detection is a dict:
        {
            "bbox": [x1, y1, x2, y2],
            "conf": float
        }
    """

    # COCO class index for "person"
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        conf_thres: float = 0.35,
        iou_thres: float = 0.50,
        device: str = "cpu"
    ) -> None:
        """
        Initialize the YOLOv8 detector.

        Parameters
        ----------
        weights_path : str
            Path or name of YOLOv8 weights (e.g., yolov8n.pt).
        conf_thres : float
            Confidence threshold for filtering detections.
        iou_thres : float
            IoU threshold used internally by YOLO NMS.
        device : str
            'cpu' or 'cuda:0' (if available).
        """

        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)

        # Load model
        self.model = YOLO(weights_path)

        # Set device explicitly
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device

        self.model.to(self.device)

    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Run human detection on a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input image in BGR format (as provided by OpenCV).

        Returns
        -------
        detections : List[Dict]
            List of detections, each with:
                - bbox : [x1, y1, x2, y2]
                - conf : confidence score
        """

        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # YOLOv8 accepts numpy arrays directly
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False
        )

        detections: List[Dict] = []

        # YOLOv8 returns a list of Results (one per image)
        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes

        # Iterate through detections
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())

            # Filter only "person" class
            if cls_id != self.PERSON_CLASS_ID:
                continue

            conf = float(boxes.conf[i].item())

            # Bounding box in xyxy format
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf
                }
            )

        return detections