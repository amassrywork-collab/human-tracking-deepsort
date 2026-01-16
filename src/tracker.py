"""
src/tracker.py
--------------
Multi-object tracking module using DeepSORT (deep-sort-realtime).

Responsibilities:
- Convert detector outputs into the format required by DeepSORT
- Maintain track identities across frames
- Return confirmed, stable tracks only (avoid noisy early IDs)

Design notes:
- We keep this module independent from the detector implementation.
- The detector provides: bbox in xyxy + confidence
- DeepSORT expects detections in: [x, y, w, h] (top-left width height) + confidence + class_name
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


@dataclass
class TrackResult:
    """Strongly-typed internal representation (useful for debugging/testing)."""
    track_id: int
    bbox_xyxy: List[int]


class DeepSortTracker:
    """
    DeepSORT tracker wrapper with a clean interface for the project.

    Public API:
    ----------
    tracker = DeepSortTracker(max_age=30, n_init=3)
    tracks = tracker.update(frame_bgr, detections)

    Where:
      detections: List[Dict] each item:
        {
          "bbox": [x1, y1, x2, y2],  # ints preferred (xyxy)
          "conf": float
        }

      tracks: List[Dict] each item:
        {
          "track_id": int,
          "bbox": [x1, y1, x2, y2]   # ints (xyxy)
        }
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.2,
        nn_budget: Optional[int] = None
    ) -> None:
        """
        Initialize DeepSORT.

        Parameters
        ----------
        max_age : int
            Number of frames to keep a track "alive" without detections.
        n_init : int
            Number of consecutive detections before a track is confirmed.
            Higher values reduce false IDs but may delay tracking appearance.
        max_iou_distance : float
            Gating threshold for IoU association (DeepSORT setting).
        max_cosine_distance : float
            Appearance similarity threshold (DeepSORT setting).
        nn_budget : Optional[int]
            Maximum size of the appearance descriptors gallery.
        """
        self.max_age = int(max_age)
        self.n_init = int(n_init)

        # DeepSort from deep-sort-realtime internally handles appearance embedding.
        self.ds = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            max_iou_distance=float(max_iou_distance),
            max_cosine_distance=float(max_cosine_distance),
            nn_budget=nn_budget,
        )

    @staticmethod
    def _xyxy_to_xywh(bbox_xyxy: List[int]) -> List[int]:
        """Convert [x1,y1,x2,y2] -> [x,y,w,h]."""
        x1, y1, x2, y2 = bbox_xyxy
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        return [x1, y1, w, h]

    @staticmethod
    def _validate_detection(det: Dict) -> Optional[Tuple[List[int], float]]:
        """
        Validate and normalize one detection dict.
        Returns (bbox_xyxy_int, conf) if valid, else None.
        """
        if not isinstance(det, dict):
            return None
        if "bbox" not in det or "conf" not in det:
            return None

        bbox = det["bbox"]
        conf = det["conf"]

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            conf_f = float(conf)
        except (ValueError, TypeError):
            return None

        # Reject degenerate boxes
        if (x2 - x1) <= 1 or (y2 - y1) <= 1:
            return None
        if conf_f <= 0.0:
            return None

        return [x1, y1, x2, y2], conf_f

    def update(self, frame_bgr: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker state with the current frame detections.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Current frame (BGR). DeepSORT uses it to compute appearance embeddings.
        detections : List[Dict]
            Detector outputs (xyxy + conf).

        Returns
        -------
        tracks : List[Dict]
            Confirmed tracks with stable IDs and bounding boxes (xyxy).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # Convert detections to DeepSORT expected format:
        # each item: ([x, y, w, h], confidence, class_name)
        ds_detections = []
        for det in detections or []:
            validated = self._validate_detection(det)
            if validated is None:
                continue
            bbox_xyxy, conf = validated
            bbox_xywh = self._xyxy_to_xywh(bbox_xyxy)

            # We track only persons in this project; class name can be fixed.
            ds_detections.append((bbox_xywh, conf, "person"))

        # Update DeepSORT
        tracks = self.ds.update_tracks(ds_detections, frame=frame_bgr)

        # Extract confirmed tracks only
        outputs: List[Dict] = []
        for tr in tracks:
            # tr is a Track object (from deep-sort-realtime)
            if not tr.is_confirmed():
                continue
            if tr.time_since_update > 0:
                # Track not updated this frame -> skip for clean visualization
                continue

            track_id = int(tr.track_id)

            # to_ltrb(): left, top, right, bottom (xyxy)
            ltrb = tr.to_ltrb()
            x1, y1, x2, y2 = [int(round(float(v))) for v in ltrb]

            # Reject degenerate boxes defensively
            if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                continue

            outputs.append({"track_id": track_id, "bbox": [x1, y1, x2, y2]})

        return outputs