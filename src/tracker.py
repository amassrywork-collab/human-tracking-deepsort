"""
src/tracker.py
--------------
Lightweight wrapper for YOLO11 native tracking (ByteTrack).

Since YOLO11 handles tracking internally via model.track(), this module
acts as an adapter to keep the rest of the application stable.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np


class TrackerWrapper:
    """
    Simplified tracker that leverages YOLO11's internal tracking state.
    """

    def __init__(self) -> None:
        pass

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Convert detection results (which already contain track_id from YOLO11) 
        into the format expected by the frontend.
        """
        tracks: List[Dict] = []
        for det in detections:
            if det.get("track_id") is not None:
                tracks.append({
                    "track_id": det["track_id"],
                    "bbox": det["bbox"]
                })
        return tracks