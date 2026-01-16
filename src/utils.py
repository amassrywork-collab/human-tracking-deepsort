"""
src/utils.py
------------
Utility functions shared across the project.

This module contains:
- Filesystem helpers
- Bounding box helpers
- Drawing utilities
- FPS measurement helper

Keeping these functions here improves:
- Code readability
- Separation of concerns
- Maintainability
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np


# -------------------------
# Filesystem utilities
# -------------------------
def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str
        Directory path.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# -------------------------
# Bounding box utilities
# -------------------------
def clip_bbox_xyxy(
    bbox: List[int],
    width: int,
    height: int
) -> List[int]:
    """
    Clip bounding box to image boundaries.

    Parameters
    ----------
    bbox : [x1, y1, x2, y2]
    width : image width
    height : image height

    Returns
    -------
    bbox_clipped : [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


# -------------------------
# Drawing utilities
# -------------------------
def draw_bbox_with_id(
    frame: np.ndarray,
    bbox: List[int],
    track_id: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> None:
    """
    Draw a bounding box with a track ID label.

    Parameters
    ----------
    frame : np.ndarray
        Image (BGR).
    bbox : [x1, y1, x2, y2]
    track_id : int
        Unique track identifier.
    color : tuple
        BGR color for drawing.
    thickness : int
        Rectangle thickness.
    """
    x1, y1, x2, y2 = bbox

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f"ID {track_id}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    y_text = max(0, y1 - 10)
    cv2.rectangle(
        frame,
        (x1, y_text - th - baseline),
        (x1 + tw, y_text + baseline),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )


def draw_fps(
    frame: np.ndarray,
    fps: float,
    frame_idx: int
) -> None:
    """
    Draw FPS and frame index on the frame.

    Parameters
    ----------
    frame : np.ndarray
    fps : float
    frame_idx : int
    """
    text = f"FPS: {fps:.2f} | Frame: {frame_idx}"
    cv2.putText(
        frame,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# -------------------------
# FPS helper
# -------------------------
class FPSMeter:
    """
    Helper class for estimating FPS using exponential smoothing.
    """

    def __init__(self, alpha: float = 0.9) -> None:
        """
        Parameters
        ----------
        alpha : float
            Smoothing factor (higher = smoother, slower response).
        """
        self.alpha = float(alpha)
        self._prev_time: Optional[float] = None
        self._fps: float = 0.0

    def update(self) -> float:
        """
        Update FPS estimate.

        Returns
        -------
        fps : float
            Smoothed FPS value.
        """
        now = time.time()

        if self._prev_time is None:
            self._prev_time = now
            return 0.0

        dt = max(1e-6, now - self._prev_time)
        inst_fps = 1.0 / dt

        if self._fps == 0.0:
            self._fps = inst_fps
        else:
            self._fps = self.alpha * self._fps + (1.0 - self.alpha) * inst_fps

        self._prev_time = now
        return self._fps