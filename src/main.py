"""
src/main.py
-----------
Entry point for the Human Tracking project (YOLOv8 + DeepSORT).

Responsibilities:
- Parse CLI arguments
- Load video (file or webcam)
- Run detection per frame
- Run multi-object tracking
- Visualize results
- Save output video
- Count unique persons (based on track IDs)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from detector import YOLOv8PersonDetector
from tracker import DeepSortTracker
from utils import (
    ensure_dir,
    clip_bbox_xyxy,
    draw_bbox_with_id,
    draw_fps,
    FPSMeter
)


# -------------------------
# Configuration dataclass
# -------------------------
@dataclass
class AppConfig:
    source: str
    output: Optional[str]
    show: bool
    max_frames: int

    # Detection
    weights: str
    conf: float
    iou: float
    device: str

    # Tracking
    max_age: int
    n_init: int

    # Output video
    out_fps: Optional[float]
    out_width: Optional[int]
    out_height: Optional[int]


# -------------------------
# Video helpers
# -------------------------
def is_webcam_source(src: str) -> bool:
    return src.isdigit()


def open_video_capture(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source)) if is_webcam_source(source) else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    return cap


def get_video_props(cap: cv2.VideoCapture) -> Tuple[int, int, float, int]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return w, h, fps, count


def build_video_writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {path}")
    return writer


# -------------------------
# Main application
# -------------------------
def run_app(cfg: AppConfig) -> None:
    cap = open_video_capture(cfg.source)
    in_w, in_h, in_fps, in_count = get_video_props(cap)

    out_w = cfg.out_width or in_w or 1280
    out_h = cfg.out_height or in_h or 720
    out_fps = cfg.out_fps or in_fps or 25.0

    writer = None
    if cfg.output:
        ensure_dir(os.path.dirname(cfg.output))
        writer = build_video_writer(cfg.output, out_w, out_h, out_fps)

    detector = YOLOv8PersonDetector(
        weights_path=cfg.weights,
        conf_thres=cfg.conf,
        iou_thres=cfg.iou,
        device=cfg.device
    )

    tracker = DeepSortTracker(
        max_age=cfg.max_age,
        n_init=cfg.n_init
    )

    fps_meter = FPSMeter(alpha=0.9)

    # -------------------------
    # Person Counter (unique IDs)
    # -------------------------
    unique_person_ids = set()

    total_frames = None
    if in_count > 0:
        total_frames = min(in_count, cfg.max_frames) if cfg.max_frames > 0 else in_count
    elif cfg.max_frames > 0:
        total_frames = cfg.max_frames

    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    frame_idx = 0

    try:
        while True:
            if cfg.max_frames > 0 and frame_idx >= cfg.max_frames:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))

            # -------------------------
            # Detection
            # -------------------------
            detections = detector.detect(frame)

            valid_dets: List[Dict] = []
            for det in detections:
                bbox = clip_bbox_xyxy(det["bbox"], out_w, out_h)
                if (bbox[2] - bbox[0]) > 1 and (bbox[3] - bbox[1]) > 1:
                    valid_dets.append(
                        {
                            "bbox": bbox,
                            "conf": det["conf"]
                        }
                    )

            # -------------------------
            # Tracking
            # -------------------------
            tracks = tracker.update(frame, valid_dets)

            for tr in tracks:
                track_id = tr["track_id"]
                bbox = tr["bbox"]

                # Update person counter
                unique_person_ids.add(track_id)

                draw_bbox_with_id(frame, bbox, track_id)

            # -------------------------
            # Overlay information
            # -------------------------
            fps = fps_meter.update()
            draw_fps(frame, fps, frame_idx)

            # Person counter text
            person_count_text = f"Total Persons: {len(unique_person_ids)}"
            cv2.putText(
                frame,
                person_count_text,
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            if writer is not None:
                writer.write(frame)

            if cfg.show:
                cv2.imshow("Human Tracking (YOLOv8 + DeepSORT)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        if writer:
            writer.release()
        if cfg.show:
            cv2.destroyAllWindows()


# -------------------------
# CLI
# -------------------------
def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Human Tracking in Videos using YOLOv8 + DeepSORT"
    )

    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)

    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--n-init", type=int, default=3)

    parser.add_argument("--out-fps", type=float, default=None)
    parser.add_argument("--out-width", type=int, default=None)
    parser.add_argument("--out-height", type=int, default=None)

    args = parser.parse_args()

    return AppConfig(
        source=args.source,
        output=args.output,
        show=args.show,
        max_frames=args.max_frames,

        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,

        max_age=args.max_age,
        n_init=args.n_init,

        out_fps=args.out_fps,
        out_width=args.out_width,
        out_height=args.out_height,
    )


def main() -> None:
    cfg = parse_args()
    run_app(cfg)


if __name__ == "__main__":
    main()