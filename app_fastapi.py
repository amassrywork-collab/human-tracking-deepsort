import cv2
import os
import sys
import uuid
import asyncio
from typing import Generator, List
from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from werkzeug.utils import secure_filename
import base64
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector import HumanDetector
from tracker import TrackerWrapper
from pose_detector import PoseDetector
from database import init_db, start_session, log_detection, get_stats
from utils import draw_bbox_with_id, clip_bbox_xyxy

app = FastAPI()

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED_FOLDER = os.path.join('static', 'processed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize DB
init_db()
current_session_id = start_session("live")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Detector, Tracker, and Pose
detector = HumanDetector()
tracker = TrackerWrapper()
pose_detector = PoseDetector()

# Global dictionary to track progress of video processing tasks
processing_tasks = {}

async def frame_generator(source=0, save_path=None, task_id=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    out = None
    if save_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if task_id:
            processing_tasks[task_id]["total_frames"] = total_frames

    unique_person_ids = set()
    frame_count = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            h, w = frame.shape[:2]

            # 1. Detect & Track (YOLOv11 native)
            detections = detector.detect_and_track(frame)
            
            # 2. Extract and format tracks
            tracks = tracker.update(detections)
            
            # 3. Action Recognition (Pose)
            poses = pose_detector.estimate_pose(frame, [t["bbox"] for t in tracks])
            
            for i, tr in enumerate(tracks):
                track_id = tr["track_id"]
                bbox = tr["bbox"]
                unique_person_ids.add(track_id)
                draw_bbox_with_id(frame, bbox, track_id)
                
                # Draw Action Label if pose was detected
                if i < len(poses):
                    action = poses[i]["action"]
                    cv2.putText(frame, action, (bbox[0], bbox[1] - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Overlay Count
            count = len(unique_person_ids)
            cv2.putText(frame, f"Count: {count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Log to DB every 30 frames (approx 1 second)
            if frame_count % 30 == 0:
                log_detection(current_session_id, count)

            # Save frame
            if out:
                out.write(frame)
                if task_id and "total_frames" in processing_tasks[task_id]:
                    progress = int((frame_count / processing_tasks[task_id]["total_frames"]) * 100)
                    processing_tasks[task_id]["progress"] = min(progress, 99)

            # Encode for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Yield control back to event loop for better concurrency
            await asyncio.sleep(0.01)

    finally:
        cap.release()
        if out:
            out.release()
            if task_id:
                processing_tasks[task_id]["progress"] = 100
                processing_tasks[task_id]["complete"] = True

@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(frame_generator(0), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post('/process_frame')
async def process_frame(request: Request):
    data = await request.json()
    if not data or 'image' not in data:
        return JSONResponse({"error": "No image data"}, status_code=400)
    
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Detect & Track
    detections = detector.detect_and_track(frame)
    tracks = tracker.update(detections)
    poses = pose_detector.estimate_pose(frame, [t["bbox"] for t in tracks])

    unique_person_ids = set()
    for i, tr in enumerate(tracks):
        track_id = tr["track_id"]
        bbox = tr["bbox"]
        unique_person_ids.add(track_id)
        draw_bbox_with_id(frame, bbox, track_id)
        if i < len(poses):
            cv2.putText(frame, poses[i]["action"], (bbox[0], bbox[1] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"Count: {len(unique_person_ids)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    ret, buffer = cv2.imencode('.jpg', frame)
    processed_img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {"image": f"data:image/jpeg;base64,{processed_img_base64}"}

@app.post('/upload_video')
async def upload_video(file: UploadFile = File(...)):
    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
        
    output_filename = f"processed_{filename}"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    
    processing_tasks[task_id] = {
        "progress": 0,
        "filename": output_filename,
        "complete": False,
        "source": filepath,
        "output_path": output_path
    }
    
    return {"task_id": task_id, "filename": filename}

@app.get('/stream_processing/{task_id}')
async def stream_processing(task_id: str):
    if task_id not in processing_tasks:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    
    task = processing_tasks[task_id]
    return StreamingResponse(frame_generator(task["source"], task["output_path"], task_id), 
                           media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/get_progress/{task_id}')
async def get_progress(task_id: str):
    if task_id not in processing_tasks:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return processing_tasks[task_id]

@app.get('/download_video/{task_id}')
async def download_video(task_id: str):
    if task_id not in processing_tasks or not processing_tasks[task_id]["complete"]:
        return JSONResponse({"error": "File not ready"}, status_code=404)
    
    task = processing_tasks[task_id]
    return FileResponse(task["output_path"], media_type='video/mp4', filename=task["filename"])

@app.get('/stats/realtime')
async def get_realtime_stats():
    # Fetch last 50 data points from DB
    stats = get_stats()
    return stats

@app.get('/stats/summary')
async def get_stats_summary():
    # Simple summary: Total detections in session
    # This can be expanded with more complex SQL queries
    stats = get_stats()
    total_detections = len(stats)
    max_count = max([s['count'] for s in stats]) if stats else 0
    return {
        "total_datapoints": total_detections,
        "max_person_count": max_count,
        "current_session": current_session_id
    }

@app.post('/set_roi')
async def api_set_roi(roi: List[int]):
    # Expects [x1, y1, x2, y2]
    detector.set_roi(roi)
    return {"status": "ROI updated", "roi": roi}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
