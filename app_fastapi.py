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
from database import init_db, start_session, log_detection, get_stats, log_activity, get_behavior_stats, get_activity_logs
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
current_analysis_mode = "human" # Global state for behavior toggle

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Detector, Tracker, and Pose
detector = HumanDetector()
tracker = TrackerWrapper()
pose_detector = PoseDetector()

@app.get("/get_video/{filename}")
async def get_video(filename: str):
    file_path = os.path.join("static", "videos", filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"message": "Video not found"})
    return FileResponse(file_path, media_type="video/mp4")


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
            
            # Tracking
            tracks = tracker.update(detections)
            unique_person_ids = set()
            
            # Action Recognition if behavior mode is active
            poses = []
            if current_analysis_mode == "behavior" and detections:
                poses = pose_detector.estimate_pose(frame, [d["bbox"] for d in detections])
            
            # Draw Bounding Boxes
            for i, tr in enumerate(tracks):
                track_id = tr["track_id"]
                bbox = tr["bbox"]
                unique_person_ids.add(track_id)
                
                # Assign action if poses are available
                action = "Standing/Walking"
                if poses and i < len(poses):
                    action = poses[i]["action"]
                    tr["action"] = action # Add to track dict for logging
                
                draw_bbox_with_id(frame, bbox, track_id)
                
                # Draw Action Label if behavior mode
                if current_analysis_mode == "behavior":
                    (tw, th), baseline = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    tx = (bbox[0] + bbox[2]) // 2 - tw // 2
                    ty = bbox[1] - 10
                    cv2.putText(frame, action, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Overlay Count
            count = len(unique_person_ids)
            if tracks:
                print(f"[DEBUG] Frame {frame_count}: Tracks={len(tracks)}, Count={count}")
                
            cv2.putText(frame, f"Count: {count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (238, 211, 34), 2)
            
            # Log periodically (e.g., every 30 frames) to DB
            if frame_count % 30 == 0:
                log_detection(current_session_id, count)
                # Log individual track activities
                for i, tr in enumerate(tracks):
                    action = "Standing/Walking" # Default if no pose detector
                    if 'action' in tr:
                        action = tr['action']
                    log_activity(current_session_id, tr["track_id"], action, 0.95)

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
    # Tracking
    tracks = tracker.update(detections)
    unique_person_ids = set()

    # Action Recognition
    poses = []
    if current_analysis_mode == "behavior" and detections:
        poses = pose_detector.estimate_pose(frame, [d["bbox"] for d in detections])

    for i, tr in enumerate(tracks):
        track_id = tr["track_id"]
        bbox = tr["bbox"]
        unique_person_ids.add(track_id)
        
        action = "Standing/Walking"
        if poses and i < len(poses):
            action = poses[i].get("action", "Standing/Walking")
            tr["action"] = action

        draw_bbox_with_id(frame, bbox, track_id)
        if current_analysis_mode == "behavior":
            (tw, th), baseline = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            tx = (bbox[0] + bbox[2]) // 2 - tw // 2
            ty = bbox[1] - 10
            cv2.putText(frame, action, (tx, ty), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Overlay Count
    cv2.putText(frame, f"Count: {len(unique_person_ids)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (238, 211, 34), 2)

    # Encode and Return
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    
    # Update DB for this interactive frame
    if current_session_id:
        log_detection(current_session_id, len(unique_person_ids))
        for tr in tracks:
            act = tr.get("action", "Standing/Walking")
            log_activity(current_session_id, tr["track_id"], act, 0.95)
            
    return {"image": encoded_image}

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

@app.get('/stats/behavior')
async def get_behavioral_stats():
    return get_behavior_stats()

@app.get('/stats/history')
async def get_history_logs():
    return get_activity_logs()

@app.get('/export_csv')
async def export_csv_file():
    import csv
    import io
    from fastapi.responses import StreamingResponse
    
    logs = get_activity_logs(limit=1000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Entity ID", "Action", "Confidence"])
    
    for log in logs:
        writer.writerow([log["timestamp"], log["track_id"], log["action"], log["confidence"]])
        
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=human_tracking_logs.csv"}
    )

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

@app.post('/reset_roi')
async def api_reset_roi():
    detector.set_roi(None)
    return {"status": "ROI reset to default"}

@app.post('/set_analysis_mode')
async def api_set_analysis_mode(data: dict):
    global current_analysis_mode
    current_analysis_mode = data.get("mode", "human")
    return {"status": "Analysis mode updated", "mode": current_analysis_mode}

if __name__ == '__main__':
    import uvicorn
    try:
        from pycloudflared import try_cloudflare
        print("\n[INFO] Starting Cloudflare Tunnel...")
        # Simpler call without unsupported arguments
        public_url = try_cloudflare(port=5000)
        print(f"\n[SUCCESS] Public URL: {public_url}")
    except Exception as e:
        print(f"\n[WARNING] Could not start Cloudflare Tunnel: {e}")
    
    uvicorn.run(app, host="0.0.0.0", port=5000)
