
from flask import Flask, render_template, Response, request, send_from_directory, jsonify
import cv2
import os
import sys
import uuid
from werkzeug.utils import secure_filename

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector import YOLOv8PersonDetector
from tracker import DeepSortTracker
from utils import draw_bbox_with_id, clip_bbox_xyxy

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED_FOLDER = os.path.join('static', 'processed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Initialize Detector and Tracker
detector = YOLOv8PersonDetector()
tracker = DeepSortTracker(max_age=30, n_init=3)

# Global dictionary to track progress of video processing tasks
# Format: { task_id: { "progress": int, "filename": str, "complete": bool } }
processing_tasks = {}

def generate_frames(source=0, save_path=None, task_id=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Video Writer setup if saving
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

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]

        # 1. Detect
        detections = detector.detect(frame)
        
        valid_dets = []
        for det in detections:
             bbox = clip_bbox_xyxy(det["bbox"], w, h)
             if (bbox[2] - bbox[0]) > 1 and (bbox[3] - bbox[1]) > 1:
                 valid_dets.append({"bbox": bbox, "conf": det["conf"]})

        # 2. Track
        tracks = tracker.update(frame, valid_dets)
        
        for tr in tracks:
            track_id = tr["track_id"]
            bbox = tr["bbox"]
            unique_person_ids.add(track_id)
            draw_bbox_with_id(frame, bbox, track_id)

        # Overlay Count
        cv2.putText(frame, f"Count: {len(unique_person_ids)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save frame
        if out:
            out.write(frame)
            if task_id and "total_frames" in processing_tasks[task_id]:
                progress = int((frame_count / processing_tasks[task_id]["total_frames"]) * 100)
                processing_tasks[task_id]["progress"] = min(progress, 99) # Keep at 99 until finished

        # Encode for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    if out:
        out.release()
        if task_id:
            processing_tasks[task_id]["progress"] = 100
            processing_tasks[task_id]["complete"] = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        task_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        processing_tasks[task_id] = {
            "progress": 0,
            "filename": output_filename,
            "complete": False,
            "source": filepath,
            "output_path": output_path
        }
        
        return jsonify({"task_id": task_id, "filename": filename})

@app.route('/stream_processing/<task_id>')
def stream_processing(task_id):
    if task_id not in processing_tasks:
        return "Task not found", 404
    
    task = processing_tasks[task_id]
    return Response(generate_frames(task["source"], task["output_path"], task_id), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_progress/<task_id>')
def get_progress(task_id):
    if task_id not in processing_tasks:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(processing_tasks[task_id])

@app.route('/download_video/<task_id>')
def download_video(task_id):
    if task_id not in processing_tasks or not processing_tasks[task_id]["complete"]:
        return "File not ready", 404
    
    task = processing_tasks[task_id]
    return send_from_directory(app.config['PROCESSED_FOLDER'], task["filename"], as_attachment=True)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    import base64
    import numpy as np
    data = request.get_json()
    if not data or 'image' not in data:
        return "No image data", 400
    
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return "Invalid image", 400

    h, w = frame.shape[:2]
    unique_person_ids = set()
    detections = detector.detect(frame)
    
    valid_dets = []
    for det in detections:
         bbox = clip_bbox_xyxy(det["bbox"], w, h)
         if (bbox[2] - bbox[0]) > 1 and (bbox[3] - bbox[1]) > 1:
             valid_dets.append({"bbox": bbox, "conf": det["conf"]})

    tracks = tracker.update(frame, valid_dets)
    for tr in tracks:
        track_id = tr["track_id"]
        bbox = tr["bbox"]
        unique_person_ids.add(track_id)
        draw_bbox_with_id(frame, bbox, track_id)

    cv2.putText(frame, f"Count: {len(unique_person_ids)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    ret, buffer = cv2.imencode('.jpg', frame)
    processed_img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {"image": f"data:image/jpeg;base64,{processed_img_base64}"}

if __name__ == '__main__':
    try:
        from flask_cloudflared import run_with_cloudflared
        run_with_cloudflared(app)
        print(f"\n\n * [CLOUDFLARE] Starting secure tunnel...")
    except Exception as e:
        print(f" * Could not start Cloudflare Tunnel: {e}")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
