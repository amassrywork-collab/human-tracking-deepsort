
from flask import Flask, render_template, Response, request, send_from_directory
import cv2
import os
import sys
from werkzeug.utils import secure_filename

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector import YOLOv8PersonDetector
from tracker import DeepSortTracker
from utils import draw_bbox_with_id, clip_bbox_xyxy

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Detector and Tracker
# Using default weights (yolov8n.pt) or your specific ones if needed
detector = YOLOv8PersonDetector()
tracker = DeepSortTracker(max_age=30, n_init=3)

def generate_frames(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    unique_person_ids = set()

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Resize for performance and web display usually good around 640x480 or 1280x720
        # frame = cv2.resize(frame, (1280, 720)) 
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

        # Encode
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # 0 is usually the default camera
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Return a page or stream component that plays this video processed? 
        # For simplicity in this demo, we might just stream it immediately or return a separate viewer page.
        # Let's try to stream the uploaded file.
        return Response(generate_frames(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')

import base64
import numpy as np

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Receive frame from client (base64)
    data = request.get_json()
    if not data or 'image' not in data:
        return "No image data", 400
    
    # Decode base64 image
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return "Invalid image", 400

    h, w = frame.shape[:2]
    unique_person_ids = set()

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

    # Encode back to JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    processed_img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {"image": f"data:image/jpeg;base64,{processed_img_base64}"}

if __name__ == '__main__':
    # Initialize Cloudflare Tunnel for secure access (HTTPS) without an account
    try:
        from flask_cloudflared import run_with_cloudflared
        run_with_cloudflared(app)
        print(f"\n\n * [CLOUDFLARE] Starting secure tunnel...")
        print(f" * Look for the 'TryCloudflare' link in the output below!\n\n")
    except Exception as e:
        print(f" * Could not start Cloudflare Tunnel: {e}")
        print(f" * Please ensure you have flask-cloudflared installed: pip install flask-cloudflared")

    app.run(host="0.0.0.0", port=5000, debug=False)
