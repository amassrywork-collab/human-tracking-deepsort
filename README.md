# 🚀 Human Tracking Pro: AI-Powered Intelligence Dashboard

[![YOLO11](https://img.shields.io/badge/Model-YOLO11-blue.svg)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art Graduation Project featuring real-time human tracking, pose estimation, and a premium glassmorphic dashboard. Powered by **YOLO11**, **ByteTrack**, and **FastAPI**.

---

## ✨ Features

- 🧠 **Next-Gen Detection**: Integrated with **YOLO11** (Nano/Small/Medium support).
- 👣 **Stable Tracking**: Uses **ByteTrack** for consistent identity retention.
- 🧘‍♂️ **Action Recognition**: Real-time pose estimation to detect "Walking", "Standing", and "Falling".
- 📱 **Mobile Optimized**: Fully responsive sidebar and adaptive grid layouts for smartphones and tablets.
- 🎨 **Premium UI/UX**: Modern Glassmorphism dashboard with **Staggered Entry Animations** and Scroll-reveal effects.
- 📊 **Live Analytics**: Real-time charts powered by **Chart.js** with **CSV Engagement Export** capabilities.
- ⚡ **Asynchronous Backend**: **FastAPI** integration for ultra-low latency video streaming.
- 📂 **Persistent Storage**: Automated logging of all detections into a **SQLite** database.
- 🛡️ **ROI Config**: Interactive "Region of Interest" (ROI) drawing tool to focus detection on specific zones.

---

## 🛠️ Technical Stack

- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6+), Chart.js.
- **Backend**: FastAPI, Uvicorn, Jinja2 Templates.
- **AI/ML**: Ultralytics YOLO11, ByteTrack, OpenCV.
- **Database**: SQLite3.
- **Deployment**: Docker support included.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Installation
```bash
git clone https://github.com/amassrywork-collab/human-tracking-deepsort.git
cd human-tracking-deepsort
pip install -r requirements.txt
```

### 3. Run the Server
- **Windows (Recommended)**: Double-click `run_server.bat`.
- **Manual**:
```bash
python app_fastapi.py
```

Open your browser at `http://localhost:5000`.

---

## 📁 Project Structure

```text
├── app_fastapi.py      # Main FastAPI server
├── src/                # AI logic (Detector, Tracker, Pose)
├── templates/          # Dashboard UI (Modern)
├── static/             # CSS (Glassmorphism), JS (Analytics)
├── models/             # YOLO weights
├── data/               # SQLite database exports
└── run_server.bat      # Quick-launch script
```

---

## 👤 Author
**Ahmed Monir Almassri**
*Student ID: 120220138*
*Graduation Project - Computer Engineering at IUGaza*

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
