# Human Tracking & Real-Time Monitoring 🚀

A powerful real-time human detection and tracking system using **YOLOv8** and **DeepSORT**, featuring a web interface that supports multi-device camera streaming.

---

## 🌟 Key Features

- **Real-Time Detection:** Powered by Ultralytics YOLOv8 for high-speed person detection.
- **Advanced Tracking:** DeepSORT algorithm maintains unique IDs for individuals across frames.
- **Multi-Device Support:** Use any smartphone or laptop camera as a video source via the web browser.
- **Auto-Secure Tunneling:** Built-in Serveo integration via SSH for instant HTTPS access on mobile devices without any configuration.
- **Interactive Web UI:** Modern, responsive dashboard built with HTML, CSS, and JavaScript.

---

## 🛠️ Technology Stack

- **Backend:** Python, Flask
- **Computer Vision:** OpenCV, Ultralytics (YOLOv8), Deep-Sort-Realtime
- **Frontend:** Vanilla JS, CSS (Glassmorphism design), HTML5
- **Networking:** Serveo (via SSH tunnel)

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/amassrywork-collab/human-tracking-deepsort.git
cd human-tracking-deepsort
pip install -r requirements.txt
```

### 3. Running the Server
Simply run the Flask application:
```bash
python app.py
```

### 4. Accessing the System
Once the server starts, check the console output for two links:
1. **Local Access:** `http://10.x.x.x:5000` (for devices on the same network).
2. **Secure Remote Access:** Look for the `serveo.net` link (e.g., `https://xxxx.serveo.net`). **Use this link on your mobile phone to enable camera access.**

---

## 📱 Mobile Camera Usage
1. Open the **HTTPS** link on your smartphone.
2. Scroll to **"Experience It Live"**.
3. Click **"Start Camera"** and **Allow** camera permissions.
4. Watch the real-time tracking from your phone's feed!

---

## 👥 Team
- **Yahya Jihad Abu Saqr** (120213375)
- **Ahmed Monir Almassri** (120220138)
- **Ahmed M. M. Abu Sabha** (120220304)

---

## 📄 License
This project is for educational purposes as part of the Human Tracking Project 2026.
