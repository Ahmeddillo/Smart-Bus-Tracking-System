# 🚌 Smart Bus Tracking System

<img width="1244" height="628" alt="Screenshot 2026-06-04 210501" src="https://github.com/user-attachments/assets/849d4cf8-11a5-4838-bbf4-f67438c41fa6" />

An AI-powered security and monitoring system for public buses. An integrated platform comprising passenger counting, driver fatigue detection, fire detection, and pothole detection modules.

---

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Architecture Overview](#architecture-overview)
- [License](#license)

---

## System Overview

This system is designed for real-time safety and operational monitoring in urban public transit vehicles. By processing camera feeds with computer vision and artificial intelligence models, it provides passenger counts, driver status, fire hazard alerts, and road condition (pothole) insights to administrators via a centralized API.

---

## Features

### 👤 Passenger Counting (`kafasayan.py` / `1_kafasayan3.py`)
- Real-time human detection using YOLOv8
- Instantaneous and moving-average passenger counts
- Support for video files, webcams, and RTSP/HTTP live streams
- Automatic reconnection on connection loss
- Thread-based parallel processing (decoupled video reading and frame analysis)
- Live JPEG video streaming over the API

### 😴 Driver Fatigue Detection (`2_yorgunluk_tespiti_parlakortam2.py`)
- Real-time facial analysis using Dlib's 68-landmark model
- **EAR** (Eye Aspect Ratio) — eye closure detection
- **MAR** (Mouth Aspect Ratio) — yawning detection
- Head posture and tilt monitoring (nose-to-chin distance)
- CLAHE image enhancement for backlight and high-brightness environments
- Automated personalized calibration during the initial 60 frames
- Audio alarms via `winsound.Beep`
- Alert classification levels: `normal` / `warning` / `critical`
- Asynchronous PostgreSQL logging via background threads

### 🔥 Fire Detection (`3_yangin_tespiti.py`)
- Zero-shot image classification powered by OpenAI's CLIP model (`clip-vit-base-patch32`)
- Grid-based spatial analysis (3×3 grid)
- HSV color thresholding (red/orange/yellow spectrum) for dual verification
- Multi-layered confidence evaluation (overall score + spatial score + color filter)
- Bounding box rendering with detection coordinates
- Real-time metric transmission to the API

### 🛣️ Pothole Detection (API Integration)
- Camera-based pothole recognition module
- Spatial database persistence with GPS coordinates
- Geographic data endpoints for map visualization

### 🌐 Centralized API (`api.py`)
- RESTful API powered by FastAPI
- Multi-bus video source management (RTSP, local video, webcam)
- Real-time and historical telemetry endpoints
- CSV and JSON data export functionality
- Live video feed (MJPEG)
- PostgreSQL primary database with SQLite fallback support
- CORS middleware for seamless web/mobile client integration

---

## Project Structure

```
Smart-Bus-Tracking-System/
│
├── api.py                              # Main FastAPI application (central hub)
├── database.py                         # Database connection and ORM models
│
├── kafasayan.py                        # Passenger counter (v1)
├── kafasayan2.py                       # Passenger counter (v2 - API integrated)
├── 1_kafasayan3.py                     # Passenger counter (v3 - standalone test mode)
│
├── 2_yorgunluk_tespiti_parlakortam2.py # Driver fatigue detection
├── 5_yorgunluk_tespiti_parlak_çYedek.py# Backup driver module
│
├── 3_yangin_tespiti.py                 # Fire detection module
│
├── Arayuz/                             # Web interface (HTML/CSS)
│
├── Dipnotes.txt                        # Developer notes
└── kaynaklar (Yüklenmeyen Dosyalar)   # Reference directory for model assets
```

---

## Technologies Used

| Layer | Technology |
|-------|------------|
| Programming Language | Python 3.x |
| Computer Vision | OpenCV, NumPy |
| Object Detection | YOLOv8 (Ultralytics) |
| Facial Landmarks | Dlib + imutils |
| Image Classification | Hugging Face Transformers (CLIP) |
| Web API | FastAPI + Uvicorn |
| Database | PostgreSQL (fallback: SQLite) |
| ORM | SQLAlchemy |
| Environment Management | python-dotenv |
| Web Interface | HTML / C# |

---

## Installation

### Prerequisites

```bash
pip install ultralytics opencv-python dlib imutils numpy
pip install fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv
pip install transformers torch pillow requests
```

> **Note:** Compiling Dlib requires CMake and Visual Studio Build Tools. Ensure these development tools are present on your system prior to installing `dlib`.

### Model Weights

Download Dlib's 68-point facial landmark predictor and place it in the project root directory:

```
shape_predictor_68_face_landmarks.dat
```

Place the YOLO weight files (`medium.pt` or `yolov8n.pt`) inside the root directory for the passenger counting module.

### Database Configuration

Create a `.env` file in the project root directory:

```env
DATABASE_URL=postgresql://user:password@localhost/otobus_db
```

Create the `otobus_db` database in PostgreSQL. Tables are auto-generated when the system starts up for the first time.

---

## Running the Project

### 1. Start the API Server

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 5000 --reload
```

The passenger counting subsystem automatically initializes alongside the API.

### 2. Launch Driver Fatigue Detection

```bash
python 2_yorgunluk_tespiti_parlakortam2.py
```

### 3. Launch Fire Detection

```bash
python 3_yangin_tespiti.py
```

### 4. Standalone Passenger Counter Testing

```bash
# Using the default video stream
python 1_kafasayan3.py

# Using a custom RTSP stream and model
python 1_kafasayan3.py --video rtsp://192.168.1.100/stream --model yolov8n.pt
```

---

## API Documentation

When the service is active, interactive Swagger documentation is available at:

```
http://localhost:5000/docs
```

### Key Endpoints

#### General

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API metadata and endpoint directory |
| GET | `/durum` | System and database connectivity status |
| GET | `/test` | Health check probe |

#### Passenger Counting

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/canli` | Instantaneous and average passenger count |
| GET | `/video` | Live MJPEG video stream |
| GET | `/gecmis` | In-memory telemetry log (max 1000 entries) |
| GET | `/gecmis/saatlik` | Aggregated statistics for the last hour |
| GET | `/gecmis/saatlik/detayli` | Granular log records for the last hour |
| GET | `/export?format=csv` | Download telemetry data as CSV |
| GET | `/db/gecmis` | Historical passenger records from PostgreSQL |
| GET | `/db/oturumlar` | Logged operational sessions |
| GET | `/db/rapor/{oturum_id}` | Session summary report |

#### Resource Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/kaynaklar` | List available onboard camera sources |
| POST | `/kaynaklar/{id}/sec` | Switch active camera source |

#### Driver Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/surucu/guncelle` | Receive telemetry from camera module |
| GET | `/surucu/durum` | Current driver alert state |
| GET | `/surucu/gecmis` | Historical warning events |
| GET | `/surucu/istatistik` | Daily driver warning summary |

#### Fire Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/yangin/guncelle` | Receive telemetry from fire module |
| GET | `/yangin/durum` | Current fire hazard status |
| GET | `/yangin/gecmis` | Historical fire detection records |

#### Pothole Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/cukur/guncelle` | Receive data from pothole camera |
| GET | `/cukur/durum` | Real-time road condition status |
| GET | `/cukur/gecmis` | Historical pothole detection events |
| GET | `/cukur/harita` | GPS coordinates for spatial mapping |

---

## Database Schema

```
oturumlar           → System runtime sessions
yolcu_sayimlari     → Time-series passenger counting logs
oturum_raporlari    → Session-level aggregated summary reports
surucu_loglari      → Driver state changes and alert logs
yangin_loglari      → Fire detection records (camera ID, confidence score, alarm state)
cukur_loglari       → Pothole detection records (with GPS coordinates)
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   api.py (FastAPI)                  │
│  /canli  /video  /surucu  /yangin  /cukur  /kaynaklar│
└────────────┬──────────┬──────────┬──────────────────┘
             │          │          │
    ┌────────┘   ┌──────┘   ┌──────┘
    ▼            ▼          ▼
kafasayan2.py   Driver     Fire
(YOLO Thread)   Module    Module
    │          (Dlib)     (CLIP)
    ▼            │          │
PostgreSQL ◄─────┴──────────┘
```

All edge modules send autonomous POST requests to push data to the API. The core API manages dual persistence: caching telemetry in RAM for low-latency live reads and committing entries to PostgreSQL for long-term storage.

---

## License

This project is developed for educational purposes.
