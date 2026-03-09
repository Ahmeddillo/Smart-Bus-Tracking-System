import cv2
import dlib
import numpy as np
from imutils import face_utils
import winsound
import time
import datetime
import threading
import os
import requests
from collections import deque

# --- VERİTABANI İMPORTLARI ---
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, text
from sqlalchemy.orm import declarative_base, Session
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. VERİTABANI VE API KURULUMU
# ---------------------------------------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Hello@localhost/otobus_db")

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base = declarative_base()
    print("✅ Veritabanı bağlantısı kuruldu.")
except Exception as e:
    print(f"❌ Veritabanı hatası: {e}")
    engine = create_engine("sqlite:///./surucu_yedek.db", connect_args={"check_same_thread": False})
    Base = declarative_base()

class SurucuLogu(Base):
    __tablename__ = "surucu_loglari"
    id             = Column(Integer, primary_key=True)
    oturum_id      = Column(Integer, nullable=True)
    zaman          = Column(DateTime, default=datetime.datetime.now)
    ear_degeri     = Column(Float)
    mar_degeri     = Column(Float)
    bas_mesafesi   = Column(Float)
    goz_kapali     = Column(Boolean, default=False) 
    esniyor        = Column(Boolean, default=False)
    bas_egildi     = Column(Boolean, default=False)
    uyari_seviyesi = Column(String)
    uyari_mesaji   = Column(String)

try:
    Base.metadata.create_all(engine)
except Exception as e:
    pass

# --- DB KUYRUK VE THREAD ---
db_kuyruk = deque(maxlen=200)

def db_kayit_thread():
    while True:
        if db_kuyruk:
            kayit_verisi = db_kuyruk.popleft()
            try:
                with Session(engine) as session:
                    log = SurucuLogu(**kayit_verisi)
                    session.add(log)
                    session.commit()
            except Exception as e:
                print(f"⚠️ DB kayit hatasi: {e}")
        time.sleep(0.1)

threading.Thread(target=db_kayit_thread, daemon=True).start()

# --- RAM DURUMU (API İÇİN) ---
son_durum = {}
son_durum_lock = threading.Lock()

# ---------------------------------------------------------
# 2. MODEL VE AYARLAR
# ---------------------------------------------------------
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("✅ Model yuklendi!")
except:
    print("❌ HATA: shape_predictor_68_face_landmarks.dat bulunamadi!")
    exit()

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    if C == 0: return 0.0
    return (A + B) / (2.0 * C)

def calculate_mar(shape):
    A = np.linalg.norm(shape[62] - shape[66])
    B = np.linalg.norm(shape[63] - shape[65])
    C = np.linalg.norm(shape[60] - shape[64])
    if C == 0: return 0.0
    return (A + B) / (2.0 * C)

# Ayarlar
EYE_AR_THRESH = 0.22 
HEAD_DIST_THRESH = 0 
MOUTH_AR_THRESH = 0.65
EYE_AR_CONSEC_FRAMES = 20

COUNTER = 0
frame_count = 0
calibration_frames = 50
calibration_ears = []
calibration_dists = []

# ---------------------------------------------------------
# 3. KAMERA VE ANA DÖNGÜ
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera acilmiyor! Lutfen Kernel Restart yapin.")
    exit()
else:
    print("🚀 Kamera acildi. Kalibrasyon basliyor, lütfen normal bakin...")

rects = []
son_db_kayit = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if frame_count % 2 == 0:
        rects = detector(gray, 0)
    
    if len(rects) == 0:
        cv2.putText(frame, "SURUCU BULUNAMADI", (10, 30), 2, 0.7, (0, 0, 255), 2)
        with son_durum_lock:
            son_durum["uyari_mesaji"] = "Surucu bulunamadi"
            son_durum["yuz_tespit"] = False

    for rect in rects:
        shape_raw = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape_raw)

        leftEye = shape[36:42]
        rightEye = shape[42:48]
        ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0
        mar = calculate_mar(shape)
        nose_chin_dist = float(shape[8][1] - shape[30][1])

        # --- KALİBRASYON ---
        if frame_count <= calibration_frames:
            calibration_ears.append(ear)
            calibration_dists.append(nose_chin_dist)
            cv2.putText(frame, f"KALIBRASYON: {frame_count}/{calibration_frames}", (10, 30), 1, 1.5, (255, 255, 0), 2)
            if frame_count == calibration_frames:
                EYE_AR_THRESH = np.mean(calibration_ears) * 0.75
                HEAD_DIST_THRESH = np.mean(calibration_dists) * 0.80
                print(f"✅ Kalibrasyon Bitti. EAR: {EYE_AR_THRESH:.2f}, DIST: {HEAD_DIST_THRESH:.0f}")
            continue

        # --- DURUM KONTROLÜ VE UYARI YÖNETİMİ ---
        goz_kapali_mi = ear < EYE_AR_THRESH
        bas_egildi_mi = nose_chin_dist < HEAD_DIST_THRESH
        esniyor_mi    = mar > MOUTH_AR_THRESH

        uyari_seviyesi = "normal"
        uyari_mesaji   = ""

        if goz_kapali_mi or bas_egildi_mi:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                uyari_seviyesi = "kritik"
                uyari_mesaji   = "TEHLIKE: UYUKLAMA!"
                cv2.putText(frame, "!!! TEHLIKE: UYUKLAMA !!!", (10, 40), 2, 0.8, (0, 0, 255), 3)
                winsound.Beep(1000, 250) 
            elif goz_kapali_mi:
                uyari_seviyesi = "dikkat"
                uyari_mesaji   = "Gozler kapaniyor"
        else:
            COUNTER = 0
            cv2.putText(frame, "SURUCU AKTIF", (10, 40), 2, 0.7, (0, 255, 0), 2)

        if esniyor_mi:
            cv2.putText(frame, "ESNEME TESPIT EDILDI", (10, 80), 2, 0.7, (255, 255, 0), 2)
            if uyari_seviyesi != "kritik":
                uyari_seviyesi = "dikkat"
                uyari_mesaji   = "Esneme tespiti"

        # --- RAM GÜNCELLE VE API'YE GÖNDER ---
        with son_durum_lock:
            son_durum.update({
                "ear"           : round(float(ear), 3),
                "mar"           : round(float(mar), 3),
                "bas_mesafesi"  : round(nose_chin_dist, 1),
                "goz_kapali"    : bool(COUNTER >= EYE_AR_CONSEC_FRAMES),
                "esniyor"       : bool(esniyor_mi),
                "bas_egildi"    : bool(bas_egildi_mi),
                "uyari_seviyesi": uyari_seviyesi,
                "uyari_mesaji"  : uyari_mesaji,
                "kalibrasyon"   : False,
                "yuz_tespit"    : True,
                "zaman"         : datetime.datetime.now().isoformat()
            })

        # API'ye fırlat (Çökmemesi için timeout var)
        try:
            requests.post("http://127.0.0.1:5000/surucu/guncelle", json=son_durum, timeout=0.1)
        except:
            pass

        # --- VERİTABANINA KAYIT (Kuyruğa Ekleme) ---
        simdi = time.time()
        if simdi - son_db_kayit >= 5 and uyari_seviyesi in ("dikkat", "kritik"):
            db_kuyruk.append({
                "ear_degeri"    : round(float(ear), 3),
                "mar_degeri"    : round(float(mar), 3),
                "bas_mesafesi"  : round(nose_chin_dist, 1),
                "goz_kapali"    : bool(COUNTER >= EYE_AR_CONSEC_FRAMES),
                "esniyor"       : bool(esniyor_mi),
                "bas_egildi"    : bool(bas_egildi_mi),
                "uyari_seviyesi": uyari_seviyesi,
                "uyari_mesaji"  : uyari_mesaji
            })
            son_db_kayit = simdi

        # Alt Bilgi Paneli
        cv2.putText(frame, f"EAR: {ear:.2f} DIST: {nose_chin_dist:.0f} (Limit: {HEAD_DIST_THRESH:.0f})", 
                    (10, frame.shape[0]-20), 1, 1, (255, 255, 255), 1)

    cv2.imshow("Driver Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()