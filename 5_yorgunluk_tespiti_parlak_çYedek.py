# 5_yorgunluk_tespiti_parlak_çYedek.py

import cv2
import dlib
import numpy as np
from imutils import face_utils
import winsound
import datetime
import requests
import time
son_api_gonderi = 0

# 1. Modeller
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def improve_light(image):
    """Ters ışıkta yüzün kararmasını engellemek için CLAHE uygular."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def calculate_mar(shape):
    # Hocanın düzelttiği doğru iç dudak indeksleri
    A = np.linalg.norm(shape[62] - shape[66])
    B = np.linalg.norm(shape[63] - shape[65])
    C = np.linalg.norm(shape[60] - shape[64])
    return (A + B) / (2.0 * C)

# --- AYARLAR ---
EYE_AR_THRESH = 0.22 
HEAD_DIST_THRESH = 0 
MOUTH_AR_THRESH = 0.65
EYE_AR_CONSEC_FRAMES = 15 #15 idi

COUNTER = 0
frame_count = 0
calibration_frames = 60
calibration_ears = []
calibration_dists = []
log_file = "driver_logs.txt"
rects = [] # <-- BURASI KRİTİK: Hata almamak için döngü dışında da tanımlıyoruz

cap = cv2.VideoCapture(0)

# Log dosyası başlatma
with open(log_file, "a") as f:
    f.write(f"\n--- Oturum Basladi: {datetime.datetime.now()} ---\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    
    # Aydınlatma müdahalesi
    frame = improve_light(frame)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # HATA ÇÖZÜMÜ: rects her zaman tanımlı olmalı
    if frame_count % 2 == 0:
        rects = detector(gray, 0)
    
    # Yüz yokken uyarı ver
    if len(rects) == 0:
        cv2.putText(frame, "SURUCU ARANIYOR...", (10, 30), 2, 0.7, (0, 255, 255), 2)

    for rect in rects:
        shape_raw = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape_raw)

        leftEye = shape[36:42]
        rightEye = shape[42:48]
        ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0
        mar = calculate_mar(shape)
        nose_chin_dist = shape[8][1] - shape[30][1]

        # --- DİNAMİK KALİBRASYON ---
        if frame_count <= calibration_frames:
            calibration_ears.append(ear)
            calibration_dists.append(nose_chin_dist)
            cv2.rectangle(frame, (0,0), (640, 40), (0,0,0), -1)
            cv2.putText(frame, f"KALIBRASYON: %{int((frame_count/60)*100)}", (180, 25), 1, 1.2, (0, 255, 255), 2)
            
            if frame_count == calibration_frames:
                EYE_AR_THRESH = np.mean(calibration_ears) * 0.78
                HEAD_DIST_THRESH = np.mean(calibration_dists) * 0.85
            continue

        # --- KARAR MEKANİZMASI ---
        if ear < EYE_AR_THRESH or nose_chin_dist < HEAD_DIST_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.rectangle(frame, (0,0), (640, 60), (0,0,255), -1)
                cv2.putText(frame, "!!! TEHLIKE: UYUKLAMA !!!", (120, 40), 2, 1, (255, 255, 255), 3)
                winsound.Beep(1500, 300)
                with open(log_file, "a") as f:
                    f.write(f"IHLAL: {datetime.datetime.now()}\n")
        else:
            COUNTER = 0
            cv2.putText(frame, "DURUM: AKTIF", (10, 35), 2, 0.7, (0, 255, 0), 2)

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "ESNEME TESPITI", (10, 70), 2, 0.7, (255, 255, 0), 2)

        # Bilgi Paneli
        cv2.rectangle(frame, (0, 440), (640, 480), (50, 50, 50), -1)
        info = f"EAR: {ear:.2f} | MAR: {mar:.2f} | DIST: {nose_chin_dist:.0f}"
        cv2.putText(frame, info, (10, 465), 1, 1, (255, 255, 255), 1)
        
        # API'ye gönder (her 1 saniyede bir)
        simdi = time.time()
        if simdi - son_api_gonderi >= 1:
            uyari_seviyesi = "normal"
            uyari_mesaji   = ""
            goz_kapali     = ear < EYE_AR_THRESH
            bas_egildi     = nose_chin_dist < HEAD_DIST_THRESH
            esniyor        = mar > MOUTH_AR_THRESH
        
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                uyari_seviyesi = "kritik"
                uyari_mesaji   = "Uyuklama tespiti!"
            elif esniyor:
                uyari_seviyesi = "dikkat"
                uyari_mesaji   = "Esneme tespiti"
            elif goz_kapali:
                uyari_seviyesi = "dikkat"
                uyari_mesaji   = "Gözler kapanıyor"
        
            try:
                requests.post("http://localhost:5000/surucu/guncelle", json={
                    "ear"            : round(float(ear), 3),
                    "mar"            : round(float(mar), 3),
                    "bas_mesafesi"   : round(float(nose_chin_dist), 1),
                    "goz_kapali"     : bool(goz_kapali),
                    "esniyor"        : bool(esniyor),
                    "bas_egildi"     : bool(bas_egildi),
                    "uyari_seviyesi" : uyari_seviyesi,
                    "uyari_mesaji"   : uyari_mesaji,
                    "yuz_tespit"     : True,
                    "zaman"          : datetime.datetime.now().isoformat()
                }, timeout=1)
                son_api_gonderi = simdi
            except Exception:
                pass

    cv2.imshow("Driver Safety System v2.0", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
