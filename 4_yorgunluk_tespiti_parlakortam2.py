import cv2
import dlib
import numpy as np
from imutils import face_utils
import winsound
import datetime
import time
import threading
import os
from collections import deque

# --- VERİTABANI ---
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, text
from sqlalchemy.orm import declarative_base, Session
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Hello@localhost/otobus_db")

engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    conn.execute(text("SELECT 1"))
print("✅ Baglanti OK")

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base = declarative_base()
    print("✅ Veritabani baglantisi kuruldu.")
except Exception as e:
    print(f"❌ Veritabani hatasi: {e}")
    engine = create_engine("sqlite:///./surucu_yedek.db", connect_args={"check_same_thread": False})
    Base = declarative_base()

# --- TABLO ---
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
    print("✅ Tablo kontrol edildi.")
except Exception as e:
    print(f"❌ Tablo hatasi: {e}")

# ===================== MODEL VE AYARLAR =====================
print("🔄 Dlib modeli yukleniyor...")
detector  = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("✅ Model yuklendi!")
except:
    print("❌ HATA: shape_predictor_68_face_landmarks.dat bulunamadi!")
    exit()

# Eşikler (kalibrasyon sonrası güncellenecek)
EYE_AR_THRESH      = 0.22
HEAD_DIST_THRESH   = 0
MOUTH_AR_THRESH    = 0.65
EYE_AR_CONSEC_FRAMES = 20

# Sayaçlar
COUNTER            = 0
frame_count        = 0
calibration_frames = 60
calibration_ears   = []
calibration_dists  = []
rects              = []

# RAM'de son durum (API için)
son_durum = {
    "ear"           : 0.0,
    "mar"           : 0.0,
    "bas_mesafesi"  : 0,
    "goz_kapali"    : False,
    "esniyor"       : False,
    "bas_egildi"    : False,
    "uyari_seviyesi": "normal",
    "uyari_mesaji"  : "",
    "kalibrasyon"   : True,
    "zaman"         : datetime.datetime.now().isoformat()
}
son_durum_lock = threading.Lock()

# DB kayıt kuyruğu (ana thread'i yavaşlatmamak için)
db_kuyruk = deque(maxlen=200)

# ===================== FONKSİYONLAR =====================
def improve_light(image):
    """Ters ışıkta yüzün kararmasını engellemek için CLAHE uygular."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

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

def db_kayit_thread():
    """Arka planda DB'ye kayıt yapan thread."""
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

# DB thread'ini başlat
db_thread = threading.Thread(target=db_kayit_thread, daemon=True)
db_thread.start()

# Log dosyası
log_file = "driver_logs.txt"
with open(log_file, "a") as f:
    f.write(f"\n--- Oturum Basladi: {datetime.datetime.now()} ---\n")

# ===================== ANA DÖNGÜ =====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera acilamadi!")
    exit()

print("🚀 Sistem baslatildi. Kalibrasyon basliyor...")

son_db_kayit = time.time()  # DB'ye her 5 saniyede bir yaz

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = improve_light(frame)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Her 2 karede bir yüz ara (performans)
    if frame_count % 2 == 0:
        rects = detector(gray, 0)

    if len(rects) == 0:
        cv2.putText(frame, "SURUCU BULUNAMADI", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
        with son_durum_lock:
            son_durum["uyari_mesaji"] = "Surucu bulunamadi"
            son_durum["yuz_tespit"] = False

    for rect in rects:
        shape_raw = predictor(gray, rect)
        shape     = face_utils.shape_to_np(shape_raw)

        leftEye  = shape[36:42]
        rightEye = shape[42:48]
        ear      = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0
        mar      = calculate_mar(shape)
        nose_chin_dist = float(shape[8][1] - shape[30][1])

        # --- KALİBRASYON ---
        if frame_count <= calibration_frames:
            calibration_ears.append(ear)
            calibration_dists.append(nose_chin_dist)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 0), -1)
            cv2.putText(frame,
                        f"KALIBRASYON: %{int((frame_count / calibration_frames) * 100)}",
                        (150, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
            if frame_count == calibration_frames:
                EYE_AR_THRESH    = np.mean(calibration_ears) * 0.78
                HEAD_DIST_THRESH = np.mean(calibration_dists) * 0.85
                print(f"✅ Kalibrasyon tamamlandi: EAR={EYE_AR_THRESH:.3f}, DIST={HEAD_DIST_THRESH:.1f}")
                with son_durum_lock:
                    son_durum["kalibrasyon"] = False
            continue

        # --- DURUM KONTROLÜ ---
        goz_kapali_mi = ear < EYE_AR_THRESH
        bas_egildi_mi = nose_chin_dist < HEAD_DIST_THRESH
        esniyor_mi    = mar > MOUTH_AR_THRESH

        if goz_kapali_mi or bas_egildi_mi:
            COUNTER += 1
        else:
            COUNTER = 0

        # Uyarı seviyesi belirle
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            uyari_seviyesi = "kritik"
            uyari_mesaji   = "TEHLIKE: UYUKLAMA!"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (0, 0, 255), -1)
            cv2.putText(frame, "!!! TEHLIKE: UYUKLAMA !!!", (80, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)
            winsound.Beep(1500, 300)
            # Log dosyasına yaz
            with open(log_file, "a") as f:
                f.write(f"IHLAL: {datetime.datetime.now()} | EAR={ear:.3f} MAR={mar:.3f}\n")
        elif esniyor_mi:
            uyari_seviyesi = "dikkat"
            uyari_mesaji   = "Esneme tespiti"
            cv2.putText(frame, "ESNEME TESPITI", (10, 75),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 2)
        elif goz_kapali_mi:
            uyari_seviyesi = "dikkat"
            uyari_mesaji   = "Gozler kapaniyor"
        else:
            uyari_seviyesi = "normal"
            uyari_mesaji   = ""
            cv2.putText(frame, "SURUCU AKTIF", (10, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

        # --- RAM GÜNCELLE (API için) ---
        with son_durum_lock:
            son_durum.update({
                "ear"           : round(float(ear), 3),
                "mar"           : round(float(mar), 3),
                "bas_mesafesi"  : round(nose_chin_dist, 1),
                "goz_kapali"    : COUNTER >= EYE_AR_CONSEC_FRAMES,
                "esniyor"       : esniyor_mi,
                "bas_egildi"    : bas_egildi_mi,
                "uyari_seviyesi": uyari_seviyesi,
                "uyari_mesaji"  : uyari_mesaji,
                "kalibrasyon"   : False,
                "yuz_tespit"    : True,
                "zaman"         : datetime.datetime.now().isoformat()
            })

        # --- DB KAYIT (her 5 saniyede bir, sadece dikkat/kritik) ---
        simdi = time.time()
        if simdi - son_db_kayit >= 5 and uyari_seviyesi in ("dikkat", "kritik"):
            db_kuyruk.append({
                "ear_degeri"    : round(float(ear), 3),
                "mar_degeri"    : round(float(mar), 3),
                "bas_mesafesi"  : round(nose_chin_dist, 1),
                # BURADAKI int()'ler bool() OLARAK DÜZELTİLDİ:
                "goz_kapali"    : bool(COUNTER >= EYE_AR_CONSEC_FRAMES),
                "esniyor"       : bool(esniyor_mi),
                "bas_egildi"    : bool(bas_egildi_mi),
                "uyari_seviyesi": uyari_seviyesi,
                "uyari_mesaji"  : uyari_mesaji
            })
            son_db_kayit = simdi

        # --- Alt Bilgi Paneli ---
        cv2.rectangle(frame, (0, frame.shape[0]-35), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
        cv2.putText(frame,
                    f"EAR: {ear:.3f}  MAR: {mar:.3f}  DIST: {nose_chin_dist:.0f}  (Limit: {HEAD_DIST_THRESH:.0f})",
                    (10, frame.shape[0]-10), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 1)

        # Sağ üst durum göstergesi
        renk = {"normal": (0,255,0), "dikkat": (0,255,255), "kritik": (0,0,255)}.get(uyari_seviyesi, (255,255,255))
        cv2.rectangle(frame, (frame.shape[1]-160, 0), (frame.shape[1], 40), renk, -1)
        cv2.putText(frame, uyari_seviyesi.upper(),
                    (frame.shape[1]-150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    cv2.imshow("Driver Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Sistem kapatildi.")