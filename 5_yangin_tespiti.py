# 5_yangin_tespiti.py
import cv2
import time
import threading
import numpy as np
from transformers import pipeline
from PIL import Image

print("🚀 Modeller yukleniyor... Lutfen bekleyin.")
detector = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
print("✅ Model hazir!")

# --- AYARLAR ---
GRID_BOYUTU = 3       
YANGIN_ESIGI = 0.65   # Değerlendirmede önerildiği gibi 0.55'ten 0.65'e çıkarıldı (Daha emin olsun)

mevcut_kare = None
tespit_edilen_kutular = []
analiz_aktif = True

# ==========================================================
# YENİ EKLENEN FİLTRE: RENK KONTROLÜ (HSV)
# ==========================================================
def yangin_rengi_var_mi(bgr_parca, esik=0.12):
    """
    Parçada yangın rengi (turuncu/kırmızı/sarı) var mı?
    esik: Piksellerin kaçının yangın renginde olması gerektiği (0.12 = %12)
    """
    hsv = cv2.cvtColor(bgr_parca, cv2.COLOR_BGR2HSV)
    
    # Kırmızı ton 1 (0-10 derece)
    maske1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    # Kırmızı ton 2 (170-180 derece)      
    maske2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    # Turuncu/Sarı ton (10-35 derece)
    maske3 = cv2.inRange(hsv, (10, 100, 100), (35, 255, 255))
    
    toplam_maske = maske1 | maske2 | maske3
    oran = np.count_nonzero(toplam_maske) / toplam_maske.size
    
    return oran > esik

# ==========================================================
# ARKA PLAN İŞÇİSİ (YAPAY ZEKA MOTORU)
# ==========================================================
def analiz_motoru():
    global tespit_edilen_kutular
    while analiz_aktif:
        if mevcut_kare is not None:
            kare_kopya = mevcut_kare.copy()
            pil_tam = Image.fromarray(cv2.cvtColor(kare_kopya, cv2.COLOR_BGR2RGB))
            
            res = detector(pil_tam, candidate_labels=["fire", "no fire"])
            genel_skor = next(r for r in res if r["label"] == "fire")["score"]
            
            yeni_kutular = []
            
            if genel_skor > 0.40:
                h, w = kare_kopya.shape[:2]
                adim_x = w // GRID_BOYUTU
                adim_y = h // GRID_BOYUTU
                
                for row in range(GRID_BOYUTU):
                    for col in range(GRID_BOYUTU):
                        x1 = col * adim_x
                        y1 = row * adim_y
                        x2 = x1 + adim_x
                        y2 = y1 + adim_y

                        parca = kare_kopya[y1:y2, x1:x2]
                        pil_parca = Image.fromarray(cv2.cvtColor(parca, cv2.COLOR_BGR2RGB))

                        p_res = detector(pil_parca, candidate_labels=["fire", "no fire"])
                        skor = next(r for r in p_res if r["label"] == "fire")["score"]

                        # --- DEĞİŞTİRİLEN ÇİFT FİLTRE KISMI ---
                        # Hem yapay zeka yangın diyecek (skor > 0.65)
                        # Hem de HSV filtresi o parçada gerçek ateş rengi bulacak (oran > 0.12)
                        if skor > YANGIN_ESIGI and yangin_rengi_var_mi(parca):
                            yeni_kutular.append((x1, y1, x2, y2, skor))
            
            tespit_edilen_kutular = yeni_kutular
        time.sleep(0.5)

ai_thread = threading.Thread(target=analiz_motoru, daemon=True)
ai_thread.start()

# ==========================================================
# ANA KAMERA DONGUSU
# ==========================================================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    
    # Görüntüyü arka plandaki yapay zekaya gönder
    mevcut_kare = frame.copy()

    # Arka plandan gelen kutuları ekrana çiz
    for (x1, y1, x2, y2, skor) in tespit_edilen_kutular:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"YANGIN: %{int(skor*100)}", (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    # Bilgi Paneli
    cv2.rectangle(frame, (0, 0), (250, 40), (0, 0, 0), -1)
    cv2.putText(frame, "DURUM: AKTIF (AI+HSV)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Yangin Tespiti v2.0", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        analiz_aktif = False
        break

cap.release()
cv2.destroyAllWindows()