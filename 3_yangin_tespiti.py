# 3_yangin_tespiti.py
import cv2
import time
import threading
import numpy as np
from transformers import pipeline
from PIL import Image
import requests
from datetime import datetime

# Burada api linki ve kamera ID'si tanımlanmıştır
API_URL   = "http://localhost:5000/yangin/guncelle"
KAMERA_ID = "kamera_1"

# Burada hangi modeli kullandığımız tanımlanmıştır: CLIP modeli kullanılıp, onun kullanıldığı bildirilmiştir
print("🚀 Modeller yukleniyor... Lutfen bekleyin.")
detector = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
print("✅ Model hazir!")

# --- AYARLAR ---
GRID_BOYUTU = 3       
YANGIN_ESIGI = 0.65   # Değerlendirmede önerildiği gibi 0.55'ten 0.65'e çıkarıldı (Daha emin olsun)

# İlk parametreler sıfırlanmıştır
mevcut_kare = None
tespit_edilen_kutular = []
analiz_aktif = True

# ---- YENİ EKLENEN FİLTRE: RENK KONTROLÜ (HSV) ----
def yangin_rengi_var_mi(bgr_parca, esik=0.12):
    """
    Parçada yangın rengi (turuncu/kırmızı/sarı) var mı?
    esik: Piksellerin kaçının yangın renginde olması gerektiği (0.12 = %12)
    """
    hsv = cv2.cvtColor(bgr_parca, cv2.COLOR_BGR2HSV)
    
    # Farklı tonlar için maskeler tanımlanmıştır, sonra da toplam maske tüm maskeleri OR operatörü ile toplanması sağlanmıştır
    # Kırmızı ton 1 (0-10 derece)
    maske1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    # Kırmızı ton 2 (170-180 derece)      
    maske2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    # Turuncu/Sarı ton (10-35 derece)
    maske3 = cv2.inRange(hsv, (10, 100, 100), (35, 255, 255))
    
    toplam_maske = maske1 | maske2 | maske3
    # Maskedeki sıfır olmayan (yani beyaz olan, yani yangın rengine uyan) piksellerin toplam sayısı
    # Görüntüdeki toplam piksel sayısına bölünmesi altta sağlanmıştır
    # Eğer oran eşik değerinden büyükse True döner, yani yangın vardır
    # Eğer oran eşik değerinden düşükse False döner
    oran = np.count_nonzero(toplam_maske) / toplam_maske.size
    return oran > esik


# ---- ARKA PLAN İŞÇİSİ (YAPAY ZEKA MOTORU) ----
# Sistemin beyni halindedir
# Sürekli çalışan bir döngü içinde kameradan gelen görüntüyü alır, 
# parçalara böler ve her parçada "Yangın var mı?"
# Hem Renk filtresine hem de yapay zekaya yangın var mı diye danışır

def analiz_motoru():
    global tespit_edilen_kutular
    while analiz_aktif:
        if mevcut_kare is not None:
            kare_kopya = mevcut_kare.copy()
            pil_tam = Image.fromarray(cv2.cvtColor(kare_kopya, cv2.COLOR_BGR2RGB))
            
            # aday etiketler tanımlanmıştır, genel skor da tanımlanmıştır
            res = detector(pil_tam, candidate_labels=["fire", "no fire"])
            genel_skor = next(r for r in res if r["label"] == "fire")["score"]
            
            yeni_kutular = []
            
            if genel_skor > 0.40:
                # eğer fire etiketi 0.40'den fazla aktifse görüntü alınır ve GRID BOYUTUNA bölünür
                h, w = kare_kopya.shape[:2]
                adim_x = w // GRID_BOYUTU
                adim_y = h // GRID_BOYUTU
                
                # Burada resim hem satır hem de sütun olarak gezilir
                for row in range(GRID_BOYUTU):
                    for col in range(GRID_BOYUTU):
                        # parçanın sol üst koordinatı hesaplanır
                        x1 = col * adim_x
                        y1 = row * adim_y
                        # parçanın sağ alt korrdinatı hesaplanır
                        x2 = x1 + adim_x
                        y2 = y1 + adim_y

                        parca = kare_kopya[y1:y2, x1:x2]# parçayı bir değişkene kopyalayıp
                        # o değişkeni matris dönüştürürüz
                        pil_parca = Image.fromarray(cv2.cvtColor(parca, cv2.COLOR_BGR2RGB))
                        
                        # yangın var mı diye skor hesaplanır
                        p_res = detector(pil_parca, candidate_labels=["fire", "no fire"])
                        skor = next(r for r in p_res if r["label"] == "fire")["score"]

                        # Hem yapay zeka yangın diyecek (skor > 0.65)
                        # Hem de HSV filtresi o parçada gerçek ateş rengi bulacak (oran > 0.12)
                        if skor > YANGIN_ESIGI and yangin_rengi_var_mi(parca):
                            yeni_kutular.append((x1, y1, x2, y2, skor))
            
            try:
                # yeni_kutular listesinde her kutunun bir skoru vardı. Bu satır, listenin içindeki en yüksek güven puanını seçer.
                en_yuksek_skor = max(s for *_, s in yeni_kutular) if yeni_kutular else 0.0
                payload = {
                    "kamera_id"  : KAMERA_ID,
                    "yangin_var" : len(yeni_kutular) > 0,
                    "skor"       : round(en_yuksek_skor, 3),
                    "kutu_sayisi": len(yeni_kutular),
                    "alarm"      : en_yuksek_skor > 0.80,
                    "zaman"      : datetime.now().isoformat()
                }
                # api'ye gönderilecek veri paketi yukarıda hazırlanmış oldu
                requests.post(API_URL, json=payload, timeout=2)
                # veri paketi api'ye gönderilir
            except Exception as e:
                print(f"⚠️ API gönderilemedi: {e}")
                # eğer hata oluşursa yukarıda hatanın açıklaması iletilir

            tespit_edilen_kutular = yeni_kutular
            # kutular, tespit edilen kutular'a atanır
        time.sleep(0.5)

# analiz için bir thread tanımlanır, ve çalıştırılması sağlanır
ai_thread = threading.Thread(target=analiz_motoru, daemon=True)
ai_thread.start()

# ==========================================================
# ANA KAMERA DONGUSU
# ==========================================================
# bilgisayarın kamerası açılır:
cap = cv2.VideoCapture(0)

while True:
    # anlık kareyi oku
    ret, frame = cap.read()
    if not ret: break
    
    # kamerayı aynalar, gerçek bir görüntü çıksın diye
    frame = cv2.flip(frame, 1)
    
    # Görüntüyü arka plandaki yapay zekaya gönder
    mevcut_kare = frame.copy() #analiz motorundaki fonksiyonun değişkenine atama yaptı ya...

    # Arka plandan gelen kutuları ekrana çiz, tespit edilen her kutu için
    for (x1, y1, x2, y2, skor) in tespit_edilen_kutular:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"YANGIN: %{int(skor*100)}", (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    # Bilgi Paneli
    # Ekranın sol üst köşesine siyah bir dikdörtgen üzerine yeşil yazıyla sistemin çalıştığını 
    # belirten bir bilgi paneli ekler.
    cv2.rectangle(frame, (0, 0), (250, 40), (0, 0, 0), -1)
    cv2.putText(frame, "DURUM: AKTIF (AI+HSV)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Yangin Tespiti v2.0", frame)
    
    # q'ya basılırsa
    if cv2.waitKey(1) & 0xFF == ord("q"):
        analiz_aktif = False # analizi durdur
        break # döngüden çıkılsın

cap.release() # kaynakları serbest bırak
cv2.destroyAllWindows() # tüm pencereleri kapat.
