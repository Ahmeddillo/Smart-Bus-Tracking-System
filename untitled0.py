import cv2
import numpy as np
from ultralytics import YOLO
import time

class SurucuAsistani:
    def __init__(self):
        print("🔄 Modeller yükleniyor...")
        self.yuz_model = YOLO('yolov8n-face.pt') 
        
        # --- Eşik Değerleri ---
        self.GOZ_KAPALI_SURE = 1.5  
        self.ESNE_SURE = 2.0        
        self.BAS_ACI_LIMIT = 25     
        
        # --- Sayaçlar ---
        self.goz_baslangic = 0
        self.esne_baslangic = 0
        self.uyari_metni = "TAKIP AKTIF"
        self.uyari_rengi = (0, 255, 0)

    def analiz_et(self, frame):
        if frame is None: return frame
        simdi = time.time()
        
        # 1. Tespit (conf değeri 0.25'e düşürüldü, imgsz kaldırıldı)
        results = self.yuz_model.predict(frame, verbose=False, conf=0.25)
        
        # Eğer kutu (box) yoksa veya results boşsa
        if not results or len(results[0].boxes) == 0:
            self.uyari_metni = "YUZ ARANIYOR..."
            self.uyari_rengi = (0, 165, 255)
            return frame

        # En büyük yüzü al (Sürücü)
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yuz_w = x2 - x1 

        # Landmarkları kontrol et
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            lmks = results[0].keypoints.xy[0].cpu().numpy()
            
            # 5 nokta gelmeli: 0:Sol Göz, 1:Sağ Göz, 2:Burun, 3:Sol Ağız, 4:Sağ Ağız
            if len(lmks) >= 5:
                sol_g, sag_g = lmks[0], lmks[1]
                sol_a, sag_a = lmks[3], lmks[4]

                # --- A. GÖZ ANALİZİ ---
                goz_mesafesi = np.linalg.norm(sol_g - sag_g)
                # Hassasiyeti biraz artırdık (0.14 -> 0.16)
                if goz_mesafesi < (yuz_w * 0.16):
                    if self.goz_baslangic == 0: self.goz_baslangic = simdi
                else:
                    self.goz_baslangic = 0

                # --- B. AĞIZ ANALİZİ (ESNEME) ---
                agiz_mesafesi = np.linalg.norm(sol_a - sag_a)
                if agiz_mesafesi > (yuz_w * 0.45):
                    if self.esne_baslangic == 0: self.esne_baslangic = simdi
                else:
                    self.esne_baslangic = 0

                # --- C. BAŞ AÇISI ---
                dy = sag_g[1] - sol_g[1]
                dx = sag_g[0] - sol_g[0]
                aci = abs(np.degrees(np.arctan2(dy, dx)))

                # --- D. KARAR MEKANİZMASI ---
                self.uyari_rengi = (0, 255, 0)
                self.uyari_metni = "SURUCU AKTIF"

                if self.goz_baslangic != 0 and (simdi - self.goz_baslangic) > self.GOZ_KAPALI_SURE:
                    self.uyari_metni = "TEHLIKE: UYUKLAMA!"
                    self.uyari_rengi = (0, 0, 255)
                elif aci > self.BAS_ACI_LIMIT:
                    self.uyari_metni = "KRITIK: BAS EGILDI!"
                    self.uyari_rengi = (0, 0, 255)
                elif self.esne_baslangic != 0 and (simdi - self.esne_baslangic) > self.ESNE_SURE:
                    self.uyari_metni = "UYARI: ESNEME TESPITI"
                    self.uyari_rengi = (0, 255, 255)

                # Görselleştirme (Yeşil Kutu ve Landmarklar)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.uyari_rengi, 2)
                for p in lmks:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 4, (255, 0, 0), -1)
        
        return frame

def main():
    # cv2.CAP_DSHOW bazı kameralarda sorun çıkarabilir, gerekirse kaldırabilirsin
    cap = cv2.VideoCapture(0)
    asistan = SurucuAsistani()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Aynalama
        frame = asistan.analiz_et(frame)

        # Alt panel
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h-50), (w, h), (10, 10, 10), -1)
        cv2.putText(frame, asistan.uyari_metni, (20, h-15), 1, 1.8, asistan.uyari_rengi, 2)
        
        cv2.imshow('Surucu Yorgunluk Analizi', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()