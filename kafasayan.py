# kafasayan.py
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import argparse
import threading

class KafaSayaci:
    def __init__(self, video_dosyasi='ornek.mp4', model_adi='medium.pt'):
        print(f"🔄 Kafa tespit modeli yükleniyor: {model_adi}")
        self.model = YOLO(model_adi)
        print("✅ Model yüklendi!")
        
        self.video_dosyasi = video_dosyasi
        self.cap = cv2.VideoCapture(video_dosyasi)
        
        if not self.cap.isOpened():
            print(f"❌ Video açılamadı: {video_dosyasi}")
            exit()
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Videonun orijinal hızı için ms cinsinden bekleme süresi
        self.bekleme_suresi = int(1000 / self.fps) if self.fps > 0 else 33 
        self.toplam_kare = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_uzunluk = self.toplam_kare / self.fps if self.fps > 0 else 0
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.son_sayilar = deque(maxlen=10)
        self.toplam_kisi = 0
        self.kare_sayaci = 0
        
        self.display_width = min(self.width, 1280)
        self.display_height = int(self.display_width * self.height / self.width)
        
        # --- Threading Değişkenleri ---
        self.guncel_kare = None         
        self.yolo_sonuc_karesi = None   
        self.yolo_anlik_kisi = 0
        self.yolo_ortalama = 0.0
        self.program_calisiyor = True
        self.yeni_kare_var = False # YOLO'nun yeni kare alması için sinyal
        
        self.yolo_thread = threading.Thread(target=self.yolo_islem_dongusu)
        self.yolo_thread.daemon = True 
        self.yolo_thread.start()

        print(f"📹 Video: {video_dosyasi} | FPS: {self.fps}")
        print("=" * 60)
    
    def yolo_islem_dongusu(self):
        """Arka planda bağımsız çalışan YOLO işçisi"""
        while self.program_calisiyor:
            if self.guncel_kare is not None and self.yeni_kare_var:
                self.yeni_kare_var = False
                islem_karesi = self.guncel_kare.copy()
                
                # YOLO tespiti
                results = self.model(islem_karesi, verbose=False)
                
                anlik = len(results[0].boxes)
                self.yolo_anlik_kisi = anlik
                self.son_sayilar.append(anlik)
                self.toplam_kisi += anlik
                self.yolo_ortalama = sum(self.son_sayilar) / len(self.son_sayilar)
                
                # Sadece kutuların çizildiği kareyi hazırla
                self.yolo_sonuc_karesi = results[0].plot()
            else:
                time.sleep(0.001) # CPU'yu yormamak için çok kısa bekleme
    
    def run(self):
        """Ana döngü - Sadece Video Oynatır"""
        cv2.namedWindow('Kafa Sayaci', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Kafa Sayaci', self.display_width, self.display_height)
        
        paused = False
        prev_time = time.time()
        
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    self.kare_sayaci += 1
                    self.guncel_kare = frame # YOLO'ya son kareyi sun
                    self.yeni_kare_var = True # YOLO'ya "işleyebilirsin" sinyali ver
                    
                    # Görüntüleme: YOLO sonucu varsa onu bas, yoksa ham kareyi bas
                    if self.yolo_sonuc_karesi is not None:
                        gosterilecek = self.yolo_sonuc_karesi.copy()
                    else:
                        gosterilecek = frame.copy()
                    
                    # Bilgi Yazıları
                    self._yazilari_ekle(gosterilecek, prev_time)
                    prev_time = time.time()
                    
                    cv2.imshow('Kafa Sayaci', gosterilecek)
                
                # BURASI KRİTİK: waitKey(self.bekleme_suresi) videoyu gerçek hızına sabitler
                key = cv2.waitKey(self.bekleme_suresi) & 0xFF
                if key == ord('q'): break
                elif key == ord(' '): paused = not paused
                    
        finally:
            self.cleanup()

    def _yazilari_ekle(self, img, p_time):
        """Ekrana istatistikleri yazar"""
        fps = 1 / (time.time() - p_time) if (time.time() - p_time) > 0 else 0
        cv2.putText(img, f"KAFA: {self.yolo_anlik_kisi}", (20, 50), 2, 1.2, (0, 255, 0), 2)
        cv2.putText(img, f"ORT: {self.yolo_ortalama:.1f}", (20, 90), 2, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"FPS: {fps:.1f}", (20, 130), 2, 0.5, (200, 200, 200), 1)

    def cleanup(self):
        self.program_calisiyor = False
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='ornek.mp4')
    parser.add_argument('--model', type=str, default='medium.pt')
    args = parser.parse_args()
    KafaSayaci(args.video, args.model).run()

if __name__ == "__main__":
    main()
