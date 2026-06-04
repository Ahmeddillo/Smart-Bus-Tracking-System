#kafasayan2.py
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import threading
from datetime import datetime
import argparse

class KafaSayaci:
    def __init__(self, video_dosyasi='ornek.mp4', model_adi='medium.pt'):
        print(f"🔄 Model yükleniyor: {model_adi}")
        self.model = YOLO(model_adi)
        print("✅ Model yüklendi!")
        
        self.video_dosyasi = video_dosyasi
        
        # --- Yayın Kontrolü ---
        self.canli_yayin = False
        # Eğer girdi sayıysa (Webcam 0) veya 'rtsp/http' içeriyorsa canlıdır
        if isinstance(video_dosyasi, int) or str(video_dosyasi).lower().startswith(('rtsp', 'http', 'https')):
            self.canli_yayin = True
            print(f"📡 Canlı Yayın Modu: {video_dosyasi}")
        else:
            print(f"📁 Dosya Modu: {video_dosyasi}")

        self.cap = cv2.VideoCapture(video_dosyasi)
        
        if not self.cap.isOpened():
            print(f"❌ Video kaynağı açılamadı: {video_dosyasi}")
            self.fps = 30
            self.width = 1280
            self.height = 720
        else:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            # RTSP bazen hatalı FPS döner, sabitleyelim
            if self.fps <= 0 or self.fps > 120 or np.isnan(self.fps): 
                self.fps = 30
            
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ekran/Yayın Boyutu
        self.display_width = 1280
        self.display_height = int(self.display_width * self.height / self.width) if self.width > 0 else 720
        
        # --- Değişkenler ---
        self.guncel_kare = None          
        self.output_frame = None         
        self.son_kutular = []            
        
        self.yolo_anlik_kisi = 0
        self.yolo_ortalama = 0.0
        self.son_sayilar = deque(maxlen=20)
        
        # --- İstatistik (RAM) ---
        self.gecmis_veriler = [] 
        
        self.program_calisiyor = True
        self.yeni_kare_var = False
        self.lock = threading.Lock()
        
        # Threadleri Başlat (Daemon=True: Ana program kapanınca bunlar da ölür)
        self.yolo_thread = threading.Thread(target=self.yolo_islem_dongusu)
        self.yolo_thread.daemon = True 
        self.yolo_thread.start()
        
        self.video_thread = threading.Thread(target=self.video_okuma_dongusu)
        self.video_thread.daemon = True
        
    def start(self):
        """API veya Main tarafından sistemi başlatır"""
        if not self.video_thread.is_alive():
            self.video_thread.start()

    def yolo_islem_dongusu(self):
        while self.program_calisiyor:
            islem_karesi = None
            with self.lock:
                if self.guncel_kare is not None and self.yeni_kare_var:
                    islem_karesi = self.guncel_kare.copy()
                    self.yeni_kare_var = False
            
            if islem_karesi is not None:
                try:
                    # HIZ İÇİN: imgsz=640, classes=[0] (Sadece insan)
                    results = self.model(islem_karesi, verbose=False, imgsz=640, classes=[0])
                    
                    temp_kutular = []
                    temp_kisi = 0
                    for box in results[0].boxes:
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        temp_kutular.append(coords)
                        temp_kisi += 1
                    
                    # İstatistik
                    self.yolo_anlik_kisi = temp_kisi
                    self.son_sayilar.append(temp_kisi)
                    if len(self.son_sayilar) > 0:
                        self.yolo_ortalama = sum(self.son_sayilar) / len(self.son_sayilar)
                    
                    # Geçmişe Kayıt (5 saniyede bir)
                    if len(self.gecmis_veriler) == 0 or (datetime.now() - self.gecmis_veriler[-1]['zaman']).total_seconds() > 5:
                        self.gecmis_veriler.append({
                            'zaman': datetime.now(),
                            'kisi': temp_kisi
                        })
                    
                    with self.lock:
                        self.son_kutular = temp_kutular
                except Exception as e:
                    print(f"YOLO Hatası: {e}")
            else:
                time.sleep(0.01)
    
    def video_okuma_dongusu(self):
        prev_time = time.time()
        scale_x = self.display_width / self.width
        scale_y = self.display_height / self.height
        
        while self.program_calisiyor:
            if not self.cap.isOpened():
                if self.canli_yayin:
                    print("⚠️ Bağlantı koptu, tekrar deneniyor...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.video_dosyasi)
                else:
                    time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                if self.canli_yayin:
                    print("⚠️ Yayın akışı durdu, yeniden bağlanılıyor...")
                    self.cap.release()
                    continue 
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Başa sar
                    continue
            
            # YOLO'ya veri gönder
            with self.lock:
                self.guncel_kare = frame.copy()
                self.yeni_kare_var = True
            
            # Görüntüyü Hazırla
            try:
                display_frame = cv2.resize(frame, (self.display_width, self.display_height))
                
                # Kutuları Çiz
                kutular = []
                with self.lock:
                    kutular = list(self.son_kutular)
                
                for box in kutular:
                    x1, y1, x2, y2 = box
                    sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
                    sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
                    cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
                
                # Bilgi Paneli
                cv2.rectangle(display_frame, (0, 0), (380, 85), (0, 0, 0), -1)
                cv2.putText(display_frame, f"Kisi: {self.yolo_anlik_kisi}", (20, 40), 0, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Ort: {self.yolo_ortalama:.1f}", (20, 75), 0, 0.7, (255, 255, 255), 1)
                
                # API Çıkışı
                with self.lock:
                    self.output_frame = display_frame

            except Exception as e:
                print(f"Görüntü hatası: {e}")
            
            # FPS Limiti (Sadece dosya ise bekle, RTSP ise bekleme)
            if not self.canli_yayin:
                time.sleep(1/self.fps if self.fps > 0 else 0.03)

    def get_frame(self):
        """API için JPEG döner"""
        with self.lock:
            if self.output_frame is None:
                return None
            ret, jpeg = cv2.imencode('.jpg', self.output_frame)
            if ret:
                return jpeg.tobytes()
            return None

    def get_stats_instant(self):
        return {
            "anlik_kisi": self.yolo_anlik_kisi,
            "ortalama": round(self.yolo_ortalama, 1),
            "zaman": datetime.now().strftime("%H:%M:%S")
        }

    def get_stats_history(self, sure_dakika=60):
        simdi = datetime.now()
        filtrelenmis = [v for v in self.gecmis_veriler if (simdi - v['zaman']).total_seconds() <= sure_dakika * 60]
        
        if not filtrelenmis:
            return {"mesaj": "Yeterli veri yok"}
            
        kisiler = [v['kisi'] for v in filtrelenmis]
        return {
            "toplam_kayit": len(kisiler),
            "en_yuksek_kisi": max(kisiler),
            "en_dusuk_kisi": min(kisiler),
            "ortalama_kisi": round(sum(kisiler) / len(kisiler), 1),
            "baslangic": filtrelenmis[0]['zaman'].strftime("%H:%M:%S"),
            "bitis": filtrelenmis[-1]['zaman'].strftime("%H:%M:%S")
        }

    def stop(self):
        self.program_calisiyor = False
        time.sleep(0.5)
        if self.cap.isOpened(): self.cap.release()

# -----------------------------------------------------------
# MAIN: TEK BAŞINA ÇALIŞTIRMA BLOĞU

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafa Sayacı Test Modu")
    parser.add_argument('--video', type=str, default='ornek.mp4', help="Video dosya yolu veya RTSP linki")
    parser.add_argument('--model', type=str, default='medium.pt', help="Model dosyası (yolov8n.pt, medium.pt vb.)")
    args = parser.parse_args()

    print("🚀 Sistem Başlatılıyor...")
    print("ℹ️ Çıkmak için 'q' tuşuna basın.")
    
    # Uygulamayı başlat
    app = KafaSayaci(video_dosyasi=args.video, model_adi=args.model)
    app.start()

    try:
        while True:
            # API simülasyonu: Sınıftan JPEG baytlarını alıp ekrana basıyoruz
            jpeg_bytes = app.get_frame()
            
            if jpeg_bytes is not None:
                # Byte verisini tekrar resme çevir (Decode)
                nparr = np.frombuffer(jpeg_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    cv2.imshow("Kafa Sayaci (Test Modu)", frame)
            
            # 'q' ile çıkış
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # CPU'yu rahatlat (RTSP modunda bile GUI için minik bekleme iyidir)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.")
    finally:
        print("Kapatılıyor...")
        app.stop()
        cv2.destroyAllWindows()
