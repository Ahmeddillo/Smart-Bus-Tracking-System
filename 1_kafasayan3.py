#kafasayan2.py
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import threading
from datetime import datetime
import argparse

# Bu fonksiyonda: medium modeli yüklendikten sonra, model olarak belirlenir, parametreyi atama işlemi ile
# Yapılandırıcı fonksiyonda: model belirleniyor, kaynak da paramtere olarak atanıyor
# rstp - http canlı video akışları için de bir yer tahsis ediliyor
# kaynak açılmazsa kaynağının açılmadığı ifade edilebiliyor
# ekran boyutları ayarlanır, threadların başlatılması sağlanır, aynı zamanda ayrı ayrı çalışması da sağlanır
class KafaSayaci:
    # Yapılandırıcı fonksiyon:
    def __init__(self, video_dosyasi='ornek.mp4', model_adi='medium.pt'): #burada model adını parametre olarak girilmiş
        print(f"🔄 Model yükleniyor: {model_adi}")
        self.model = YOLO(model_adi) # model: YOLO tabanlı model belirlenmiştir
        print("✅ Model yüklendi!")
        
        self.video_dosyasi = video_dosyasi # Kullanılcak Video: ornek.mp4 videosu
        
        # --- RTSP / CANLI YAYIN KONTROLÜ ---
        self.canli_yayin = False
        # Eğer girdi sayıysa (Webcam 0) veya 'rtsp/http' içeriyorsa canlıdır
        if isinstance(video_dosyasi, int) or str(video_dosyasi).lower().startswith(('rtsp', 'http', 'https')):
            self.canli_yayin = True
            print(f"📡 Canlı Yayın Modu: {video_dosyasi}")
        else:
            print(f"📁 Dosya Modu: {video_dosyasi}")

        self.cap = cv2.VideoCapture(video_dosyasi)
        
        # Eğer kaynak açılmamışsa, kaynağın açılmadığı ifade edilir:
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
        
        # --- DEĞİŞKENLER ---: initial parametreler sıfırlanır; kare, frame, son kutuların hepsi none
        self.guncel_kare = None          
        self.output_frame = None         
        self.son_kutular = []            
        
        self.yolo_anlik_kisi = 0
        self.yolo_ortalama = 0.0
        self.son_sayilar = deque(maxlen=20)
        
        # NOT: böylece gecmis_veriler listesi kaldırılmış olur, ama API tarafında tutulacak.
        
        self.program_calisiyor = True
        self.yeni_kare_var = False
        self.lock = threading.Lock()
        
        # Threadleri Başlat (Daemon=True: Ana program kapanınca bunlar da ölür)
        # Threadlerin kısaca işlevi: bir thread videoyu okurken, diğer thread video'yu YOLO modeline sokar
        self.yolo_thread = threading.Thread(target=self.yolo_islem_dongusu)
        self.yolo_thread.daemon = True 
        self.yolo_thread.start()
        
        self.video_thread = threading.Thread(target=self.video_okuma_dongusu)
        self.video_thread.daemon = True

# Burada başlatma fonksiyonu bulunmaktadır        
    def start(self):
        """API veya Main tarafından sistemi başlatır"""
        if not self.video_thread.is_alive():
            self.video_thread.start()

# Alttaki fonksiyonda, kutular boşaltılır, anlık kişi sayısı sıfırlanır, ve 0 etiketli her bir kutu bulunduğunda
# o kutu tüm kutulara dahil edilip anlık kişi sayısı 1 olarak arttırlır
# en alt satırlarda yine anlık ve ortalama kişi sayısı bulunur ve tespit edilir
    def yolo_islem_dongusu(self):
        while self.program_calisiyor:
            islem_karesi = None
            with self.lock:
                if self.guncel_kare is not None and self.yeni_kare_var:
                    islem_karesi = self.guncel_kare.copy()
                    self.yeni_kare_var = False
            
            if islem_karesi is not None: # işlem karesi boş değilse
                try:
                    # HIZ İÇİN: imgsz=640, classes=[0] (Sadece insan)
                    results = self.model(islem_karesi, verbose=False, imgsz=640, classes=[0])
                    
                    temp_kutular = []
                    temp_kisi = 0 # Önce anlık kişi sayısını 0 olarak setle
                    for box in results[0].boxes:
                        coords = box.xyxy[0].cpu().numpy().astype(int) # tespit edilen kutuyu dahil et
                        temp_kutular.append(coords)
                        temp_kisi += 1 # ve anlık kişi sayısı artsın
                    
                    # İstatistik (Thread güvenliği için kilitli alanda yazmak daha iyidir ama 
                    # deque ve int atamaları Python'da genelde atomiktir. Yine de okuma tarafında kilit şart.)
                    with self.lock:
                        self.yolo_anlik_kisi = temp_kisi
                        self.son_sayilar.append(temp_kisi)
                        if len(self.son_sayilar) > 0:
                            self.yolo_ortalama = sum(self.son_sayilar) / len(self.son_sayilar)
                        
                        self.son_kutular = temp_kutular

                except Exception as e:
                    print(f"YOLO Hatası: {e}")
            else:
                time.sleep(0.01)

# Alttaki fonksiyonda kareler okunur, eğer donma varsa donanım kaynakları serbest bırakılıp, tekrar çalıştırılır.
# lock işleminin yapılması ile birlikte, birbirinden farklı olan thread'ların işlem yapmasını engellemek için
# tespit edilen nesneler üzerinde kare çizilir, koordinatları ayarlanır
# en son'da da anlık kişi ve ortalama kişi sayısı için siyah bir köşe çizilir ve üzerinde anlık ve ortalama kişi sayısı oluşturulur.
    
    def video_okuma_dongusu(self):
        # Başlangıç zamanını ve görüntü ölçeklendirme katsayılarını hesapla
        prev_time = time.time()
        scale_x = self.display_width / self.width
        scale_y = self.display_height / self.height
        
        # Program çalıştığı sürece devam eden ana döngü
        while self.program_calisiyor:
            # Video kaynağı (kamera veya dosya) açık değilse kontrol et
            if not self.cap.isOpened():
                if self.canli_yayin:
                    # Canlı yayında bağlantı koptuysa serbest bırak ve 2 saniye sonra tekrar dene
                    print("⚠️ Bağlantı koptu, tekrar deneniyor...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.video_dosyasi)
                else:
                    # Video dosyasıysa bekle ve devam et
                    time.sleep(1)
                continue

            # Bir kare oku
            ret, frame = self.cap.read()
            # Kare okunamadıysa (video bitti veya yayın koptu)
            if not ret:
                if self.canli_yayin:
                    # Canlı yayında kare gelmiyorsa akışı serbest bırakıp döngüye devam et
                    print("⚠️ Yayın akışı durdu, yeniden bağlanılıyor...")
                    self.cap.release()
                    continue 
                else:
                    # Yerel videonun sonuna gelindiyse videoyu başa sar
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Başa sar
                    continue
            
            # --- Kritik Bölge (YOLO İşleme Hazırlığı) ---
            # Lock kullanarak kareyi güncelliyoruz ki başka bir thread aynı anda veriyi değiştirmesin
            with self.lock:
                self.guncel_kare = frame.copy() # Orijinal kareyi kopyala
                self.yeni_kare_var = True       # YOLO thread'ine yeni veri geldiğini haber ver
            
            # --- Görüntüyü Hazırla ve Ekrana Çiz ---
            try:
                # Görüntüyü belirlenen ekran boyutlarına yeniden boyutlandır
                display_frame = cv2.resize(frame, (self.display_width, self.display_height))
                
                # Çizilecek kutuları thread-safe (güvenli) bir şekilde al
                kutular = []
                with self.lock:
                    kutular = list(self.son_kutular) # Mevcut tespit edilen kutuları listeye aktar
                
                # Tespit edilen nesnelerin üzerine dikdörtgenleri çiz
                for box in kutular:
                    x1, y1, x2, y2 = box
                    # Koordinatları ekran boyutuna göre ölçeklendir
                    sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
                    sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
                    cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
                
                # --- Bilgi Paneli Çizimi ---
                # İstatistiksel verileri lock ile alıyoruz (çakışmaları önlemek için)
                with self.lock:
                    anlik_kisi = self.yolo_anlik_kisi
                    ortalama_kisi = self.yolo_ortalama

                # Siyah bir bilgi kutusu oluştur ve metinleri üzerine yaz
                cv2.rectangle(display_frame, (0, 0), (380, 85), (0, 0, 0), -1)
                cv2.putText(display_frame, f"Kisi: {anlik_kisi}", (20, 40), 0, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Ort: {ortalama_kisi:.1f}", (20, 75), 0, 0.7, (255, 255, 255), 1)
                
                # İşlenmiş ve üzerine çizim yapılmış son kareyi dışarıya (API'ye) hazır hale getir
                with self.lock:
                    self.output_frame = display_frame

            except Exception as e:
                print(f"Görüntü hatası: {e}")
            
            # --- Kaynak Yönetimi ve CPU Koruması ---
            if not self.canli_yayin:
                # Yerel videoda hedef FPS'ye göre bekleme yap (Video hızını korumak için)
                time.sleep(1/self.fps if self.fps > 0 else 0.03)
            else:
                # RTSP/Canlı yayında CPU'nun gereksiz yere %100 yüklenmesini engellemek için çok kısa mola
                time.sleep(0.001)

    def get_frame(self):
        """API üzerinden web arayüzüne veya başka bir yere JPEG formatında görüntü döner"""
        with self.lock:
            if self.output_frame is None:
                return None
            # OpenCV karesini JPEG formatına encode et
            ret, jpeg = cv2.imencode('.jpg', self.output_frame)
            if ret:
                return jpeg.tobytes() # Byte formatında gönder
            return None

    def get_stats_instant(self):
        """O anki tespit istatistiklerini JSON formatına uygun sözlük yapısında döner"""
        # Verilerin o an başka bir thread tarafından değiştirilmediğinden emin olmak için Lock
        with self.lock:
            anlik = self.yolo_anlik_kisi
            ortalama = self.yolo_ortalama

        return {
            "anlik_kisi": anlik,
            "ortalama": round(ortalama, 1),
            "zaman": datetime.now().strftime("%H:%M:%S")
        }

    def stop(self):
        """Döngüleri durdurur ve kamera bağlantısını güvenli bir şekilde kapatır"""
        self.program_calisiyor = False
        time.sleep(0.5) # Thread'lerin kapanması için kısa bir süre tanı
        if self.cap.isOpened(): 
            self.cap.release()

# ==========================================
# MAIN: TEK BAŞINA ÇALIŞTIRMA BLOĞU
# ==========================================
# Script doğrudan çalıştırıldığında (import edilmediğinde) devreye girer
# Aşağıda bulunan main kısmında argument kısmı bulunmaktadır, varsayılan değerleri atamak ve kullanıcının istersen farklı model veya kaynak
# kullanması için ona yol açar
# app değişken tanımlandıktan sonra 
# app başlatılır
# while döngüsünde görüntünün başarılı gelmesi halinde, byte dizi uint8 numpy dizisine dönüştürülür
# sonra da numpy dizisi opencv'nin anlayacağı formata dönüştürürüz
# Sonra da program kapatılır.
if __name__ == "__main__":
    # Komut satırı argümanlarını yönetmek için bir parser (ayrıştırıcı) oluşturur
    parser = argparse.ArgumentParser(description="Kafa Sayacı Test Modu")
    # Kullanıcının dışarıdan video yolu vermesini sağlar (Varsayılan: 'ornek.mp4')
    parser.add_argument('--video', type=str, default='ornek.mp4', help="Video dosya yolu veya RTSP linki")
    # Kullanıcının hangi YOLO modelini kullanacağını seçmesini sağlar (Varsayılan: 'medium.pt')
    parser.add_argument('--model', type=str, default='medium.pt', help="Model dosyası (yolov8n.pt, medium.pt vb.)")
    
    # Verilen argümanları parse eder ve 'args' değişkenine atar
    args = parser.parse_args()
    print("🚀 Sistem Başlatılıyor...")
    print("ℹ️ Çıkmak için 'q' tuşuna basın.")
    
    # Ana sınıfı (KafaSayaci), argümanlarla gelen video ve model bilgisiyle başlatır
    app = KafaSayaci(video_dosyasi=args.video, model_adi=args.model)
    # Arka plandaki iş parçacıklarını (thread'leri) başlatır
    app.start()

    try:
        # Ana döngü: Program durdurulana kadar çalışır
        while True:
            # Sınıf içindeki get_frame metodunu çağırarak güncel kareyi byte (JPEG) formatında alır
            # Bu kısım aslında bir web API'sinin yapacağı işlemi simüle eder
            jpeg_bytes = app.get_frame()
            
            # Eğer görüntü başarılı bir şekilde geldiyse
            if jpeg_bytes is not None:
                # Byte dizisini numpy dizisine (uint8) dönüştürür
                nparr = np.frombuffer(jpeg_bytes, np.uint8)
                # Numpy dizisindeki veriyi OpenCV'nin anlayacağı bir görüntü (matris) haline getirir
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Görüntü bozulmadan decode edildiyse ekranda göster
                if frame is not None:
                    cv2.imshow("Kafa Sayaci (Test Modu)", frame)
            
            # Klavyeden 'q' tuşuna basılıp basılmadığını kontrol eder (1ms bekler)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # CPU'yu gereksiz yormamak için çok kısa bir bekleme süresi
            time.sleep(0.01)

    except KeyboardInterrupt:
        # Kullanıcı terminalden Ctrl+C yaparsa yakalar
        print("\nKullanıcı tarafından durduruldu.")
    
    finally:
        # Program 'q' ile de bitse hata ile de bitse burası mutlaka çalışır
        print("Kapatılıyor...")
        # Kamera kaynağını serbest bırakır ve thread'leri durdurur
        app.stop()
        # Açık olan tüm OpenCV pencerelerini kapatır
        cv2.destroyAllWindows()
