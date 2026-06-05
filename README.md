# 🚌 Akıllı Otobüs Takip Sistemi

<img width="1244" height="628" alt="Ekran görüntüsü 2026-06-04 210501" src="https://github.com/user-attachments/assets/849d4cf8-11a5-4838-bbf4-f67438c41fa6" />

Otobüslerde yapay zeka destekli güvenlik ve izleme sistemi. Yolcu sayımı, sürücü yorgunluk tespiti, yangın algılama ve çukur tespiti modüllerinden oluşan entegre bir platform.

---

## 📋 İçindekiler

- [Sistem Özeti](#sistem-özeti)
- [Özellikler](#özellikler)
- [Proje Yapısı](#proje-yapısı)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [API Dokümantasyonu](#api-dokümantasyonu)
- [Veritabanı Şeması](#veritabanı-şeması)

---

## Sistem Özeti

Bu sistem, şehir içi toplu taşıma araçlarında gerçek zamanlı güvenlik ve operasyon izlemesi için tasarlanmıştır. Kamera görüntülerini yapay zeka modelleri ile işleyerek; yolcu sayısını, sürücü durumunu, olası yangın tehlikelerini ve yol koşullarını (çukur) merkezi bir API üzerinden yöneticilere sunar.

---

## Özellikler

### 👤 Yolcu Sayımı (`kafasayan.py` / `1_kafasayan3.py`)
- YOLOv8 tabanlı gerçek zamanlı insan tespiti
- Anlık ve hareketli ortalama kişi sayısı
- Video dosyası, webcam ve RTSP/HTTP canlı yayın desteği
- Bağlantı koptuğunda otomatik yeniden bağlanma
- Thread tabanlı paralel işleme (okuma ve analiz ayrı thread'lerde)
- JPEG formatında canlı video akışı (API üzerinden)

### 😴 Sürücü Yorgunluk Tespiti (`2_yorgunluk_tespiti_parlakortam2.py`)
- Dlib 68 yüz landmark modeli ile gerçek zamanlı analiz
- **EAR** (Eye Aspect Ratio) — göz kapanma tespiti
- **MAR** (Mouth Aspect Ratio) — esneme tespiti
- Baş eğimi ve diklik kontrolü (burun-çene mesafesi)
- Ters ışık ve parlak ortam için CLAHE görüntü iyileştirme
- Kişiye özel otomatik kalibrasyon (ilk 60 kare)
- Sesli alarm (`winsound.Beep`)
- Uyarı seviyeleri: `normal` / `dikkat` / `kritik`
- PostgreSQL'e asenkron log kaydı (arka plan thread)

### 🔥 Yangın Tespiti (`3_yangin_tespiti.py`)
- OpenAI CLIP modeli (`clip-vit-base-patch32`) ile sıfır-shot görüntü sınıflandırma
- Grid tabanlı bölgesel analiz (3×3 ızgara)
- HSV renk filtresi (kırmızı/turuncu/sarı tonları) ile çift doğrulama
- Yüksek hassasiyet için katmanlı eşik sistemi (genel skor + bölgesel skor + renk filtresi)
- Tespit koordinatları ile ekranda bounding box gösterimi
- API'ye gerçek zamanlı veri gönderimi

### 🛣️ Çukur Tespiti (API Entegrasyonu)
- Kamera tabanlı çukur algılama modülü
- GPS koordinatı ile birlikte veritabanına kayıt
- Harita görselleştirmesi için coğrafi veri endpoint'i

### 🌐 Merkezi API (`api.py`)
- FastAPI tabanlı RESTful API
- Çoklu otobüs kaynağı yönetimi (RTSP, dosya, webcam)
- Anlık ve geçmiş veri endpoint'leri
- CSV ve JSON dışa aktarım
- Canlı video akışı (MJPEG)
- PostgreSQL + SQLite yedek desteği
- CORS middleware (web/mobil arayüz uyumluluğu)

---

## Proje Yapısı

```
Akilli-Otobus-Takip-Sistemi/
│
├── api.py                              # Ana FastAPI uygulaması (merkezi hub)
├── database.py                         # Veritabanı bağlantı ve model tanımları
│
├── kafasayan.py                        # Yolcu sayacı (v1)
├── kafasayan2.py                       # Yolcu sayacı (v2 - API'ye entegre)
├── 1_kafasayan3.py                     # Yolcu sayacı (v3 - bağımsız test modu)
│
├── 2_yorgunluk_tespiti_parlakortam2.py # Sürücü yorgunluk tespiti
├── 5_yorgunluk_tespiti_parlak_çYedek.py# Yedek sürücü modülü
│
├── 3_yangin_tespiti.py                 # Yangın tespit modülü
│
├── Arayuz/                             # Web arayüzü (HTML/CSS)
│
├── Dipnotes.txt                        # Geliştirici notları
└── kaynaklar (Yüklenmeyen Dosyalar)   # Model dosyaları referansı
```

---

## Kullanılan Teknolojiler

| Katman | Teknoloji |
|--------|-----------|
| Dil | Python 3.x |
| Görüntü İşleme | OpenCV, NumPy |
| Nesne Tespiti | YOLOv8 (Ultralytics) |
| Yüz Landmark | Dlib + imutils |
| Görüntü Sınıflandırma | Hugging Face Transformers (CLIP) |
| Web API | FastAPI + Uvicorn |
| Veritabanı | PostgreSQL (yedek: SQLite) |
| ORM | SQLAlchemy |
| Ortam Değişkenleri | python-dotenv |
| Web Arayüzü | HTML / C# |

---

## Kurulum

### Gereksinimler

```bash
pip install ultralytics opencv-python dlib imutils numpy
pip install fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv
pip install transformers torch pillow requests
```

> **Not:** Dlib kurulumu için CMake ve Visual Studio Build Tools gerekebilir. `dlib` paketini kurmadan önce sisteminizde bu araçların mevcut olduğundan emin olun.

### Model Dosyaları

Sürücü yorgunluk tespiti için Dlib'in 68 nokta yüz landmark modelini indirip proje kök dizinine kopyalayın:

```
shape_predictor_68_face_landmarks.dat
```

Yolcu sayımı için YOLO model dosyasını (`medium.pt` veya `yolov8n.pt`) proje kök dizinine yerleştirin.

### Veritabanı Yapılandırması

Proje kök dizininde bir `.env` dosyası oluşturun:

```env
DATABASE_URL=postgresql://kullanici:sifre@localhost/otobus_db
```

PostgreSQL'de `otobus_db` veritabanını oluşturun. Tablolar sistem ilk çalıştığında otomatik oluşturulur.

---

## Çalıştırma

### 1. API Sunucusunu Başlatın

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 5000 --reload
```

API başladığında yolcu sayım sistemi de otomatik olarak devreye girer.

### 2. Sürücü Yorgunluk Tespitini Başlatın

```bash
python 2_yorgunluk_tespiti_parlakortam2.py
```

### 3. Yangın Tespitini Başlatın

```bash
python 3_yangin_tespiti.py
```

### 4. Yolcu Sayacını Bağımsız Test Etmek İçin

```bash
# Varsayılan video dosyasıyla
python 1_kafasayan3.py

# Özel kaynak ve modelle
python 1_kafasayan3.py --video rtsp://192.168.1.100/stream --model yolov8n.pt
```

---

## API Dokümantasyonu

API çalışırken interaktif dokümantasyona şu adresten erişilebilir:

```
http://localhost:5000/docs
```

### Temel Endpoint'ler

#### Genel

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/` | API bilgileri ve mevcut endpoint listesi |
| GET | `/durum` | Sistem ve veritabanı bağlantı durumu |
| GET | `/test` | Basit sağlık kontrolü |

#### Yolcu Sayımı

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/canli` | Anlık yolcu sayısı ve ortalama |
| GET | `/video` | MJPEG canlı video akışı |
| GET | `/gecmis` | RAM'deki geçmiş veriler (max 1000 kayıt) |
| GET | `/gecmis/saatlik` | Son 1 saatlik özet istatistik |
| GET | `/gecmis/saatlik/detayli` | Son 1 saatin tüm kayıtları |
| GET | `/export?format=csv` | Verileri CSV olarak indir |
| GET | `/db/gecmis` | PostgreSQL'den geçmiş yolcu verileri |
| GET | `/db/oturumlar` | Tüm çalışma oturumları |
| GET | `/db/rapor/{oturum_id}` | Oturum bazlı özet rapor |

#### Kaynak Yönetimi

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/kaynaklar` | Mevcut otobüs kamerası kaynakları |
| POST | `/kaynaklar/{id}/sec` | Aktif kamera kaynağını değiştir |

#### Sürücü İzleme

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/surucu/guncelle` | Kamera modülünden veri al |
| GET | `/surucu/durum` | Anlık sürücü durumu |
| GET | `/surucu/gecmis` | Geçmiş uyarı kayıtları |
| GET | `/surucu/istatistik` | Günlük uyarı özeti |

#### Yangın İzleme

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/yangin/guncelle` | Yangın modülünden veri al |
| GET | `/yangin/durum` | Anlık yangın durumu |
| GET | `/yangin/gecmis` | Geçmiş yangın kayıtları |

#### Çukur İzleme

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/cukur/guncelle` | Çukur kamerasından veri al |
| GET | `/cukur/durum` | Anlık çukur durumu |
| GET | `/cukur/gecmis` | Geçmiş çukur kayıtları |
| GET | `/cukur/harita` | Harita için GPS koordinatları |

---

## Veritabanı Şeması

```
oturumlar           → Sistem çalışma oturumları
yolcu_sayimlari     → Zaman serisi yolcu sayım kayıtları
oturum_raporlari    → Oturum bazlı özet raporlar
surucu_loglari      → Sürücü uyarı ve durum logları
yangin_loglari      → Yangın tespit kayıtları (kamera ID, skor, alarm)
cukur_loglari       → Çukur tespit kayıtları (GPS koordinatlı)
```

---

## Mimariye Genel Bakış

```
┌─────────────────────────────────────────────────────┐
│                     api.py (FastAPI)                 │
│  /canli  /video  /surucu  /yangin  /cukur  /kaynaklar│
└────────────┬──────────┬──────────┬──────────────────┘
             │          │          │
    ┌────────┘   ┌──────┘   ┌──────┘
    ▼            ▼          ▼
kafasayan2.py   surucu     yangin
(YOLO Thread)  modülü     modülü
    │          (Dlib)     (CLIP)
    ▼            │          │
PostgreSQL ◄─────┴──────────┘
```

Tüm modüller API'ye bağımsız POST isteği göndererek veri iletir. API, gelen verileri hem RAM'de (anlık erişim için) hem de PostgreSQL'de (kalıcı kayıt için) saklar.

---

## Lisans

Bu proje eğitim amaçlı geliştirilmiştir.
