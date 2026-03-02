# api.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
from typing import Optional, List, Dict, Any
import time
import json
import csv
import io
from collections import deque
import threading

# Senin oluşturduğun kafa sayacı dosyasını import ediyoruz
from kafasayan2 import KafaSayaci

# ==================== SİSTEM BAŞLATMA ====================
print("🚀 Kafa Sayacı Sistemi Başlatılıyor (Port: 5000)...")

# Video kaynağını belirle (RTSP, HLS veya video dosyası)
VIDEO_KAYNAK = 'ornek.mp4'  # veya RTSP: 'rtsp://...' veya HLS

# Sayaç sistemini başlat
sistem = KafaSayaci(video_dosyasi=VIDEO_KAYNAK, model_adi='medium.pt')
sistem.start()

# RAM'DE VERİ DEPOLAMA (veritabanı yok!)
son_anlik_veriler = deque(maxlen=1000)  # Son 1000 anlık ölçüm

# Arka planda veri toplama thread'i
def veri_topla():
    """Sürekli olarak sistemden veri al ve RAM'de depola"""
    while True:
        try:
            anlik = sistem.get_stats_instant()
            son_anlik_veriler.append({
                'zaman': datetime.now().isoformat(),
                'anlik_kisi': anlik['anlik_kisi'],
                'ortalama': anlik['ortalama']
            })
        except Exception as e:
            print(f"⚠️ Veri toplama hatası: {e}")
        time.sleep(1)

veri_thread = threading.Thread(target=veri_topla, daemon=True)
veri_thread.start()

# ==================== FASTAPI UYGULAMASI ====================
app = FastAPI(
    title="Kafa Sayacı API",
    description="""
    # 🚌 Akıllı Otobüs Yolcu Sayım Sistemi
    
    Bu API, otobüs içi kamera görüntülerini işleyerek **gerçek zamanlı yolcu sayımı** yapar.
    
    ## 📊 Özellikler
    * Canlı yolcu sayısı takibi
    * Anlık ve ortalama değerler
    * Geçmiş veri depolama (RAM'de)
    * JSON/CSV dışa aktarım
    
    ## 🎯 Kullanım
    - `/canli` ile anlık verileri alın
    - `/gecmis` ile geçmiş verileri görün
    - `/gecmis?limit=50` ile son 50 kaydı görün
    - `/rapor` ile istatistikleri inceleyin
    - `/export` ile verileri indirin
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ANA SAYFA ====================
@app.get("/", 
         response_class=JSONResponse,
         summary="Ana Sayfa",
         description="API ana sayfası ve endpoint listesi")
def root():
    """API ana sayfa"""
    return {
        "api_adi": "Kafa Sayacı API",
        "versiyon": "1.0.0",
        "kaynak": VIDEO_KAYNAK,
        "model": "medium.pt",
        "durum": "🟢 Aktif",
        "endpointler": {
            "/canli": "Anlık sayaç değerleri (GET)",
            "/video": "Canlı video akışı (GET)",
            "/gecmis": "Geçmiş veriler (GET) - ?limit parametresi ile kullan",
            "/gecmis/saatlik": "Saatlik özet (GET)",
            "/gecmis/saatlik/detayli": "Detaylı saatlik veri (GET)",
            "/rapor": "Genel rapor (GET)",
            "/export": "Verileri dışa aktar (GET)",
            "/durum": "Sistem durumu (GET)",
            "/test": "Test endpoint (GET)"
        },
        "dokuman": "/docs"
    }

# ==================== CANLI VERİ ENDPOINT'LERİ ====================
@app.get("/canli",
         response_model=Dict[str, Any],
         summary="Anlık Yolcu Sayısı",
         description="""
         **Gerçek zamanlı yolcu sayısı**
         
         Şu anki anlık yolcu sayısı ve hareketli ortalamayı döndürür.
         
         **Yanıt:**
         - `anlik_kisi`: O anda tespit edilen kişi sayısı
         - `ortalama`: Son 10 karenin hareketli ortalaması
         """)
def canli_veri():
    """Anlık sayaç değerleri"""
    try:
        return sistem.get_stats_instant()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sistem hatası: {e}")

@app.get("/video",
         summary="Canlı Video Akışı",
         description="""
         **MJPEG formatında canlı video akışı**
         
         Otobüs içi kameranın canlı görüntüsünü verir.
         Kafaların etrafında yeşil kutular ve ID'ler gösterilir.
         
         **Not:** Tarayıcıda açmak için direkt linki kullanın.
         """)
def video_akisi():
    """Canlı video akışı (MJPEG)"""
    def frame_generator():
        while True:
            frame_bytes = sistem.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.01)
    
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ==================== GEÇMİŞ VERİ ENDPOINT'LERİ ====================
@app.get("/gecmis",
         summary="Geçmiş Veriler",
         description="RAM'de depolanan anlık verileri getirir. 'limit' parametresi ile kaç kayıt istediğinizi belirtebilirsiniz.")
def gecmis_veriler(
    limit: int = Query(100, ge=1, le=1000, description="Getirilecek kayıt sayısı (1-1000)")
):
    """Son N veriyi getir (query parametresi ile)"""
    veriler = list(son_anlik_veriler)[-limit:]
    
    # İstatistik çıkar
    if veriler:
        ortalama = sum(v['anlik_kisi'] for v in veriler) / len(veriler)
        maks = max(v['anlik_kisi'] for v in veriler)
        min_deger = min(v['anlik_kisi'] for v in veriler)
    else:
        ortalama = maks = min_deger = 0
    
    return {
        "limit": limit,
        "mevcut_kayit": len(veriler),
        "istatistik": {
            "ortalama": round(ortalama, 2),
            "maksimum": maks,
            "minimum": min_deger
        },
        "veriler": veriler
    }

@app.get("/gecmis/saatlik",
         summary="Saatlik Özet",
         description="Son 1 saatin özet istatistikleri")
def saatlik_ozet():
    """Son 1 saatin özeti"""
    son_saat = list(son_anlik_veriler)[-360:]  # Son 360 saniye
    
    if not son_saat:
        return {"mesaj": "Yeterli veri yok (en az 1 saat bekleyin)"}
    
    degerler = [v['anlik_kisi'] for v in son_saat]
    
    return {
        "zaman_araligi": {
            "baslangic": son_saat[0]['zaman'],
            "bitis": son_saat[-1]['zaman']
        },
        "istatistik": {
            "ortalama": round(sum(degerler) / len(degerler), 2),
            "maksimum": max(degerler),
            "minimum": min(degerler),
            "standart_sapma": round((sum((x - sum(degerler)/len(degerler))**2 for x in degerler) / len(degerler))**0.5, 2)
        }
    }

@app.get("/gecmis/saatlik/detayli",
         summary="Detaylı Saatlik Veri",
         description="Son 1 saatin tüm verileri (her saniye)")
def saatlik_detayli():
    """Son 1 saatin tüm verileri"""
    son_saat = list(son_anlik_veriler)[-360:]
    
    return {
        "kayit_sayisi": len(son_saat),
        "veriler": son_saat
    }

# ==================== RAPOR ENDPOINT'LERİ ====================
@app.get("/rapor",
         summary="Genel Rapor",
         description="Sistemin genel durum raporu")
def genel_rapor():
    """Genel sistem raporu"""
    toplam_kayit = len(son_anlik_veriler)
    
    if toplam_kayit == 0:
        return {
            "mesaj": "Henüz veri yok",
            "calisma_suresi": f"{toplam_kayit} saniye"
        }
    
    degerler = [v['anlik_kisi'] for v in son_anlik_veriler]
    
    return {
        "calisma_suresi": f"{toplam_kayit} saniye",
        "toplam_veri_sayisi": toplam_kayit,
        "genel_istatistik": {
            "ortalama": round(sum(degerler) / len(degerler), 2),
            "maksimum": max(degerler),
            "minimum": min(degerler),
            "son_deger": degerler[-1] if degerler else 0
        },
        "model": "medium.pt",
        "kaynak": VIDEO_KAYNAK
    }

# ==================== VERİ AKTARIM ENDPOINT'LERİ ====================
@app.get("/export",
         summary="Verileri Dışa Aktar",
         description="""
         **JSON veya CSV formatında veri indir**
         
         Parametreler:
         - `format`: json veya csv
         - `limit`: indirilecek kayıt sayısı
         """)
def veri_aktar(
    format: str = Query("json", regex="^(json|csv)$", description="Dosya formatı"),
    limit: int = Query(100, ge=1, le=1000, description="Kayıt sayısı")
):
    """Verileri dışa aktar"""
    
    veriler = list(son_anlik_veriler)[-limit:]
    basliklar = ["zaman", "anlik_kisi", "ortalama"]
    
    if format == "json":
        return {
            "format": "json",
            "kayit_sayisi": len(veriler),
            "veriler": veriler
        }
    else:  # CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(basliklar)
        
        for v in veriler:
            writer.writerow([v['zaman'], v['anlik_kisi'], v['ortalama']])
        
        filename = f"kafa_sayaci_verileri_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

# ==================== SİSTEM DURUMU ====================
@app.get("/durum",
         summary="Sistem Durumu",
         description="Sistemin canlılık kontrolü")
def sistem_durumu():
    """Sistem durumunu göster"""
    return {
        "api_durum": "🟢 Aktif",
        "zaman": datetime.now().isoformat(),
        "kaynak": VIDEO_KAYNAK,
        "hafizada_kayit": len(son_anlik_veriler),
        "model": "medium.pt",
        "calisma_suresi": f"{len(son_anlik_veriler)} saniye"
    }

# ==================== CANLI TEST ENDPOINT'İ ====================
@app.get("/test",
         summary="Test Endpoint",
         description="API'nin çalıştığını test etmek için basit bir endpoint")
def test():
    """Test endpoint"""
    return {
        "durum": "başarılı",
        "zaman": datetime.now().isoformat(),
        "mesaj": "API çalışıyor!"
    }

# ==================== TERMİNALDEN ÇALIŞTIRMA ====================
# Bu dosyayı terminalden şu şekilde çalıştır:
# python -m uvicorn api:app --reload --host 0.0.0.0 --port 5000

