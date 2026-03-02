# api.py
# Terminalden Çalıştırma Komutu:
# python -m uvicorn api:app --host 0.0.0.0 --port 5000 --reload

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
import os

# --- VERİTABANI İMPORTLARI ---
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, text
from sqlalchemy.orm import declarative_base, Session
from dotenv import load_dotenv

# Senin oluşturduğun kafa sayacı dosyasını import ediyoruz
from kafasayan2 import KafaSayaci

# .env dosyasını yükle
load_dotenv()

# ==================== VERİTABANI AYARLARI ====================
DATABASE_URL = os.getenv("DATABASE_URL")

# Eğer .env yoksa veya boşsa otomatik olarak kodun içine yazdığın adresi al
if not DATABASE_URL:
    # BURAYA KENDİ DB BİLGİLERİNİ YAZIYORSUN
    DATABASE_URL = "postgresql://postgres:Hello@localhost/otobus_db"

print(f"🔗 Bağlanılacak Adres: {DATABASE_URL}")

try:
    # SQLite ve PostgreSQL ayrımı
    if "sqlite" in DATABASE_URL:
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    
    Base = declarative_base()
    print("✅ Veritabanı motoru başarıyla oluşturuldu.")
except Exception as e:
    print(f"❌ Veritabanı Bağlantı Hatası: {e}")
    # Hata durumunda geçici çözüm (kod çökmesin diye)
    engine = create_engine("sqlite:///./gecici_yedek.db", connect_args={"check_same_thread": False})
    Base = declarative_base()

# --- TABLO MODELLERİ ---
class Oturum(Base):
    __tablename__ = "oturumlar"
    id = Column(Integer, primary_key=True)
    baslangic = Column(DateTime, default=datetime.now)
    bitis = Column(DateTime, nullable=True)
    video_kaynak = Column(String)
    model_adi = Column(String)

class YolcuSayimi(Base):
    __tablename__ = "yolcu_sayimlari"
    id = Column(Integer, primary_key=True)
    oturum_id = Column(Integer)
    zaman = Column(DateTime, default=datetime.now)
    anlik_kisi = Column(Integer)
    ortalama = Column(Float)

class OturumRaporu(Base):
    __tablename__ = "oturum_raporlari"
    id = Column(Integer, primary_key=True)
    oturum_id = Column(Integer)
    toplam_kayit = Column(Integer)
    ortalama_kisi = Column(Float)
    maksimum_kisi = Column(Integer)
    minimum_kisi = Column(Integer)
    olusturulma = Column(DateTime, default=datetime.now)

# Tabloları oluştur (yoksa)
try:
    Base.metadata.create_all(engine)
    print("✅ Tablolar kontrol edildi/oluşturuldu.")
except Exception as e:
    print(f"❌ Tablo oluşturma hatası: {e}")

# ==================== SİSTEM BAŞLATMA ====================
print("🚀 Kafa Sayacı Sistemi Başlatılıyor (Port: 5000)...")

MODEL_ADI = 'medium.pt'

# --- YENİ EKLENEN KAYNAKLAR YAPISI ---
KAYNAKLAR = {
    1: {"ad": "Otobüs 1", "kaynak": "ornek.mp4"},
    2: {"ad": "Otobüs 2", "kaynak": "rtsp://192.168.1.101/stream"},
    3: {"ad": "Otobüs 3", "kaynak": "rtsp://192.168.1.102/stream"},
}
AKTIF_KAYNAK_ID = 1  # Başlangıçta 1. kaynak aktif
VIDEO_KAYNAK = KAYNAKLAR[AKTIF_KAYNAK_ID]["kaynak"]
# --------------------------------------

# Sayaç sistemini başlat
sistem = KafaSayaci(video_dosyasi=VIDEO_KAYNAK, model_adi=MODEL_ADI)
sistem.start()

# --- OTURUM AÇMA İŞLEMİ ---
AKTIF_OTURUM_ID = None
try:
    with Session(engine) as session:
        aktif_oturum = Oturum(
            video_kaynak=str(VIDEO_KAYNAK),
            model_adi=MODEL_ADI
        )
        session.add(aktif_oturum)
        session.commit()
        session.refresh(aktif_oturum)
        AKTIF_OTURUM_ID = aktif_oturum.id
    print(f"✅ DB Oturumu açıldı: ID={AKTIF_OTURUM_ID}")
except Exception as e:
    print(f"⚠️ Oturum açılamadı (DB hatası olabilir): {e}")

# RAM'DE VERİ DEPOLAMA
son_anlik_veriler = deque(maxlen=1000) 

# --- VERİ TOPLAMA THREAD ---
def veri_topla():
    kayit_sayaci = 0
    while True:
        try:
            if not sistem.program_calisiyor:
                break

            anlik = sistem.get_stats_instant()

            # 1. RAM'e yaz
            son_anlik_veriler.append({
                'zaman': datetime.now().isoformat(),
                'anlik_kisi': anlik['anlik_kisi'],
                'ortalama': anlik['ortalama']
            })

            # 2. DB'ye yaz (Her 5 saniyede bir)
            if AKTIF_OTURUM_ID is not None:
                kayit_sayaci += 1
                if kayit_sayaci >= 5:
                    with Session(engine) as session:
                        kayit = YolcuSayimi(
                            oturum_id=AKTIF_OTURUM_ID,
                            anlik_kisi=anlik['anlik_kisi'],
                            ortalama=anlik['ortalama']
                        )
                        session.add(kayit)
                        session.commit()
                    kayit_sayaci = 0

        except Exception as e:
            print(f"⚠️ Veri toplama hatası: {e}")
        
        time.sleep(1)

veri_thread = threading.Thread(target=veri_topla, daemon=True)
veri_thread.start()

# ==================== FASTAPI UYGULAMASI ====================
app = FastAPI(
    title="Kafa Sayacı API (DB Destekli)",
    description="Otobüs yolcu sayım sistemi. Hem RAM hem PostgreSQL kullanır.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MEVCUT ENDPOINTLER ====================
@app.get("/", 
         response_class=JSONResponse,
         summary="Ana Sayfa")
def root():
    return {
        "api_adi": "Kafa Sayacı API",
        "versiyon": "2.1.0",
        "durum": "🟢 Aktif",
        "oturum_id": AKTIF_OTURUM_ID,
        "endpointler": ["/canli", "/gecmis", "/gecmis/saatlik", "/db/gecmis", "/db/oturumlar", "/kaynaklar"]
    }

# --- YENİ EKLENEN KAYNAK YÖNETİMİ ENDPOINT'LERİ ---
@app.get("/kaynaklar", summary="Mevcut Kaynaklar")
def kaynaklar_listesi():
    return {
        "aktif_id": AKTIF_KAYNAK_ID,
        "kaynaklar": [
            {"id": k, "ad": v["ad"], "kaynak": v["kaynak"]}
            for k, v in KAYNAKLAR.items()
        ]
    }

@app.post("/kaynaklar/{kaynak_id}/sec", summary="Kaynak Seç")
def kaynak_sec(kaynak_id: int):
    global sistem, AKTIF_KAYNAK_ID, AKTIF_OTURUM_ID
    if kaynak_id not in KAYNAKLAR:
        raise HTTPException(status_code=404, detail="Kaynak bulunamadı")
    
    # Mevcut sistemi durdur
    sistem.stop()
    
    # Yeni sistemi başlat
    yeni_kaynak = KAYNAKLAR[kaynak_id]["kaynak"]
    sistem = KafaSayaci(video_dosyasi=yeni_kaynak, model_adi=MODEL_ADI)
    sistem.start()
    AKTIF_KAYNAK_ID = kaynak_id
    
    # Yeni oturum aç
    try:
        with Session(engine) as session:
            yeni_oturum = Oturum(
                video_kaynak=str(yeni_kaynak),
                model_adi=MODEL_ADI
            )
            session.add(yeni_oturum)
            session.commit()
            session.refresh(yeni_oturum)
            AKTIF_OTURUM_ID = yeni_oturum.id
    except Exception as e:
        print(f"⚠️ Oturum açılamadı: {e}")
        
    return {
        "mesaj": f"{KAYNAKLAR[kaynak_id]['ad']} seçildi",
        "aktif_id": AKTIF_KAYNAK_ID,
        "yeni_oturum_id": AKTIF_OTURUM_ID
    }
# --------------------------------------------------

@app.get("/canli", summary="Anlık Yolcu Sayısı")
def canli_veri():
    try:
        return sistem.get_stats_instant()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sistem hatası: {e}")

@app.get("/video", summary="Canlı Video Akışı")
def video_akisi():
    def frame_generator():
        while True:
            frame_bytes = sistem.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.01)
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/gecmis", summary="RAM Geçmiş Veriler")
def gecmis_veriler(limit: int = Query(100, ge=1, le=1000)):
    veriler = list(son_anlik_veriler)[-limit:]
    if veriler:
        ortalama = sum(v['anlik_kisi'] for v in veriler) / len(veriler)
        maks = max(v['anlik_kisi'] for v in veriler)
        min_deger = min(v['anlik_kisi'] for v in veriler)
    else:
        ortalama = maks = min_deger = 0
    return {
        "kaynak": "RAM",
        "limit": limit,
        "istatistik": {"ortalama": round(ortalama, 2), "maksimum": maks, "minimum": min_deger},
        "veriler": veriler
    }

@app.get("/gecmis/saatlik",
         summary="Saatlik Özet (RAM)",
         description="RAM'deki son verilerin (maksimum son 360 kayıt) özeti")
def saatlik_ozet():
    """Son 1 saatin (veya RAM'deki son kısmın) özeti"""
    son_saat = list(son_anlik_veriler)[-360:]  # Son 360 kayıt (yaklaşık)
    
    if not son_saat:
        return {"mesaj": "Yeterli veri yok (en az 1 saat bekleyin)"}
    
    degerler = [v['anlik_kisi'] for v in son_saat]
    
    # Standart sapma hesabı
    ortalama = sum(degerler) / len(degerler)
    varyans = sum((x - ortalama) ** 2 for x in degerler) / len(degerler)
    std_sapma = varyans ** 0.5
    
    return {
        "zaman_araligi": {
            "baslangic": son_saat[0]['zaman'],
            "bitis": son_saat[-1]['zaman']
        },
        "istatistik": {
            "ortalama": round(ortalama, 2),
            "maksimum": max(degerler),
            "minimum": min(degerler),
            "standart_sapma": round(std_sapma, 2)
        }
    }

@app.get("/gecmis/saatlik/detayli",
         summary="Detaylı Saatlik Veri (RAM)",
         description="Son 1 saatin tüm detaylı verileri")
def saatlik_detayli():
    """Son 1 saatin tüm verileri"""
    son_saat = list(son_anlik_veriler)[-360:]
    
    return {
        "kayit_sayisi": len(son_saat),
        "veriler": son_saat
    }

@app.get("/export", summary="Verileri Dışa Aktar")
def veri_aktar(
    format: str = Query("json", regex="^(json|csv)$"),
    limit: int = Query(100, ge=1, le=1000)
):
    veriler = list(son_anlik_veriler)[-limit:]
    if format == "json":
        return {"format": "json", "veriler": veriler}
    else:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["zaman", "anlik_kisi", "ortalama"])
        for v in veriler:
            writer.writerow([v['zaman'], v['anlik_kisi'], v['ortalama']])
        filename = f"kafa_sayaci_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

# ==================== DB ENDPOINTLERİ ====================

@app.get("/db/gecmis",
         summary="DB - Geçmiş Veriler",
         description="PostgreSQL'den geçmiş yolcu verilerini getirir")
def db_gecmis(limit: int = Query(100, ge=1, le=1000)):
    if not AKTIF_OTURUM_ID:
        return {"mesaj": "Veritabanı bağlantısı yok", "veriler": []}
        
    with Session(engine) as session:
        kayitlar = session.query(YolcuSayimi)\
            .filter(YolcuSayimi.oturum_id == AKTIF_OTURUM_ID)\
            .order_by(YolcuSayimi.zaman.desc())\
            .limit(limit)\
            .all()
    return {
        "kaynak": "PostgreSQL",
        "oturum_id": AKTIF_OTURUM_ID,
        "kayit_sayisi": len(kayitlar),
        "veriler": [
            {
                "zaman": k.zaman,
                "anlik_kisi": k.anlik_kisi,
                "ortalama": k.ortalama
            } for k in kayitlar
        ]
    }

@app.get("/db/oturumlar",
         summary="DB - Tüm Oturumlar",
         description="Sistemin geçmiş tüm çalışma oturumlarını getirir")
def db_oturumlar():
    try:
        with Session(engine) as session:
            oturumlar = session.query(Oturum)\
                .order_by(Oturum.baslangic.desc())\
                .all()
        return {
            "toplam_oturum": len(oturumlar),
            "oturumlar": [
                {
                    "id": o.id,
                    "baslangic": o.baslangic,
                    "bitis": o.bitis,
                    "video_kaynak": o.video_kaynak
                } for o in oturumlar
            ]
        }
    except Exception as e:
        return {"hata": str(e)}

@app.get("/db/rapor/{oturum_id}",
         summary="DB - Oturum Raporu",
         description="Belirli bir oturumun özet istatistikleri")
def db_oturum_raporu(oturum_id: int):
    with Session(engine) as session:
        kayitlar = session.query(YolcuSayimi)\
            .filter(YolcuSayimi.oturum_id == oturum_id)\
            .all()

    if not kayitlar:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı veya veri yok")

    degerler = [k.anlik_kisi for k in kayitlar]
    return {
        "oturum_id": oturum_id,
        "toplam_kayit": len(degerler),
        "istatistik": {
            "ortalama": round(sum(degerler) / len(degerler), 2),
            "maksimum": max(degerler),
            "minimum": min(degerler)
        }
    }

# ==================== SİSTEM DURUMU ====================
@app.get("/durum")
def sistem_durumu():
    return {
        "api_durum": "🟢 Aktif",
        "db_baglantisi": "🟢 Aktif" if AKTIF_OTURUM_ID else "🔴 Pasif",
        "oturum_id": AKTIF_OTURUM_ID,
        "aktif_kaynak_id": AKTIF_KAYNAK_ID,
        "zaman": datetime.now().isoformat()
    }

@app.get("/test")
def test():
    return {"durum": "başarılı", "mesaj": "API çalışıyor!"}