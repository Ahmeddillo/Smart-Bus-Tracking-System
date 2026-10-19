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

# VERİTABANI AYARLARI:
DATABASE_URL = os.getenv("DATABASE_URL")

# Eğer .env yoksa veya boşsa otomatik olarak kodun içine yazdığın adresi al
if not DATABASE_URL:
    # BURAYA KENDİ DB BİLGİLERİNİ YAZIYORSUN
    DATABASE_URL = "****************"

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
    
class SurucuLogu(Base):
    __tablename__ = "surucu_loglari"
    id             = Column(Integer, primary_key=True)
    oturum_id      = Column(Integer, nullable=True)
    zaman          = Column(DateTime, default=datetime.now)
    ear_degeri     = Column(Float)
    mar_degeri     = Column(Float)
    bas_mesafesi   = Column(Float)
    goz_kapali     = Column(Integer, default=0)
    esniyor        = Column(Integer, default=0)
    bas_egildi     = Column(Integer, default=0)
    uyari_seviyesi = Column(String)
    uyari_mesaji   = Column(String)

class YanginLogu(Base):
    __tablename__ = "yangin_loglari"
    id          = Column(Integer, primary_key=True)
    kamera_id   = Column(String)
    skor        = Column(Float)
    kutu_sayisi = Column(Integer)
    alarm       = Column(Integer, default=0)
    zaman       = Column(DateTime, default=datetime.now)
    
class CukurLogu(Base):
    __tablename__ = "cukur_loglari"
    id          = Column(Integer, primary_key=True)
    kamera_id   = Column(String)
    skor        = Column(Float)
    kutu_sayisi = Column(Integer)
    enlem       = Column(Float)   # GPS latitude
    boylam      = Column(Float)   # GPS longitude
    zaman       = Column(DateTime, default=datetime.now)

# Tabloları oluştur (yoksa)
try:
    Base.metadata.create_all(engine)
    print("✅ Tablolar kontrol edildi/oluşturuldu.")
except Exception as e:
    print(f"❌ Tablo oluşturma hatası: {e}")

# SİSTEM BAŞLATMA:
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
    son_saat = list(son_anlik_veriler)[-360:] 
    
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

# DB ENDPOINTLERİ:

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


# SÜRÜCÜ TAKİP ENDPOINTLERİ: 

# --- API'nin RAM'inde tutulacak boş durumlar ---
api_surucu_durumu = {}
api_yangin_durumu = {}

@app.post("/surucu/guncelle", summary="Kamera Kodundan Canlı Veri Alır", tags=["Sürücü"])
def surucu_veri_al(veri: dict):
    """
    Kamera kodun saniyede bir çalışıp verilerini buraya POST eder.
    Böylece API kilitlenmez ve veri güncel kalır.
    """
    global api_surucu_durumu
    api_surucu_durumu = veri
    return {"mesaj": "Veri başarıyla alındı"}

@app.get("/surucu/durum", summary="Sürücü Anlık Durumu", tags=["Sürücü"])
def surucu_anlik_durum():
    """
    Kameradan gelen en son sürücü durumunu döndürür.
    Frontend ve mobil uygulamalar buradan veriyi okur.
    """
    return api_surucu_durumu

@app.get("/surucu/gecmis", summary="Sürücü Geçmiş Uyarılar", tags=["Sürücü"])
def surucu_gecmis(limit: int = Query(50, ge=1, le=500)):
    """
    DB'den geçmiş dikkat/kritik uyarıları getirir.
    """
    try:
        with Session(engine) as session:
            kayitlar = session.query(SurucuLogu)\
                .order_by(SurucuLogu.zaman.desc())\
                .limit(limit)\
                .all()
        return {
            "kayit_sayisi": len(kayitlar),
            "veriler": [
                {
                    "zaman"         : k.zaman,
                    "uyari_seviyesi": k.uyari_seviyesi,
                    "uyari_mesaji"  : k.uyari_mesaji,
                    "ear"           : k.ear_degeri,
                    "mar"           : k.mar_degeri,
                    "bas_mesafesi"  : k.bas_mesafesi,
                    "goz_kapali"    : bool(k.goz_kapali),
                    "esniyor"       : bool(k.esniyor),
                    "bas_egildi"    : bool(k.bas_egildi)
                } for k in kayitlar
            ]
        }
    except Exception as e:
        return {"hata": str(e), "veriler": []}


@app.get("/surucu/istatistik", summary="Sürücü İstatistikleri", tags=["Sürücü"])
def surucu_istatistik():
    """
    Bugünkü uyarı özeti.
    """
    try:
        bugun = datetime.now().date()
        with Session(engine) as session:
            bugun_kayitlar = session.query(SurucuLogu)\
                .filter(SurucuLogu.zaman >= bugun)\
                .all()

        kritik = sum(1 for k in bugun_kayitlar if k.uyari_seviyesi == "kritik")
        dikkat = sum(1 for k in bugun_kayitlar if k.uyari_seviyesi == "dikkat")

        return {
            "bugun": str(bugun),
            "toplam_uyari": len(bugun_kayitlar),
            "kritik_sayisi": kritik,
            "dikkat_sayisi": dikkat,
        }
    except Exception as e:
        return {"hata": str(e)}

# YANGIN TAKİP ENDPOINTLERİ:

@app.post("/yangin/guncelle", summary="Yangın Kodundan Canlı Veri Alır", tags=["Yangın"])
def yangin_veri_al(veri: dict):
    global api_yangin_durumu
    api_yangin_durumu = veri

    # Yangın varsa DB'ye kaydet
    if veri.get("yangin_var"):
        try:
            with Session(engine) as session:
                kayit = YanginLogu(
                    kamera_id   = veri.get("kamera_id", "kamera_1"),
                    skor        = veri.get("skor", 0.0),
                    kutu_sayisi = veri.get("kutu_sayisi", 0),
                    alarm       = 1 if veri.get("alarm") else 0
                )
                session.add(kayit)
                session.commit()
        except Exception as e:
            print(f"⚠️ Yangın DB hatası: {e}")

    return {"mesaj": "Veri alındı"}

@app.get("/yangin/durum", summary="Yangın Anlık Durumu", tags=["Yangın"])
def yangin_anlik_durum():
    return api_yangin_durumu

@app.get("/yangin/gecmis", summary="Yangın Geçmiş Kayıtlar", tags=["Yangın"])
def yangin_gecmis(limit: int = Query(50, ge=1, le=500)):
    try:
        with Session(engine) as session:
            kayitlar = session.query(YanginLogu)\
                .order_by(YanginLogu.zaman.desc())\
                .limit(limit)\
                .all()
        return {
            "kayit_sayisi": len(kayitlar),
            "veriler": [
                {
                    "zaman"      : k.zaman,
                    "kamera_id"  : k.kamera_id,
                    "skor"       : k.skor,
                    "kutu_sayisi": k.kutu_sayisi,
                    "alarm"      : bool(k.alarm)
                } for k in kayitlar
            ]
        }
    except Exception as e:
        return {"hata": str(e), "veriler": []}
    
@app.post("/cukur/guncelle", summary="Çukur Kamerasından Veri Alır", tags=["Çukur"])
def cukur_veri_al(veri: dict):
    global api_cukur_durumu
    api_cukur_durumu = veri
 
    # Sadece kayit_edildi=True olanları DB'ye yaz (tekrar kaçınmak için)
    if veri.get("cukur_var") and veri.get("kayit_edildi"):
        try:
            with Session(engine) as session:
                kayit = CukurLogu(
                    kamera_id   = veri.get("kamera_id", "on_kamera"),
                    skor        = veri.get("skor", 0.0),
                    kutu_sayisi = veri.get("kutu_sayisi", 0),
                    enlem       = veri.get("enlem"),
                    boylam      = veri.get("boylam"),
                )
                session.add(kayit)
                session.commit()
                print(f"📍 Çukur kaydedildi: ({veri.get('enlem')}, {veri.get('boylam')})")
        except Exception as e:
            print(f"⚠️ Çukur DB hatası: {e}")
 
    return {"mesaj": "Veri alındı"}
 
 
@app.get("/cukur/durum", summary="Çukur Anlık Durum", tags=["Çukur"])
def cukur_anlik_durum():
    return api_cukur_durumu
 
 
@app.get("/cukur/gecmis", summary="Çukur Geçmiş Kayıtlar", tags=["Çukur"])
def cukur_gecmis(limit: int = Query(50, ge=1, le=500)):
    try:
        with Session(engine) as session:
            kayitlar = session.query(CukurLogu)\
                .order_by(CukurLogu.zaman.desc())\
                .limit(limit)\
                .all()
        return {
            "kayit_sayisi": len(kayitlar),
            "veriler": [
                {
                    "zaman"      : k.zaman,
                    "kamera_id"  : k.kamera_id,
                    "skor"       : k.skor,
                    "kutu_sayisi": k.kutu_sayisi,
                    "enlem"      : k.enlem,
                    "boylam"     : k.boylam,
                } for k in kayitlar
            ]
        }
    except Exception as e:
        return {"hata": str(e), "veriler": []}
 
 
@app.get("/cukur/harita", summary="Çukur Harita Verisi", tags=["Çukur"])
def cukur_harita():
    """
    Dashboard haritasında nokta olarak göstermek için
    sadece koordinat + skor döndürür.
    """
    try:
        with Session(engine) as session:
            kayitlar = session.query(CukurLogu)\
                .filter(CukurLogu.enlem.isnot(None))\
                .order_by(CukurLogu.zaman.desc())\
                .limit(200)\
                .all()
        return {
            "noktalar": [
                {
                    "enlem" : k.enlem,
                    "boylam": k.boylam,
                    "skor"  : k.skor,
                    "zaman" : k.zaman,
                } for k in kayitlar
            ]
        }
    except Exception as e:
        return {"hata": str(e), "noktalar": []}

# SİSTEM DURUMU:
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
