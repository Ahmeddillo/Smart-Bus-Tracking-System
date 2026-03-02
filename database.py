import psycopg2
from psycopg2 import pool
import json

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.connection_pool = None

    def connect(self):
        """PostgreSQL bağlantı havuzunu oluşturur"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,
                host=self.config['host'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                port=self.config['port']
            )
            print("✅ Veritabanı bağlantı havuzu oluşturuldu.")
        except Exception as e:
            print(f"❌ Veritabanına bağlanılamadı: {e}")

    def get_connection(self):
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        self.connection_pool.putconn(conn)

    def anlik_kaydet(self, kamera_id, kafa_sayisi):
        """10 saniyede bir gelen anlık veriyi tabloya yazar"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO anlik_veriler (kamera_id, kafa_sayisi, zaman) VALUES (%s, %s, NOW())",
                    (kamera_id, kafa_sayisi)
                )
            conn.commit()
        except Exception as e:
            print(f"⚠️ Anlık veri kaydedilemedi: {e}")
        finally:
            self.return_connection(conn)

    def dakikalik_kaydet(self, kamera_id, baslangic, bitis, toplam, ortalama, maks, min, fps_gercek, detaylar):
        """Dakika dolduğunda istatistikleri tabloya yazar"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO dakikalik_veriler 
                    (kamera_id, baslangic_zamani, bitis_zamani, toplam_kafa, ortalama_kafa, maks_kafa, min_kafa, fps_verimi, ek_detaylar) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (kamera_id, baslangic, bitis, toplam, ortalama, maks, min, fps_gercek, json.dumps(detaylar))
                )
            conn.commit()
        except Exception as e:
            print(f"⚠️ Dakikalık veri kaydedilemedi: {e}")
        finally:
            self.return_connection(conn)

    def close(self):
        if self.connection_pool:
            self.connection_pool.closeall()
            print("🔌 Veritabanı bağlantıları kapatıldı.")