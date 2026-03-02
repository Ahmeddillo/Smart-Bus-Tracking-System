import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

class SurucuDedektoru:
    def __init__(self):
        print("YOLO yukleniyor (telefon tespiti)...")
        self.yolo = YOLO('yolov8n.pt')

        print("Haar Cascade yukleniyor...")
        haarcascade = cv2.data.haarcascades
        self.yuz_cascade  = cv2.CascadeClassifier(haarcascade + 'haarcascade_frontalface_alt2.xml')
        self.goz_cascade  = cv2.CascadeClassifier(haarcascade + 'haarcascade_eye.xml')
        self.gulen_cascade = cv2.CascadeClassifier(haarcascade + 'haarcascade_smile.xml')
        print("Hazir!")

        self.GOZ_ORAN_ESIK   = 0.28
        self.BAS_ACI_ESIK    = 0.25
        self.GOZ_KAPALI_KARE = 20
        self.ESNE_KARE       = 12

        self.goz_kapali_sayac = 0
        self.esne_sayac       = 0
        self.son_durum = self._bos_durum()

    def _bos_durum(self):
        return {
            "goz_oran"      : 0.0,
            "agiz_acik"     : False,
            "bas_acisi"     : 0.0,
            "telefon"       : False,
            "goz_kapali"    : False,
            "esniyor"       : False,
            "bas_egildi"    : False,
            "uyari_seviyesi": "normal",
            "uyari_mesaji"  : "",
            "yuz_tespit"    : False,
            "zaman"         : datetime.now().isoformat()
        }

    def _telefon_tespit(self, frame):
        results = self.yolo(frame, verbose=False, classes=[67], imgsz=320)
        return len(results[0].boxes) > 0

    def analiz_et(self, frame):
        if frame is None:
            return self.son_durum

        h, w = frame.shape[:2]
        gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gri = cv2.equalizeHist(gri)

        # 1. Yuz tespiti - direkt tum framede ara
        yuzler = self.yuz_cascade.detectMultiScale(
            gri, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )

        if len(yuzler) == 0:
            durum = self._bos_durum()
            durum["uyari_mesaji"] = "Yuz tespit edilemedi"
            self.son_durum = durum
            return durum

        # En buyuk yuzu al
        yuz = max(yuzler, key=lambda r: r[2]*r[3])
        yx, yy, yw, yh = yuz
        yuz_frame = frame[yy:yy+yh, yx:yx+yw]
        yuz_gri   = gri[yy:yy+yh, yx:yx+yw]

        # 2. Goz analizi - yuzun ust yarisinda
        ust_yari = yuz_gri[:yh//2, :]
        gozler = self.goz_cascade.detectMultiScale(
            ust_yari, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15)
        )

        if len(gozler) > 0:
            oranlar = [gh/gw for (gx,gy,gw,gh) in gozler if gw > 0]
            goz_oran = sum(oranlar) / len(oranlar)
        else:
            goz_oran = 0.0

        # 3. Agiz analizi - yuzun alt yarisinda
        alt_yari = yuz_gri[yh//2:, :]
        gulenler = self.gulen_cascade.detectMultiScale(
            alt_yari, scaleFactor=1.7, minNeighbors=20, minSize=(25, 15)
        )
        agiz_acik = len(gulenler) > 0

        # 4. Bas acisi
        yuz_merkez = yx + yw // 2
        sapma = abs(yuz_merkez - w//2) / w
        bas_acisi = round(sapma, 3)

        # 5. Telefon
        telefon = self._telefon_tespit(frame)

        # 6. Sayaclar
        goz_kapali_mi = goz_oran < self.GOZ_ORAN_ESIK

        if goz_kapali_mi:
            self.goz_kapali_sayac += 1
        else:
            self.goz_kapali_sayac = 0

        if agiz_acik:
            self.esne_sayac += 1
        else:
            self.esne_sayac = max(0, self.esne_sayac - 1)

        goz_kapali = self.goz_kapali_sayac >= self.GOZ_KAPALI_KARE
        esniyor    = self.esne_sayac >= self.ESNE_KARE
        bas_egildi = bas_acisi > self.BAS_ACI_ESIK

        # 7. Uyari
        if telefon:
            uyari, mesaj = "kritik", "TELEFON KULLANIYOR!"
        elif goz_kapali:
            uyari, mesaj = "kritik", "UYUKLUYOR!"
        elif bas_egildi:
            uyari, mesaj = "kritik", "BAS EGILMIS!"
        elif esniyor:
            uyari, mesaj = "dikkat", "Esniyor"
        elif goz_kapali_mi:
            uyari, mesaj = "dikkat", "Gozler kapaniyor"
        else:
            uyari, mesaj = "normal", ""

        self.son_durum = {
            "goz_oran"      : round(goz_oran, 3),
            "agiz_acik"     : agiz_acik,
            "bas_acisi"     : bas_acisi,
            "telefon"       : telefon,
            "goz_kapali"    : goz_kapali,
            "esniyor"       : esniyor,
            "bas_egildi"    : bas_egildi,
            "uyari_seviyesi": uyari,
            "uyari_mesaji"  : mesaj,
            "yuz_tespit"    : True,
            "yuz_kutu"      : (yx, yy, yw, yh),
            "goz_kutulari"  : gozler.tolist() if len(gozler) > 0 else [],
            "zaman"         : datetime.now().isoformat()
        }

        return self.son_durum


# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Surucu Yorgunluk Tespiti")
    parser.add_argument('--kaynak', type=str, default='0',
                        help="Webcam icin 0, RTSP icin rtsp://..., video icin dosya.mp4")
    args = parser.parse_args()

    kaynak = int(args.kaynak) if args.kaynak.isdigit() else args.kaynak

    cap = cv2.VideoCapture(kaynak)
    if not cap.isOpened():
        print("Video kaynagi acilamadi!")
        exit()

    dedektör = SurucuDedektoru()
    print("Baslatildi. Cikmak icin q tusuna basin.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        durum = dedektör.analiz_et(frame)

        renk = {
            "normal": (0, 255, 0),
            "dikkat": (0, 255, 255),
            "kritik": (0, 0, 255)
        }.get(durum["uyari_seviyesi"], (255,255,255))

        # Yuz kutusunu ciz
        if durum["yuz_tespit"]:
            vx, vy, vw, vh = durum["yuz_kutu"]
            cv2.rectangle(frame, (vx, vy), (vx+vw, vy+vh), renk, 2)
            for (gx, gy, gw, gh) in durum["goz_kutulari"]:
                cv2.rectangle(frame, (vx+gx, vy+gy), (vx+gx+gw, vy+gy+gh), (255,255,0), 1)

        # Bilgi paneli
        cv2.rectangle(frame, (0, 0), (310, 160), (0,0,0), -1)
        cv2.putText(frame, f"Goz  : {durum['goz_oran']:.3f}",                       (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Agiz : {'ACIK' if durum['agiz_acik'] else 'KAPALI'}",  (10,60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Bas  : {durum['bas_acisi']:.3f}",                      (10,90),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Tel  : {'VAR' if durum['telefon'] else 'YOK'}",        (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Yuz  : {'VAR' if durum['yuz_tespit'] else 'YOK'}",     (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if durum["uyari_mesaji"]:
            cv2.putText(frame, durum["uyari_mesaji"],
                        (10, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, renk, 3)

        cv2.rectangle(frame, (frame.shape[1]-180, 0), (frame.shape[1], 45), renk, -1)
        cv2.putText(frame, durum["uyari_seviyesi"].upper(),
                    (frame.shape[1]-170, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

        cv2.imshow("Surucu Yorgunluk Tespiti", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program kapatildi.")