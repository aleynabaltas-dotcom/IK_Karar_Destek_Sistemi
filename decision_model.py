# =============================================================================
# src/decision_model.py
# Görev: Semantik embedding + TOPSIS ile aday sıralama ve karar üretme
# =============================================================================

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import os
import json
from datetime import datetime

# --- Sabitler ---
MODEL_ADI = "paraphrase-multilingual-MiniLM-L12-v2"

# TOPSIS kriterleri ve ağırlıkları (toplam = 1.0 olmalı)
# Semantik skor en yüksek ağırlıkta çünkü NL sorguya uyumu ölçüyor
KRITER_AGIRLIKLARI = {
    "Semantik_Skor":           0.25,
    "Deneyim_Yili":            0.20,
    "Egitim_Seviyesi_Numerik": 0.15,
    "Teknik_Puan":             0.15,
    "Iletisim_Puani":          0.10,
    "Referans_Skoru":          0.10,
    "Sertifika_Sayisi":        0.03,
    "Maas_Beklenti_Katsayisi": 0.02,
}

# Fayda yönü: True = büyük iyidir, False = küçük iyidir
FAYDA_YONU = {
    "Semantik_Skor":           True,
    "Deneyim_Yili":            True,
    "Egitim_Seviyesi_Numerik": True,
    "Teknik_Puan":             True,
    "Iletisim_Puani":          True,
    "Referans_Skoru":          True,
    "Sertifika_Sayisi":        True,
    "Maas_Beklenti_Katsayisi": False,  # Düşük maaş beklentisi daha iyi
}


class KararDestek:
    """
    Doğal Dil Tabanlı İK Karar Destek Sistemi ana sınıfı.
    Sentence Transformers + TOPSIS hibrit mimarisi kullanır.
    """

    def __init__(self, df: pd.DataFrame, log_klasoru: str = "logs"):
        """
        Args:
            df: data_preprocessing'den gelen işlenmiş DataFrame
            log_klasoru: Sorgu loglarının kaydedileceği klasör
        """
        self.df = df.copy()
        self.log_klasoru = log_klasoru
        self.model = None
        self.aday_vektorleri = None
        self._model_yukle()
        self._vektorleri_hesapla()

    def _model_yukle(self):
        """Sentence Transformer modelini yükler."""
        try:
            print(f"[BİLGİ] Model yükleniyor: {MODEL_ADI}")
            self.model = SentenceTransformer(MODEL_ADI)
            print("[BİLGİ] Model başarıyla yüklendi ✓")
        except Exception as e:
            raise RuntimeError(f"[HATA] Model yüklenemedi: {e}")

    def _vektorleri_hesapla(self):
        """Tüm adayların embedding vektörlerini önceden hesaplar."""
        try:
            print("[BİLGİ] Aday vektörleri hesaplanıyor...")
            metinler = self.df["Embedding_Metni"].tolist()
            self.aday_vektorleri = self.model.encode(
                metinler,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            print(f"[BİLGİ] {len(metinler)} aday vektörize edildi ✓")
        except Exception as e:
            raise RuntimeError(f"[HATA] Vektörizasyon başarısız: {e}")

    def _semantik_skor_hesapla(self, sorgu: str) -> np.ndarray:
        """
        Kullanıcı sorgusunu vektöre çevirip adaylarla
        cosine similarity hesaplar.

        Args:
            sorgu: Kullanıcının doğal dil sorgusu

        Returns:
            Her aday için 0-1 arası benzerlik skoru dizisi
        """
        sorgu_vektoru = self.model.encode(sorgu, convert_to_numpy=True)

        # Cosine similarity: normalize edilmiş vektörlerin iç çarpımı
        aday_norm = self.aday_vektorleri / (
            np.linalg.norm(self.aday_vektorleri, axis=1, keepdims=True) + 1e-9
        )
        sorgu_norm = sorgu_vektoru / (np.linalg.norm(sorgu_vektoru) + 1e-9)

        benzerlik = np.dot(aday_norm, sorgu_norm)

        # 0-1 aralığına normalize et
        benzerlik = (benzerlik - benzerlik.min()) / (
            benzerlik.max() - benzerlik.min() + 1e-9
        )
        return benzerlik

    def _topsis_hesapla(self, matris: np.ndarray, agirliklar: list, fayda: list) -> np.ndarray:
        """
        TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
        algoritmasını uygular.

        Adımlar:
            1. Normalize et (vektör normalizasyonu)
            2. Ağırlıklı normalize matris oluştur
            3. İdeal en iyi ve en kötü çözümleri bul
            4. Her alternatifin ideal çözümlere uzaklığını hesapla
            5. Yakınlık katsayısını hesapla

        Args:
            matris: (n_aday x n_kriter) boyutlu karar matrisi
            agirliklar: Her kriterin ağırlığı (liste)
            fayda: Her kriter için True=fayda, False=maliyet

        Returns:
            Her aday için TOPSIS yakınlık katsayısı (0-1 arası)
        """
        # 1. Vektör normalizasyonu
        norm = np.sqrt((matris ** 2).sum(axis=0))
        norm[norm == 0] = 1e-9
        normalized = matris / norm

        # 2. Ağırlıklı normalize matris
        agirlik_dizisi = np.array(agirliklar)
        agirlikli = normalized * agirlik_dizisi

        # 3. İdeal en iyi (A+) ve en kötü (A-) çözümler
        ideal_iyi = np.where(fayda, agirlikli.max(axis=0), agirlikli.min(axis=0))
        ideal_kotu = np.where(fayda, agirlikli.min(axis=0), agirlikli.max(axis=0))

        # 4. Uzaklıklar (Öklid mesafesi)
        uzaklik_iyi = np.sqrt(((agirlikli - ideal_iyi) ** 2).sum(axis=1))
        uzaklik_kotu = np.sqrt(((agirlikli - ideal_kotu) ** 2).sum(axis=1))

        # 5. Yakınlık katsayısı
        yakinlik = uzaklik_kotu / (uzaklik_iyi + uzaklik_kotu + 1e-9)

        return yakinlik

    def _aciklama_uret(self, aday_satir: pd.Series, topsis_skoru: float, semantik_skor: float) -> str:
        """
        Bir adayın neden bu sırada olduğunu açıklayan metin üretir.

        Args:
            aday_satir: Adayın DataFrame satırı
            topsis_skoru: Adayın TOPSIS skoru
            semantik_skor: Adayın semantik benzerlik skoru

        Returns:
            Açıklama metni
        """
        guclu = []
        zayif = []

        if semantik_skor >= 0.7:
            guclu.append("sorgu ile yüksek semantik uyum")
        elif semantik_skor <= 0.3:
            zayif.append("sorgu ile düşük semantik uyum")

        if aday_satir["Teknik_Puan"] >= 85:
            guclu.append(f"yüksek teknik puan ({aday_satir['Teknik_Puan']})")
        elif aday_satir["Teknik_Puan"] <= 65:
            zayif.append(f"düşük teknik puan ({aday_satir['Teknik_Puan']})")

        if aday_satir["Deneyim_Yili"] >= 6:
            guclu.append(f"güçlü deneyim ({aday_satir['Deneyim_Yili']} yıl)")
        elif aday_satir["Deneyim_Yili"] <= 2:
            zayif.append(f"sınırlı deneyim ({aday_satir['Deneyim_Yili']} yıl)")

        if aday_satir["Referans_Skoru"] >= 4.3:
            guclu.append(f"güçlü referans skoru ({aday_satir['Referans_Skoru']})")

        if aday_satir["Maas_Beklenti_Katsayisi"] <= 1.0:
            guclu.append("bütçeye uygun maaş beklentisi")
        elif aday_satir["Maas_Beklenti_Katsayisi"] >= 1.3:
            zayif.append("yüksek maaş beklentisi")

        aciklama = f"TOPSIS Skoru: {topsis_skoru:.4f} | "
        if guclu:
            aciklama += f"Güçlü yönler: {', '.join(guclu)}. "
        if zayif:
            aciklama += f"Zayıf yönler: {', '.join(zayif)}."

        return aciklama.strip()

    def _log_kaydet(self, sorgu: str, sonuclar: pd.DataFrame):
        """
        Sorgu ve sonuçları JSON formatında logs/ klasörüne kaydeder.

        Args:
            sorgu: Kullanıcının orijinal sorgusu
            sonuclar: Sıralanmış sonuç DataFrame'i
        """
        try:
            os.makedirs(self.log_klasoru, exist_ok=True)
            zaman = datetime.now().strftime("%Y%m%d_%H%M%S")
            dosya_adi = os.path.join(self.log_klasoru, f"sorgu_{zaman}.json")

            log_verisi = {
                "zaman": zaman,
                "sorgu": sorgu,
                "sonuclar": sonuclar[["Ad_Soyad", "Pozisyon", "TOPSIS_Skoru", "Aciklama"]].to_dict(orient="records")
            }

            with open(dosya_adi, "w", encoding="utf-8") as f:
                json.dump(log_verisi, f, ensure_ascii=False, indent=2)

            print(f"[BİLGİ] Log kaydedildi: {dosya_adi}")
        except Exception as e:
            print(f"[UYARI] Log kaydedilemedi: {e}")

    def aralik_filtrele(self, df: pd.DataFrame, pozisyon: str = None,
                        min_deneyim: int = None, max_deneyim: int = None) -> pd.DataFrame:
        """
        Sorgu öncesi isteğe bağlı filtre uygular.

        Args:
            df: Filtrelenecek DataFrame
            pozisyon: Pozisyon filtresi (örn: "Data Scientist")
            min_deneyim: Minimum deneyim yılı
            max_deneyim: Maksimum deneyim yılı

        Returns:
            Filtrelenmiş DataFrame
        """
        if pozisyon:
            df = df[df["Pozisyon"].str.lower() == pozisyon.lower()]
        if min_deneyim is not None:
            df = df[df["Deneyim_Yili"] >= min_deneyim]
        if max_deneyim is not None:
            df = df[df["Deneyim_Yili"] <= max_deneyim]
        return df

    def en_iyi_adaylari_bul(self, sorgu: str, top_n: int = 5,
                             pozisyon: str = None,
                             min_deneyim: int = None,
                             max_deneyim: int = None) -> pd.DataFrame:
        """
        Ana karar fonksiyonu: NL sorguya göre en iyi adayları sıralar.

        Args:
            sorgu: Doğal dil sorgusu (örn: "makine öğrenmesi bilen kıdemli veri bilimcisi")
            top_n: Kaç aday döndürülsün
            pozisyon: Opsiyonel pozisyon filtresi
            min_deneyim: Opsiyonel minimum deneyim filtresi
            max_deneyim: Opsiyonel maksimum deneyim filtresi

        Returns:
            Sıralanmış ve açıklamalı aday DataFrame'i
        """
        # Filtreleme uygula
        df_filtre = self.aralik_filtrele(
            self.df.copy(), pozisyon, min_deneyim, max_deneyim
        )

        if df_filtre.empty:
            print("[UYARI] Filtre kriterlerine uyan aday bulunamadı.")
            return pd.DataFrame()

        # Filtrelenmiş adayların indekslerini al
        indeksler = df_filtre.index.tolist()
        filtreli_vektorler = self.aday_vektorleri[indeksler]

        # Semantik skor hesapla
        sorgu_vektoru = self.model.encode(sorgu, convert_to_numpy=True)
        aday_norm = filtreli_vektorler / (
            np.linalg.norm(filtreli_vektorler, axis=1, keepdims=True) + 1e-9
        )
        sorgu_norm = sorgu_vektoru / (np.linalg.norm(sorgu_vektoru) + 1e-9)
        semantik_skorlar = np.dot(aday_norm, sorgu_norm)
        semantik_skorlar = (semantik_skorlar - semantik_skorlar.min()) / (
            semantik_skorlar.max() - semantik_skorlar.min() + 1e-9
        )

        df_filtre = df_filtre.copy()
        df_filtre["Semantik_Skor"] = semantik_skorlar

        # TOPSIS karar matrisi oluştur
        kriterler = list(KRITER_AGIRLIKLARI.keys())
        matris = df_filtre[kriterler].values.astype(float)
        agirliklar = list(KRITER_AGIRLIKLARI.values())
        fayda = [FAYDA_YONU[k] for k in kriterler]

        # TOPSIS hesapla
        topsis_skorlari = self._topsis_hesapla(matris, agirliklar, fayda)
        df_filtre["TOPSIS_Skoru"] = topsis_skorlari

        # Açıklama üret
        df_filtre["Aciklama"] = [
            self._aciklama_uret(row, row["TOPSIS_Skoru"], row["Semantik_Skor"])
            for _, row in df_filtre.iterrows()
        ]

        # Sırala ve top_n al
        sonuclar = df_filtre.sort_values("TOPSIS_Skoru", ascending=False).head(top_n)
        sonuclar = sonuclar.reset_index(drop=True)
        sonuclar.index += 1  # 1'den başlasın

        # Log kaydet
        self._log_kaydet(sorgu, sonuclar)

        return sonuclar


# =============================================================================
# TEST BLOĞU
# =============================================================================
if __name__ == "__main__":
    from data_preprocessing import veriyi_yukle_ve_isle
    
    CSV_YOLU = "resumes.csv"

    print("=" * 55)
    print("   KARAR MODELİ — TEST ÇALIŞIYOR")
    print("=" * 55)

    df = veriyi_yukle_ve_isle(CSV_YOLU)
    kds = KararDestek(df)

    # Test sorgusu
    sorgu = "makine öğrenmesi ve derin öğrenme bilen kıdemli veri bilimcisi"
    print(f"\n[SORGU] {sorgu}\n")

    sonuclar = kds.en_iyi_adaylari_bul(sorgu, top_n=5, pozisyon="Data Scientist")

    print(sonuclar[["Ad_Soyad", "Deneyim_Yili", "Teknik_Puan",
                    "TOPSIS_Skoru", "Aciklama"]].to_string())
