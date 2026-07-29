# =============================================================================
# src/data_preprocessing.py
# Görev: Ham CSV verisini yükler, temizler ve modele hazır hale getirir.
# =============================================================================

import pandas as pd
import numpy as np
import re
import os

# --- Sabitler ---
# Eğitim seviyelerinin sayısal karşılıkları (TOPSIS için gerekli)
EGITIM_HARITASI = {
    "Lisans": 1,
    "Yüksek Lisans": 2,
    "Doktora": 3
}

# TOPSIS'e girecek sayısal sütunlar (ağırlıklandırma burada başlar)
NUMERIK_KRITERLER = [
    "Deneyim_Yili",
    "Egitim_Seviyesi_Numerik",
    "Teknik_Puan",
    "Iletisim_Puani",
    "Referans_Skoru",
    "Sertifika_Sayisi",
    "Maas_Beklenti_Katsayisi"
]


def metin_temizle(metin: str) -> str:
    """
    Verilen metni küçük harfe çevirir, noktalama işaretlerini
    kaldırır ve fazla boşlukları temizler.
    Türkçe karakterler (ç, ş, ı, ğ, ö, ü) korunur.

    Args:
        metin: Temizlenecek ham metin

    Returns:
        Temizlenmiş metin dizisi
    """
    if not isinstance(metin, str):
        return ""

    # Küçük harfe çevir
    metin = metin.lower()

    # Noktalama işaretlerini kaldır (Türkçe harfleri koru)
    metin = re.sub(r"[^\w\sçşıığöüÇŞİĞÖÜ]", " ", metin)

    # Birden fazla boşluğu tek boşluğa indir
    metin = re.sub(r"\s+", " ", metin).strip()

    return metin


def egitim_seviyesi_donustur(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Egitim_Seviyesi' sütunundaki kategorik değerleri
    TOPSIS için sayısal değerlere dönüştürür.
    Bilinmeyen değerler için 0 atanır ve uyarı verilir.

    Args:
        df: Ham veri çerçevesi

    Returns:
        'Egitim_Seviyesi_Numerik' sütunu eklenmiş veri çerçevesi
    """
    df = df.copy()

    df["Egitim_Seviyesi_Numerik"] = df["Egitim_Seviyesi"].map(EGITIM_HARITASI)

    # Eşleşmeyen değerleri tespit et ve uyar
    eslesmeyen = df[df["Egitim_Seviyesi_Numerik"].isna()]["Egitim_Seviyesi"].unique()
    if len(eslesmeyen) > 0:
        print(f"[UYARI] Tanımlanamayan eğitim seviyeleri: {eslesmeyen} → 0 atandı")
        df["Egitim_Seviyesi_Numerik"] = df["Egitim_Seviyesi_Numerik"].fillna(0)

    return df


def embedding_metni_olustur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sentence Transformer modeline verilecek birleşik metin sütununu oluşturur.
    Pozisyon + Yetenekler_Ve_Ozet birleştirilerek zengin bir bağlam sağlanır.

    Args:
        df: Temizlenmiş veri çerçevesi

    Returns:
        'Embedding_Metni' sütunu eklenmiş veri çerçevesi
    """
    df = df.copy()

    df["Yetenekler_Temiz"] = df["Yetenekler_Ve_Ozet"].apply(metin_temizle)

    # Pozisyon bilgisini de ekleyerek semantik zenginlik artırılır
    df["Embedding_Metni"] = (
        df["Pozisyon"].str.lower() + " " + df["Yetenekler_Temiz"]
    )

    return df


def veriyi_yukle_ve_isle(csv_yolu: str) -> pd.DataFrame:
    """
    Ana işlev: CSV'yi yükler ve tüm ön işleme adımlarını sırayla uygular.

    Adımlar:
        1. Dosya varlığını kontrol et
        2. CSV'yi oku
        3. Eksik değerleri kontrol et
        4. Eğitim seviyesini sayısala çevir
        5. Embedding metnini oluştur
        6. Veri tiplerini doğrula

    Args:
        csv_yolu: resumes.csv dosyasının yolu

    Returns:
        İşlenmiş ve modele hazır pandas DataFrame

    Raises:
        FileNotFoundError: CSV dosyası bulunamazsa
        ValueError: Zorunlu sütunlar eksikse
    """
    # --- 1. Dosya Kontrolü ---
    if not os.path.exists(csv_yolu):
        raise FileNotFoundError(f"[HATA] Veri dosyası bulunamadı: {csv_yolu}")

    # --- 2. CSV Okuma ---
    try:
        df = pd.read_csv(csv_yolu, encoding="utf-8")
        print(f"[BİLGİ] Veri başarıyla yüklendi: {df.shape[0]} aday, {df.shape[1]} sütun")
    except Exception as e:
        raise ValueError(f"[HATA] CSV okunamadı: {e}")

    # --- 3. Zorunlu Sütun Kontrolü ---
    zorunlu_sutunlar = [
        "Aday_ID", "Ad_Soyad", "Pozisyon", "Deneyim_Yili",
        "Egitim_Seviyesi", "Teknik_Puan", "Iletisim_Puani",
        "Referans_Skoru", "Sertifika_Sayisi", "Maas_Beklenti_Katsayisi",
        "Yetenekler_Ve_Ozet"
    ]
    eksik_sutunlar = [s for s in zorunlu_sutunlar if s not in df.columns]
    if eksik_sutunlar:
        raise ValueError(f"[HATA] Eksik sütunlar: {eksik_sutunlar}")

    # --- 4. Eksik Değer Raporu ---
    eksik_sayisi = df.isnull().sum().sum()
    if eksik_sayisi > 0:
        print(f"[UYARI] Toplam {eksik_sayisi} eksik değer tespit edildi:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("[BİLGİ] Eksik değer yok ✓")

    # --- 5. Eğitim Seviyesi Dönüşümü ---
    df = egitim_seviyesi_donustur(df)
    print("[BİLGİ] Eğitim seviyesi sayısala dönüştürüldü ✓")

    # --- 6. Embedding Metni Oluşturma ---
    df = embedding_metni_olustur(df)
    print("[BİLGİ] Embedding metinleri hazırlandı ✓")

    # --- 7. Veri Tipi Doğrulama ---
    for kriter in NUMERIK_KRITERLER:
        if kriter in df.columns:
            df[kriter] = pd.to_numeric(df[kriter], errors="coerce").fillna(0)

    print("[BİLGİ] Tüm ön işleme adımları tamamlandı ✓")
    print(f"[BİLGİ] Modele hazır sütunlar: {NUMERIK_KRITERLER}")

    return df


# =============================================================================
# TEST BLOĞU — Bu dosyayı doğrudan çalıştırınca devreye girer
# =============================================================================
if __name__ == "__main__":

    # Proje kök dizininden çalıştırıldığı varsayılır
  CSV_YOLU = "resumes.csv"

    print("=" * 55)
    print("   VERİ ÖN İŞLEME MODÜLİ — TEST ÇALIŞIYOR")
    print("=" * 55)

    df_islenmis = veriyi_yukle_ve_isle(CSV_YOLU)

    print("\n--- İlk 3 Adayın İşlenmiş Verisi ---")
    print(df_islenmis[["Ad_Soyad", "Egitim_Seviyesi",
                        "Egitim_Seviyesi_Numerik",
                        "Embedding_Metni"]].head(3).to_string())

    print("\n--- Numerik Kriterlerin İstatistikleri ---")
    print(df_islenmis[NUMERIK_KRITERLER].describe().round(2))
