# 🧠 Doğal Dil Tabanlı İK Karar Destek Sistemi (KDS)

Modern İnsan Kaynakları departmanlarının işe alım süreçlerindeki operasyonel yükü hafifletmek ve aday değerlendirmesinde nesnel, ölçülebilir bir çerçeve sunmak amacıyla geliştirilmiş, yapay zeka destekli bir Karar Destek Sistemidir (KDS).

Sistem, yapılandırılmamış metin verilerini (özgeçmiş/yetenek özetleri) ve yapılandırılmış sayısal verileri (deneyim yılı, eğitim seviyesi, teknik puan vb.) bir araya getirerek **Çok Kriterli Karar Verme (ÇKKV)** metodolojisiyle bütünleşik bir aday sıralama modeli üretir.

🚀 **Canlı Demo:** [aleynabaltas-ik-kds.streamlit.app](https://aleynabaltas-ik-kds.streamlit.app)

---

## 🏗️ Sistem Mimarisi ve Kullanılan Teknolojiler

Proje, Doğal Dil İşleme (NLP) ve Çok Kriterli Karar Verme algoritmalarının hibrit bir yapıda çalışması üzerine kurulmuştur.

- **Arayüz (Frontend):** Streamlit
- **Veri Manipülasyonu:** Pandas, NumPy
- **NLP & Vektör Modelleme:** Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) — çok dilli, Türkçe destekli embedding modeli
- **Karar Algoritması:** TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
- **Görselleştirme:** Plotly Express, Plotly Graph Objects

---

## ⚙️ Teknik Altyapı ve Modüller

Proje modüler bir yapıda tasarlanmıştır:

### 1. `data_preprocessing.py` — Veri Ön İşleme & Özellik Mühendisliği
- CSV verisini okur, zorunlu sütunları doğrular ve eksik değerleri raporlar.
- Kategorik eğitim seviyelerini (Lisans / Yüksek Lisans / Doktora) ordinal sayısal değerlere dönüştürür.
- Adayın pozisyon ve yetenek bilgilerini birleştirip tek bir embedding metni oluşturur (metin temizleme: küçük harfe çevirme, noktalama temizliği, Türkçe karakter koruması).

### 2. `decision_model.py` — Skorlama ve TOPSIS Algoritması
- **Semantik Eşleştirme:** Kullanıcının serbest metin sorgusu vektöre çevrilir; Kosinüs Benzerliği ile her adayın yetenek vektörüyle karşılaştırılıp 0–1 arası bir semantik skor hesaplanır.
- **TOPSIS Uygulaması:** Semantik skor + 7 sayısal kriter (deneyim yılı, eğitim seviyesi, teknik puan, iletişim puanı, referans skoru, sertifika sayısı, maaş beklenti katsayısı) ağırlıklandırılıp karar matrisi oluşturulur.
- Matris vektör normalizasyonundan geçirilir, pozitif ve negatif ideal çözüm noktaları belirlenir, her adayın bu noktalara Öklid uzaklığı üzerinden nihai **Uygunluk Skoru (Closeness Coefficient)** hesaplanır.
- Her aday için güçlü/zayıf yönleri özetleyen otomatik bir gerekçe metni üretilir.
- Her sorgu ve sonuç, `logs/` klasörüne JSON olarak kaydedilir (yerel çalıştırmada kalıcıdır; Streamlit Cloud gibi geçici dosya sistemlerinde oturum sonunda silinir).

### 3. `app.py` — Uygulama Sunumu
- Pozisyon, deneyim aralığı ve gösterilecek aday sayısı için interaktif kenar çubuğu (sidebar) filtreleri.
- Güncel kriter ağırlıklarının görüntülendiği bir panel.
- Sonuçları TOPSIS bar chart, ilk 3 aday için radar chart, aday kartları ve detaylı tablo ile sunar.

> **Not:** Kriter ağırlıkları şu an sabittir ve sidebar'da yalnızca bilgi amaçlı gösterilir; kullanıcı tarafından anlık değiştirilemez. Bu, aşağıdaki Yol Haritası bölümünde planlanan bir geliştirmedir.

---

## 📊 Veri Seti Yapısı (`resumes.csv`)

Proje kapsamında sentetik/simüle edilmiş bir İK veri seti kullanılmaktadır (20 aday, 4 pozisyon: Data Scientist, Software Engineer, HR Specialist, Marketing Specialist). Temel sütunlar:

| Sütun | Açıklama |
|---|---|
| `Ad_Soyad` | Adayın adı soyadı |
| `Pozisyon` | Başvurduğu/uzmanlaştığı rol |
| `Deneyim_Yili` | Toplam aktif çalışma süresi (yıl) |
| `Egitim_Seviyesi` | Lisans / Yüksek Lisans / Doktora |
| `Teknik_Puan`, `Iletisim_Puani`, `Referans_Skoru`, `Sertifika_Sayisi`, `Maas_Beklenti_Katsayisi` | TOPSIS kriterleri |
| `Yetenekler_Ve_Ozet` | Serbest metin yetenek/özet açıklaması |

---

## 💻 Kurulum ve Çalıştırma

**1. Repoyu klonlayın**
```bash
git clone https://github.com/aleynabaltas-dotcom/IK_Karar_Destek_Sistemi.git
cd IK_Karar_Destek_Sistemi
```

**2. Sanal ortam oluşturun (önerilir)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**3. Gerekli kütüphaneleri yükleyin**
```bash
pip install -r requirements.txt
```

**4. Streamlit sunucusunu başlatın**
```bash
streamlit run app.py
```

Tarayıcınız otomatik açılacak ve uygulama `http://localhost:8501` adresinde çalışmaya başlayacaktır.

---

## 🗺️ Yol Haritası

- [ ] Kriter ağırlıklarını sidebar'dan anlık değiştirebilme (duyarlılık analizi)
- [ ] Semantik skorun mutlak bir referansa göre normalize edilmesi (şu an yalnızca filtrelenmiş grup içinde göreceli)
- [ ] Daha geniş / gerçekçi veri seti
- [ ] Birim testleri (pytest)

## ⚠️ Sınırlamalar

- Veri seti sentetiktir ve küçük ölçeklidir (20 kayıt); üretim ortamı performansını yansıtmaz.
- Semantik skor, her sorguda yalnızca filtrelenmiş aday grubu içinde göreceli olarak normalize edilir; mutlak bir eşleşme kalitesi eşiği değildir.
- Gerçek özgeçmiş verisiyle kullanılacaksa, kişisel veri saklama ve KVKK/GDPR uyumluluğu ayrıca değerlendirilmelidir.

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

## 🎓 Not

Bu proje akademik/portföy amaçlı geliştirilmiştir.
