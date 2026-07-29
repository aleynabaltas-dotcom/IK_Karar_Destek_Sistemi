# ============================================================================
# Görev: Streamlit tabanlı akıllı İK Karar Destek Sistemi arayüzü
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
from data_preprocessing import veriyi_yukle_ve_isle, NUMERIK_KRITERLER
from decision_model import KararDestek, KRITER_AGIRLIKLARI

# =============================================================================
# SAYFA AYARLARI
# =============================================================================
st.set_page_config(
    page_title="İK Karar Destek Sistemi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS — Profesyonel görünüm
# =============================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: white;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px 0;
    }
    .aday-kart {
        background: white;
        border-left: 5px solid #2d6a9f;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .birinci { border-left-color: #f59e0b; }
    .ikinci  { border-left-color: #9ca3af; }
    .ucuncu  { border-left-color: #b45309; }
    .aciklama-text {
        font-size: 0.85em;
        color: #64748b;
        margin-top: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        font-size: 1em;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# VERİ VE MODEL YÜKLEME (önbelleğe al — her sorguda yeniden yüklenmesin)
# =============================================================================
@st.cache_resource(show_spinner="🧠 Model yükleniyor, lütfen bekleyin...")
def sistemi_yukle():
    """Veri ve KDS modelini yükler, önbelleğe alır."""
    csv_yolu = "resumes.csv"
    df = veriyi_yukle_ve_isle(csv_yolu)
    kds = KararDestek(df)
    return df, kds

# =============================================================================
# SIDEBAR — FİLTRELER
# =============================================================================
def sidebar_olustur():
    """Sol panel filtrelerini oluşturur."""
    with st.sidebar:
        st.markdown("## ⚙️ Filtreler")
        st.markdown("---")

        pozisyon = st.selectbox(
            "📌 Pozisyon Filtresi",
            options=["Tümü", "Data Scientist", "Software Engineer",
                     "HR Specialist", "Marketing Specialist"],
            index=0
        )

        st.markdown("#### 📅 Deneyim Aralığı (Yıl)")
        col1, col2 = st.columns(2)
        with col1:
            min_deneyim = st.number_input("Min", min_value=0, max_value=20, value=0)
        with col2:
            max_deneyim = st.number_input("Max", min_value=0, max_value=20, value=20)

        top_n = st.slider("🏆 Gösterilecek Aday Sayısı", min_value=1, max_value=10, value=5)

        st.markdown("---")
        st.markdown("#### 📊 Kriter Ağırlıkları")
        st.caption("Mevcut ağırlıklar (toplam = 1.0)")
        for kriter, agirlik in KRITER_AGIRLIKLARI.items():
            st.progress(agirlik, text=f"{kriter}: {agirlik:.2f}")

        return (
            None if pozisyon == "Tümü" else pozisyon,
            min_deneyim,
            max_deneyim,
            top_n
        )

# =============================================================================
# PLOTLY GRAFİKLER
# =============================================================================
def topsis_bar_chart(sonuclar: pd.DataFrame) -> go.Figure:
    """TOPSIS skorlarını yatay bar chart olarak gösterir."""
    renkler = ["#f59e0b", "#9ca3af", "#b45309"] + ["#2d6a9f"] * (len(sonuclar) - 3)

    fig = go.Figure(go.Bar(
        x=sonuclar["TOPSIS_Skoru"].round(4),
        y=sonuclar["Ad_Soyad"],
        orientation="h",
        marker_color=renkler[:len(sonuclar)],
        text=sonuclar["TOPSIS_Skoru"].round(4),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>TOPSIS Skoru: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title="🏆 TOPSIS Sıralama Skoru",
        xaxis_title="TOPSIS Yakınlık Katsayısı",
        yaxis=dict(autorange="reversed"),
        height=350,
        margin=dict(l=20, r=60, t=50, b=30),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12)
    )
    return fig


def radar_chart(sonuclar: pd.DataFrame) -> go.Figure:
    """İlk 3 adayın kriterlerini radar chart ile karşılaştırır."""
    kriterler_goster = [
        "Teknik_Puan", "Iletisim_Puani", "Deneyim_Yili",
        "Referans_Skoru", "Sertifika_Sayisi"
    ]
    kr_etiketler = ["Teknik", "İletişim", "Deneyim", "Referans", "Sertifika"]

    fig = go.Figure()
    renkler = ["#f59e0b", "#9ca3af", "#b45309"]

    for i, (_, row) in enumerate(sonuclar.head(3).iterrows()):
        degerler = [row[k] for k in kriterler_goster]
        # Normalize (0-100 arası)
        degerler_norm = [
            degerler[0],           # Teknik_Puan zaten 0-100
            degerler[1],           # Iletisim_Puani zaten 0-100
            degerler[2] * 10,      # Deneyim_Yili → 0-100
            degerler[3] * 20,      # Referans_Skoru → 0-100
            min(degerler[4] * 20, 100)  # Sertifika_Sayisi → 0-100
        ]
        fig.add_trace(go.Scatterpolar(
            r=degerler_norm + [degerler_norm[0]],
            theta=kr_etiketler + [kr_etiketler[0]],
            fill="toself",
            name=row["Ad_Soyad"],
            line_color=renkler[i],
            opacity=0.7
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="🕸️ İlk 3 Aday Karşılaştırması",
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", y=-0.1)
    )
    return fig

# =============================================================================
# ADAY KARTLARI
# =============================================================================
def aday_karti_goster(sira: int, row: pd.Series):
    """Tek bir adayın detay kartını gösterir."""
    renk_sinif = {1: "birinci", 2: "ikinci", 3: "ucuncu"}.get(sira, "")
    rozet = {1: "🥇", 2: "🥈", 3: "🥉"}.get(sira, f"#{sira}")

    st.markdown(f"""
    <div class="aday-kart {renk_sinif}">
        <strong>{rozet} {row['Ad_Soyad']}</strong>
        &nbsp;|&nbsp; {row['Pozisyon']}
        &nbsp;|&nbsp; {row['Deneyim_Yili']} yıl deneyim
        &nbsp;|&nbsp; Eğitim: {row['Egitim_Seviyesi']}
        <br>
        <span style="color:#2d6a9f; font-weight:600;">
            TOPSIS: {row['TOPSIS_Skoru']:.4f}
        </span>
        &nbsp;|&nbsp;
        Teknik: {row['Teknik_Puan']} &nbsp;
        İletişim: {row['Iletisim_Puani']} &nbsp;
        Referans: {row['Referans_Skoru']}
        <div class="aciklama-text">💡 {row['Aciklama']}</div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# ANA UYGULAMA
# =============================================================================
def main():
    # Başlık
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:1.8em;">🧠 Doğal Dil Tabanlı İK Karar Destek Sistemi</h1>
        <p style="margin:5px 0 0 0; opacity:0.85;">
            Sentence Transformers + TOPSIS Hibrit Mimarisi ile Akıllı Personel Seçimi
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar filtreleri
    pozisyon, min_deneyim, max_deneyim, top_n = sidebar_olustur()

    # Model yükle
    try:
        df, kds = sistemi_yukle()
    except Exception as e:
        st.error(f"❌ Sistem yüklenemedi: {e}")
        return

    # Üst metrik satırı
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Toplam Aday", len(df))
    with col2:
        st.metric("📌 Pozisyon Sayısı", df["Pozisyon"].nunique())
    with col3:
        st.metric("🎓 Ort. Deneyim", f"{df['Deneyim_Yili'].mean():.1f} yıl")
    with col4:
        st.metric("⚙️ TOPSIS Kriterleri", len(KRITER_AGIRLIKLARI))

    st.markdown("---")

    # Sorgu kutusu
    st.markdown("### 💬 Doğal Dil ile Aday Arayın")
    st.caption("Örnek: *'Python bilen, en az 3 yıl deneyimli veri bilimcisi'* veya *'iletişim becerileri güçlü İK uzmanı'*")

    sorgu = st.text_area(
        label="Sorgunuzu buraya yazın:",
        placeholder="Aradığınız adayı doğal dille tanımlayın...",
        height=100,
        label_visibility="collapsed"
    )

    ara_butonu = st.button("🔍 En İyi Adayları Bul")

    # Sonuçları göster
    if ara_butonu and sorgu.strip():
        with st.spinner("⚙️ TOPSIS hesaplanıyor..."):
            try:
                sonuclar = kds.en_iyi_adaylari_bul(
                    sorgu=sorgu,
                    top_n=top_n,
                    pozisyon=pozisyon,
                    min_deneyim=min_deneyim if min_deneyim > 0 else None,
                    max_deneyim=max_deneyim if max_deneyim < 20 else None
                )
            except Exception as e:
                st.error(f"❌ Hata: {e}")
                return

        if sonuclar.empty:
            st.warning("⚠️ Kriterlere uyan aday bulunamadı. Filtreleri gevşetin.")
            return

        st.success(f"✅ {len(sonuclar)} aday bulundu ve TOPSIS ile sıralandı.")

        # Grafik + Radar yan yana
        col_bar, col_radar = st.columns(2)
        with col_bar:
            st.plotly_chart(topsis_bar_chart(sonuclar), use_container_width=True)
        with col_radar:
            if len(sonuclar) >= 2:
                st.plotly_chart(radar_chart(sonuclar), use_container_width=True)

        # Aday kartları
        st.markdown("### 📋 Aday Detayları")
        for sira, (_, row) in enumerate(sonuclar.iterrows(), start=1):
            aday_karti_goster(sira, row)

        # Detay tablosu (expander içinde)
        with st.expander("📊 Tüm Kriterleri Gösteren Detay Tablosu"):
            goster_sutunlar = [
                "Ad_Soyad", "Pozisyon", "Deneyim_Yili",
                "Egitim_Seviyesi", "Teknik_Puan", "Iletisim_Puani",
                "Referans_Skoru", "Sertifika_Sayisi",
                "Maas_Beklenti_Katsayisi", "TOPSIS_Skoru"
            ]
            st.dataframe(
                sonuclar[goster_sutunlar].style.format({"TOPSIS_Skoru": "{:.4f}"}),
                use_container_width=True
            )

    elif ara_butonu and not sorgu.strip():
        st.warning("⚠️ Lütfen bir sorgu girin.")

    # Footer
    st.markdown("---")
    st.caption("🎓 Doğal Dil Tabanlı Karar Destek Sistemi | Sentence Transformers + TOPSIS | Akademik Proje")


if __name__ == "__main__":
    main()
