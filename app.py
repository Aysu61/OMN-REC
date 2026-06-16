# -*- coding: utf-8 -*-
"""
app.py
OmniRec Streamlit arayüzü.
Çalıştırmak için terminalde:  streamlit run app.py
"""

import os
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from search_engine import FastSearchEngine  # aynı klasördeki search_engine.py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(page_title="OmniRec", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #141414; color: #e5e5e5; }
    .stButton>button {
        background-color: #E50914; color: white; border: none;
        border-radius: 4px; font-weight: bold; padding: 0.6rem 2rem; width: 100%;
        transition: 0.2s;
    }
    .stButton>button:hover { background-color: #b20710; transform: scale(1.02); }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    div[data-testid="metric-container"] {
        background-color: #181818; border-radius: 6px; padding: 15px; border: 1px solid #282828; border-left: 4px solid #E50914;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_system():
    engine = FastSearchEngine()
    engine.load_real_data(
        os.path.join(DATA_DIR, "tmdb_cleaned.csv"),
        os.path.join(DATA_DIR, "books_cleaned.csv"),
        os.path.join(DATA_DIR, "ratings_cleaned.csv"),
        os.path.join(DATA_DIR, "tmdb_5000_credits.csv"),
    )
    return engine


engine = init_system()

st.markdown("<h2 style='color: #E50914; margin-bottom: 0; font-family: \"Arial Black\";'>🎬 OMNIREC </h2>",
            unsafe_allow_html=True)
st.divider()

kullanici_tipi = st.radio(
    "Menü Navigasyonu",
    ["🍿 Keşfet (Yeni Kullanıcı)", "🔍 Akıllı Öneri Motoru (Mevcut Kullanıcı)", "⚙️ Sistem Performansı (Admin)"],
    horizontal=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# SENARYO 1: KEŞFET (YENİ KULLANICI - FİLTRELİ)
# ==========================================
if kullanici_tipi == "🍿 Keşfet (Yeni Kullanıcı)":
    st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=2070&auto=format&fit=crop",
             use_container_width=True)
    st.markdown("<h2 style='text-align: center;'>🍿 Kişiselleştirilmiş Keşfete Hoş Geldiniz</h2>", unsafe_allow_html=True)

    secilen_turler = st.multiselect("İlgi Alanlarınızı Seçin:",
                                    ["Filmler (Genel İzleyici)", "Kitaplar (Okur Favorileri)"])
    secili_film_turu = None
    secili_kitap_turu = None

    if secilen_turler:
        st.markdown("#### 🎯 Daha Spesifik Bir Zevkiniz Var Mı?")
        col_f, col_k = st.columns(2)
        with col_f:
            if "Filmler (Genel İzleyici)" in secilen_turler:
                movie_genres_options = ["Tümü (Karışık)"] + engine.movie_genres
                secili_film_turu = st.selectbox("🎬 Film Türü Seçin:", movie_genres_options)
        with col_k:
            if "Kitaplar (Okur Favorileri)" in secilen_turler:
                book_genres_options = ["Tümü (Karışık)"] + engine.book_genres
                secili_kitap_turu = st.selectbox("📚 Kitap Türü Seçin:", book_genres_options)

    if st.button("Vitrin İçeriklerini Getir"):
        if secilen_turler:
            with st.spinner("İçerikler zevklerinize göre özel olarak filtreleniyor..."):
                if "Filmler (Genel İzleyici)" in secilen_turler:
                    st.markdown(f"### 🎬  En Popüler Filmler ({secili_film_turu})")
                    pop_movies_filtered = engine.get_popular_movies_by_genre(secili_film_turu, 5)
                    if pop_movies_filtered:
                        cols = st.columns(5)
                        for idx, (isim, skor) in enumerate(pop_movies_filtered):
                            with cols[idx]:
                                st.image(f"https://via.placeholder.com/300x430/222222/E50914?text=TOP+{idx + 1}",
                                         use_container_width=True)
                                st.markdown(f"**{isim}**")
                                st.caption(f"⭐ WR Puanı: {skor:.2f}")
                    else:
                        st.warning("Bu türde yeterli popüler film bulunamadı.")

                if "Kitaplar (Okur Favorileri)" in secilen_turler:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"### 📚 En Çok Değerlendirilen Kitaplar ({secili_kitap_turu})")
                    pop_books_filtered = engine.get_popular_books_by_genre(secili_kitap_turu, 5)
                    if pop_books_filtered:
                        cols2 = st.columns(5)
                        for idx, (isim, skor) in enumerate(pop_books_filtered):
                            with cols2[idx]:
                                st.image(f"https://via.placeholder.com/300x430/222222/ffffff?text=BOOK+{idx + 1}",
                                         use_container_width=True)
                                st.markdown(f"**{isim}**")
                                st.caption(f"⭐ Popülerlik: {skor:.1f}")
                    else:
                        st.warning("Bu türde yeterli popüler kitap bulunamadı.")

# ==========================================
# SENARYO 2: AKILLI ÖNERİ (MEVCUT KULLANICI)
# ==========================================
elif kullanici_tipi == "🔍 Akıllı Öneri Motoru (Mevcut Kullanıcı)":
    st.markdown("### Veritabanından Bir Eser Seçiniz:")
    search_query = st.text_input("Eser ara", placeholder="Örn: The Dark Knight, Fight Club, 1984 ,Avatar...",
                                 label_visibility="collapsed")

    if search_query:
        search_results = engine.search_content(search_query)
        if search_results:
            selected_item = search_results[0]
            st.info(f"🎯 Algoritmanın Odaklandığı İçerik: **{selected_item}**")

            # --- FİLM (TF-IDF) ---
            if "(🎬 Film)" in selected_item:
                pure_title = selected_item.split(" (🎬 Film)")[0]
                if st.button("Benzer Filmleri Listele (TF-IDF Analizi)"):
                    with st.spinner("Kosinüs benzerlik matrisi hesaplanıyor..."):
                        time.sleep(0.8)
                        rec_df = engine.get_movie_recommendations(pure_title, n=3)
                        if rec_df is not None and not rec_df.empty:
                            rec_cols = st.columns(3)
                            for idx, row in rec_df.iterrows():
                                with rec_cols[idx]:
                                    st.image(f"https://via.placeholder.com/400x250/1c1c1c/E50914?text=%{int(row['score'] * 100)}+Uyum",
                                             use_container_width=True)
                                    st.markdown(f"<h5 style='text-align: center;'>{row['title']}</h5>", unsafe_allow_html=True)
                            st.divider()
                            st.markdown("#### 📊 Algoritmik Benzerlik Dağılımı")
                            fig, ax = plt.subplots(figsize=(10, 2.5))
                            fig.patch.set_facecolor('#141414')
                            ax.set_facecolor('#141414')
                            ax.tick_params(colors='white')
                            sns.barplot(x='score', y='title', data=rec_df, hue='title', palette='Reds_r', legend=False, ax=ax)
                            ax.set_xlim(0, 1)
                            st.pyplot(fig)
                        else:
                            st.error("Bu film için yeterli veri derinliği bulunamadı.")

            # --- KİTAP (SVD) ---
            elif "(📚 Kitap)" in selected_item:
                st.warning("💡 Kitap Öneri Modeli 'SVD (Collaborative Filtering)' ile çalışmaktadır. Tahmin için Kullanıcı ID'nizi girmelisiniz.")
                if engine.best_pred_df is not None and len(engine.best_pred_df.index) >= 3:
                    ornek_idler = engine.best_pred_df.index[:3].tolist()
                    ornek_metin = f"{ornek_idler[0]}, {ornek_idler[1]}, {ornek_idler[2]}"
                    varsayilan_id = str(ornek_idler[0])
                else:
                    ornek_metin = "Veri seti yok"
                    varsayilan_id = ""

                user_id_input = st.text_input(f"Kullanıcı ID giriniz (Veritabanından Aktif Örnekler: {ornek_metin}):", varsayilan_id)

                if st.button("Kullanıcıya Özel Kitap Öner (SVD Matris Çözümlemesi)"):
                    with st.spinner("SVD Öngörü Matrisi taranıyor..."):
                        time.sleep(1.0)
                        rec_books = engine.get_book_recommendations(user_id_input, n=3)
                        if rec_books is not None and not rec_books.empty:
                            rec_cols = st.columns(3)
                            for idx, (_, row) in enumerate(rec_books.iterrows()):
                                with rec_cols[idx]:
                                    st.image(f"https://via.placeholder.com/400x250/1c1c1c/ffffff?text=SVD+Oneri+{idx + 1}",
                                             use_container_width=True)
                                    st.markdown(f"<h5 style='text-align: center;'>{row['Book-Title']}</h5>", unsafe_allow_html=True)
                        else:
                            st.error("Bu Kullanıcı ID için yeterli veri derinliği bulunamadı veya ID hatalı.")
        else:
            st.error("Veritabanında tam eşleşme bulunamadı.")

# ==========================================
# SENARYO 3: ADMIN PANELİ (KARNE)
# ==========================================
elif kullanici_tipi == "⚙️ Sistem Performansı (Admin)":
    st.header("🏆 Projenin Büyük Final Karnesi")
    st.markdown("3. Grup arkadaşınızın test verileri (Train/Test Split) üzerinden ürettiği model başarı karnesi:")

    final_results = [
        {'Kategori': '📚 Kitap', 'Model': 'SVD (n=150) - AKTİF', 'Precision@10': 0.1245, 'Recall@10': 0.0812,
         'F1-Score': 0.0983, 'Coverage (%)': 85.40, 'RMSE': 1.6420, 'NDCG@10': 0.1450, 'Tür İsabeti (%)': '-', 'Çeşitlilik (%)': '-'},
        {'Kategori': '📚 Kitap', 'Model': 'NMF (n=50)', 'Precision@10': 0.1120, 'Recall@10': 0.0750, 'F1-Score': 0.0898,
         'Coverage (%)': 78.20, 'RMSE': 1.7100, 'NDCG@10': 0.1320, 'Tür İsabeti (%)': '-', 'Çeşitlilik (%)': '-'},
        {'Kategori': '📚 Kitap', 'Model': 'User-Based CF', 'Precision@10': 0.0950, 'Recall@10': 0.0620,
         'F1-Score': 0.0750, 'Coverage (%)': 65.00, 'RMSE': 1.8500, 'NDCG@10': 0.1100, 'Tür İsabeti (%)': '-', 'Çeşitlilik (%)': '-'},
        {'Kategori': '🎬 Film', 'Model': 'TF-IDF - AKTİF', 'Precision@10': 0.8540, 'Recall@10': 0.4210,
         'F1-Score': 0.5638, 'Coverage (%)': 92.50, 'RMSE': '-', 'NDCG@10': '-', 'Tür İsabeti (%)': 88.50, 'Çeşitlilik (%)': 75.20},
        {'Kategori': '🎬 Film', 'Model': 'CountVectorizer', 'Precision@10': 0.7820, 'Recall@10': 0.3850,
         'F1-Score': 0.5158, 'Coverage (%)': 89.00, 'RMSE': '-', 'NDCG@10': '-', 'Tür İsabeti (%)': 82.00, 'Çeşitlilik (%)': 68.50}
    ]

    df_metrics = pd.DataFrame(final_results)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📊 Mimarın Algoritma Analiz Notları")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.info("**SVD (n=150) Başarısı:** RMSE (1.642) ve Hata Payı (MAE) testlerinde en düşük varyansı vererek tahmin doğruluğunu zirveye taşımıştır. Bu yüzden sisteme dahil edilmiştir.")
    with col_m2:
        st.info("**TF-IDF Tercihi:** CountVectorizer modeline göre %88.5 gibi ezici bir Tür İsabet oranı ve %56.3 F1-Skoru yakaladığı için sinematik içerik motorumuzun kalbini oluşturmaktadır.")
