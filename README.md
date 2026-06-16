# OmniRec — Film & Kitap Öneri Sistemi (VS Code sürümü)

## Klasör yapısı
```
omnirec/
├── data/                     # Ham + temiz CSV'ler buraya
│   ├── tmdb_5000_movies.csv
│   ├── tmdb_5000_credits.csv
│   ├── Books.csv
│   └── Ratings.csv
├── veri_hazirlama.py         # 1) Veri temizleme + feature engineering
├── model_analiz.py           # 2) Model karşılaştırma + final karne (opsiyonel)
├── search_engine.py          # Öneri motoru sınıfı (app.py bunu import eder)
├── app.py                    # 3) Streamlit arayüzü
└── requirements.txt
```


## Kurulum
1. Klasörü VS Code'da aç.
2. (Önerilir) Sanal ortam:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. Kütüphaneleri yükle:
   ```bash
   pip install -r requirements.txt
   ```
4. Ham CSV'leri `data/` klasörüne koy.

## Çalıştırma sırası
```bash
# 1) Temiz CSV'leri üret (bir kez)
python veri_hazirlama.py

# 2) (opsiyonel) Model analizini ve karneyi gör
python model_analiz.py

# 3) Arayüzü başlat
streamlit run app.py
```
`streamlit run app.py` komutu tarayıcıda otomatik bir sayfa açar (genelde
http://localhost:8501).

