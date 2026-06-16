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

## "search_engine kısmını nereye koyacağım?"
`search_engine.py` zaten ayrı bir dosya olarak **app.py ile aynı klasörde** duruyor.
`app.py` içindeki `from search_engine import FastSearchEngine` satırı, dosya aynı
klasörde olduğu için VS Code'da hatasız çalışır. Ekstra bir şey yapmana gerek yok —
sadece bu klasörü olduğu gibi VS Code'da aç.

> Not: Colab'da tek hücrede `from search_engine import ...` yazıp aynı yere sınıfı da
> koyduğun için hata veriyordu. Burada o kısım gerçek bir `.py` dosyasına ayrıldı.

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

