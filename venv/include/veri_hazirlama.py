# -*- coding: utf-8 -*-
"""
veri_hazirlama.py
Ham CSV'leri temizler, feature engineering yapar ve temiz CSV'leri data/ klasörüne kaydeder.
Çalıştırmak için:  python veri_hazirlama.py

GEREKLİ HAM DOSYALAR (data/ klasörüne koyun):
    data/tmdb_5000_movies.csv
    data/tmdb_5000_credits.csv
    data/Books.csv
    data/Ratings.csv
"""

import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def yol(dosya):
    return os.path.join(DATA_DIR, dosya)


# ============================================================
#  BÖLÜM 1: FİLM VERİSİ (Feature Engineering)
# ============================================================
movies  = pd.read_csv(yol("tmdb_5000_movies.csv"))
credits = pd.read_csv(yol("tmdb_5000_credits.csv"), engine='python', on_bad_lines='skip')

# Verileri isim yerine eşsiz ID'ler üzerinden birleştiriyoruz
movies = movies.merge(credits, left_on="id", right_on="movie_id")
movies.drop(columns=["title_y", "movie_id"], inplace=True)
movies.rename(columns={"title_x": "title"}, inplace=True)

movies.dropna(subset=["overview", "genres", "vote_average"], inplace=True)
movies.drop_duplicates(subset="title", inplace=True)

esik = movies["vote_count"].quantile(0.10)
movies = movies[movies["vote_count"] >= esik]

C = movies["vote_average"].mean()
m = movies["vote_count"].quantile(0.60)


def weighted_rating(row, m=m, C=C):
    v = row["vote_count"]
    R = row["vote_average"]
    return (v / (v + m)) * R + (m / (v + m)) * C


movies["weighted_rating"] = movies.apply(weighted_rating, axis=1)


def parse_genres(text):
    try:
        return [g["name"] for g in ast.literal_eval(text)]
    except:
        return []


movies["genre_list"] = movies["genres"].apply(parse_genres)

movies["popularity_score"] = (
    movies["weighted_rating"] * 0.7 +
    (movies["vote_count"] / movies["vote_count"].max()) * 100 * 0.3
)

movies["runtime"] = movies["runtime"].fillna(movies["runtime"].median())


def runtime_category(mins):
    if mins < 90:
        return "kısa"
    elif mins < 120:
        return "orta"
    else:
        return "uzun"


movies["runtime_category"] = movies["runtime"].apply(runtime_category)

print(f"📊 Toplam film sayısı   : {len(movies)}")
print(f"📊 Ortalama oy sayısı   : {movies['vote_count'].mean():.0f}")
print(f"📊 Weighted rating ort. : {movies['weighted_rating'].mean():.2f}")

movies.to_csv(yol("tmdb_cleaned.csv"), index=False)
print("✅ tmdb_cleaned.csv kaydedildi")


# ============================================================
#  BÖLÜM 2: KİTAP VERİSİ (Feature Engineering)
# ============================================================
books   = pd.read_csv(yol("Books.csv"),   encoding="latin-1", low_memory=False)
ratings = pd.read_csv(yol("Ratings.csv"), encoding="latin-1")

books.dropna(subset=["Book-Title", "Book-Author", "Year-Of-Publication"], inplace=True)
books.drop_duplicates(subset="ISBN", inplace=True)
ratings = ratings[ratings["Book-Rating"] > 0]

# Aktif kullanıcı filtrele (>= 10 rating veren)
# ISBN'leri string'e sabitle (eşleşme kaybını önler)
ratings["ISBN"] = ratings["ISBN"].astype(str)
books["ISBN"]   = books["ISBN"].astype(str)
ratings = ratings[ratings["ISBN"].isin(books["ISBN"])]

# İTERATİF filtre: eşik gerçekten sağlanana kadar tekrarla
MIN_USER, MIN_BOOK = 5, 5
prev = None
while True:
    uc = ratings["User-ID"].value_counts()
    ratings = ratings[ratings["User-ID"].isin(uc[uc >= MIN_USER].index)]
    bc = ratings["ISBN"].value_counts()
    ratings = ratings[ratings["ISBN"].isin(bc[bc >= MIN_BOOK].index)]
    if prev == len(ratings):   # değişiklik durunca yakınsadı demektir
        break
    prev = len(ratings)
rating_count = ratings.groupby("ISBN")["Book-Rating"].count().reset_index()
rating_count.columns = ["ISBN", "rating_count"]

rating_mean = ratings.groupby("ISBN")["Book-Rating"].mean().reset_index()
rating_mean.columns = ["ISBN", "rating_mean"]

books = books.merge(rating_count, on="ISBN", how="inner")
books = books.merge(rating_mean,  on="ISBN", how="inner")

books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
books["Year-Of-Publication"] = books["Year-Of-Publication"].replace(0, np.nan)
books["Year-Of-Publication"] = books["Year-Of-Publication"].fillna(books["Year-Of-Publication"].median())
books["book_age"] = 2026 - books["Year-Of-Publication"]

books["popularity_score"] = (
    books["rating_mean"] * 0.6 +
    (books["rating_count"] / books["rating_count"].max()) * 10 * 0.4
)

total_users   = ratings["User-ID"].nunique()
total_books   = ratings["ISBN"].nunique()
total_ratings = len(ratings)
sparsity      = 1 - (total_ratings / (total_users * total_books))

print(f"📊 Kullanıcı sayısı : {total_users}")
print(f"📊 Kitap sayısı     : {total_books}")
print(f"📊 Rating sayısı    : {total_ratings}")
print(f"📊 Sparsity         : %{sparsity*100:.2f}")

books.to_csv(yol("books_cleaned.csv"),     index=False)
ratings.to_csv(yol("ratings_cleaned.csv"), index=False)
print("✅ books_cleaned.csv ve ratings_cleaned.csv kaydedildi")



# ============================================================
#  GÖRSELLEŞTİRME & VERİ ÖZETİ
# ============================================================
top10 = movies.nlargest(10, "weighted_rating")[["title", "weighted_rating"]]
plt.figure(figsize=(10, 5))
plt.barh(top10["title"], top10["weighted_rating"], color="steelblue")
plt.title("En Yüksek Weighted Rating'e Sahip 10 Film")
plt.xlabel("Weighted Rating")
plt.tight_layout()
plt.savefig(yol("top10_film.png"), dpi=150)
plt.show()

top10_books = books.nlargest(10, "popularity_score")[["Book-Title", "popularity_score"]]
plt.figure(figsize=(10, 5))
plt.barh(top10_books["Book-Title"], top10_books["popularity_score"], color="coral")
plt.title("En Popüler 10 Kitap")
plt.xlabel("Popularity Score")
plt.tight_layout()
plt.savefig(yol("top10_kitap.png"), dpi=150)
plt.show()

plt.figure(figsize=(8, 4))
ratings["Book-Rating"].value_counts().sort_index().plot(kind="bar", color="green")
plt.title("Kitap Rating Dağılımı")
plt.xlabel("Rating")
plt.ylabel("Adet")
plt.tight_layout()
plt.savefig(yol("rating_dagilimi.png"), dpi=150)
plt.show()

print("=" * 40)
print("📋 VERİ ÖZETİ")
print("=" * 40)
print(f"🎬 Film sayısı       : {len(movies)}")
print(f"📚 Kitap sayısı      : {len(books)}")
print(f"👥 Aktif kullanıcı   : {total_users}")
print(f"⭐ Toplam rating     : {total_ratings}")
print(f"🕳️  Sparsity          : %{sparsity*100:.2f}")
print("=" * 40)
print("\n✅ Veri hazırlama tamamlandı. Şimdi model_analiz.py veya app.py çalıştırabilirsiniz.")

