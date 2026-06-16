# -*- coding: utf-8 -*-
"""
model_analiz.py
3. Kısım: Model geliştirme, karşılaştırma ve final karne.
ÖNCE veri_hazirlama.py çalıştırılmış olmalı (temiz CSV'lere ihtiyaç var).
Çalıştırmak için:  python model_analiz.py
"""

import os
import ast
import math
import random
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def yol(dosya):
    return os.path.join(DATA_DIR, dosya)


# ============================================================
#  TEMİZ VERİLERİ YÜKLE
# ============================================================
books   = pd.read_csv(yol("books_cleaned.csv"))
ratings = pd.read_csv(yol("ratings_cleaned.csv"))
movies  = pd.read_csv(yol("tmdb_cleaned.csv"))
credits = pd.read_csv(yol("tmdb_5000_credits.csv"), engine='python', on_bad_lines='skip')

print("Books:", books.shape, "| Ratings:", ratings.shape, "| Movies:", movies.shape)


def parse_list(x):
    try:
        return [i['name'] for i in ast.literal_eval(str(x))]
    except:
        return []


def get_director(x):
    try:
        for i in ast.literal_eval(str(x)):
            if i['job'] == 'Director':
                return i['name']
    except:
        return ''
    return ''


movies = movies.drop(columns=['cast', 'crew'], errors='ignore')
movies = movies.merge(credits[['title', 'cast', 'crew']], on='title', how='left')

movies['cast_list']     = movies['cast'].apply(lambda x: parse_list(x)[:3])
movies['director']      = movies['crew'].apply(get_director)
movies['keywords_list'] = movies['keywords'].apply(parse_list)
print("✅ Parse tamamlandı")


# ============================================================
#  KİTAP MATRİSİ + SVD (n=150) BASELINE
# ============================================================
user_item_matrix = ratings.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
print("Matris boyutu:", user_item_matrix.shape)

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_matrix = train_data.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0).astype(float)

train_matrix_centered = train_matrix.copy()
train_matrix_centered[train_matrix_centered == 0] = np.nan
user_means = train_matrix_centered.mean(axis=1)
train_matrix_centered = train_matrix_centered.sub(user_means, axis=0).fillna(0)

svd = TruncatedSVD(n_components=150, random_state=42)
train_svd = svd.fit_transform(train_matrix_centered)
best_pred_df = pd.DataFrame(np.dot(train_svd, svd.components_), index=train_matrix.index, columns=train_matrix.columns)
print(f"✅ SVD tamamlandı | Açıklanan varyans: %{round(svd.explained_variance_ratio_.sum()*100, 2)}")

y_true, y_pred = [], []
for _, row in test_data.iterrows():
    uid, isbn, true_r = row['User-ID'], row['ISBN'], row['Book-Rating']
    if uid in best_pred_df.index and isbn in best_pred_df.columns:
        pred = max(1, min(10, best_pred_df.loc[uid, isbn] + user_means[uid]))
        y_true.append(true_r)
        y_pred.append(pred)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
print(f"📊 SVD baseline -> RMSE: {rmse:.4f} | MAE: {mae:.4f}")


# ============================================================
#  FİLM TF-IDF
# ============================================================
def weighted_soup(row):
    director = (str(row['director']) + ' ') * 3
    genres   = (' '.join(row['genre_list'] if isinstance(row['genre_list'], list)
                         else ast.literal_eval(str(row['genre_list']))) + ' ') * 2
    cast     = (' '.join(row['cast_list']) + ' ') * 2
    keywords = ' '.join(row['keywords_list'])
    overview = str(row['overview']) if pd.notna(row['overview']) else ''
    return f"{director}{genres}{cast}{keywords} {overview}"


movies['soup'] = movies.apply(weighted_soup, axis=1)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies['soup'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movies = movies.reset_index(drop=True)
indices = pd.Series(movies.index, index=movies['title'])
print("✅ TF-IDF hazır, matris:", cosine_sim.shape)


# ============================================================
#  3. KISIM — SVD BİLEŞEN SAYISI ARAMASI (F1 SWEET-SPOT)
# ============================================================
def evaluate_model(pred_df, test_df, k=20, threshold=7.0): # k=20 YAPILDI
    y_t, y_p, precisions, recalls = [], [], [], []
    test_users = test_df['User-ID'].unique()

    for _, row in test_df.iterrows():
        uid, isbn, true_r = row['User-ID'], row['ISBN'], row['Book-Rating']
        if uid in pred_df.index and isbn in pred_df.columns:
            y_t.append(true_r)
            y_p.append(max(1, min(10, pred_df.loc[uid, isbn])))

    rmse_ = np.sqrt(mean_squared_error(y_t, y_p))
    mae_ = np.mean(np.abs(np.array(y_t) - np.array(y_p)))

    for uid in test_users:
        if uid not in pred_df.index:
            continue
        user_test_data = test_df[test_df['User-ID'] == uid]
        relevant_items = set(user_test_data[user_test_data['Book-Rating'] >= threshold]['ISBN'])
        if len(relevant_items) == 0:
            continue
        top_k_items = set(pred_df.loc[uid].nlargest(k).index)
        hits = len(top_k_items.intersection(relevant_items))
        precisions.append(hits / k)
        recalls.append(hits / len(relevant_items))

    avg_p = np.mean(precisions) if precisions else 0
    avg_r = np.mean(recalls) if recalls else 0
    return rmse_, mae_, avg_p, avg_r


print("\n⏳ SVD için en iyi F1 dengesi aranıyor...")
n_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 150, 200]
svd_search_results = []

for n in n_list:
    s = TruncatedSVD(n_components=n, random_state=42)
    ts = s.fit_transform(train_matrix_centered)
    pdf = pd.DataFrame(np.dot(ts, s.components_), index=train_matrix_centered.index,
                       columns=train_matrix_centered.columns).add(user_means, axis=0)
    _, _, prec, rec = evaluate_model(pdf, test_data, k=20) # k=20 YAPILDI
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    svd_search_results.append({'n': n, 'Precision@20': prec, 'Recall@20': rec, 'F1_Score': f1})

df_search = pd.DataFrame(svd_search_results)
best_n = df_search.loc[df_search['F1_Score'].idxmax()]['n']
print(f"🏆 En iyi denge n={int(best_n)}")
print(df_search.round(4).to_string(index=False))

plt.figure(figsize=(10, 6))
plt.plot(df_search['n'], df_search['Precision@20'], marker='o', linewidth=2, label='Precision@20', color='#66b3ff')
plt.plot(df_search['n'], df_search['Recall@20'], marker='s', linestyle='--', linewidth=2, label='Recall@20', color='#99ff99')
plt.plot(df_search['n'], df_search['F1_Score'], marker='^', linewidth=3, label='F1-Score', color='#ff9999')
plt.axvline(x=best_n, color='red', linestyle=':', linewidth=2, label=f'Optimum n = {int(best_n)}')
plt.title('SVD Bileşen Sayısına Göre Öneri Başarısı (k=20)', fontweight='bold', fontsize=14)
plt.xlabel('n_components', fontweight='bold')
plt.ylabel('Skor', fontweight='bold')
plt.xticks(n_list)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(yol("svd_sweet_spot.png"), dpi=300)
plt.show()


# ============================================================
#  ALGORİTMA KARŞILAŞTIRMASI (SVD / NMF / CF)
# ============================================================
results = []
print("\n⏳ Algoritmalar yarışıyor...")

# SVD (n=35)
svd35 = TruncatedSVD(n_components=35, random_state=42)
pred_df_svd_35 = pd.DataFrame(np.dot(svd35.fit_transform(train_matrix_centered), svd35.components_),
                              index=train_matrix_centered.index, columns=train_matrix_centered.columns).add(user_means, axis=0)
rmse_, mae_, prec, rec = evaluate_model(pred_df_svd_35, test_data, k=20)
results.append({'Algoritma': 'SVD (Truncated)', 'RMSE': rmse_, 'MAE': mae_, 'Precision@20': prec, 'Recall@20': rec})

# NMF (n=50)
nmf_clean = NMF(n_components=50, init='nndsvd', max_iter=500, random_state=42)
pred_df_nmf_clean = pd.DataFrame(np.dot(nmf_clean.fit_transform(train_matrix), nmf_clean.components_),
                                 index=train_matrix.index, columns=train_matrix.columns)
rmse_, mae_, prec, rec = evaluate_model(pred_df_nmf_clean, test_data, k=20)
results.append({'Algoritma': 'NMF', 'RMSE': rmse_, 'MAE': mae_, 'Precision@20': prec, 'Recall@20': rec})

# User-Based CF
user_similarity = cosine_similarity(train_matrix)
np.fill_diagonal(user_similarity, 0)
user_sim_pos = np.clip(user_similarity, 0, None)
pred_cf = user_sim_pos.dot(train_matrix) / (user_sim_pos.sum(axis=1, keepdims=True) + 1e-8)
pred_df_cf = pd.DataFrame(pred_cf, index=train_matrix.index, columns=train_matrix.columns)
rmse_, mae_, prec, rec = evaluate_model(pred_df_cf, test_data, k    =20)        