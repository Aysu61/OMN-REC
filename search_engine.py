# -*- coding: utf-8 -*-
"""
search_engine.py
OmniRec öneri motoru. app.py bu dosyadan `FastSearchEngine` sınıfını import eder.
Bu dosya app.py ile AYNI klasörde olmalıdır.
"""

import os
import ast
import warnings
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Verilerin bulunduğu klasör (bu dosyanın yanındaki "data" klasörü)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# --- Yardımcı Fonksiyonlar ---
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


def weighted_soup(row):
    director = (str(row.get('director', '')) + ' ') * 3
    genres_val = row.get('genre_list', [])
    if not isinstance(genres_val, list):
        try:
            genres_val = ast.literal_eval(str(genres_val))
        except:
            genres_val = []
    genres = (' '.join(genres_val) + ' ') * 2
    cast_val = row.get('cast_list', [])
    cast = (' '.join(cast_val) + ' ') * 2
    keywords_val = row.get('keywords_list', [])
    keywords = ' '.join(keywords_val)
    overview = str(row.get('overview', '')) if pd.notna(row.get('overview')) else ''
    return f"{director}{genres}{cast}{keywords} {overview}"


class FastSearchEngine:
    def __init__(self):
        self.content_database = {}
        self.popular_movies = []
        self.popular_books = []
        self.movies_df = None
        self.books_df = None
        self.cosine_sim = None
        self.movie_indices = None
        self.user_item_matrix = None
        self.best_pred_df = None
        self.user_means = None
        self.movie_genres = []
        self.book_genres = []

    def load_dummy_data(self):
        self.popular_movies = [("Inception", 8.8), ("Interstellar", 8.6), ("The Matrix", 8.7)]
        self.popular_books = [("1984", 4.8), ("Dune", 4.7)]
        self.movie_genres = ["Science Fiction", "Action", "Adventure"]
        self.book_genres = ["Fiction", "Science Fiction", "Dystopia"]
        for name, _ in self.popular_movies:
            self.content_database[name.lower()] = f"{name} (🎬 Film)"
        for name, _ in self.popular_books:
            self.content_database[name.lower()] = f"{name} (📚 Kitap)"

    def load_real_data(self, movies_path=None, books_path=None,
                       ratings_path=None, credits_path=None):
        # Yol verilmediyse data/ klasöründeki dosyaları kullan
        movies_path  = movies_path  or os.path.join(DATA_DIR, "tmdb_cleaned.csv")
        books_path   = books_path   or os.path.join(DATA_DIR, "books_cleaned.csv")
        ratings_path = ratings_path or os.path.join(DATA_DIR, "ratings_cleaned.csv")
        credits_path = credits_path or os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

        if not (os.path.exists(movies_path) and os.path.exists(books_path) and os.path.exists(ratings_path)):
            print("⚠️ Temiz CSV'ler bulunamadı, demo verisi yükleniyor. Önce veri_hazirlama.py çalıştırın.")
            self.load_dummy_data()
            return False

        print("📥 Gerçek veri setleri yükleniyor...")
        df_movies = pd.read_csv(movies_path, low_memory=False)
        df_books = pd.read_csv(books_path, low_memory=False)
        df_ratings = pd.read_csv(ratings_path, low_memory=False)
        self.books_df = df_books

        for movie_name in df_movies['title'].dropna().values:
            self.content_database[str(movie_name).lower()] = f"{movie_name} (🎬 Film)"
        for book_name in df_books['Book-Title'].dropna().values:
            self.content_database[str(book_name).lower()] = f"{book_name} (📚 Kitap)"

        if 'popularity_score' in df_movies.columns:
            top_movies = df_movies.sort_values(by='popularity_score', ascending=False).head(5)
            self.popular_movies = list(zip(top_movies['title'], top_movies['weighted_rating']))
        if 'popularity_score' in df_books.columns:
            top_books = df_books.sort_values(by='popularity_score', ascending=False).head(5)
            self.popular_books = list(zip(top_books['Book-Title'], top_books['popularity_score']))

        if 'genre_list' in df_movies.columns:
            all_m_genres = set()
            for g in df_movies['genre_list'].dropna():
                try:
                    g_list = ast.literal_eval(str(g)) if not isinstance(g, list) else g
                    all_m_genres.update(g_list)
                except:
                    pass
            self.movie_genres = sorted(list(all_m_genres))

        if 'genre' in df_books.columns:
            self.book_genres = sorted(
                df_books['genre']
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            self.book_genres = [
                "Fiction",
                "Fantasy",
                "Mystery",
                "Romance",
                "Historical",
                "Biography",
                "Classic",
                "Science Fiction"
            ]

        if os.path.exists(credits_path) and 'cast' not in df_movies.columns:
            df_credits = pd.read_csv(credits_path, engine='python', on_bad_lines='skip')
            if 'title' in df_credits.columns and 'title' in df_movies.columns:
                df_movies = df_movies.merge(df_credits[['title', 'cast', 'crew']], on='title', how='left')

        if 'cast' not in df_movies.columns:
            df_movies['cast'] = df_movies['cast_x'] if 'cast_x' in df_movies.columns else "[]"
        if 'crew' not in df_movies.columns:
            df_movies['crew'] = df_movies['crew_x'] if 'crew_x' in df_movies.columns else "[]"

        df_movies['cast_list'] = df_movies['cast'].apply(lambda x: parse_list(x)[:3] if pd.notna(x) else [])
        df_movies['director'] = df_movies['crew'].apply(get_director) if 'crew' in df_movies.columns else ""
        if 'keywords' not in df_movies.columns:
            df_movies['keywords'] = "[]"
        df_movies['keywords_list'] = df_movies['keywords'].apply(lambda x: parse_list(x) if pd.notna(x) else [])

        df_movies['soup'] = df_movies.apply(weighted_soup, axis=1)
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df_movies['soup'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.movies_df = df_movies.reset_index(drop=True)
        self.movie_indices = pd.Series(self.movies_df.index, index=self.movies_df['title'])

        self.user_item_matrix = df_ratings.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
        train_data, _ = train_test_split(df_ratings, test_size=0.2, random_state=42)
        train_matrix = train_data.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0).astype(float)

        train_matrix_centered = train_matrix.copy()
        train_matrix_centered[train_matrix_centered == 0] = np.nan
        self.user_means = train_matrix_centered.mean(axis=1)
        train_matrix_centered = train_matrix_centered.sub(self.user_means, axis=0).fillna(0)

        svd = TruncatedSVD(n_components=50, random_state=42)
        train_svd = svd.fit_transform(train_matrix_centered)
        reconstructed = np.dot(train_svd, svd.components_)
        self.best_pred_df = pd.DataFrame(reconstructed, index=train_matrix.index, columns=train_matrix.columns)
        return True

    def search_content(self, query):
        query_lower = query.lower().strip()
        return [display_name for key, display_name in self.content_database.items() if query_lower in key][:5]

    def get_popular_movies_by_genre(self, genre_filter=None, n=5):
        if not genre_filter or genre_filter == "Tümü (Karışık)":
            return self.popular_movies[:n]
        filtered = []
        if 'popularity_score' in self.movies_df.columns:
            sorted_df = self.movies_df.sort_values(by='popularity_score', ascending=False)
        else:
            sorted_df = self.movies_df
        for _, row in sorted_df.iterrows():
            try:
                g_list = ast.literal_eval(str(row['genre_list'])) if not isinstance(row['genre_list'], list) else row['genre_list']
                if genre_filter in g_list:
                    filtered.append((row['title'], row.get('weighted_rating', 0)))
            except:
                continue
            if len(filtered) == n:
                break
        return filtered

    def get_popular_books_by_genre(self, genre_filter=None, n=5):
        if self.books_df is None:
            return []

        if not genre_filter or genre_filter == "Tümü (Karışık)":
            return self.popular_books[:n]

        if 'genre' not in self.books_df.columns:
            return []

        filtered_df = self.books_df[
            self.books_df['genre'] == genre_filter
        ].sort_values(
            by='popularity_score',
            ascending=False
        )

        return list(
            zip(
                filtered_df['Book-Title'].head(n),
                filtered_df['popularity_score'].head(n)
            )
        )

    def get_movie_recommendations(self, title, n=3, genre_filter=None):
        if self.movie_indices is None or title not in self.movie_indices:
            return None
        idx = self.movie_indices[title]
        sim_scores = sorted(list(enumerate(self.cosine_sim[idx])), key=lambda x: x[1], reverse=True)
        rec_data = []
        for i, score in sim_scores[1:]:
            row = self.movies_df.iloc[i]
            if genre_filter:
                try:
                    g_list = ast.literal_eval(str(row['genre_list'])) if not isinstance(row['genre_list'], list) else row['genre_list']
                    if genre_filter not in g_list:
                        continue
                except:
                    continue
            rec_data.append({'title': row['title'], 'score': round(score, 2)})
            if len(rec_data) == n:
                break
        return pd.DataFrame(rec_data)

    def get_book_recommendations(self, user_id, n=3, genre_filter=None):
        try:
            user_id = int(user_id)
        except:
            return None
        if self.best_pred_df is None or user_id not in self.best_pred_df.index:
            return None
        okunan = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
        user_preds = (self.best_pred_df.loc[user_id] + self.user_means[user_id]).clip(1, 10)
        oneriler = user_preds[~user_preds.index.isin(okunan)].sort_values(ascending=False)
        rec_books = []
        for isbn, score in oneriler.items():
            book_row = self.books_df[self.books_df['ISBN'] == isbn]
            if book_row.empty:
                continue
            title = book_row['Book-Title'].values[0]
            if genre_filter and 'genre' in book_row.columns:
                if book_row['genre'].values[0] != genre_filter:
                    continue
            if title not in [b['Book-Title'] for b in rec_books]:
                rec_books.append({
                    'Book-Title': title,
                    'score': round(float(score), 2)
                })  
            if len(rec_books) == n:
                break
        return pd.DataFrame(rec_books)