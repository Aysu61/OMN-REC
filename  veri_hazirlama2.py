### Kodda Yapılan Değişikliklerin Özeti (Kitap Bölümü):
```python
# ============================================================
#  BÖLÜM 2: KİTAP VERİSİ (Tür İyileştirme & Feature Engineering)
# ============================================================
books   = pd.read_csv(yol("Books.csv"), low_memory=False)
ratings = pd.read_csv(yol("Ratings.csv"))

# Kitap başlıklarından daha zengin türler türetmek için fonksiyon
def basliktan_tur_bul(title):
    if not isinstance(title, str): return "General"
    title_lower = title.lower()
    
    # Genişletilmiş anahtar kelime haritası
    tur_haritasi = {
        'Fiction': ['novel', 'stories', 'fiction', 'tales', 'classic'],
        'Mystery & Thriller': ['mystery', 'murder', 'thriller', 'crime', 'suspense', 'detective', 'death'],
        'Romance': ['love', 'romance', 'heart', 'marriage', 'passion'],
        'Sci-Fi & Fantasy': ['science fiction', 'fantasy', 'magic', 'witch', 'dragon', 'star', 'alien'],
        'History & Biography': ['history', 'biography', 'historical', 'war', 'ancient', 'king', 'queen'],
        'Science & Tech': ['science', 'computer', 'data', 'physics', 'mathematics', 'technology'],
        'Self-Help & Health': ['health', 'diet', 'motivation', 'life', 'guide', 'self']
    }
    
    for tur, anahtar_kelimeler in tur_haritasi.items():
        if any(keyword in title_lower for keyword in anahtar_kelimeler):
            return tur
    return "Fiction" # Varsayılan zengin kategori

# Dinamik tür ataması gerçekleştiriliyor
books['genre'] = books['Book-Title'].apply(basliktan_tur_bul)