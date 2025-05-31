import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Veri dosyaları yükleme
movies = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python',
                     names=['movieId', 'title', 'genres'], encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                      names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin-1')

#Kullanıcı-film puan matrisi
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Filmler arası benzerlik (cosine similarity) — NaN'leri sıfırla
item_similarity = cosine_similarity(user_movie_matrix.T.fillna(0))

#DataFrame olarak kaydet
item_similarity_df = pd.DataFrame(item_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

#Belirli bir film için benzer filmleri öner
def recommend_similar_movies(movie_title, movies_df, similarity_df, top_n=5):
    
    # movieId'yi bul
    movie_id = movies_df[movies_df['title'].str.contains(movie_title, case=False, regex=False)]
    if movie_id.empty:
        print("Film bulunamadı.")
        return
    movie_id = movie_id.iloc[0]['movieId']

    # Benzer filmleri sırala
    similar_scores = similarity_df[movie_id].sort_values(ascending=False)
    similar_scores = similar_scores.drop(movie_id)  # kendisini çıkar

    top_movies = movies_df[movies_df['movieId'].isin(similar_scores.head(top_n).index)]
    print(f"\n'{movie_title}' filmine benzer öneriler:")
    for title in top_movies['title']:
        print("•", title)

#FİLM YAZMA-ÖNERİ
recommend_similar_movies("Toy Story", movies, item_similarity_df)
