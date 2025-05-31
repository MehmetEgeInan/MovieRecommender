import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


movies = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python',
                     names=['movieId', 'title', 'genres'], encoding='latin-1')

ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                      names=['userId', 'movieId', 'rating', 'timestamp'],
                      encoding='latin-1')

#Kullanıcı-film matrisi oluşturma
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

#Matris faktörizasyonu için SVD uygula
R = user_movie_matrix.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# 50 latent faktör ile SVD
U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)

# Tahmin edilen puanları hesapla
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_movie_matrix.columns)

# Öneri fonksiyonu
def recommend_movies_svd(user_id, preds_df, movies_df, original_ratings_df, num_recommendations=5):
    # Kullanıcı indexi (userId 1'den başlayıp index 0'dan başlıyor)
    user_row_number = user_id - 1
    
    # Tahmin edilen puanları sırala
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)

    # Kullanıcının zaten oy verdiği filmler
    user_data = original_ratings_df[original_ratings_df.userId == user_id]
    user_history = user_data.merge(movies_df, on='movieId').sort_values(by='rating', ascending=False)

    print(f"User {user_id} daha önce oy verdiği filmler:")
    for title, rating in zip(user_history['title'], user_history['rating']):
        print(f"• {title} : {rating}")

    # Kullanıcının izlemediği filmler için tahmini puanlar
    preds = pd.DataFrame(sorted_user_predictions).reset_index()
    preds.columns = ['movieId', 'PredictedRating']

    recommendations = movies_df[~movies_df['movieId'].isin(user_data['movieId'])]
    recommendations = recommendations.merge(preds, on='movieId')
    recommendations = recommendations.sort_values('PredictedRating', ascending=False).head(num_recommendations)

    print(f"\nUser {user_id} için önerilen filmler:")
    for title in recommendations['title']:
        print("•", title)

# USER GİRME/ USER=1 ----------------------------------------
recommend_movies_svd(1, preds_df, movies, ratings, num_recommendations=5)
