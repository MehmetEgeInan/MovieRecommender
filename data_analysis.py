import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ratings_path = 'ml-1m/ratings.dat'
movies_path = 'ml-1m/movies.dat'
users_path = 'ml-1m/users.dat'

#Dosya oku, latin-1 encoding
ratings = pd.read_csv(ratings_path, sep='::', engine='python', names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin-1')
movies = pd.read_csv(movies_path, sep='::', engine='python', names=['movieId', 'title', 'genres'], encoding='latin-1')
users = pd.read_csv(users_path, sep='::', engine='python', names=['userId', 'gender', 'age', 'occupation', 'zip'], encoding='latin-1')

#Eksik veri kontrolü
print("Eksik veri durumu:")
print(ratings.isnull().sum())
print(movies.isnull().sum())
print(users.isnull().sum())

# Kullanıcı başına ortalama oy sayısı
ratings_per_user = ratings.groupby('userId').size()
print("Kullanıcı başına ortalama oy sayısı:", ratings_per_user.mean())

# Film başına ortalama puan
avg_rating_per_movie = ratings.groupby('movieId')['rating'].mean()

# Puan dağılımı histogramı
plt.figure(figsize=(8,5))
sns.histplot(ratings['rating'], bins=10, kde=False)
plt.title('Puan Dağılımı')
plt.xlabel('Puan')
plt.ylabel('Sayı')
plt.show()

# Türlere göre dağılım (film başına en çok olan türler)
movies['genres'] = movies['genres'].str.split('|')
all_genres = movies.explode('genres')
plt.figure(figsize=(12,6))
sns.countplot(data=all_genres, y='genres', order=all_genres['genres'].value_counts().index)
plt.title('Türlere Göre Film Sayısı')
plt.show()
