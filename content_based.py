import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Veri setini yükleme .DAT SETİ OLDUĞU İÇİN :: İLE AYIRDIM
movies = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python',
                     names=['movieId', 'title', 'genres'], encoding='latin-1')

ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                      names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin-1')

users = pd.read_csv('ml-1m/users.dat', sep='::', engine='python',
                    names=['userId', 'gender', 'age', 'occupation', 'zip'], encoding='latin-1')

#Filmlerin türünü kullanarak TF-IDF matrisi
tfidf = TfidfVectorizer(stop_words='english')

# genres sütun
tfidf_matrix = tfidf.fit_transform(movies['genres'])

#Film benzerlik matrisini hesapla (cosine similarity)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Film başlıklarına göre indeks 
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Öneri fonk
def get_recommendations(title, cosine_sim=cosine_sim):
    # Girilen filmin indeksi
    idx = indices[title]

    # Tüm filmlerin benzerlik skorları
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Benzerlik skorlarına göre sıralama (yüksekten düşüğe)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # İlk filmi çıkar (kendisi)
    sim_scores = sim_scores[1:11]

    # Film indekslerini al
    movie_indices = [i[0] for i in sim_scores]

    # Benzer filmleri döndür
    return movies['title'].iloc[movie_indices]

#ÖNERİ FİLM
print("... filmine benzer öneriler:")

#YAZACAĞI FİLM

#print(get_recommendations('Toy Story (1995)'))
print(get_recommendations('GoldenEye (1995)'))