import pandas as pd
import os
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split





# ratings.dat
ratings_path = os.path.join(os.path.dirname(__file__), "ratings.dat")
reader = Reader(line_format="user item rating timestamp", sep="::")
data = Dataset.load_from_file(ratings_path, reader=reader)

# movies.dat -> DataFrame DF olarak oku
movies_path = os.path.join(os.path.dirname(__file__), "movies.dat")
movies_df = pd.read_csv(movies_path, sep="::", engine="python", encoding="latin1",
                        names=["movieId", "title", "genres"])


# Eğitim Verisi
# ----------------------------------------------
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


#User-Based CF Model

sim_options = {
    "name": "cosine",
    "user_based": True  # USER-BASED CF
}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)


# KULLANICI GİR------------------------------------------------

user_id = "75"  

# Kullanıcının zaten oyladığı filmleri çıkar
user_inner_id = trainset.to_inner_uid(user_id)
rated_items = {trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[user_inner_id]}

# Tüm film ID'lerini al (string formatta)
all_movie_ids = set(movies_df["movieId"].astype(str).tolist())

# Önerilecek filmleri al
unrated_items = all_movie_ids - rated_items

# Her birine tahmini puan hesapla
predictions = []
for movie_id in unrated_items:
    pred = algo.predict(user_id, movie_id)
    predictions.append((movie_id, pred.est))

# Tahminlere göre sırala
top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]

# Film adlarını getir
print(f"\n🎬 Kullanıcı {user_id} için En İyi 10 Film Önerisi:")
for movie_id, est_rating in top_n:
    title = movies_df[movies_df["movieId"] == int(movie_id)]["title"].values[0]
    print(f"{title} - Tahmini Puan: {est_rating:.2f}")
