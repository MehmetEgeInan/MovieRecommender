import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Current working directory:", os.getcwd())


# Veri yükle
movies = pd.read_csv('movies.dat', sep='::', engine='python', encoding='latin1',
                     names=['movieId', 'title', 'genres'])
ratings = pd.read_csv('ratings.dat', sep='::', engine='python',
                      names=['userId', 'movieId', 'rating', 'timestamp'])
users = pd.read_csv('users.dat', sep='::', engine='python',
                    names=['userId', 'gender', 'age', 'occupation', 'zip'])

#ID'leri sıfırdan başlayan indexlere çevir
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()

user2idx = {user: idx for idx, user in enumerate(user_ids)}
movie2idx = {movie: idx for idx, movie in enumerate(movie_ids)}

ratings['user'] = ratings['userId'].map(user2idx)
ratings['movie'] = ratings['movieId'].map(movie2idx)

#Eğitim ve test verisini ayır
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

#  Tensorlara çevir
train_user = torch.LongTensor(train['user'].values)
train_movie = torch.LongTensor(train['movie'].values)
train_rating = torch.FloatTensor(train['rating'].values)

test_user = torch.LongTensor(test['user'].values)
test_movie = torch.LongTensor(test['movie'].values)
test_rating = torch.FloatTensor(test['rating'].values)

## Model tanımı
class DeepRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embed_size=50):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_size)
        self.movie_embed = nn.Embedding(num_movies, embed_size)
        self.fc1 = nn.Linear(embed_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, user, movie):
        user_emb = self.user_embed(user)
        movie_emb = self.movie_embed(movie)
        x = torch.cat([user_emb, movie_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

# Model, loss ve optimizer
num_users = len(user2idx)
num_movies = len(movie2idx)
model = DeepRecommender(num_users, num_movies)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Eğitim döngüsü
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_user, train_movie)
    loss = loss_fn(outputs, train_rating)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

#Test performansı
model.eval()
with torch.no_grad():
    test_preds = model(test_user, test_movie)
    test_loss = loss_fn(test_preds, test_rating)
    print(f"Test MSE Loss: {test_loss.item():.4f}")

#USER ID GİRME -----------------------
user_id = 0
user_idx = torch.LongTensor([user_id] * num_movies)
movie_idx = torch.LongTensor(list(range(num_movies)))

model.eval()
with torch.no_grad():
    scores = model(user_idx, movie_idx)

_, top5_idx = torch.topk(scores, 5)
print(f"Kullanıcı {user_id} için önerilen ilk 5 film indeksleri:")
print(top5_idx.numpy())

print("Film isimleri:")
print(movies.loc[movies['movieId'].isin([movie_ids[i] for i in top5_idx.numpy()]), 'title'].values)
    