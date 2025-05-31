import os
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNBasic, SVD, NormalPredictor
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict


file_path = os.path.join(os.path.dirname(__file__), 'ratings.dat')
reader = Reader(line_format='user item rating timestamp', sep='::')
data = Dataset.load_from_file(file_path, reader=reader)


#Eğitim ve Test Verisi

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# Precision@K ve Recall@K Fonksiyonları

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []
    recalls = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in top_k)

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recall = n_rel_and_rec_k / n_rel if n_rel else 0

        precisions.append(precision)
        recalls.append(recall)

    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


#Modeller, Tanımlama

models = {
    "User-based CF": KNNBasic(sim_options={'user_based': True}),
    "Item-based CF": KNNBasic(sim_options={'user_based': False}),
    "SVD": SVD(),
    "Random": NormalPredictor()
}


#SONUÇ KAYIT

results = []

for model_name, algo in models.items():
    print(f"\n🧠 Model: {model_name}")
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions, k=10, threshold=3.5)

    results.append({
        'Model': model_name,
        'RMSE': rmse,
        'Precision@10': precision,
        'Recall@10': recall
    })


#DataFrame ve Grafikler
# -----------------------------------------------------------------------
df_results = pd.DataFrame(results)
print("\n📊 Karşılaştırma Tablosu:")
print(df_results)

# RMSE grafiği
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.bar(df_results['Model'], df_results['RMSE'], color='coral')
plt.title('RMSE Karşılaştırması')
plt.ylabel('RMSE')
plt.xticks(rotation=15)

# Precision@10 grafiği
plt.subplot(1, 3, 2)
plt.bar(df_results['Model'], df_results['Precision@10'], color='skyblue')
plt.title('Precision@10')
plt.ylabel('Precision')
plt.xticks(rotation=15)

#Recall@10 grafiği
plt.subplot(1, 3, 3)
plt.bar(df_results['Model'], df_results['Recall@10'], color='lightgreen')
plt.title('Recall@10')
plt.ylabel('Recall')
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()
