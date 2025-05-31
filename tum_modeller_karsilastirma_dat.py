import os
from surprise import Dataset, Reader, KNNBasic, SVD, NormalPredictor
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict


#Veri Seti Yükleme

file_path = os.path.join(os.path.dirname(__file__), 'ratings.dat')
reader = Reader(line_format='user item rating timestamp', sep='::')
data = Dataset.load_from_file(file_path, reader=reader)


#  Eğitim ve Test Verisi

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


#Precision@K(DOĞRULUK) ve Recall@K(KAPSANMA) Fonksiyonları /DOĞRULUK ÖLÇME
# --------------------------
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


#Modelleri Tanımlama

models = {
    "User-based CF": KNNBasic(sim_options={'user_based': True}),
    "Item-based CF": KNNBasic(sim_options={'user_based': False}),
    "SVD": SVD(),
    "Random": NormalPredictor()
}


#Her Modeli Eğit, Test Et ve Değerlendir

for model_name, algo in models.items():
    print(f"\n🧠 Model: {model_name}")

    # Eğit
    algo.fit(trainset)

    # Test
    predictions = algo.test(testset)

    # RMSE(root mean squared error/MODELİN HATA ORANINI ÖLÇEN METRİK)
    rmse = accuracy.rmse(predictions, verbose=False)

    # Precision@K, Recall@K  (kalita/isabet  ,  kapsam)
    precision, recall = precision_recall_at_k(predictions, k=10, threshold=3.5)

    # Sonuçları Yazdır
    print(f"📊 RMSE: {rmse:.4f}")
    print(f"🎯 Precision@10: {precision:.4f}")
    print(f"📥 Recall@10: {recall:.4f}")

#Content-based filtering için film türü, 
#açıklama gibi metin veya içerik verisi gerekir, bu veriler Surprise'ta kullanılmaz.

#Deep learning modelleri ise PyTorch gibi framework'lerle özel olarak kurulmalıdır,
#Surprise bunu desteklemez. 

#Surprise kütüphanesi, sadece kullanıcı-puan verisiyle 
#çalışan klasik modelleri (CF, SVD vs.) destekler.