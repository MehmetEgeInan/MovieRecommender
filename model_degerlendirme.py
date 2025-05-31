import numpy as np

def rmse(true_ratings, pred_ratings):
    """
    RMSE hesaplar.
    :param true_ratings: Gerçek puanlar (list veya numpy array)
    :param pred_ratings: Tahmin edilen puanlar (list veya numpy array)
    :return: RMSE float
    """
    true_ratings = np.array(true_ratings)
    pred_ratings = np.array(pred_ratings)
    return np.sqrt(np.mean((true_ratings - pred_ratings) ** 2))

def precision_at_k(recommended, relevant, k):
    """
    Precision@K hesaplar.
    :param recommended: Modelin önerdiği film ID listesi (sıralı)
    :param relevant: Kullanıcının gerçekten beğendiği film ID listesi
    :param k: Öneri listesinde kaç film değerlendirilecek
    :return: Precision@K float
    """
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k if k > 0 else 0

def recall_at_k(recommended, relevant, k):
    """
    Recall@K hesaplar.
    :param recommended: Modelin önerdiği film ID listesi (sıralı)
    :param relevant: Kullanıcının gerçekten beğendiği film ID listesi
    :param k: Öneri listesinde kaç film değerlendirilecek
    :return: Recall@K float
    """
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0

def main():
    #Örnek gerçek ve tahmin puanları
    true_ratings = [4, 5, 3, 2, 4]
    pred_ratings = [3.8, 4.9, 2.5, 2.2, 3.9]

    print("RMSE:", rmse(true_ratings, pred_ratings))

    # Örnek öneri listesi (film ID'leri)
    recommended = [10, 20, 30, 40, 50, 60, 70]
    relevant = [20, 30, 80, 90]  #kullanıcının beğendiği filmler

    k = 5
    print(f"Precision@{k}:", precision_at_k(recommended, relevant, k))
    print(f"Recall@{k}:", recall_at_k(recommended, relevant, k))

if __name__ == "__main__":
    main()

#model çıktısı varmış gibi varsayılan veriler kullanılarak, 
#bu çıktıları RMSE, Precision@K ve Recall@K ile ölçülüyor.

#RMSE == Puan tahmin doğruluğu
#Precision@K == Önerilenler arasında isabet oranı
#Recall@K == Beğenilenler içinde önerilenlerin oranı