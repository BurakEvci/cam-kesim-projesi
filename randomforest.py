import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Excel dosyasını oku
file_path = 'verisilinmis2.xlsx'  # Excel dosyanın yolunu buraya ekle
excel_data = pd.read_excel(file_path, sheet_name=None)  # Tüm sayfaları oku


# Sütun isimlerindeki boşlukları temizleyerek sütun adlarını kontrol et
def fix_format(s):
    s = s.replace('[', '').replace(']', '').split()
    return [int(s[0]), int(s[1])]


# İki nokta arasındaki Euclidean mesafesini hesaplayan fonksiyon
def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# Özellik çıkarımı (Euclidean ve Hausdorff Distance)
def extract_features(prev_shape, curr_shape):
    # Euclidean mesafesi hesaplama
    euclidean_distances = np.linalg.norm(prev_shape - curr_shape, axis=1)

    # Hausdorff Distance hesaplama
    hausdorff_dist1 = directed_hausdorff(prev_shape, curr_shape)[0]
    hausdorff_dist2 = directed_hausdorff(curr_shape, prev_shape)[0]
    hausdorff_distance = max(hausdorff_dist1, hausdorff_dist2)

    # Ortalama Euclidean mesafesi ve Hausdorff mesafesini özellik olarak kullan
    features = [np.mean(euclidean_distances), hausdorff_distance]
    return features


# Veri setini ve etiketleri oluştur
data = []
labels = []  # Aynı seriler için 0, farklı seriler için 1

# Tüm sayfaları dolaş
for sheet_name, df in excel_data.items():
    # Sütun isimlerindeki boşlukları tekrar temizleyelim
    df.columns = df.columns.str.strip()

    # 'Prev.' ve 'Curr.' sütunlarının isimlerinin doğru olduğunu doğrula
    if 'Prev.' in df.columns and 'Curr.' in df.columns:
        # Eksik verileri kaldır
        df = df.dropna()

        # 'Prev.' ve 'Curr.' sütunlarını temizle ve numpy array'e çevir
        prev = np.array([fix_format(x) for x in df['Prev.']])
        curr = np.array([fix_format(x) for x in df['Curr.']])

        # Özellik çıkarımı
        features = extract_features(prev, curr)
        data.append(features)

        # Aynı seri için 0, farklı seri için 1 olacak şekilde etiket ver

        if sheet_name.startswith('AYN'):
            label = 0  # Aynı seri
        else:
            label = 1  # Farklı seri

        labels.append(label)


    else:
        print(f"Sheet {sheet_name} contains unexpected column names: {df.columns}")

# Random Forest modeli
rf_model = RandomForestClassifier()

# Cross-validation (k=5) ile modelin performansını değerlendirme
scores = cross_val_score(rf_model, data, labels, cv=5)

# Her fold için doğruluk oranı ve ortalama doğruluk
print(f"Cross-validation scores: {scores}")
print(f"Average accuracy: {np.mean(scores)}")

# Modeli tüm veri seti üzerinde eğit ve tahmin yap
rf_model.fit(data, labels)
y_pred = rf_model.predict(data)

# Tüm verilerde tahminleri yazdır
for i in range(len(data)):
    actual_label = "Aynı Seri" if labels[i] == 0 else "Farklı Seri"
    predicted_label = "Aynı Seri" if y_pred[i] == 0 else "Farklı Seri"
    print(f"Veri {i + 1}: Gerçek Sonuç: {actual_label}, Model Tahmini: {predicted_label}")
