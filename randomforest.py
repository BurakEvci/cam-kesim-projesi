# import numpy as np
# import pandas as pd
# import math
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import directed_hausdorff
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
#
# # Excel dosyasını oku
# file_path = 'verisilinmis.xlsx'  # Excel dosyanın yolunu buraya ekle
# excel_data = pd.read_excel(file_path, sheet_name=None)  # Tüm sayfaları oku
#
#
# # Sütun isimlerindeki boşlukları temizleyerek sütun adlarını kontrol et
# def fix_format(s):
#     s = s.replace('[', '').replace(']', '').split()
#     return [int(s[0]), int(s[1])]
#
#
# # İki nokta arasındaki Euclidean mesafesini hesaplayan fonksiyon
# def euclidean_distance(p1, p2):
#     return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
#
#
# # Her iki şekil için kenar uzunluklarını hesapla
# def calculate_sides(shape):
#     return [euclidean_distance(shape[i], shape[(i + 1) % len(shape)]) for i in range(len(shape))]
#
#
# # Kenar uzunluklarının ortalamasını al
# def average_side_length(sides):
#     return sum(sides) / len(sides)
#
#
# # Özellik çıkarımı (örnek veri seti)
# def extract_features(prev_shape, curr_shape):
#     # Euclidean mesafe hesaplama
#     euclidean_distances = np.linalg.norm(prev_shape - curr_shape, axis=1)
#
#     # Hausdorff mesafesi hesaplama
#     hausdorff_dist1 = directed_hausdorff(prev_shape, curr_shape)[0]
#     hausdorff_dist2 = directed_hausdorff(curr_shape, prev_shape)[0]
#     hausdorff_distance = max(hausdorff_dist1, hausdorff_dist2)
#
#     # Ortalama X ve Y koordinatları
#     avg_prev_x, avg_prev_y = np.mean(prev_shape, axis=0)
#     avg_curr_x, avg_curr_y = np.mean(curr_shape, axis=0)
#
#     # Özellikleri birleştir
#     features = [np.mean(euclidean_distances), hausdorff_distance, avg_prev_x, avg_prev_y, avg_curr_x, avg_curr_y]
#     return features
#
#
# # Veri setini ve etiketleri oluştur
# data = []
# labels = []  # Aynı seriler için 0, farklı seriler için 1
#
# # Tüm sayfaları dolaş
# for sheet_name, df in excel_data.items():
#     # Sütun isimlerindeki boşlukları tekrar temizleyelim
#     df.columns = df.columns.str.strip()
#
#     # 'Prev.' ve 'Curr.' sütunlarının isimlerinin doğru olduğunu doğrula
#     if 'Prev.' in df.columns and 'Curr.' in df.columns:
#         # Eksik verileri kaldır
#         df = df.dropna()
#
#         # 'Prev.' ve 'Curr.' sütunlarını temizle ve numpy array'e çevir
#         prev = np.array([fix_format(x) for x in df['Prev.']])
#         curr = np.array([fix_format(x) for x in df['Curr.']])
#
#         # Özellik çıkarımı ve veriyi cam_data yapısına entegre et
#         features = extract_features(prev, curr)
#         data.append(features)
#
#         # Burada label'ı senin belirlemen gerek. Aynı seriler için 0, farklı seriler için 1:
#         # Örneğin:
#         label = 0 if sheet_name == 'same_series' else 1
#         labels.append(label)
#     else:
#         print(f"Sheet {sheet_name} contains unexpected column names: {df.columns}")
#
# # Veri setini eğitim ve test olarak ayır
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
#
# # Random Forest modeli eğit
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
#
# # Test sonuçları
# y_pred = rf_model.predict(X_test)
#
# # Doğruluk oranı ve performans değerlendirmesi
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
#
# # Modeli test etmek için örnek bir grafik çizelim
# for i in range(len(X_test)):
#     print(f"Test {i + 1}:")
#     prev_shape = np.array([[X_test[i][2], X_test[i][3]]])  # Test örneği için prev noktaları
#     curr_shape = np.array([[X_test[i][4], X_test[i][5]]])  # Test örneği için curr noktaları
#
#     plt.figure(figsize=(8, 8))
#     plt.scatter(prev_shape[:, 0], prev_shape[:, 1], color='red', label='Prev.', s=100)
#     plt.scatter(curr_shape[:, 0], curr_shape[:, 1], color='blue', label='Curr.', s=100)
#
#     plt.title(f"Hausdorff Distance: {X_test[i][1]:.2f}")
#     plt.legend()
#     plt.xlabel("X")
#     plt.ylabel("Y")
#
#     plt.grid(True)
#     plt.show()

#**************************************************************

# import numpy as np
# import pandas as pd
# import math
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import directed_hausdorff
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
#
# # Excel dosyasını oku
# file_path = 'verisilinmis.xlsx'  # Excel dosyanın yolunu buraya ekle
# excel_data = pd.read_excel(file_path, sheet_name=None)  # Tüm sayfaları oku
#
#
# # Sütun isimlerindeki boşlukları temizleyerek sütun adlarını kontrol et
# def fix_format(s):
#     s = s.replace('[', '').replace(']', '').split()
#     return [int(s[0]), int(s[1])]
#
#
# # İki nokta arasındaki Euclidean mesafesini hesaplayan fonksiyon
# def euclidean_distance(p1, p2):
#     return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
#
#
# # Özellik çıkarımı (Euclidean ve Hausdorff Distance)
# def extract_features(prev_shape, curr_shape):
#     # Euclidean mesafesi hesaplama
#     euclidean_distances = np.linalg.norm(prev_shape - curr_shape, axis=1)
#
#     # Hausdorff Distance hesaplama
#     hausdorff_dist1 = directed_hausdorff(prev_shape, curr_shape)[0]
#     hausdorff_dist2 = directed_hausdorff(curr_shape, prev_shape)[0]
#     hausdorff_distance = max(hausdorff_dist1, hausdorff_dist2)
#
#     # Ortalama Euclidean mesafesi ve Hausdorff mesafesini özellik olarak kullan
#     features = [np.mean(euclidean_distances), hausdorff_distance]
#     return features
#
#
# # Veri setini ve etiketleri oluştur
# data = []
# labels = []  # Aynı seriler için 0, farklı seriler için 1
#
# # Tüm sayfaları dolaş
# for sheet_name, df in excel_data.items():
#     # Sütun isimlerindeki boşlukları tekrar temizleyelim
#     df.columns = df.columns.str.strip()
#
#     # 'Prev.' ve 'Curr.' sütunlarının isimlerinin doğru olduğunu doğrula
#     if 'Prev.' in df.columns and 'Curr.' in df.columns:
#         # Eksik verileri kaldır
#         df = df.dropna()
#
#         # 'Prev.' ve 'Curr.' sütunlarını temizle ve numpy array'e çevir
#         prev = np.array([fix_format(x) for x in df['Prev.']])
#         curr = np.array([fix_format(x) for x in df['Curr.']])
#
#         # Özellik çıkarımı
#         features = extract_features(prev, curr)
#         data.append(features)
#
#         # Aynı seri için 0, farklı seri için 1 olacak şekilde etiket ver
#         label = 0 if sheet_name == 'same_series' else 1
#         labels.append(label)
#     else:
#         print(f"Sheet {sheet_name} contains unexpected column names: {df.columns}")
#
# # Veri setini eğitim ve test olarak ayır
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
#
# # Random Forest modeli eğit
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
#
# # Test sonuçları
# y_pred = rf_model.predict(X_test)
#
# # Doğruluk oranı ve performans değerlendirmesi
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
#
# # Modeli test etmek için örnek bir grafik çizelim
# for i in range(len(X_test)):
#     print(f"Test {i + 1}:")
#     prev_shape = np.array([[X_test[i][0], X_test[i][0]]])  # Test örneği için prev noktaları
#     curr_shape = np.array([[X_test[i][1], X_test[i][1]]])  # Test örneği için curr noktaları
#
#     plt.figure(figsize=(8, 5))
#     plt.scatter(prev_shape[:, 0], prev_shape[:, 1], color='red', label='Prev.', s=100)
#     plt.scatter(curr_shape[:, 0], curr_shape[:, 1], color='blue', label='Curr.', s=100)
#
#     plt.title(f"Hausdorff Distance: {X_test[i][1]:.2f}")
#     plt.legend()
#     plt.xlabel("X")
#     plt.ylabel("Y")
#
#     plt.grid(True)
#     plt.show()

#******   Cross Validation Olmadan **********
# import numpy as np
# import pandas as pd
# import math
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import directed_hausdorff
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
#
# # Excel dosyasını oku
# file_path = 'verisilinmis.xlsx'  # Excel dosyanın yolunu buraya ekle
# excel_data = pd.read_excel(file_path, sheet_name=None)  # Tüm sayfaları oku
#
#
# # Sütun isimlerindeki boşlukları temizleyerek sütun adlarını kontrol et
# def fix_format(s):
#     s = s.replace('[', '').replace(']', '').split()
#     return [int(s[0]), int(s[1])]
#
#
# # İki nokta arasındaki Euclidean mesafesini hesaplayan fonksiyon
# def euclidean_distance(p1, p2):
#     return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
#
#
# # Özellik çıkarımı (Euclidean ve Hausdorff Distance)
# def extract_features(prev_shape, curr_shape):
#     # Euclidean mesafesi hesaplama
#     euclidean_distances = np.linalg.norm(prev_shape - curr_shape, axis=1)
#
#     # Hausdorff Distance hesaplama
#     hausdorff_dist1 = directed_hausdorff(prev_shape, curr_shape)[0]
#     hausdorff_dist2 = directed_hausdorff(curr_shape, prev_shape)[0]
#     hausdorff_distance = max(hausdorff_dist1, hausdorff_dist2)
#
#     # Ortalama Euclidean mesafesi ve Hausdorff mesafesini özellik olarak kullan
#     features = [np.mean(euclidean_distances), hausdorff_distance]
#     return features
#
#
# # Veri setini ve etiketleri oluştur
# data = []
# labels = []  # Aynı seriler için 0, farklı seriler için 1
#
# # Tüm sayfaları dolaş
# for sheet_name, df in excel_data.items():
#     # Sütun isimlerindeki boşlukları tekrar temizleyelim
#     df.columns = df.columns.str.strip()
#
#     # 'Prev.' ve 'Curr.' sütunlarının isimlerinin doğru olduğunu doğrula
#     if 'Prev.' in df.columns and 'Curr.' in df.columns:
#         # Eksik verileri kaldır
#         df = df.dropna()
#
#         # 'Prev.' ve 'Curr.' sütunlarını temizle ve numpy array'e çevir
#         prev = np.array([fix_format(x) for x in df['Prev.']])
#         curr = np.array([fix_format(x) for x in df['Curr.']])
#
#         # Özellik çıkarımı
#         features = extract_features(prev, curr)
#         data.append(features)
#
#         # Aynı seri için 0, farklı seri için 1 olacak şekilde etiket ver
#         label = 0 if sheet_name == 'same_series' else 1
#         labels.append(label)
#     else:
#         print(f"Sheet {sheet_name} contains unexpected column names: {df.columns}")
#
# # Veri setini eğitim ve test olarak ayır
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
#
# # Random Forest modeli eğit
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
#
# # Test sonuçları
# y_pred = rf_model.predict(X_test)
#
# # Doğruluk oranı ve performans değerlendirmesi
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
#
# # Test sonuçlarını ve tahminleri yazdırma
# for i in range(len(X_test)):
#     actual_label = "Aynı Seri" if y_test[i] == 0 else "Farklı Seri"
#     predicted_label = "Aynı Seri" if y_pred[i] == 0 else "Farklı Seri"
#     print(f"Test {i + 1}: Gerçek Sonuç: {actual_label}, Model Tahmini: {predicted_label}")


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
