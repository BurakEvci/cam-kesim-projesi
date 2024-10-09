import math
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import cv2  # OpenCV'yi ekleyelim

# Hausdorff mesafesi fonksiyonu
def hausdorff_distance(shape1, shape2):
    hausdorff_dist1 = directed_hausdorff(shape1, shape2)[0]
    hausdorff_dist2 = directed_hausdorff(shape2, shape1)[0]
    return max(hausdorff_dist1, hausdorff_dist2)

# Öklid mesafesinin ortalamasını hesaplayan fonksiyon
def calculate_average_euclidean_distance(shape1, shape2):
    distances = [euclidean_distance(p1, p2) for p1, p2 in zip(shape1, shape2)]
    return sum(distances) / len(distances) if distances else 0



# İki nokta arasındaki mesafeyi hesaplayan fonksiyon
def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Her iki şekil için kenar uzunluklarını hesapla
def calculate_sides(shape):
    return [euclidean_distance(shape[i], shape[(i + 1) % len(shape)]) for i in range(len(shape))]

# Kenar uzunluklarının ortalamasını al
def average_side_length(sides):
    return sum(sides) / len(sides)

#     #         # Şekillerin kenar uzunluklarını hesapla
#     #         sides_prev_shape = calculate_sides(prev_shape)
#     #         sides_curr_shape = calculate_sides(curr_shape)
#     #
#     #         # Benzerlik oranını hesapla
#     #         avg_prev_shape = average_side_length(sides_prev_shape)
#     #         avg_curr_shape = average_side_length(sides_curr_shape)
#     #         similarity_ratio = avg_prev_shape / avg_curr_shape

# OpenCV Shape Matching kullanımı
def match_shapes(prev_shape, curr_shape):
    # Koordinatları OpenCV'nin kabul edeceği kontur formatına dönüştür
    prev_shape = np.array(prev_shape).astype(np.int32).reshape((-1, 1, 2))
    curr_shape = np.array(curr_shape).astype(np.int32).reshape((-1, 1, 2))

    # cv2.matchShapes ile şekilleri karşılaştır
    match_value = cv2.matchShapes(prev_shape, curr_shape, cv2.CONTOURS_MATCH_I1, 0.0)
    return match_value
