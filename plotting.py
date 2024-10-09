import matplotlib.pyplot as plt
import math
from distance_calculations import hausdorff_distance
from distance_calculations import euclidean_distance
from distance_calculations import calculate_sides
from distance_calculations import average_side_length
from scipy.spatial.distance import directed_hausdorff

def plot_shapes_hausdorff(cam_name, prev_shape, curr_shape, hausdorff_distance):
    plt.figure(figsize=(8, 5))
    plt.scatter(prev_shape[:, 0], prev_shape[:, 1], color='red', label='Prev.', s=100)
    plt.scatter(curr_shape[:, 0], curr_shape[:, 1], color='blue', label='Curr.', s=100)

    for a in prev_shape:
        for b in curr_shape:
            plt.plot([a[0], b[0]], [a[1], b[1]], 'k--', alpha=0.2)

    for i, point in enumerate(prev_shape):
        plt.text(point[0] + 0.1, point[1], f'P{i}', fontsize=8, color='red')
    for i, point in enumerate(curr_shape):
        plt.text(point[0] + 0.1, point[1], f'C{i}', fontsize=8, color='blue')

    plt.title(f"{cam_name} - Hausdorff Distance: {hausdorff_distance:.2f}")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")

    if hausdorff_distance < 24.39:  # Eşik değer olarak 24.39 kullandım, ihtiyacına göre ayarlayabilirsin
        plt.text(0.7, 0.05, "Aynı Kesim", fontsize=20, color='green', transform=plt.gca().transAxes)
    else:
        plt.text(0.7, 0.05, "Farklı Kesim", fontsize=20, color='red', transform=plt.gca().transAxes)

    plt.grid(True)
    plt.show()


def plot_shapes_euclidean(cam_name, prev_shape, curr_shape, calculate_sides, average_side_length):
    # Şekillerin kenar uzunluklarını hesapla
    sides_prev_shape = calculate_sides(prev_shape)
    sides_curr_shape = calculate_sides(curr_shape)

    # Benzerlik oranını hesapla
    avg_prev_shape = average_side_length(sides_prev_shape)
    avg_curr_shape = average_side_length(sides_curr_shape)
    similarity_ratio = avg_prev_shape / avg_curr_shape

    plt.figure(figsize=(8, 5))
    plt.plot(prev_shape[:, 0], prev_shape[:, 1], 'ro-', label="Prev")
    plt.plot(curr_shape[:, 0], curr_shape[:, 1], 'bo-', label="Curr")
    plt.title(f"{cam_name} Kesim Noktaları - Benzerlik Oranı: {similarity_ratio:.4f}")
    plt.ylabel('Y Koordinatları')
    plt.xlabel('X Koordinatları')

    if 0.8 < similarity_ratio < 1.1:
        plt.text(0.7, 0.05, "Aynı Kesim", fontsize=20, color='green', transform=plt.gca().transAxes)
    else:
        plt.text(0.7, 0.05, "Farklı Kesim", fontsize=20, color='red', transform=plt.gca().transAxes)

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_shapes_opencv(cam_name, prev_shape, curr_shape, match_value):
    # Grafik çiz ve Shape Matching sonucunu ekle
    plt.figure(figsize=(8, 5))
    plt.scatter(prev_shape[:, 0], prev_shape[:, 1], color='red', label='Prev.', s=100)
    plt.scatter(curr_shape[:, 0], curr_shape[:, 1], color='blue', label='Curr.', s=100)

    plt.title(f"{cam_name} - Shape Matching Value: {match_value:.4f}")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")

    # Shape Matching sonucuna göre metin ekle
    if match_value < 0.2:  # Daha düşük değerler daha benzer demektir
        plt.text(0.7, 0.05, "Aynı Kesim", fontsize=20, color='green', transform=plt.gca().transAxes)
    else:
        plt.text(0.7, 0.05, "Farklı Kesim", fontsize=20, color='red', transform=plt.gca().transAxes)

    plt.grid(True)
    plt.show()