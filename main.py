import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from data_processing import fix_format, read_excel_data
from distance_calculations import hausdorff_distance, calculate_average_euclidean_distance
from plotting import plot_shapes_hausdorff, plot_shapes_opencv
from plotting import plot_shapes_euclidean
from scipy.spatial.distance import directed_hausdorff
from distance_calculations import calculate_sides, average_side_length
from distance_calculations import hausdorff_distance
from distance_calculations import match_shapes
# Excel dosyasını oku
file_path = 'veriler.xlsx'
excel_data = read_excel_data(file_path)

# Tüm sayfaları dolaş
for sheet_name, df in excel_data.items():
    if 'Prev.' in df.columns and 'Curr.' in df.columns:

        # NaN olan hücreler için interpolasyon yapmadan önce string veriyi işliyoruz
        df['Prev.'] = df['Prev.'].apply(lambda x: fix_format(x))
        df['Curr.'] = df['Curr.'].apply(lambda x: fix_format(x))

        # 'Prev.' ve 'Curr.' sütunlarını iki ayrı X ve Y sütununa ayır
        prev_df = pd.DataFrame(df['Prev.'].tolist(), columns=['Prev_X', 'Prev_Y'])
        curr_df = pd.DataFrame(df['Curr.'].tolist(), columns=['Curr_X', 'Curr_Y'])

        # Interpolasyon işlemi uygulayalım
        prev_df['Prev_X'] = prev_df['Prev_X'].interpolate(method='linear')
        prev_df['Prev_Y'] = prev_df['Prev_Y'].interpolate(method='linear')
        curr_df['Curr_X'] = curr_df['Curr_X'].interpolate(method='linear')
        curr_df['Curr_Y'] = curr_df['Curr_Y'].interpolate(method='linear')

        # Interpolasyon ile doldurulmuş verileri tekrar birleştir
        prev = np.array(prev_df[['Prev_X', 'Prev_Y']].values)
        curr = np.array(curr_df[['Curr_X', 'Curr_Y']].values)


        # # Eksik olan noktaları lineer interpolasyon ile doldur
        # df['Prev.'] = df['Prev.'].interpolate(method='linear')
        # df['Curr.'] = df['Curr.'].interpolate(method='linear')
        #
        # # Lineer interpolasyonla doldurulan verileri numpy array'e çevir
        # prev = np.array([fix_format(str(x)) for x in df['Prev.']])
        # curr = np.array([fix_format(str(x)) for x in df['Curr.']])


        # df = df.dropna()


        cam_data = {
            sheet_name: {
                "prev": prev,
                "curr": curr
            }
        }

        for cam_name, points in cam_data.items():
            prev_shape = points["prev"]
            curr_shape = points["curr"]

            # # Hausdorff Distance hesapla
            # h_distance = hausdorff_distance(prev_shape, curr_shape)
            #
            # # Öklid mesafesinin ortalamasını hesapla
            # avg_euclidean_distance = calculate_average_euclidean_distance(prev_shape, curr_shape)

            # # Debug için Euclidean mesafesini kontrol et
            # print(f"Euclidean distance for {cam_name}: {avg_euclidean_distance}")

            plot_shapes_euclidean(cam_name, prev_shape, curr_shape, calculate_sides, average_side_length)

            # Hausdorff Distance hesapla
            h_distance = hausdorff_distance(prev_shape, curr_shape)
            # Grafik çiz ve Hausdorff Distance'ı ekle
            plot_shapes_hausdorff(cam_name, prev_shape, curr_shape, h_distance)


            # OpenCV Shape Matching kullanarak benzerlik hesaplama
            match_value = match_shapes(prev_shape, curr_shape)

            plot_shapes_opencv(cam_name, prev_shape, curr_shape, match_value)

    else:
        print(f"Sheet {sheet_name} contains unexpected column names: {df.columns}")



#eşik değerini her yeni sayfa eklendiğinde güncelle