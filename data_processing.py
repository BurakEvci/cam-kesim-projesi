import pandas as pd
import numpy as np

def fix_format(s):
    # NaN kontrolü
    if pd.isna(s):
        return [np.nan, np.nan]
    # String veriyi temizle ve liste haline getir
    s = s.replace('[', '').replace(']', '').split()
    return [int(s[0]), int(s[1])]


def read_excel_data(file_path):
    excel_data = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in excel_data.items():
        df.columns = [str(col).strip() for col in df.columns]  # Sütun isimlerindeki boşlukları temizle
    return excel_data
