import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("master_housing_dataset.csv")
df.head()

def soru_cevap(df):
    print(f"QA1 = Pandas {pd.__version__} ")
    print(f"QA2 = {df.shape[1]} ")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"QA3 = {col} sütunu, {df[col].isnull().sum()} ")
    print(f"QA4 = {df.ocean_proximity.nunique()} ")
    x = round(df[df.ocean_proximity == "NEAR BAY"]["median_house_value"].mean(), 0)
    print(f"QA5 = {x} 'dir")
    print(f"QA6 = {round(df.total_bedrooms.mean(), 4)} ")
    df["total_bedrooms"].fillna(df["total_bedrooms"].mean(), inplace=True)
    print(f"QA6 Sonuç = {round(df.total_bedrooms.mean(), 4)} ")
    island_options = df[df["ocean_proximity"] == "ISLAND"]
    selected_columns = island_options[["housing_median_age", "total_rooms", "total_bedrooms"]]
    X = selected_columns.values
    XTX = X.T @ X  # Matris Çarpımı (X ve X Transpoze Matrisi)
    XTX_inverse = np.linalg.inv(XTX)  # Matrisin tersini hesapla
    y = np.array([950, 1300, 800, 1000, 1300])
    a = XTX_inverse @ X.T
    w = a @ y   # Burda hata alıyorum. Çünkü matrislerden y'nin boyutu 5 X'in boyutu 20640 bu sebeple matris uyumsuzlupu hatası veriyor.
    print(f"QA7 Sonuç = {round(w[-1], 4)}")


soru_cevap(df)








