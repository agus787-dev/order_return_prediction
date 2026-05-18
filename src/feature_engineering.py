
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

class FeatureEngineer:
    def __init__(self):
        self.transformer = PowerTransformer(method="yeo-johnson")  # skewness uchun
        self.skewed_cols  = []

    def add_features(self, df):
        df = df.copy()

        # 1. Yetkazib berish tezligi
        df["is_fast_delivery"] = (df["delivery_days"] <= 3).astype(int)
        df["is_late_delivery"] = (df["delivery_days"] > 7).astype(int)

        # 2. Narx kategoriyasi
        df["price_category"] = pd.cut(
            df["item_price"],
            bins=[0, 20, 50, 100, 999999],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # 3. Foydalanuvchi yoshi kategoriyasi
        df["age_group"] = pd.cut(
            df["user_age"],
            bins=[0, 25, 35, 50, 999],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # 4. Hafta oxiri buyurtma
        df["is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)

        # 5. Narx * yetkazib berish kunlari
        df["price_x_delivery"] = df["item_price"] * df["delivery_days"]

        print(f"✓ Yangi featurelar qo'shildi: {df.shape[1]} ustun")
        return df

    def fix_skewness(self, df, threshold=0.75):
        df = df.copy()

        # Skew bo'lgan raqamli ustunlarni topish
        num_cols = df.select_dtypes(include="number").columns.tolist()

        # "return" target ustunini chiqarib tashlash
        if "return" in num_cols:
            num_cols.remove("return")

        skewness = df[num_cols].skew().abs()
        self.skewed_cols = skewness[skewness > threshold].index.tolist()

        print(f"✓ Skewed ustunlar ({len(self.skewed_cols)} ta): {self.skewed_cols}")

        if self.skewed_cols:
            df[self.skewed_cols] = self.transformer.fit_transform(df[self.skewed_cols])
            print("✓ Skewness tuzatildi (Yeo-Johnson)")

        return df

    def check_skewness(self, df):
        # Skewness darajasini ko'rsatish
        num_cols = df.select_dtypes(include="number").columns.tolist()
        skew_df  = df[num_cols].skew().abs().sort_values(ascending=False)
        print("\n📊 Skewness darajalari:")
        print(skew_df)
        return skew_df