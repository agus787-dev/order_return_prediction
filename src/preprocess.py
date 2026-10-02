import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler   = StandardScaler()
        self.encoders = {}
    
    def fillna_train(self, df):   
        fill_values = {}
        df=df.copy()
        num_cols = df.select_dtypes(include='number').columns
        cat_cols = df.select_dtypes(exclude='number').columns

        for col in cat_cols:
            fill_values[col] = df[col].mode()[0]
            df[col] = df[col].fillna(fill_values[col])

        for col in num_cols:
            fill_values[col] = df[col].mean()
            df[col] = df[col].fillna(fill_values[col])

        return df, fill_values

    def fillna_test(self, df, fill_values):
        df=df.copy()

        for col, value in fill_values.items():
            df[col] = df[col].fillna(value)

        return df
    
    def encode_train(self, df, threshold=0):
        df = df.copy()
        encoders = {}
        cat_cols = df.select_dtypes(exclude='number').columns

        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        return df, encoders

    def encode_test(self, df, encoders, train_columns):
        df = df.copy()

        # Label encoding
        for col, le in encoders.items():
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Align columns
        df = df.reindex(columns=train_columns, fill_value=0)

        return df
    
    def scale_train(self, df):
        df = df.copy()
        scalers = {}
        num_cols = df.select_dtypes(include='number').columns


        for col in num_cols:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler

        return df, scalers

    def scale_test(self, df, scalers):
        df = df.copy()

        for col in df.columns:
            if col in scalers:
                df[col] = scalers[col].transform(df[[col]])

        return df