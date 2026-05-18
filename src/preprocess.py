import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler   = StandardScaler()
        self.encoders = {}
    
    def fillna_train(self, df):   
        fill_values = {}
        df=df.copy()

        for col in df.columns:
            if df[col].dtype == 'object':
                fill_values[col] = df[col].mode()[0]
                df[col].fillna(fill_values[col], inplace=True)
            else:
                fill_values[col] = df[col].mean()
                df[col].fillna(fill_values[col], inplace=True)

        return df, fill_values

    def fillna_test(self, df, fill_values):
        df=df.copy()

        for col, value in fill_values.items():
            df[col].fillna(value, inplace=True)

        return df
    
    def encode_train(self, df, threshold=0):
        df = df.copy()
        encoders = {}
        onehot_cols = []

        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].nunique() <= threshold:
                    onehot_cols.append(col)
                    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                    df = pd.concat([df.drop(columns=col), dummies], axis=1)
                else:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le

        return df, encoders, onehot_cols

    def encode_test(self, df, encoders, onehot_cols, train_columns):
        df = df.copy()

        # Label encoding
        for col, le in encoders.items():
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # One-hot ONLY for known columns
        for col in onehot_cols:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df.drop(columns=col), dummies], axis=1)

        # Align columns
        df = df.reindex(columns=train_columns, fill_value=0)

        return df
    
    def scale_train(self, df):
        df = df.copy()
        scalers = {}

        for col in df.columns:
            if df[col].dtype != 'object':
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