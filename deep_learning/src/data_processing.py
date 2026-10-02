import os
import joblib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataProcessingConfig:
    DATE_COLS = ['order_date', 'delivery_date', 'user_dob', 'user_reg_date']
    NUM_COLS  = ['item_price', 'delivery_days', 'user_age', 'account_age_days']
    CAT_COLS  = ['item_size', 'item_color', 'brand_id', 'user_title', 'user_state']
    FLAG_COLS = ['delivery_missing', 'dob_missing']

    def __init__(self, path_to_data="../../data/raw", load_data=True):
        self.path_to_data = path_to_data
        self.df = None
        if load_data:  # predict does not require loading the CSV
            self.df = pd.read_csv(os.path.join(self.path_to_data, "order_return_data.csv"))
        self.preprocessor = None

    # ---------- Training and Prediction ----------
    def make_features(self, df):
        df = df.copy()

        # predict does not require all date columns to be present
        for c in self.DATE_COLS:
            if c not in df.columns:
                df[c] = pd.NaT
            df[c] = pd.to_datetime(df[c], errors='coerce')

        df['delivery_days'] = (df['delivery_date'] - df['order_date']).dt.days
        df['user_age'] = (df['order_date'] - df['user_dob']).dt.days / 365.25
        df['account_age_days'] = (df['order_date'] - df['user_reg_date']).dt.days

        # missing value indicators
        df['delivery_missing'] = df['delivery_date'].isna().astype(int)
        df['dob_missing'] = df['user_dob'].isna().astype(int)

        df['brand_id'] = df['brand_id'].astype(str)  
        # for c in ['item_size', 'item_color', 'brand_id', 'user_title', 'user_state']:
        #     print(c, df[c].nunique())

        # if there is no such column, it will be ignored
        return df.drop(columns=self.DATE_COLS + ['order_item_id', 'user_id'],
                       errors='ignore')

    def _build_preprocessor(self):
        return ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
            ]), self.NUM_COLS),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist',
                             min_frequency=50,
                             sparse_output=False)),
            ]), self.CAT_COLS),
            ('flags', 'passthrough', self.FLAG_COLS),
        ])

    # ---------- TRAINING: split + fit ----------
    def preprocess_data(self):
        df = self.make_features(self.df)

        X = df.drop(columns='return')
        y = df['return']

        # 1. Train-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 2. Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )

        # 3. Pipeline: imputing + scaling + encoding
        self.preprocessor = self._build_preprocessor()

        # 4. fit only on train data
        X_train_p = self.preprocessor.fit_transform(X_train)
        X_val_p   = self.preprocessor.transform(X_val)
        X_test_p  = self.preprocessor.transform(X_test)

        print(f"Train: {X_train_p.shape}, Val: {X_val_p.shape}, Test: {X_test_p.shape}")

        # 5. Tensor conversion
        X_train_t = torch.tensor(X_train_p, dtype=torch.float32)
        X_val_t   = torch.tensor(X_val_p,   dtype=torch.float32)
        X_test_t  = torch.tensor(X_test_p,  dtype=torch.float32)

        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        y_val_t   = torch.tensor(y_val.values,   dtype=torch.float32).unsqueeze(1)
        y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1)

        return (X_train_t, X_val_t, X_test_t,
                y_train_t, y_val_t, y_test_t)

    # ---------- INFERENCE: raw -> tensor ----------
    def transform(self, raw):
        """ raw is a dict (single row) or DataFrame. The 'return' column is not required."""
        if self.preprocessor is None:
            raise RuntimeError("First call preprocess_data() or load() before transform().")
        if isinstance(raw, dict):
            raw = pd.DataFrame([raw])

        df = self.make_features(raw)
        x = self.preprocessor.transform(df)   # no fit, just transform
        return torch.tensor(x, dtype=torch.float32)

    # ---------- Save / Load ----------
    def save(self, path="preprocessor.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.preprocessor, path)

    def load(self, path="preprocessor.joblib"):
        self.preprocessor = joblib.load(path)