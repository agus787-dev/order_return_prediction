
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class FeatureEngineer:
    def __init__(self):
        self.skewed_cols  = []

    def add_features(self, df):
        df = df.copy()
        
        df["order_date"] = pd.to_datetime(df["order_date"])

        df["delivery_date"] = pd.to_datetime(df["delivery_date"])

        df["fe_delivery_days"] = (
            df["delivery_date"] - df["order_date"]
        ).dt.days
        

        df["fe_is_fast_delivery"] = (df["fe_delivery_days"] <= 3).astype(int)
        df["fe_is_late_delivery"] = (df["fe_delivery_days"] > 7).astype(int)
        
        
        df["user_dob"] = pd.to_datetime(df["user_dob"])

        df["fe_user_age"] = (
            pd.Timestamp.today() - df["user_dob"]
        ).dt.days // 365

        print(f"✓ Yangi featurelar qo'shildi: {df.shape[1]} ustun")
        return df

    def fix_skewness(self, df, threshold=0.75):
        df = df.copy()
        skewness = df.select_dtypes(include="number").skew()
        features_log = skewness[(skewness >= threshold)].index.tolist()
        
        # for col in features_log:
        #     if (self.df[col] > 0).all():  # only positive values
        #         self.df[col] = np.log1p(self.df[col])
        return features_log

    def check_skewness(self, df):
        # Skewness darajasini ko'rsatish
        num_cols = df.select_dtypes(include="number").columns.tolist()
        skew_df  = df[num_cols].skew().abs().sort_values(ascending=False)
        print("\n📊 Skewness darajalari:")
        print(skew_df)
        return skew_df
    
    def lasso_selector(self, x_train, x_test, y_train ):
        X_train_lasso = x_train.copy()
        X_test_lasso = x_test.copy()


        scaler = StandardScaler()
        X_train_scaled_lasso = scaler.fit_transform(X_train_lasso)
        X_test_scaled_lasso = scaler.transform(X_test_lasso)


        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_train_scaled_lasso, y_train)


        selected_features_lasso = X_train_lasso.columns[lasso.coef_ != 0].tolist()

        print("Selected features for Linear Models:", selected_features_lasso)

        # Reduce train/test to selected features
        X_train_lasso_selected = X_train_lasso[selected_features_lasso]
        X_test_lasso_selected = X_test_lasso[selected_features_lasso]
        
        return X_train_lasso_selected, X_test_lasso_selected
    def random_forest_selector(self, x_train, x_test, y_train):
        X_train_tree = x_train.copy()
        X_test_tree = x_test.copy()


        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_tree, y_train)

        # Get feature importances
        importances = pd.Series(rf.feature_importances_, index=X_train_tree.columns)
        importances_sorted = importances.sort_values(ascending=False)


        top_features_tree = importances_sorted.head(10).index.tolist()
        print("Top features for Tree Models:", top_features_tree)


        X_train_tree_selected = X_train_tree[top_features_tree]
        X_test_tree_selected = X_test_tree[top_features_tree]
        
        return X_train_tree_selected, X_test_tree_selected
        