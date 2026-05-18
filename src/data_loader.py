import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.df   = None

    def load(self):
        self.df = pd.read_csv(self.path)
        print(f"Data yuklandi: {self.df.shape[0]} qator, {self.df.shape[1]} ustun")
        return self.df

    def split(self, target_column, test_size=0.2):
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        # output_folder="../data/raw"
        # os.makedirs(output_folder, exist_ok=True)
        # output_path_train=os.path.join(output_folder, "train.csv")
        # X_train.to_csv(output_path_train, index=False)
        # output_path_test=os.path.join(output_folder, "test.csv")
        # X_test.to_csv(output_path_test, index=False)
        
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test