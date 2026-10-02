import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

sys.path.append("..")

from src.data_processing import DataProcessingConfig   



class OfflineTester:
    def __init__(self,
                 model_path="../models/best_model.pth",
                 preprocessor_path="../models/preprocessor.joblib",
                 threshold=0.5):
        self.threshold = threshold

        # 1. Preprocessor 
        self.processor = DataProcessingConfig(load_data=False)
        self.processor.load(preprocessor_path)

        # 2. Model 
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        input_dim = state_dict["0.weight"].shape[1]   # set weights

        self.model = nn.Sequential(                   # same with train() 
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    # ---------- Prediction ----------
    @torch.no_grad()
    def predict_proba(self, raw):
        x = self.processor.transform(raw)             # make_features + transform + tensor
        logits = self.model(x)
        return torch.sigmoid(logits).numpy().ravel()

    def predict(self, raw):
        return (self.predict_proba(raw) > self.threshold).astype(int)

    def explain(self, raw):
        """Human-readable result for a single order."""
        p = float(self.predict_proba(raw)[0])   # probability of return
        label = "WILL BE RETURNED" if p > self.threshold else "WILL NOT BE RETURNED"

        print(f"{p * 100:.0f}% chance the order will be returned")
        print(f"{(1 - p) * 100:.0f}% chance the order will not be returned")
        print(f"Prediction: {label}")

        return p