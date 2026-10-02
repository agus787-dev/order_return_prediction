import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

torch.manual_seed(42)


class TrainConfig:
    def __init__(self, learning_rate=0.001, batch_size=256, num_epochs=40,
                 x_train_tensor=None, y_train_tensor=None,
                 x_val_tensor=None, y_val_tensor=None,
                 x_test_tensor=None, y_test_tensor=None, 
                 early_stop_patience=5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.x_train_tensor = x_train_tensor
        self.y_train_tensor = y_train_tensor
        self.x_val_tensor = x_val_tensor
        self.y_val_tensor = y_val_tensor
        self.x_test_tensor = x_test_tensor    
        self.y_test_tensor = y_test_tensor 
        self.early_stop_patience = early_stop_patience

    def make_loaders(self, x_train, y_train, x_val, y_val):
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def train_one_epoch(self, model, loader, loss_fn, optimizer):
        model.train()
        total_loss = 0

        for bx, by in loader:
            pred = model(bx)
            loss = loss_fn(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, model, loader, loss_fn):         
        model.eval()
        total_loss = 0
        y_true, y_prob = [], []

        for bx, by in loader:
            logits = model(bx)
            loss = loss_fn(logits, by)
            total_loss += loss.item() * bx.size(0)

            probs = torch.sigmoid(logits)                # logit -> probability (0..1)
            y_true.extend(by.cpu().numpy().ravel())
            y_prob.extend(probs.cpu().numpy().ravel())

        avg_loss = total_loss / len(loader.dataset)

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (y_prob > 0.5).astype(int)              # threshold 0.5

        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        auc       = roc_auc_score(y_true, y_prob)        

        return avg_loss, accuracy, precision, recall, f1, auc

    def fit_model(self, model, train_loader, val_loader, optimizer, loss_fn):
        train_losses, val_losses = [], []
        best_val = float("inf")
        patience_left = self.early_stop_patience
        best_state = None

        for epoch in range(1, self.num_epochs + 1):
            tr_loss = self.train_one_epoch(model, train_loader, loss_fn, optimizer)
            val_loss, acc, prec, rec, f1, auc = self.evaluate(model, val_loader, loss_fn)

            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            # Metrikalar endi har safar chiqadi (early stop'ga bog'liq emas)
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | train={tr_loss:.6f} | val={val_loss:.6f}")
                print(f"  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} "
                      f"F1={f1:.4f} AUC={auc:.4f}")

            # Early stopping
            if self.early_stop_patience is not None:
                if val_loss < best_val:
                    best_val = val_loss
                    patience_left = self.early_stop_patience

                    best_state = copy.deepcopy(model.state_dict())   # <-- haqiqiy nusxa
                    os.makedirs("../models", exist_ok=True)
                    torch.save(best_state, "../models/best_model.pth")
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(f"Early stop at epoch {epoch} (best val={best_val:.4f})")
                        print(f"  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} "
                              f"F1={f1:.4f} AUC={auc:.4f}")
                        break

        # Best model
        if best_state is not None:
            model.load_state_dict(best_state)

        return train_losses, val_losses

    def train(self):
        input_dim = self.x_train_tensor.shape[1]  # features soni

        train_loader, val_loader = self.make_loaders(
            self.x_train_tensor, self.y_train_tensor,
            self.x_val_tensor, self.y_val_tensor
        )

        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate,
                               weight_decay=1e-3)

        train_losses, val_losses = self.fit_model(
            model, train_loader, val_loader, optimizer, loss_fn
        )
        print("Training completed.")
        print(f"Final train loss: {train_losses[-1]:.6f}, "
              f"Final val loss: {val_losses[-1]:.6f}")

        return model, train_losses, val_losses          
    
    def test(self, model):
        
        test_ds = TensorDataset(self.x_test_tensor, self.y_test_tensor)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        loss_fn = nn.BCEWithLogitsLoss()
        loss, acc, prec, rec, f1, auc = self.evaluate(model, test_loader, loss_fn)

        print("===== TEST Result =====")
        print(f"Loss     : {loss:.4f}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1       : {f1:.4f}")
        print(f"AUC      : {auc:.4f}")

        return {"loss": loss, "accuracy": acc, "precision": prec,
                "recall": rec, "f1": f1, "auc": auc}  