from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

class ModelTrainer:
    def __init__(self):
        self.models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        }
        self.trained   = {}
        self.best_name = None
        self.best_mdl  = None
        self.best_score = 0

    def train_all(self, X_train, y_train):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained[name] = model
            print(f"✓ {name} o'rgatildi")
        return self.trained

    def select_best(self, X_test, y_test):
        for name, model in self.trained.items():
            score = accuracy_score(y_test, model.predict(X_test))
            print(f"  {name}: {score:.4f}")
            if score > self.best_score:
                self.best_score = score
                self.best_name  = name
                self.best_mdl   = model
        print(f"\n✓ Eng yaxshi model: {self.best_name} ({self.best_score:.4f})")
        return self.best_name, self.best_mdl

