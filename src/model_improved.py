from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelTrainerImproved:
   def __init__(self):
        self.trained   = {}
        self.best_name = None
        self.best_mdl  = None
        self.best_score = 0
    
   def logReg(self, x_train, y_train):
     log_reg = LogisticRegression(max_iter=1000, random_state=42)
    
     log_reg.fit(x_train, y_train)
     
     self.trained["Logistic Regression"] = log_reg
     print("Logistic Regression o'rgatildi")
     return self.trained
   
   def RandomForest(self, x_train, y_train):
     rftree = RandomForestClassifier(random_state=42)
     rftree.fit(x_train, y_train)
     self.trained["Random Forest"] = rftree
     print("Random Forest o'rgatildi")
     
     
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
    