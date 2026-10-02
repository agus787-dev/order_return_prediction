import sys

sys.path.append("..")

from src.data_processing import DataProcessingConfig

# --- Training ---
cfg = DataProcessingConfig()
X_train, X_val, X_test, y_train, y_val, y_test = cfg.preprocess_data()
cfg.save(path="../models/preprocessor.joblib")