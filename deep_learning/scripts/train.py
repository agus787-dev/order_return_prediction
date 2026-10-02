import sys

sys.path.append("..")
from src.train import TrainConfig
from src.data_processing import DataProcessingConfig


prepconfig = DataProcessingConfig()
X_train, X_val, X_test, y_train, y_val, y_test = prepconfig.preprocess_data()


trainconfig = TrainConfig(
    x_train_tensor=X_train, y_train_tensor=y_train,
    x_val_tensor=X_val,     y_val_tensor=y_val,
    x_test_tensor=X_test,   y_test_tensor=y_test,
)



model, train_losses, val_losses = trainconfig.train()   # Train + val
results = trainconfig.test(model)                       # test