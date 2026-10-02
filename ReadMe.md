# Order Return Prediction

This project is designed to predict the likelihood that an order will be returned using machine learning and deep learning models.

The main goal is to help estimate whether a customer will return a purchased order before or after the transaction is completed.

## Key links

- [Dataset (raw)](data/raw/order_return_data.csv)
- [Preprocessed train data](data/preprocessed/x_train.csv)
- [Preprocessed test data](data/preprocessed/x_test.csv)
- [Train labels](data/preprocessed/y_train.csv)
- [Test labels](data/preprocessed/y_test.csv)
- [Baseline model](models/baseline/best_model.joblib)
- [Deep learning model](deep_learning/models/best_model.pth)
- [Deep learning preprocessor](deep_learning/models/preprocessor.joblib)
- [EDA notebook](notebooks/01_eda.ipynb)
- [Modeling notebook](notebooks/03_modeling.ipynb)
- [Training script](deep_learning/scripts/train.py)
- [Offline testing script](deep_learning/scripts/offline_testing.py)

## Project structure

- [data](data/)
- [deep_learning](deep_learning/)
- [models](models/)
- [notebooks](notebooks/)
- [src](src/)
- [requirements.txt](requirements.txt)

## Technologies used

- Python 3.12+
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- FastAPI
- Joblib

## Setup

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python deep_learning/scripts/train.py
```

## Evaluation / testing

```bash
python deep_learning/scripts/offline_testing.py
```

## Main source files

- [src/model.py](src/model.py) — baseline ML model and trainer
- [src/model_improved.py](src/model_improved.py) — improved model variants
- [src/preprocess.py](src/preprocess.py) — preprocessing logic
- [src/feature_engineering.py](src/feature_engineering.py) — feature engineering
- [src/data_loader.py](src/data_loader.py) — data loading logic
- [deep_learning/src/train.py](deep_learning/src/train.py) — deep learning training code
- [deep_learning/src/data_processing.py](deep_learning/src/data_processing.py) — deep learning data preprocessing

## Dataset information

The raw dataset is loaded from [data/raw/order_return_data.csv](data/raw/order_return_data.csv). After that, the data is preprocessed and stored in the [data/preprocessed](data/preprocessed) folder.

## Model artifacts

- Baseline model: [models/baseline/best_model.joblib](models/baseline/best_model.joblib)
- Deep learning model: [deep_learning/models/best_model.pth](deep_learning/models/best_model.pth)

## Repository layout

```text
order_return_prediction/
├── data/
├── deep_learning/
├── models/
├── notebooks/
├── src/
├── .gitignore
├── ReadMe.md
├── requirements.txt
└── ...
```

## Author

This project is focused on building an ML/DL pipeline for predicting order return probability.

If needed, the next steps can include:

- improving the model performance,
- creating a FastAPI service,
- adding Docker support,
- or preparing a more complete GitHub-ready README.
