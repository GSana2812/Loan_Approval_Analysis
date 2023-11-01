# In this file we will perform predictions and store predictions in pickle file
from typing import Type, Dict
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle


def score(model: Type[BaseEstimator], components: Dict[str, pd.DataFrame])-> float:

    model.fit(components['X_train'], components['y_train'])
    y_pred = model.predict(components['X_test'])
    return accuracy_score(y_pred, components['y_test'])

# We use this file to store predictions and load predictions through the pickle file
def save_model(model: Type[BaseEstimator], file_path: str)-> None:
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


def load_model(file_path: str)-> Type[BaseEstimator]:
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model
