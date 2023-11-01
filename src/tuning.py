from typing import Type, Dict
from sklearn.base import BaseEstimator
import numpy as np
from numpy import ndarray
from sklearn.model_selection import GridSearchCV

class Tuning:

    def __init__(self, model: Type[BaseEstimator], param_grid: Dict[str, str])-> None:

        self.model = model
        self.param_grid = param_grid

    def tune(self, X: ndarray, y: ndarray)-> Dict[str, float]:

        grid_search = GridSearchCV(self.model, self.param_grid, cv = 5, scoring = 'accuracy')
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        return best_params
