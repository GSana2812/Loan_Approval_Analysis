# support vector machine
SVC_PARAM_GRID = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.1, 1, 'scale', 'auto'],
}

# logistic regression
LOGISTIC_REGRESSION_PARAM_GRID = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
}

# decision trees
DECISION_TREES_PARAM_GRID = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# random forest
RANDOM_FOREST_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# XGBoost
XGB_PARAM_GRID = {
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
}

