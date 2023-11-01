import pandas as pd
from typing import List
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def drop_column(data: pd.DataFrame, col_name: List[str])-> pd.DataFrame:
    return data.drop(columns = col_name, axis=1)

def convert_float_to_object(data: pd.DataFrame, col_name: List[str])-> pd.DataFrame:
    data[col_name] = data[col_name].astype('object')
    return data[col_name]

# Performing some feature engineering

def add_total_income(data: pd.DataFrame, old_col_1:str , old_col_2:str, new_col: str)-> pd.DataFrame:
    data[new_col] = data[old_col_1] + data[old_col_2]
    return data

def fill_nan_values(data: pd.DataFrame, cols: List[str])->None:

    """
        Fill missing values in a specified column of a DataFrame based on data type.

        Parameters:
        data (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column to fill.

        Note:
        - If the column data type is 'object', it fills missing values with the most common value.
        - If the column data type is 'int', it fills missing values with the mean.

        The function modifies the DataFrame in place and does not return a new DataFrame.
        """
    for col_name in cols:
        if data[col_name].dtype == 'O':
            data[col_name] = data[col_name].fillna(data[col_name].value_counts().sort_values(ascending=False).index[0])
        elif data[col_name].dtype == 'int64' or data[col_name].dtype == 'float64':
            data[col_name] = data[col_name].fillna(round(data[col_name].mean(),2))

    return data

def remove_outliers(data: pd.DataFrame, col_name: str)-> pd.DataFrame:
    if col_name == "ApplicantIncome":
        data = data[~(data['ApplicantIncome'] >= 10408)]
    elif col_name == "CoapplicantIncome":
        data = data[~(data['CoapplicantIncome'] >= 6250)]
    elif col_name == "LoanAmount":
        data = data[~(data['LoanAmount'] >= 400)]
    elif col_name == "Loan_Amount_Term":
        data = data[~(data['Loan_Amount_Term'] >= 375)]
    else:
        return "Error"

    return data

def scale_features(data: pd.DataFrame, cols: List[str])-> pd.DataFrame:

    scaler = MinMaxScaler()
    data[cols] = np.round(scaler.fit_transform(data[cols]),2)

    return data

def encode_features(data: pd.DataFrame, cols: List[str])-> pd.DataFrame:
    label_encoder = LabelEncoder()

    for col in cols:

        data[col] = label_encoder.fit_transform(data[col]).astype(float)

    return data

def get_unique_values(data: pd.DataFrame, col: str) -> List[str]:
    return list(data[col].unique())










