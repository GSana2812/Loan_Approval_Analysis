import sys
import pandas as pd
import numpy as np
sys.path.append("/Users/gscerberus/Desktop/Loan_Prediction_Analysis/")
import fastapi
import uvicorn
from src.predictions import load_model
from src.preprocessing import scale_features,encode_features
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List,Dict
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()
MODEL = load_model('serials/logistic_regression.pkl')
result = None


class LoanPrediction(BaseModel):
    Gender: str
    Married: str
    Education: str
    Self_Employed: str
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str
    TotalIncome: float


def get_scales(scaler: Dict[str, MinMaxScaler],
               cols: List[str],
               data_dict: Dict[str, float],
               scaler_str: str)-> Dict:

    scale = []

    for col in cols:
        if col in data_dict:
            scale.append(data_dict[col])

    # Adjust the dimensionality to work in scaling
    data = np.array(scale).reshape(-1, 1)

    # performing scaling
    scaled_data = scaler[scaler_str].fit_transform(data)

    # turn it back to the previous dim
    scaled_data_flatten = scaled_data.flatten()

    for key, value in zip(cols, scaled_data_flatten):
        data_dict[key] = value

    return data_dict

def get_encoded_cols(encoder: Dict[str, str],
                     cols: List[str],
                     data_dict: Dict[str, str]
                     )-> Dict:

    for col in cols:
        if col in data_dict:
            data_dict[col] =  float(encoder[f'label_encoder_{col}'].transform([data_dict[col]]).item())

    return data_dict



@app.post('/predict')
async def predictions(item: LoanPrediction):

     data_dict = item.dict()

     # loading the scaler
     scaler = load_model('serials/scaler.pkl')

     #columns to be scaled
     cols_to_be_scaled = ["LoanAmount", "Loan_Amount_Term","TotalIncome"]

     # applying the method to scale the numerical features
     data_dict_scaled = get_scales(scaler, cols_to_be_scaled, data_dict, 'min_max_scaler')

     # getting the str columns
     cols_to_be_encoded = [key for key, value in data_dict.items() if isinstance(value, str)]

     # loading the encoder
     encoder = load_model('serials/encoder.pkl')

     data_dict_encoded = get_encoded_cols(encoder, cols_to_be_encoded, data_dict_scaled)

     global result
     # Perform predictions

     input = list(data_dict_encoded.values())

     result = input #to show the value in get


     return MODEL.predict([input]).tolist()




@app.get("/get_result")
def read_root():
    # http://localhost:8000/get_result (Will display the message)
    return {"prediction": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)