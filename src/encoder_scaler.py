from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
from src.preprocessing import get_unique_values

# cols for scaler 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'
# cols for encoder 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
#        'Property_Area'
# there is also credit history

# getting the data
df = pd.read_csv('/Users/gscerberus/Desktop/Loan_Prediction_Analysis/data/pre_final_data_set.csv')

# initializing the label encoder and standard scaler
label_encoder_Gender = LabelEncoder()
label_encoder_Married = LabelEncoder()
label_encoder_Education = LabelEncoder()
label_encoder_Self_Employed = LabelEncoder()
label_encoder_Property_Area = LabelEncoder()


# get unique values for each categorical value

gender = get_unique_values(df, 'Gender')
married = get_unique_values(df, 'Married')
education = get_unique_values(df, 'Education')
self_employed = get_unique_values(df, 'Self_Employed')
property_area = get_unique_values(df, 'Property_Area')
credit_history = get_unique_values(df, 'Credit_History')

results_gender = label_encoder_Gender.fit(df['Gender'])
results_married = label_encoder_Married.fit(df['Married'])
results_education = label_encoder_Education.fit(df['Education'])
results_self_employed = label_encoder_Self_Employed.fit(df['Self_Employed'])
results_property_area = label_encoder_Property_Area.fit(df['Property_Area'])

DATA_ENCODER = {"label_encoder_Gender":label_encoder_Gender,
               "label_encoder_Married":label_encoder_Married,
                "label_encoder_Education":label_encoder_Education,
               "label_encoder_Self_Employed":label_encoder_Self_Employed,
                "label_encoder_Property_Area":label_encoder_Property_Area,
                "gender":gender,
                "married":married,
                "education":education,
                "self_employed":self_employed,
                "property_area":property_area,
                "credit_history":credit_history}


# scaler turn

min_max_scaler = MinMaxScaler()
encoded_cols = min_max_scaler.fit_transform(df[['LoanAmount','Loan_Amount_Term','TotalIncome']])

DATA_SCALER = {"min_max_scaler":min_max_scaler}



