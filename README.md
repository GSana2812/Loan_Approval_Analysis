## Loan Approval Analysis

In this project, I wanted to further improve my knowledge in the front end and back end part of a machine learning web app, dwelling
further into <b>streamlit</b> and <b>fastapi</b> libraries. 

First i have performed a data analysis and cleaning, removing null values, making data type necessary convertions, removing outliers,
scaling features (either numerical or categorical to numerical using LabelEncoder) and in the end I have saved the final data set in csv format.
I have performed hyperparameter tuning, taking several models for classification tasks such as:

1. Logistic Regression
2. SVC (Support Vector Machine)
3. DecisionTreeClassifier
4. RandomForest
5. XGB

After finding the best hyperparameters that may yield the best results, I have decided to test the models, and so far Logistic Regression
has resulted as the best performing model. I have tested it in inference, and somewhat is producing satisfiable results, but unfortunately
we are still limited, since we are dealing with a very small dataset (534 rows after cleaning) and I am not satisfied with the overall performance
in inference level of the model. However as I said, this projected intented more my understanding of streamlit and fastapi, being able in
the future, to create even more dynamic ML web apps, to connect these 2 libraries functionalities with each other. 

Feel free to look the code, and a reminder is that, both backend and frontend must run in separate servers, thus the port numbers should be
different, as well as they both should run in different terminals.

- For backend: `run src/backend/main.py` <br>
- For frontend: `run src/frontend/main.py` 
