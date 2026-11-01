import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib


df = pd.read_csv(r"D:\shreeram codes\AIML Project\Logistic Regression\logistic regression dataset-Social_Network_Ads.csv") #change the path of csv
# print(df.head())
# print(df.describe())
# print(df.info())

df = df.iloc[:,1:] #deletes the first column
# plt.scatter(df["Age"],df["EstimatedSalary"],c=df["Purchased"])
# plt.show()

X = df.iloc[:,0:3]
y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

numeric_features = ["Age","EstimatedSalary"]
categorical_features = ["Gender"]

numeric_pipeline = Pipeline([("impute",SimpleImputer(strategy="median")),
                             ("scale",StandardScaler())])

categorical_pipeline = Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
                                 ("encode", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

model_pipeline = Pipeline([("prep",preprocessor),
                           ("clf",LogisticRegression(random_state=42))])
model_pipeline.fit(X_train,y_train)

y_prep = model_pipeline.predict(X_test)
score = accuracy_score(y_test, y_prep)
print(score)
joblib.dump(model_pipeline,"logistic_Regression_model.joblib")

#streamlit 
import streamlit as st

model = joblib.load("logistic_Regression_model.joblib")
st.title("Will This customer purchase via AD?")
st.write("Enter the customers details model will predicts that the customer will purchase the product through the Ad")

Gender = st.selectbox("Gender", ["Male","Female"])
Age = st.slider("Age of customer", 0,100,5)
EstimatedSalary = st.slider("Estimates salary of customer", 0, 250000, 5000)

if st.button("Predict"):

    input_row = pd.DataFrame([{
        "Gender" : Gender,
        "Age" : Age,
        "EstimatedSalary" : EstimatedSalary
    }])

    prediction = model.predict(input_row)[0]

    probability = model.predict_proba(input_row)[0][1]

    if prediction == 1:

        st.success(f"Prediction: Purchase (confidence: {probability:.0%})")
    else:

        st.error(f"Prediction: Not Purchase (confidence: {1 - probability:.0%})")