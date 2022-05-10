output_text = {
    0 : "Extremely Weak",
    1 : "Weak",
    2 : "Normal",
    3 : "Overweight",
    4 : "Obesity",
    5 : "Extreme Obesity"
}

#Coding Part
#region start Coding
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import pickle

import os
path = os.path.dirname(os.path.realpath(__file__))

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder


def cvt_features(df, ohe):
    arr = ohe.transform(df["Gender"].to_numpy().reshape(-1,1)).toarray()
    new_df = pd.DataFrame(arr, columns=ohe.categories_[0])
    return pd.concat([df.reset_index(), new_df], axis=1).drop(["Gender","index"], axis=1)

def get_data():
    data = pd.read_csv(path + r"/dataset/data.csv")
    features = data.iloc[:,:-1]
    target = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = tts(features, target)
    return (X_train, X_test, y_train, y_test)

@st.cache
def load_model():
    with open(path + r"/models/RandomForest", "rb") as f:
        rf = pickle.load(f)
    
    return rf

X_train, X_test, y_train, y_test = get_data()
model = load_model()

ohe = OneHotEncoder(dtype=np.int32)
ohe.fit(X_train["Gender"].to_numpy().reshape(-1,1))    

X_train = cvt_features(X_train, ohe)
X_test = cvt_features(X_test, ohe)
#endregion


#Streamlit part
#region start Streamlit
st.header("BODY MASS INDEX (BMI) APP")
st.write("Tells your BMI using Machine Learning Models")

with st.form("input_form"):
    st.write("### Enter your details")
    gender = st.radio("Select your gender", ("Male", "Female"))
    height = st.number_input("Enter your height in cm", min_value=0, max_value=250, step=1)
    weight = st.number_input("Enter your weight in kg", min_value=0, max_value=500, step=1)

    submitted = st.form_submit_button("Submit")
    if submitted:
        row = pd.DataFrame([[gender, height, weight]], index=[0], columns=["Gender", "Height", "Weight"])
        cvted = cvt_features(row, ohe)
        res = model.predict(cvted)
        st.write(f"# You are {output_text[res[0]]}")


col1, col2 = st.columns([2,1])
with col1:
    st.write("# ")
    st.write("# ")
    gender = st.radio("Gender",("Male", "Female"))
    height = st.slider("Height", min_value=50, max_value=225, step=1, value=170)
    weight = st.slider("Weight", min_value=25, max_value=200, step=1, value=67)

with col2:
    st.write("# ")
    st.write("# ")
    row = pd.DataFrame([[gender, height, weight]], index=[0], columns=["Gender", "Height", "Weight"])
    cvted = cvt_features(row, ohe)
    res = model.predict(cvted)
    st.write(f"# {output_text[res[0]]}")
#endregion
