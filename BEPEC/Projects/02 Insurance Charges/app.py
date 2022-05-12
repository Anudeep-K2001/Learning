import streamlit as st

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.metrics import r2_score, mean_squared_error
import pickle


st.set_page_config(page_title="Insurance Charge Prediction", page_icon="📈")


def get_models():
    with open(r"./models/SVR", "rb") as f:
        svr = pickle.load(f)

    with open(r"./models/RandomForest", "rb") as f:
        rf = pickle.load(f)

    return svr, rf

@st.cache
def get_data():
    data = pd.read_csv(r"./datasets/data.csv")
    return data

@st.cache
def split_data(data):
    d1 = data[data.smoker == "no"]  #splitting into 2 datasets based on smoker or not
    d2 = data[data.smoker == "yes"]
    f1 = d1.iloc[:,:-1]
    T1 = d1.iloc[:,-1]
    f2 = d2.iloc[:,:-1]
    T2 = d2.iloc[:,-1]
    t1 = np.log(T1)
    t2 = np.log(T2)
    X_train_1, X_test_1, y_train_1, y_test_1 = tts(f1, t1)
    X_train_2, X_test_2, y_train_2, y_test_2 = tts(f2, t2)
    ohe1 = OneHotEncoder(dtype=np.int32)
    ohe2 = OneHotEncoder(dtype=np.int32)

    ohe1.fit(X_train_1[['sex', 'smoker', 'region']])
    ohe2.fit(X_train_2[['sex', 'smoker', 'region']])

    mms1 = MinMaxScaler()
    mms2 = MinMaxScaler()

    mms1.fit(X_train_1[['age', 'bmi']])
    mms2.fit(X_train_2[['age', 'bmi']])
    return ohe1, ohe2, mms1, mms2


def cvt_data(df, encoders):
    ohe, mms = encoders
    arr_1 = ohe.transform(df[['sex', 'smoker', 'region']]).toarray()
    df_2 = pd.DataFrame(arr_1,columns=[i for k in ohe.categories_ for i in k], index=range(arr_1.shape[0]))
    df_3 = pd.concat([df.reset_index(), df_2], axis=1).drop(["index", "sex", "smoker", "region"],axis=1)
    arr_2 = mms.transform(df[['age','bmi']])
    df_4 = pd.DataFrame(arr_2, columns = mms.feature_names_in_, index=range(arr_2.shape[0]))
    df_3[['age', 'bmi']] = df_4
    drop1 = ["female", "no"]
    drop2 = ["female", "yes"]
    if "no" in df_3.columns:
        return df_3.drop(drop1, axis=1)
    else:
        return df_3.drop(drop2, axis=1)


data = get_data()
ohe1, ohe2, mms1, mms2 = split_data(data)
svr, rf = get_models()




st.title("Insurance Charge Prediction")
st.markdown("""The current web app predicts the insurance charge that might cost form the following [dataset](https://www.kaggle.com/code/sudhirnl7/linear-regression-tutorial/data)""")

st.sidebar.title("Insurance Charge Prediction")
st.sidebar.markdown("""The current web app predicts the insurance charge that might cost form the following dataset with below factors""")
st.sidebar.markdown("""
                - age
                - sex
                - BMI
                - children
                - smoker
                - region
            """)






with st.form("input_form"):
    st.write("### Enter the details")
    age = st.number_input("Enter age", min_value=18, max_value=100, step=1)
    bmi = st.number_input("Enter BMI", min_value=15. , max_value=50. , step=0.5)
    children = st.number_input("Children", min_value=0, max_value=5, step=1)
    sex = st.radio("Gender", ("male", "female"))
    smoker = st.radio("Smoker", ("no", "yes"))
    region = st.selectbox("Enter a region", ("north-east", "north-west", "south-east", "south-west"))
    submitted = st.form_submit_button("Submit")
    if submitted:
        region_format = {
            "north-east" : "northeast",
            "north-west" : "northwest",
            "south-east" : "southeast",
            "south-west" : "southwest"
        }
        entry = [age, sex, bmi, children, smoker, region_format[region]]
        if smoker == "yes":
            df = cvt_data(pd.DataFrame([entry], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'], index=range(1)), [ohe2,mms2])
            st.write(f"### Insurance Charge is : {round(np.exp(rf.predict(df)[0]),2)} $")
        else:
            df = cvt_data(pd.DataFrame([entry], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'], index=range(1)), [ohe1, mms1])
            st.write(f"### Insurance Charge is : {round(np.exp(svr.predict(df)[0]),2)} $")
        
