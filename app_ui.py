import streamlit as st
import requests
import pandas as pd

st.title("Enhanced AutoML Pipeline Generator")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    algorithm = st.selectbox("Select AutoML Algorithm", ["h2o", "autosklearn", "tpot"])

    if st.button("Train Model"):
        response = requests.post("http://127.0.0.1:8000/train/", files={"file": uploaded_file}, data={"algorithm": algorithm})
        st.write(response.json())

    if st.button("Make Prediction"):
        response = requests.post("http://127.0.0.1:8000/predict/", files={"file": uploaded_file})
        st.write(response.json())

st.write("### Model Performance Metrics")
st.image("metrics.png")  # Visualization of classification report (to be generated separately)
