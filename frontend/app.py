import streamlit as st
import requests

st.title("AI Exam Anxiety Detector")

st.write("Enter a sentence about your exam feelings")

user_input = st.text_area("Your text:")

if st.button("Predict"):

    url = "http://127.0.0.1:8000/predict"

    data = {
        "text": user_input
    }

    response = requests.post(url, json=data)

    result = response.json()

    prediction = result["prediction"]

    if prediction == "Low":
        st.success("Low Anxiety 😊")

    elif prediction == "Moderate":
        st.warning("Moderate Anxiety 😐")

    elif prediction == "High":
        st.error("High Anxiety 😟")