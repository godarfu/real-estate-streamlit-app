import streamlit as st
import joblib
import numpy as np
import os

st.title("Real Estate Investment Advisor")

st.write("✅ App started")

st.write("📂 Files available in app folder:")
st.write(os.listdir())

try:
    clf = joblib.load("investment_model.pkl")
    reg = joblib.load("price_model.pkl")
    st.success("✅ Models loaded successfully")
except Exception as e:
    st.error("❌ Model loading failed")
    st.error(e)
    st.stop()

bhk = st.slider("BHK", 1, 5, 2)
size = st.number_input("Size in SqFt", 500, 5000, 1000)
pps = st.number_input("Price per SqFt", 2000, 20000, 6000)
age = st.slider("Property Age", 0, 30, 5)
schools = st.slider("Nearby Schools", 0, 10, 3)
hospitals = st.slider("Nearby Hospitals", 0, 10, 2)
parking = st.slider("Parking Spaces", 0, 3, 1)

input_data = np.array([[bhk, size, pps, age, schools, hospitals, parking]])

if st.button("Predict"):
    invest = clf.predict(input_data)[0]
    price = reg.predict(input_data)[0]

    st.write("Prediction result:")
    st.write("Good Investment:", invest)
    st.write("Future Price:", price)
