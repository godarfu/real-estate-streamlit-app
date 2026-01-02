import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("🏠 Real Estate Investment Advisor")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("india_housing_prices_small.csv")


df = load_data()
st.success("Dataset loaded successfully")

# Feature engineering
df['Price_per_SqFt'] = (df['Price_in_Lakhs'] * 100000) / df['Size_in_SqFt']
median_price = df['Price_in_Lakhs'].median()
df['Good_Investment'] = ((df['Price_in_Lakhs'] <= median_price) & (df['BHK'] >= 2)).astype(int)

# Features and targets
X = df[['BHK', 'Size_in_SqFt', 'Price_per_SqFt']]
y_class = df['Good_Investment']
y_reg = df['Price_in_Lakhs']

# Train models
@st.cache_resource
def train_models():
    X_train, X_test, y1_train, y1_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y2_train, y2_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100)
    reg = RandomForestRegressor(n_estimators=100)

    clf.fit(X_train, y1_train)
    reg.fit(X_train, y2_train)

    return clf, reg

clf, reg = train_models()
st.success("Models trained successfully")

# User input
bhk = st.slider("BHK", 1, 5, 2)
size = st.number_input("Size in SqFt", 500, 5000, 1000)
pps = st.number_input("Price per SqFt", 2000, 20000, 6000)

input_data = np.array([[bhk, size, pps]])

if st.button("Predict"):
    invest = clf.predict(input_data)[0]
    price = reg.predict(input_data)[0]

    st.write("✅ Good Investment" if invest == 1 else "❌ Not a Good Investment")
    st.write("💰 Estimated Price:", round(price, 2), "Lakhs")
