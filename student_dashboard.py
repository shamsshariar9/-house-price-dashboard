import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="House Price Prediction Dashboard", layout="wide")

st.title("🏠 House Price Prediction Dashboard")

# Load dataset
df = pd.read_csv("HousingData.csv")

# Clean data
df = df.dropna()
df = df.select_dtypes(include=['number'])
df = df.reset_index(drop=True)

# Features
X = df[["RM", "LSTAT"]]
y = df["MEDV"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction section
st.subheader("🔮 Predict House Price")

rm = st.slider("Average Rooms (RM)", float(df["RM"].min()), float(df["RM"].max()), float(df["RM"].mean()))
lstat = st.slider("Lower Status % (LSTAT)", float(df["LSTAT"].min()), float(df["LSTAT"].max()), float(df["LSTAT"].mean()))

prediction = model.predict([[rm, lstat]])

st.success(f"Predicted House Price: ${prediction[0]:.2f}k")

# Dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Graph
st.subheader("📈 Actual vs Predicted")

y_pred = model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")

st.pyplot(fig)

st.success("Model trained successfully!")
