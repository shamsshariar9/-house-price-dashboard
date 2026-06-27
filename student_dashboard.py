import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="House Price AI Dashboard",
    layout="wide",
    page_icon="🏠"
)

st.title("🏠 House Price Prediction AI Dashboard")
st.markdown("### 📊 Predict house prices using Machine Learning (Linear Regression)")
st.divider()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="House Price Prediction Dashboard", layout="wide")

st.title("🏠 House Price Prediction Dashboard")

# Load dataset
df = pd.read_csv("HousingData.csv")

# STEP 1: Clean missing values
df = df.dropna()

# STEP 2: Keep only numeric columns
df = df.select_dtypes(include=['number'])

# STEP 3: Reset index (important for Streamlit)
df = df.reset_index(drop=True)

# Features
X = df[["RM", "LSTAT"]]
y = df["MEDV"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
st.subheader("🔮 Predict House Price")

st.sidebar.header("🏠 Input Features")

rm = st.sidebar.slider(
    "Average Number of Rooms (RM)",
    float(df["RM"].min()),
    float(df["RM"].max()),
    float(df["RM"].mean())
)

lstat = st.sidebar.slider(
    "Lower Status Population (%)",
    float(df["LSTAT"].min()),
    float(df["LSTAT"].max()),
    float(df["LSTAT"].mean())
)
pred = model.predict([[rm, lstat]])

st.success(f"Predicted House Price: ${pred[0]:.2f}k")

# Data preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Graph
st.subheader("📈 Actual vs Predicted")

y_pred = model.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")

st.pyplot(fig)
