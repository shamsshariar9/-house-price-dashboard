import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Dashboard", layout="wide")

st.title("🎓 Student Performance Dashboard")

data = {
    "Student": ["Ali", "Sara", "John", "Aina", "David"],
    "Math": [85, 92, 78, 88, 95],
    "Science": [90, 89, 80, 91, 87],
    "English": [88, 94, 75, 90, 85]
}

df = pd.DataFrame(data)

st.subheader("Student Scores")
st.dataframe(df)

col1, col2, col3 = st.columns(3)
col1.metric("Students", len(df))
col2.metric("Average Math", round(df["Math"].mean(), 2))
col3.metric("Average Science", round(df["Science"].mean(), 2))

student = st.selectbox("Select Student", df["Student"])

selected = df[df["Student"] == student]

scores = selected[["Math", "Science", "English"]].iloc[0]

fig, ax = plt.subplots()
ax.bar(scores.index, scores.values)
ax.set_ylabel("Score")
ax.set_ylim(0, 100)

st.pyplot(fig)

st.subheader("Class Average")
avg_scores = df[["Math", "Science", "English"]].mean()
st.bar_chart(avg_scores)
