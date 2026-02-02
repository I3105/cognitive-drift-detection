%%writefile app.py
import streamlit as st

st.title("Streamlit in Google Colab")

name = st.text_input("Enter your name:")

if st.button("Greet me"):
    st.write(f"Hello, {name}!")

age = st.slider("Select your age:", 0, 100, 25)
st.write(f"Your age is: {age}")
