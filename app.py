import pickle
import pandas as pd
import streamlit as st

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("House Price Prediction")
st.write(
    "Enter the house details below to estimate its price "
    "using a Linear Regression model."
)

st.divider()

# User inputs
area = st.number_input(
    "House Area (square feet)",
    min_value=100,
    max_value=10000,
    value=1200,
    step=100
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

age = st.number_input(
    "Age of House (years)",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)

# Predict price
if st.button("Predict House Price", type="primary"):
    input_data = pd.DataFrame(
        [[area, bedrooms, age]],
        columns=["area", "bedrooms", "age"]
    )

    predicted_price = model.predict(input_data)[0]

    st.subheader("Predicted House Price")
    st.success(f"₹{predicted_price:,.2f}")

    st.caption(
        "This is an educational estimate based on synthetic sample data, "
        "not a reliable estimate of actual market value."
    )

st.divider()
st.caption("Built with Python, Scikit-learn and Streamlit")
