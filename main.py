import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle

# Load the trained model
def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

# Define the risk classification function
def classify_risk(score):
    if score > 75:
        return "High Risk"
    elif 40 <= score <= 75:
        return "Low Risk"
    else:
        return "No Risk"

# Define the Streamlit app
def main():
    st.title("Climate Impact Prediction App")
    st.write("Enter climate-related parameters to predict the Impact Score and assess climate risk.")

    # Input fields
    temperature = st.number_input("Temperature", value=25.0)
    co2_emissions = st.number_input("CO2 Emissions", value=400.0)
    sea_level_rise = st.number_input("Sea Level Rise", value=3.0)
    precipitation = st.number_input("Precipitation", value=50.0)
    humidity = st.number_input("Humidity", value=60.0)
    wind_speed = st.number_input("Wind Speed", value=10.0)

    model = load_model()
    features = np.array([[temperature, co2_emissions, sea_level_rise, precipitation, humidity, wind_speed]])
    
    if st.button("Predict"):
        prediction = model.predict(features)[0]
        risk_level = classify_risk(prediction)
        st.write(f"Predicted Impact Score: {prediction:.2f}")
        st.write(f"Climate Risk Level: {risk_level}")

if __name__ == "__main__":
    main()
