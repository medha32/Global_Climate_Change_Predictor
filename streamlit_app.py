import streamlit as st
import pandas as pd

import numpy as np

# Define input fields for user input
st.title("Impact Score Prediction App")
st.write("Enter environmental data to predict the Impact Score.")

# Create input fields
co2 = st.number_input("CO2 Emissions", min_value=0.0, format="%.2f")
sea_level = st.number_input("Sea Level Rise", min_value=0.0, format="%.2f")
temperature = st.number_input("Temperature", min_value=-50.0, max_value=60.0, format="%.2f")
precipitation = st.number_input("Precipitation", min_value=0.0, format="%.2f")
humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, format="%.2f")
wind_speed = st.number_input("Wind Speed", min_value=0.0, format="%.2f")

# Predict button
if st.button("Predict Impact Score"):
    input_data = np.array([[temperature, co2, sea_level, precipitation, humidity, wind_speed]])
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Impact Score: {prediction[0]:.4f}")
