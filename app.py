import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load("penguins_rf_model.pkl")
label_encoder = joblib.load("penguins_label_encoder.pkl")

# Mapping for categorical variables
island_mapping = {"Torgersen": 0, "Biscoe": 1, "Dream": 2}
sex_mapping = {"male": 0, "female": 1}

# Streamlit UI
st.title("Penguin Species Predictor 🐧")
st.write("Enter the penguin features below to predict its species.")

# User inputs
island = st.selectbox("Island", options=["Torgersen", "Biscoe", "Dream"])
bill_length = st.number_input("Bill Length (mm)", min_value=30.0, max_value=70.0, step=0.1)
bill_depth = st.number_input("Bill Depth (mm)", min_value=10.0, max_value=25.0, step=0.1)
flipper_length = st.number_input("Flipper Length (mm)", min_value=170.0, max_value=250.0, step=1.0)
body_mass = st.number_input("Body Mass (g)", min_value=2500, max_value=6500, step=10)
sex = st.radio("Sex", options=["male", "female"])

# Prediction button
if st.button("Predict Species"):
    # Convert inputs to DataFrame
    input_data = pd.DataFrame(
        [[island_mapping[island], bill_length, bill_depth, flipper_length, body_mass, sex_mapping[sex]]],
        columns=["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
    )

    # Make prediction
    prediction = model.predict(input_data)
    predicted_species = label_encoder.inverse_transform(prediction)[0]

    # Display result
    st.success(f"Predicted Penguin Species: **{predicted_species}** 🐧")