import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Forest Cover Type Predictor")

model = joblib.load("best_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("🌲 Forest Cover Type Prediction")

st.write("Enter terrain and soil details")

user_input = {
    "Elevation": st.number_input("Elevation", 0),
    "Aspect": st.number_input("Aspect", 0),
    "Slope": st.number_input("Slope", 0),
    "Horizontal_Distance_To_Hydrology": st.number_input("Horizontal_Distance_To_Hydrology", 0),
    "Vertical_Distance_To_Hydrology": st.number_input("Vertical_Distance_To_Hydrology", 0),
    "Horizontal_Distance_To_Roadways": st.number_input("Horizontal_Distance_To_Roadways", 0),
    "Hillshade_9am": st.number_input("Hillshade_9am", 0),
    "Hillshade_Noon": st.number_input("Hillshade_Noon", 0),
    "Hillshade_3pm": st.number_input("Hillshade_3pm", 0),
    "Horizontal_Distance_To_Fire_Points": st.number_input("Horizontal_Distance_To_Fire_Points", 0),
    "Wilderness_Area": st.selectbox("Wilderness_Area", [1, 2, 3, 4]),
    "Soil_Type": st.selectbox("Soil_Type", list(range(1, 41)))
}

# Feature engineering (same as training)
user_input["Hillshade_Diff"] = (
    user_input["Hillshade_Noon"] - user_input["Hillshade_9am"]
)

input_df = pd.DataFrame([user_input])
X_input = preprocessor.transform(input_df)

if st.button("Predict Forest Cover Type"):
    prediction = model.predict(X_input)[0]
    st.success(f"🌲 Predicted Forest Cover Type: {prediction}")

