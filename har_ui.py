import streamlit as st
import numpy as np
import joblib
import os

# Load model and scaler from the models directory
model_path = os.path.join("models", "har_model.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("Human Activity Recognition (100 Features)")

st.write("Enter 100 comma-separated values from the sensor:")

user_input = st.text_area("Input Data", height=150)

if st.button("Predict Activity"):
    try:
        # Convert input string to list of floats
        input_list = [float(x.strip()) for x in user_input.split(",") if x.strip()]
        
        if len(input_list) != 100:
            st.error("Please enter exactly 100 values.")
        else:
            input_array = np.array(input_list).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            
            # Optional: Map labels to names
            activity_map = {
                1: "WALKING",
                2: "WALKING_UPSTAIRS",
                3: "WALKING_DOWNSTAIRS",
                4: "SITTING",
                5: "STANDING",
                6: "LAYING"
            }

            activity_name = activity_map.get(prediction, "Standing")
            st.success(f"Predicted Activity: **{activity_name}**")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
