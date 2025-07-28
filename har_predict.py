import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("models/har_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Activity label mapping based on your model's training order
activity_map = {
    0: "LAYING",
    1: "SITTING",
    2: "STANDING",
    3: "WALKING",
    4: "WALKING_UPSTAIRS",
    5: "WALKING_DOWNSTAIRS"
}

def predict_activity(raw_input_str):
    try:
        raw_data = [float(x.strip()) for x in raw_input_str.split(",")]
        if len(raw_data) != 100:
            raise ValueError("❌ Error: Input must contain exactly 100 feature values.")

        # Scale the input
        input_scaled = scaler.transform([raw_data])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get activity name
        activity_name = activity_map.get(prediction, "Unknown Activity")
        
        print(f"✅ Predicted Activity: {activity_name} (Label: {prediction})")

    except Exception as e:
        print("❌ Error:", e)

# Run if file executed directly
if __name__ == "__main__":
    input_str = input("🧠 Enter 100 comma-separated feature values: ")
    predict_activity(input_str)
