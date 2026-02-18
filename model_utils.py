import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/linear_regression_model.save")

# Get feature names used during training
feature_names = model.feature_names_in_

def predict_energy(input_dict):
    """
    input_dict: dictionary with feature_name: value
    returns predicted energy consumption
    """
    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure correct feature order
    input_df = input_df[feature_names]

    prediction = model.predict(input_df)

    return float(prediction[0])
