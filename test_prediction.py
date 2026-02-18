from model_utils import predict_energy

sample_input = {
    "Home ID": 1,
    "Outdoor Temperature (°C)": 25,
    "Household Size": 3,
    "hour": 12,
    "day": 5,
    "weekday": 2,
    "month": 8,
    "lag_1": 0.5,
    "lag_24": 0.6,
    "rolling_mean_24": 0.7
}

print("Predicted Energy:", predict_energy(sample_input))
