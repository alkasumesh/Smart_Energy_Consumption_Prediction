Week 6 – Model Evaluation 
Compared Models

Linear Regression (Baseline Model)

LSTM (Long Short-Term Memory)

Metrics Used

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Evaluation Summary

Linear Regression was implemented as a baseline regression model using engineered time-based features.

LSTM was implemented as a deep learning time-series forecasting model using sequential daily energy data.

After evaluation:

LSTM achieved lower MAE and RMSE

LSTM captured temporal patterns better

LSTM handled sequential dependencies effectively

Best Model Selected

LSTM was selected as the final model due to superior time-series forecasting capability and improved prediction accuracy.

Model saved as:

models/lstm_energy_model.h5