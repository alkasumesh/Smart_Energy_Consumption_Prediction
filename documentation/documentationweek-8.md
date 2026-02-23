Week 8 – Web Application Deployment 
Flask API Routes

GET / → Loads main dashboard

POST /api/predict → Returns predicted kWh and bill

User Inputs

Electricity rate (₹ per unit)

Hidden ML Logic (Backend Only)

Daily energy series resampling

MinMax scaling

7-day sequence preparation

LSTM inference

Inverse scaling

No engineered lag features are exposed to the user.

Prediction Method 

The final system uses:

Last 7 days of daily energy consumption

LSTM time-series forecasting model

Model predicts:

→ Next day energy consumption

Monthly consumption is estimated as:

Predicted Daily × 30

Electricity bill:

Monthly kWh × User-defined rate