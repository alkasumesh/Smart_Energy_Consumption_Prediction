Final Project Report 
Project Title

Smart Energy Consumption Analysis and Prediction System

Objective

To analyze household energy usage trends and forecast future electricity consumption using deep learning-based time-series prediction, presented through a user-friendly web dashboard.

Technologies Used

Python

Pandas

NumPy

Scikit-learn

TensorFlow / Keras

Flask

Matplotlib

HTML, CSS, JavaScript

Dataset

smart_home_energy_consumption_large.csv

Primary feature used:

Energy_kWh (daily aggregated)

Data Processing

Combined Date + Time into Datetime index

Sorted chronologically

Resampled to daily total energy consumption

Applied MinMax scaling

Created sequential data (7-day lookback window)

Models Developed

Linear Regression (Baseline)

LSTM (Final Selected Model)

Final Model

LSTM selected based on:

Lower RMSE

Lower MAE

Better time-series learning capability

Sequential dependency modeling

Saved as:

models/lstm_energy_model.h5
Dashboard Features

Hourly consumption visualization

Daily consumption visualization

Weekly consumption visualization

Monthly consumption visualization

Device-wise energy share chart

Monthly energy prediction

Smart energy-saving suggestions

Web Application

Built using Flask backend

Frontend built using HTML/CSS/JavaScript

LSTM model integrated into backend

Prediction API returns monthly kWh and bill

Deployment:

http://127.0.0.1:5000

Run command:

python app.py