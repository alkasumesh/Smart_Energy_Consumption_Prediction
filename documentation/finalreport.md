# Final Project Report

## Project Title
Smart Energy Consumption Analysis and Prediction System

## Objective
To analyze home energy usage patterns and predict next month's electricity consumption using machine learning — presented through a simple, user-friendly web dashboard accessible to common users.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Flask
- Matplotlib
- HTML, CSS, JavaScript

## Dataset
smart_home_energy_consumption_large.csv
Features used: Home ID, Outdoor Temperature, Household Size, hour, day, weekday, month, lag_1, lag_24, rolling_mean_24

## Data Processing
- Combined Date + Time into Datetime index
- Created time-based features (hour, day, weekday, month)
- Created lag features (lag_1, lag_24)
- Created rolling_mean_24
- Removed missing values

## Models Developed
- Linear Regression
- LSTM

## Best Model
Linear Regression selected based on better R² score and stable performance.
Saved as: models/linear_regression_model.save

## Dashboard Features
- Hourly average consumption chart
- Daily consumption chart (last 30 days)
- Weekly consumption chart (last 12 weeks)
- Monthly consumption chart
- Device-wise energy share chart (donut)
- Smart energy saving tips

## Web Application
- Built using Flask backend and HTML/CSS/JavaScript frontend
- 4 pages: Home, Charts, Predict, Tips
- 3-step prediction wizard (members → appliances → month)
- Prediction shows estimated kWh and monthly bill in ₹
- All ML features (lag_1, lag_24, rolling_mean_24) hidden from user
- Deployed on localhost: http://127.0.0.1:5000

## Prediction Method
User inputs: number of members, appliances used, target month, electricity rate.
Backend auto-generates all ML features and blends physics-based estimate with the trained model.
Output: Estimated monthly kWh + Bill (₹) + Appliance-wise breakdown + Personalised tips

## Results
- Complete data processing pipeline built
- Linear Regression and LSTM models trained and evaluated
- Best model integrated into Flask web application
- 5 energy consumption charts generated using Matplotlib
- Interactive prediction wizard built for non-technical users
- 21 test cases passed — 100% pass rate

## Conclusion
The system successfully predicts next month's energy consumption based on household size and appliance usage. The web interface is designed for everyday users with no technical background — no ML terms, no complex inputs. Smart tips are generated automatically based on the user's appliance selection.

## Future Improvements
- Deploy to cloud (Render or AWS)
- Add user login and monthly history tracking
- Integrate real-time smart meter data
- Add regional language support (Tamil, Hindi)
- Extend to appliance-level individual predictions