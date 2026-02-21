# Testing Report

## Objective
To verify correctness, reliability, and robustness of the Smart Energy Consumption Prediction System.

---

## 1️⃣ Functional Testing

| Test Case ID | Scenario | Input | Expected Output | Result |
|--------------|----------|--------|----------------|--------|
| TC01 | Load dashboard | Open localhost URL | Dashboard loads without error | Pass |
| TC02 | View hourly chart | Navigate to charts section | Hourly chart displayed | Pass |
| TC03 | View daily chart | Navigate to charts section | Daily chart displayed | Pass |
| TC04 | View weekly chart | Navigate to charts section | Weekly chart displayed | Pass |
| TC05 | View monthly chart | Navigate to charts section | Monthly chart displayed | Pass |
| TC06 | View device chart | Navigate to charts section | Device-wise chart displayed | Pass |
| TC07 | Prediction with valid input | 4 members + AC selected | kWh + bill displayed | Pass |
| TC08 | Prediction without appliance selection | 3 members | Default average prediction | Pass |
| TC09 | High consumption scenario | Multiple heavy appliances | High usage warning displayed | Pass |
| TC10 | Invalid numeric input | Text in number field | Validation prevents submission | Pass |

---

## 2️⃣ Model Testing

- Verified model loads correctly using joblib.
- Verified feature order matches model training order.
- Verified predictions are numeric and reasonable.
- Compared predictions across different appliance combinations.

---

## 3️⃣ Performance Testing

- Dashboard loads in under 2 seconds.
- Chart generation completes without memory issues.
- Model prediction time < 0.1 seconds.

---

## 4️⃣ Edge Case Testing

- Zero appliance selection
- Single member household
- Large household size
- Multiple heavy appliance selection

All scenarios handled without application crash.

---

## Test Summary

Total Test Cases: 21  
Passed: 21  
Failed: 0  

Pass Rate: 100%

---

## Conclusion

The system is stable, functionally correct, and ready for deployment.