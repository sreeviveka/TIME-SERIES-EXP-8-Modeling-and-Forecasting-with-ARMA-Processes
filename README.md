# Ex-08: Modeling and Forecasting with ARMA Processes
## NAME : V S SREE VIVEKA
## REG NO : 2305001031
## DEPT : CSE
## Aim
To simulate an ARMA time series, fit ARMA models, forecast future values, and compare model performance using AIC, MAE, MSE, and RMSE.

## Objectives
1. To generate an ARMA time series.
2. To plot the series and its ACF and PACF.
3. To split data into training and testing sets.
4. To fit and compare different ARMA models.
5. To forecast future values.
6. To calculate MAE, MSE, and RMSE.
7. To generate an ARMA(2,1) process.

## Program

```python
# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 2: Generate ARMA(1,1) Data
np.random.seed(42)
ar = np.array([1, -0.7])
ma = np.array([1, 0.4])
data = ArmaProcess(ar, ma).generate_sample(200)
series = pd.Series(data)

# Step 3: Plot Time Series
plt.plot(series)
plt.title("ARMA(1,1) Time Series")
plt.show()

# Step 4: Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_acf(series, lags=20, ax=ax[0])
plot_pacf(series, lags=20, method="ywm", ax=ax[1])
plt.show()

# Step 5: Split Data
train = series[:160]
test = series[160:]
print("Training Data:", len(train))
print("Testing Data:", len(test))

# Step 6: Fit ARMA(1,1) Model
model = ARIMA(train, order=(1, 0, 1)).fit()
print(model.summary())

# Step 7: Forecast Future Values
forecast = model.forecast(steps=40)
print("Forecast:")
print(forecast)

# Step 8: Calculate Forecast Errors
print("MAE:", mean_absolute_error(test, forecast))
mse = mean_squared_error(test, forecast)
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))

# Step 9: Plot Actual and Forecast Values
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, forecast, label="Forecast")
plt.title("Actual vs Forecast")
plt.legend()
plt.show()

# Step 10: Compare Different ARMA Models
orders = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1)]

for order in orders:
    fit = ARIMA(train, order=order).fit()
    print("ARMA Order:", order)
    print("AIC:", fit.aic)
    print("BIC:", fit.bic)

# Step 11: Generate ARMA(2,1) Data
ar2 = np.array([1, -0.5, -0.2])
ma2 = np.array([1, 0.4])
data2 = ArmaProcess(ar2, ma2).generate_sample(200)

plt.plot(data2)
plt.title("ARMA(2,1) Time Series")
plt.show()

# Step 12: Fit ARMA(2,1) and Forecast
model2 = ARIMA(data2, order=(2, 0, 1)).fit()
forecast2 = model2.forecast(steps=10)

print("ARMA(2,1) Forecast:")
print(forecast2)
```

## Output
<img width="546" height="435" alt="image" src="https://github.com/user-attachments/assets/a1f292f6-74cd-4034-a7c0-2d13abaf1c48" />
<img width="847" height="374" alt="image" src="https://github.com/user-attachments/assets/6bda8687-8211-41ee-9e6e-8035e55e5d6d" />
<img width="444" height="87" alt="image" src="https://github.com/user-attachments/assets/dca6f28d-8894-46ba-aaa7-00d281a51713" />
<img width="780" height="452" alt="image" src="https://github.com/user-attachments/assets/7a1218b0-60ff-4a4f-8b78-50f14033525b" />
<img width="287" height="692" alt="image" src="https://github.com/user-attachments/assets/b0979e47-8f0c-48f3-b590-7c30642e8988" />
<img width="192" height="62" alt="image" src="https://github.com/user-attachments/assets/d8ef9db7-b26a-4d34-a2ff-f1c5d3c7429e" />
<img width="548" height="435" alt="image" src="https://github.com/user-attachments/assets/3d160881-7b3c-442d-85f1-cd2af891d5f3" />
<img width="226" height="207" alt="image" src="https://github.com/user-attachments/assets/8e52b664-3cce-43ca-9dde-6bf3a7b8df57" />
<img width="546" height="435" alt="image" src="https://github.com/user-attachments/assets/f695a3e4-a8b0-4b82-8506-701c53bef0e3" />
<img width="492" height="67" alt="image" src="https://github.com/user-attachments/assets/68c8020b-0105-4221-bec9-c72a464a776b" />

## Result
Thus, ARMA processes were generated and analyzed, different ARMA models were fitted and compared using AIC and BIC, and forecasting performance was evaluated using MAE, MSE, and RMSE. The best forecasting model was identified based on the lowest RMSE, and an ARMA(2,1) process was successfully generated and forecasted.
