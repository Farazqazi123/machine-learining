from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_true = [10, 20]
y_pred = [8, 25]

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)
