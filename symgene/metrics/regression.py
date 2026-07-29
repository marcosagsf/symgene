import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mse(y_true, y_pred):   return float(mean_squared_error(y_true, y_pred))
def rmse(y_true, y_pred):  return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def mae(y_true, y_pred):   return float(mean_absolute_error(y_true, y_pred))
def r2(y_true, y_pred):    return float(r2_score(y_true, y_pred))
def nrmse(y_true, y_pred):
    rng = float(np.max(y_true) - np.min(y_true))
    return rmse(y_true, y_pred) / rng if rng > 1e-9 else rmse(y_true, y_pred)
def mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-9
    if not mask.any(): return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
