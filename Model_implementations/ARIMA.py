
## Do imports

import pandas as pd
import time
from statsmodels.tsa.arima.model import ARIMA
import sys
sys.path.append("../Utilities")
from Preprocessing import *
from Backtest import *
from Plotting import *



## Choose configs
label_name, asset_mode = 'EURUSD Curncy', 'macro'
num_lags = 15
frac_diff_on = False



## Get data and arrange into correct format

# Get data (no pre-processing)
data_all_df, data_train_df, data_val_df, data_test_df = get_datasets(asset_mode)

# Pick column of interest
data_all_df = data_all_df[[label_name]]
data_train_df = data_train_df[[label_name]]
data_val_df = data_val_df[[label_name]]
data_test_df = data_test_df[[label_name]]

# Choose start point in training set (ARIMA needs at least a few points)
arima_start = 20

# Get labels
Y_train = take_log_returns_np(data_train_df.to_numpy())[arima_start:]
Y_val = take_log_returns_np(data_val_df.to_numpy())

# Fractional differencing
if frac_diff_on is True:
    data_all_df, data_train_df, data_val_df, data_test_df = frac_diff_data(data_all_df, data_train_df, data_val_df, data_test_df)

# Make numpy arrays
X_train = data_train_df.to_numpy()
X_val = data_val_df.to_numpy()

# Containers used during prediction
X_running = X_train[0:arima_start]
Y_pred = np.array([])



## Predict on the training set and then the validation set

print(f"Training model and predicting on training set...")

# Start timer
start = time.perf_counter()

for pred_idx in range(arima_start, X_train.shape[0]):
    
    model = ARIMA(X_running, order = (num_lags, 1, 0))
    model_fit = model.fit(method_kwargs = {"warn_convergence": False})
    Y_pred = np.append(Y_pred, model_fit.forecast()[0])
    X_running = np.append(X_running, X_train[pred_idx])


print(f"Training model and predicting on validation set...")

for pred_idx in range(X_val.shape[0]):
    
    model = ARIMA(X_running, order = (num_lags, 1, 0))
    model_fit = model.fit(method_kwargs = {"warn_convergence": False})
    Y_pred = np.append(Y_pred, model_fit.forecast()[0])
    X_running = np.append(X_running, X_val[pred_idx])


# End timer
end = time.perf_counter()
print(f"Total run time: {time.strftime('%H:%M:%S', time.gmtime(end - start))}")



## Post-process predictions
Y_pred_logret = take_log_returns_np(Y_pred)
Y_pred_train = Y_pred_logret[:-Y_val.shape[0]]
Y_pred_val = Y_pred_logret[-Y_val.shape[0]:]



## Generate backtest results
num_pos, pred_pos, frac_correct_sign, precision, recall = sign_stats(Y_train, Y_pred_train)
print(f"Train fraction of data positive: {num_pos:.3f}")
print(f"Train fraction of predictions positive: {pred_pos:.3f}")
print(f"Train fraction with correct sign: {frac_correct_sign:.3f}")
print(f"Train precision: {precision:.3f}")
print(f"Train recall: {recall:.3f}")
backtest_results_train = pnl_backtest(Y_train, Y_pred_train)

num_pos, pred_pos, frac_correct_sign, precision, recall = sign_stats(Y_val, Y_pred_val)
print(f"Val fraction of data positive: {num_pos:.3f}")
print(f"Val fraction of predictions positive: {pred_pos:.3f}")
print(f"Val fraction with correct sign: {frac_correct_sign:.3f}")
print(f"Val precision: {precision:.3f}")
print(f"Val recall: {recall:.3f}")
backtest_results_val = pnl_backtest(Y_val, Y_pred_val)



## Plot results
plot_time_series([Y_train, Y_pred_train], start = 0, end = 150)
plot_time_series([Y_val, Y_pred_val])

plot_backtest_results(backtest_results_train)
plot_backtest_results(backtest_results_val)
