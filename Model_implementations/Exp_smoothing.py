
## Do imports

import time
import sys
sys.path.append("../Utilities")
from Preprocessing import *
from Backtest import *
from Plotting import *



## Get pre-processed and split data

print(f"Fitting smoother...")

# Start timer
start = time.perf_counter()

# Choose data configs
label_name, asset_mode = 'EURUSD Curncy', 'macro'
#label_name, asset_mode = 'AAPL', 'stock'

# Get data
X_train, Y_train, X_val, Y_val, _, _ = get_preprocessed_datasets(label_name, asset_mode, diff_mode = None, smoothing = True)

# End timer
end = time.perf_counter()

print(f"Data smoothed in {time.strftime('%H:%M:%S', time.gmtime(end - start))}.")

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")



## Get smoothed data and post-process (should really be using label_name to get this)
Y_train = Y_train[1:]
Y_pred = np.concatenate([X_train[:, 0], X_val[:, 0]])
Y_pred_logret = take_log_returns_np(Y_pred)
Y_pred_train = Y_pred_logret[:-Y_val.shape[0]]
Y_pred_val = Y_pred_logret[-Y_val.shape[0]:]


print(Y_pred_train.shape, Y_train.shape)
print(Y_pred_val.shape, Y_val.shape)


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
plot_time_series([Y_train, Y_pred_train], start = 200, end = 300)
plot_time_series([Y_val, Y_pred_val])

plot_backtest_results(backtest_results_train)
plot_backtest_results(backtest_results_val)
