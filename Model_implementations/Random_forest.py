
## Do imports

from sklearn.ensemble import RandomForestRegressor
import time
import sys
sys.path.append("../Utilities")
from Preprocessing import *
from Backtest import *
from Plotting import *
np.random.seed(1)



## Get pre-processed and split data

print(f"Pre-processing...")

# Start timer
start = time.perf_counter()

# Choose data configs
label_name, asset_mode = 'EURUSD Curncy', 'macro'
#label_name, asset_mode = 'AAPL', 'stock'
diff_mode = 'logret'
#diff_mode = 'fracdiff'
smoothing = False
scaling = False

# Get data
X_train, Y_train, X_val, Y_val, _, _ = get_preprocessed_datasets(label_name, asset_mode, diff_mode, smoothing, scaling)

# Dimension reduction
compression_mode = None
X_train = reduce_dimension(X_train, compression_mode, asset_mode, diff_mode, smoothing, scaling)
X_val = reduce_dimension(X_val, compression_mode, asset_mode, diff_mode, smoothing, scaling)

# Laggify the Xs and chop the Ys
num_lags = 15
X_train_lagged = laggify_data(X_train, num_lags)
Y_train_chopped = Y_train[num_lags:]
X_val_with_train_tail = stitch_tail(X_train, X_val, num_lags)
X_val_lagged = laggify_data(X_val_with_train_tail, num_lags)

# End timer
end = time.perf_counter()

print(f"Data pre-processed with smoothing = {smoothing} and diff_mode = {diff_mode} in {time.strftime('%H:%M:%S', time.gmtime(end - start))}.")

print(f"Training data shape: {X_train_lagged.shape}")
print(f"Validation data shape: {X_val_lagged.shape}")



## Train the model

print(f"Training model...")

# Start timer
start = time.perf_counter()

random_forest = RandomForestRegressor(verbose = 2, max_depth = None)
random_forest.fit(X_train_lagged, Y_train_chopped.reshape(Y_train_chopped.shape[0]))

# End timer
end = time.perf_counter()
print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(end - start))}")



## Make predictions
Y_pred_train = random_forest.predict(X_train_lagged).reshape(Y_train_chopped.shape)
Y_pred_val = random_forest.predict(X_val_lagged).reshape(Y_val.shape)



## Generate backtest results
num_pos, pred_pos, frac_correct_sign, precision, recall = sign_stats(Y_train_chopped, Y_pred_train)
print(f"Train fraction of data positive: {num_pos:.3f}")
print(f"Train fraction of predictions positive: {pred_pos:.3f}")
print(f"Train fraction with correct sign: {frac_correct_sign:.3f}")
print(f"Train precision: {precision:.3f}")
print(f"Train recall: {recall:.3f}")
backtest_results_train = pnl_backtest(Y_train_chopped, Y_pred_train)

num_pos, pred_pos, frac_correct_sign, precision, recall = sign_stats(Y_val, Y_pred_val)
print(f"Val fraction of data positive: {num_pos:.3f}")
print(f"Val fraction of predictions positive: {pred_pos:.3f}")
print(f"Val fraction with correct sign: {frac_correct_sign:.3f}")
print(f"Val precision: {precision:.3f}")
print(f"Val recall: {recall:.3f}")
backtest_results_val = pnl_backtest(Y_val, Y_pred_val)



## Plot results
plot_time_series([Y_train[num_lags:], Y_pred_train], start = 200, end = 300)
plot_time_series([Y_val, Y_pred_val])

plot_backtest_results(backtest_results_train)
plot_backtest_results(backtest_results_val)
