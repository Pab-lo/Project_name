
import time
import xgboost as xgb
from Preprocessing import *
from Backtest import *
from Plotting import *



## Get pre-processed and split data

print(f"Pre-processing...")

# Start timer
start = time.perf_counter()

# Choose data configs
#label_name = 'EURUSD Curncy'
#asset_mode = 'macro'
label_name = 'AAPL'
asset_mode = 'stock'
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

num_round = 2
param = {'max_depth':2, 'eta':1, 'objective':'reg:squarederror' }
train_xgb = xgb.DMatrix(X_train_lagged, label=Y_train_chopped)
val_xgb = xgb.DMatrix(X_val_lagged, label=Y_val)
xgboost_model = xgb.train(param, train_xgb, num_round)

# End timer
end = time.perf_counter()
print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(end - start))}")



## Make predictions
y_pred_train = xgboost_model.predict(train_xgb).reshape(Y_train_chopped.shape)
y_pred_val = xgboost_model.predict(val_xgb).reshape(Y_val.shape)



## Generate backtest results
frac_correct_sign, precision, recall = sign_stats(Y_train_chopped, y_pred_train)
print(f"Train fraction with correct sign: {frac_correct_sign:.3f}")
print(f"Train precision: {precision:.3f}")
print(f"Train recall: {recall:.3f}")
portfolio_value_train = pnl_backtest(Y_train_chopped, y_pred_train)

frac_correct_sign, precision, recall = sign_stats(Y_val, y_pred_val)
print(f"Val fraction with correct sign: {frac_correct_sign:.3f}")
print(f"Val precision: {precision:.3f}")
print(f"Val recall: {recall:.3f}")
portfolio_value_val = pnl_backtest(Y_val, y_pred_val)



## Plot results
plot_backtest_results(portfolio_value_train)
plot_backtest_results(portfolio_value_val)
