
## Do imports

import torch
import time
import sys
sys.path.append("../Utilities")
from Preprocessing import *
from Models import *
from Training import *
from Backtest import *
from Plotting import *



## Get pre-processed and split data

print(f"Pre-processing...")

# Start timer
start = time.perf_counter()

# Choose data configs
#label_name, asset_mode = 'GBPUSD Curncy', 'macro'
label_name, asset_mode = 'AAPL', 'stock'
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

# Categorise data
cutoffs = get_category_cutoffs(X_train)
X_train_cat = categorise(X_train, cutoffs)
X_val_cat = categorise(X_val, cutoffs)
Y_train_cat = categorise(Y_train, cutoffs)
Y_val_cat = categorise(Y_val, cutoffs)

# End timer
end = time.perf_counter()

print(f"Data pre-processed with smoothing = {smoothing} and diff_mode = {diff_mode} in {time.strftime('%H:%M:%S', time.gmtime(end - start))}.")

print(f"Training data shape: {X_train_cat.shape}")
print(f"Validation data shape: {X_val_cat.shape}")



## Create model instance and deploy to device

# Choose series type
series_type = 'multi_dim'
if series_type == 'multi_dim':
    data_dim = X_train_cat.shape[1]
elif series_type == 'one_dim':
    data_dim = 1

model = ForecastingModelCat(data_dim = data_dim, hidden_dim = 32, num_layers = 2, model_type = 'GRU')



## Train the model

num_lags = 15

num_epochs = 20

print(f"Training model...")

# Start timer
start = time.perf_counter()

loss_by_epoch = train_forecasting_model(
    model,
    series_type,
    X_train_cat,
    Y_train_cat,
    X_val_cat,
    Y_val_cat,
    num_lags,
    num_epochs,
    verbose = True,
    categorical = True
)

# End timer
end = time.perf_counter()
print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(end - start))}")



## Plot learning curves
plot_learning_curves(loss_by_epoch)



## Save trained model to disk
model_string = f"{model.name}_{num_lags}_{label_name.replace(' ', '_')}_{num_epochs}_{asset_mode}_{diff_mode}_{smoothing}_{scaling}"
save_model(model, model_string)



## Make predictions

# Val data should immediately follow train data in time; we join them here
X_train_tail_plus_val = stitch_tail(X_train_cat, X_val_cat, num_lags)
Y_train_tail_plus_val = stitch_tail(Y_train_cat, Y_val_cat, num_lags)

data_train_torch = PricesDatasetMultiDim(X_train_cat, Y_train_cat, x_len = num_lags)
data_val_torch = PricesDatasetMultiDim(X_train_tail_plus_val, Y_train_tail_plus_val, x_len = num_lags)

Y_pred_train = run_model_on_data(model, data_train_torch).detach().numpy()
Y_pred_val = run_model_on_data(model, data_val_torch).detach().numpy()

Y_pred_train = np.argmax(Y_pred_train, axis = 1)
Y_pred_val = np.argmax(Y_pred_val, axis = 1)



## Generate backtest results
num_pos, pred_pos, frac_correct_sign, precision, recall = sign_stats(Y_train_cat[Y_train.shape[0] - Y_pred_train.shape[0]:], Y_pred_train)
print(f"Train fraction of data positive: {num_pos:.3f}")
print(f"Train fraction of predictions positive: {pred_pos:.3f}")
print(f"Train fraction with correct sign: {frac_correct_sign:.3f}")
print(f"Train precision: {precision:.3f}")
print(f"Train recall: {recall:.3f}")
backtest_results_train = pnl_backtest_cat(Y_train[Y_train.shape[0] - Y_pred_train.shape[0]:], Y_pred_train)

num_pos, pred_pos, frac_correct_sign, precision, recall = sign_stats(Y_val_cat, Y_pred_val)
print(f"Val fraction of data positive: {num_pos:.3f}")
print(f"Val fraction of predictions positive: {pred_pos:.3f}")
print(f"Val fraction with correct sign: {frac_correct_sign:.3f}")
print(f"Val precision: {precision:.3f}")
print(f"Val recall: {recall:.3f}")
backtest_results_val = pnl_backtest_cat(Y_val, Y_pred_val)



## Plot results
plot_time_series([Y_train_cat[num_lags:], Y_pred_train], start = 200, end = 300)
plot_time_series([Y_val_cat, Y_pred_val])

plot_backtest_results(backtest_results_train)
plot_backtest_results(backtest_results_val)
