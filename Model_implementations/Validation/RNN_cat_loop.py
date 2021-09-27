
import torch
import time
import sys
sys.path.append("../Utilities")
from Preprocessing import *
from Models import *
from Training import *
from Backtest import *
from Plotting import *


def rnn_run(label_name, num_lags, num_epochs, diff_mode, smoothing = False, scaling = False, compression_mode = None):
    
    # Choose data configs
    asset_mode = 'macro'

    # Get data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_preprocessed_datasets(label_name, asset_mode, diff_mode, smoothing, scaling)

    # Dimension reduction
    X_train = reduce_dimension(X_train, compression_mode, asset_mode, diff_mode, smoothing, scaling)
    X_val = reduce_dimension(X_val, compression_mode, asset_mode, diff_mode, smoothing, scaling)
    X_test = reduce_dimension(X_test, compression_mode, asset_mode, diff_mode, smoothing, scaling)
    
    # Categorise data
    cutoffs = get_category_cutoffs(X_train)
    X_train_cat = categorise(X_train, cutoffs)
    X_val_cat = categorise(X_val, cutoffs)
    X_test_cat = categorise(X_test, cutoffs)
    Y_train_cat = categorise(Y_train, cutoffs)
    Y_val_cat = categorise(Y_val, cutoffs)
    Y_test_cat = categorise(Y_test, cutoffs)
    
    ## Create model instance and deploy to device
    
    # Choose series type
    series_type = 'multi_dim'
    if series_type == 'multi_dim':
        data_dim = X_train.shape[1]
    elif series_type == 'one_dim':
        data_dim = 1
    
    model = ForecastingModelCat(data_dim = data_dim, hidden_dim = 16, num_layers = 2, model_type = 'GRU', drop_prob = 0.2)
    
    
    ## Train the model

    loss_by_epoch = train_forecasting_model(
        model,
        series_type,
        X_train_cat,
        Y_train_cat,
        X_val_cat,
        Y_val_cat,
        num_lags,
        num_epochs,
        verbose = False,
        categorical = True
    )
    
    
    ## Make predictions

    # Val data should immediately follow train data in time; we join them here
    X_train_tail_plus_val = stitch_tail(X_train_cat, X_val_cat, num_lags)
    Y_train_tail_plus_val = stitch_tail(Y_train_cat, Y_val_cat, num_lags)

    X_val_tail_plus_test = stitch_tail(X_val_cat, X_test_cat, num_lags)
    Y_val_tail_plus_test = stitch_tail(Y_val_cat, Y_test_cat, num_lags)

    data_train_torch = PricesDatasetMultiDim(X_train_cat, Y_train_cat, x_len = num_lags)
    data_val_torch = PricesDatasetMultiDim(X_train_tail_plus_val, Y_train_tail_plus_val, x_len = num_lags)
    data_test_torch = PricesDatasetMultiDim(X_val_tail_plus_test, Y_val_tail_plus_test, x_len = num_lags)

    Y_pred_train = run_model_on_data(model, data_train_torch).detach().numpy()
    Y_pred_val = run_model_on_data(model, data_val_torch).detach().numpy()
    Y_pred_test = run_model_on_data(model, data_test_torch).detach().numpy()
    
    Y_pred_train = np.argmax(Y_pred_train, axis = 1)
    Y_pred_val = np.argmax(Y_pred_val, axis = 1)
    Y_pred_test = np.argmax(Y_pred_test, axis = 1)

    backtest_results_train = pnl_backtest_cat(Y_train[Y_train.shape[0] - Y_pred_train.shape[0]:], Y_pred_train)
    backtest_results_val = pnl_backtest_cat(Y_val, Y_pred_val)
    backtest_results_test = pnl_backtest_cat(Y_test, Y_pred_test)
    
    
    # Return alpha and Sharpe
    _, data_train_df, data_val_df, data_test_df = get_datasets(asset_mode)
    alpha_train = calc_alpha(data_train_df, backtest_results_train)
    alpha_val = calc_alpha(data_val_df, backtest_results_val)
    alpha_test = calc_alpha(data_test_df, backtest_results_test)
    
    alphas = [alpha_train, alpha_val, alpha_test]

    sharpe_train = calc_rel_sharpe(data_train_df, backtest_results_train)
    sharpe_val = calc_rel_sharpe(data_val_df, backtest_results_val)
    sharpe_test = calc_rel_sharpe(data_test_df, backtest_results_test)
    
    sharpes = [sharpe_train, sharpe_val, sharpe_test]

    return alphas, sharpes
    
    
all_pairs = ["GBPUSD Curncy", "EURUSD Curncy", "USDJPY Curncy", "AUDUSD Curncy", "USDCHF Curncy", "XAUUSD Curncy", "XAGUSD Curncy", "AUDJPY Curncy", "NOKUSD Curncy", "SEKUSD Curncy", "USDMXN Curncy", "NZDUSD Curncy"]

num_lags = 15
num_epochs = 200
diff_mode = 'logret'
smoothing = False
scaling = False
compression_mode = None

all_alphas = {"Train": [], "Val": [], "Test": []}
all_sharpes = {"Train": [], "Val": [], "Test": []}

# Start timer
start = time.perf_counter()

for pair in all_pairs:
    
    alphas, sharpes = rnn_run(pair, num_lags, num_epochs, diff_mode, smoothing, scaling, compression_mode)
    all_alphas["Train"].append(alphas[0])
    all_alphas["Val"].append(alphas[1])
    all_alphas["Test"].append(alphas[2])
    all_sharpes["Train"].append(sharpes[0])
    all_sharpes["Val"].append(sharpes[1])
    all_sharpes["Test"].append(sharpes[2])
    
    print(f"{pair} done at {time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start))}")
    
# End timer
end = time.perf_counter()

print(f"RNN loop run in {time.strftime('%H:%M:%S', time.gmtime(end - start))}.")
    
print_model_results_table(all_pairs, all_alphas, all_sharpes, f"GRU results using ${num_lags}$ lags", f"tab:gru_{num_lags}")
