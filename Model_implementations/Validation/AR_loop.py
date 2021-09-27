
import pandas as pd
import time
from statsmodels.tsa.arima.model import ARIMA
import sys
sys.path.append("../Utilities")
from Preprocessing import *
from Backtest import *
from Plotting import *


def ar_run(label_name, p):
    
    # Choose data configs
    asset_mode = 'macro'
    
    
    ## Get data and arrange into correct format

    # Get data (no pre-processing)
    data_all_df, data_train_df, data_val_df, data_test_df = get_datasets(asset_mode)
    
    
    data_all_df = data_all_df[:50]
    data_train_df = data_train_df[-30:]
    data_val_df = data_val_df[:10]
    data_test_df = data_test_df[:10]
    
    
    # Pick column of interest
    data_all_df = data_all_df[[label_name]]
    data_train_df = data_train_df[[label_name]]
    data_val_df = data_val_df[[label_name]]
    data_test_df = data_test_df[[label_name]]

    # Choose start point in training set (ARIMA needs at least a few points)
    arima_start = 20

    # Make numpy arrays
    X_train = data_train_df.to_numpy()
    X_val = data_val_df.to_numpy()
    X_test = data_test_df.to_numpy()
    
    # Get labels
    Y_true_logret = take_log_returns_np(data_all_df.to_numpy())
    Y_train = Y_true_logret[arima_start:data_train_df.shape[0] - 1]
    Y_val = Y_true_logret[data_train_df.shape[0] - 1:data_train_df.shape[0] + data_val_df.shape[0] - 1]
    Y_test = Y_true_logret[data_train_df.shape[0] + data_val_df.shape[0] - 1:]

    # Containers used during prediction
    X_running = X_train[0:arima_start]
    Y_pred = np.array([])



    ## Predict on the training set and then the validation set

    for pred_idx in range(arima_start, X_train.shape[0]):

        model = ARIMA(X_running, order = (p, 0, 0))
        model_fit = model.fit(method_kwargs = {"warn_convergence": False})
        Y_pred = np.append(Y_pred, model_fit.forecast()[0])
        X_running = np.append(X_running, X_train[pred_idx])


    for pred_idx in range(X_val.shape[0]):

        model = ARIMA(X_running, order = (p, 0, 0))
        model_fit = model.fit(method_kwargs = {"warn_convergence": False})
        Y_pred = np.append(Y_pred, model_fit.forecast()[0])
        X_running = np.append(X_running, X_val[pred_idx])


    for pred_idx in range(X_test.shape[0]):

        model = ARIMA(X_running, order = (p, 0, 0))
        model_fit = model.fit(method_kwargs = {"warn_convergence": False})
        Y_pred = np.append(Y_pred, model_fit.forecast()[0])
        X_running = np.append(X_running, X_test[pred_idx])


    ## Post-process predictions
    Y_pred_logret = take_log_returns_np(Y_pred).reshape(-1, 1)
    Y_pred_train = Y_pred_logret[:data_train_df.shape[0] - 1 - arima_start]
    Y_pred_val = Y_pred_logret[data_train_df.shape[0] - 1 - arima_start:data_train_df.shape[0] + data_val_df.shape[0] - 1 - arima_start]
    Y_pred_test = Y_pred_logret[data_train_df.shape[0] + data_val_df.shape[0] - 1 - arima_start:]    

    
    # Run backtests
    backtest_results_train = pnl_backtest(Y_train[Y_train.shape[0] - Y_pred_train.shape[0]:], Y_pred_train)
    backtest_results_val = pnl_backtest(Y_val, Y_pred_val)
    backtest_results_test = pnl_backtest(Y_test, Y_pred_test)
    
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


all_alphas = {"Train": [], "Val": [], "Test": []}
all_sharpes = {"Train": [], "Val": [], "Test": []}

# Start timer
start = time.perf_counter()

for p in [1, 2, 3]:
        
    for pair in all_pairs:

        alphas, sharpes = ar_run(pair, p)
        all_alphas["Train"].append(alphas[0])
        all_alphas["Val"].append(alphas[1])
        all_alphas["Test"].append(alphas[2])
        all_sharpes["Train"].append(sharpes[0])
        all_sharpes["Val"].append(sharpes[1])
        all_sharpes["Test"].append(sharpes[2])

        print(f"{pair} done at {time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start))}")

    print(f"{p} done: {100.*np.mean(all_alphas['Val'])}")
    
# End timer
end = time.perf_counter()

print(f"AR loop run in {time.strftime('%H:%M:%S', time.gmtime(end - start))}.")

print_model_results_table(all_pairs, all_alphas, all_sharpes, f"AR results", f"tab:arima_results")
