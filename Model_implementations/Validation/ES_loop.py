
import time
import sys
sys.path.append("../Utilities")
from Preprocessing import *
from Backtest import *
from Plotting import *

import warnings
warnings.filterwarnings("ignore")


# Choose data configs
asset_mode = 'macro'
diff_mode = 'logret'
smoothing = True
scaling = False




def es_run(label_name):
    
    # Get data
    data_all_df, data_train_df, data_val_df, data_test_df = get_datasets(asset_mode)
    data_all_sm, data_train_sm, data_val_sm, data_test_sm = smooth_data(data_all_df, data_train_df, data_val_df, data_test_df, verbose = False)
    
    # Get predictions
    Y_pred = data_all_sm[label_name].to_numpy()
    Y_pred_logret = take_log_returns_np(Y_pred).reshape(-1, 1)
    Y_pred_train = Y_pred_logret[:data_train_df.shape[0]]
    Y_pred_val = Y_pred_logret[data_train_df.shape[0]:data_train_df.shape[0] + data_val_df.shape[0]]
    Y_pred_test = Y_pred_logret[data_train_df.shape[0] + data_val_df.shape[0]:]
    
    Y_true = data_all_df[label_name].to_numpy()
    Y_true_logret = take_log_returns_np(Y_true).reshape(-1, 1)
    Y_train = Y_true_logret[:data_train_df.shape[0]]
    Y_val = Y_true_logret[data_train_df.shape[0]:data_train_df.shape[0] + data_val_df.shape[0]]
    Y_test = Y_true_logret[data_train_df.shape[0] + data_val_df.shape[0]:]
    
    # Run backtests
    backtest_results_train = pnl_backtest(Y_train[Y_train.shape[0] - Y_pred_train.shape[0]:], Y_pred_train)
    backtest_results_val = pnl_backtest(Y_val, Y_pred_val)
    backtest_results_test = pnl_backtest(Y_test, Y_pred_test)
    
    # Return alpha and Sharpe
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

for pair in all_pairs:
    
    alphas, sharpes = es_run(pair)
    all_alphas["Train"].append(alphas[0])
    all_alphas["Val"].append(alphas[1])
    all_alphas["Test"].append(alphas[2])
    all_sharpes["Train"].append(sharpes[0])
    all_sharpes["Val"].append(sharpes[1])
    all_sharpes["Test"].append(sharpes[2])
    
    print(f"{pair} done")
    
# End timer
end = time.perf_counter()

print(f"ES loop run in {time.strftime('%H:%M:%S', time.gmtime(end - start))}.")

print_model_results_table(all_pairs, all_alphas, all_sharpes, f"Exponential smoothing results", f"tab:es_results")
