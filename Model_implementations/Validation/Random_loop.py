
import torch
import time
import sys
sys.path.append("../Utilities")
from Preprocessing import *
from Models import *
from Training import *
from Backtest import *
from Plotting import *


def random_run(label_name, seed):
    
    # Choose data configs
    asset_mode = 'macro'
    diff_mode = 'logret'
    smoothing = False
    scaling = False

    # Get data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_preprocessed_datasets(label_name, asset_mode, diff_mode, smoothing, scaling)

    # Get number of positive returns
    frac_pos_returns = 0.5#np.sum(Y_train > 0.) / Y_train.shape[0]
    
    # Randomly generate a list of signs
    total_num = Y_train.shape[0] + Y_val.shape[0] + Y_test.shape[0]
    random_signs = np.random.choice([1., -1.], size = total_num, p = [frac_pos_returns, 1. - frac_pos_returns])
    
    # Predict randomly
    Y_pred_train = random_signs[:X_train.shape[0]]
    Y_pred_val = random_signs[X_train.shape[0]:X_train.shape[0] + X_val.shape[0]]
    Y_pred_test = random_signs[X_train.shape[0] + X_val.shape[0]:]
    
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

num_seeds = 250

# Start timer
start = time.perf_counter()

for pair in all_pairs:
    
    a0 = 0.
    a1 = 0.
    a2 = 0.
    s0 = 0.
    s1 = 0.
    s2 = 0.
    
    for seed in range(num_seeds):
    
        np.random.seed(seed)
        alphas_seed, sharpes_seed = random_run(pair, seed)
        a0 += alphas_seed[0] / num_seeds
        a1 += alphas_seed[1] / num_seeds
        a2 += alphas_seed[2] / num_seeds
        s0 += sharpes_seed[0] / num_seeds
        s1 += sharpes_seed[1] / num_seeds
        s2 += sharpes_seed[2] / num_seeds
    
    alphas, sharpes = random_run(pair, seed)
    all_alphas["Train"].append(a0)
    all_alphas["Val"].append(a1)
    all_alphas["Test"].append(a2)
    all_sharpes["Train"].append(s0)
    all_sharpes["Val"].append(s1)
    all_sharpes["Test"].append(s2)
    
    print(f"{pair} done at {time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start))}")
    
# End timer
end = time.perf_counter()

print(f"Random loop run in {time.strftime('%H:%M:%S', time.gmtime(end - start))}.")
    
print_model_results_table(all_pairs, all_alphas, all_sharpes, f"Random guessing results using ${num_seeds}$ seeds", f"tab:random_res")
