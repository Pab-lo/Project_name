
from sklearn.ensemble import RandomForestRegressor
import time
import sys
sys.path.append("../Utilities")
from Preprocessing import *
from Backtest import *
from Plotting import *
np.random.seed(1)


def rf_run(label_name, num_lags, n_estimators, max_depth, min_samples_leaf, max_features, diff_mode, smoothing = False, scaling = False, compression_mode = None):
    
    # Choose data configs
    asset_mode = 'macro'

    # Get data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_preprocessed_datasets(label_name, asset_mode, diff_mode, smoothing, scaling)

    # Dimension reduction
    X_train = reduce_dimension(X_train, compression_mode, asset_mode, diff_mode, smoothing, scaling)
    X_val = reduce_dimension(X_val, compression_mode, asset_mode, diff_mode, smoothing, scaling)
    
    
    # Laggify the Xs and chop the Ys
    X_train_lagged = laggify_data(X_train, num_lags)
    Y_train_chopped = Y_train[num_lags:]
    X_val_with_train_tail = stitch_tail(X_train, X_val, num_lags)
    X_val_lagged = laggify_data(X_val_with_train_tail, num_lags)
    X_test_with_val_tail = stitch_tail(X_val, X_test, num_lags)
    X_test_lagged = laggify_data(X_test_with_val_tail, num_lags)
    
    
    ## Train the model

    random_forest = RandomForestRegressor(n_estimators = n_estimators,
                                          max_depth = max_depth,
                                          min_samples_leaf = min_samples_leaf,
                                          max_features = max_features,
                                          verbose = 0)
    random_forest.fit(X_train_lagged, Y_train_chopped.reshape(Y_train_chopped.shape[0]))
    
    
    ## Make predictions

    Y_pred_train = random_forest.predict(X_train_lagged).reshape(Y_train_chopped.shape)
    Y_pred_val = random_forest.predict(X_val_lagged).reshape(Y_val.shape)
    Y_pred_test = random_forest.predict(X_test_lagged).reshape(Y_test.shape)
    
    backtest_results_train = pnl_backtest(Y_train_chopped, Y_pred_train)
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

num_lags = 100
n_estimators = 100
#max_depth = 2
#min_samples_leaf = 1
#max_features = 'sqrt'
diff_mode = 'logret'
smoothing = False
scaling = False
compression_mode = None

all_alphas = {"Train": [], "Val": [], "Test": []}
all_sharpes = {"Train": [], "Val": [], "Test": []}

# Start timer
start = time.perf_counter()

for max_depth in [2, 5, None]:
    for min_samples_leaf in [1, 150, 750]:
        for max_features in ['auto', 'sqrt', 10]:
            
            max_depth = 2
            min_samples_leaf = 1
            max_features = 'sqrt'

            for pair in all_pairs:

                alphas, sharpes = rf_run(pair, num_lags, n_estimators, max_depth, min_samples_leaf, max_features, diff_mode, smoothing, scaling, compression_mode)
                all_alphas["Train"].append(alphas[0])
                all_alphas["Val"].append(alphas[1])
                all_alphas["Test"].append(alphas[2])
                all_sharpes["Train"].append(sharpes[0])
                all_sharpes["Val"].append(sharpes[1])
                all_sharpes["Test"].append(sharpes[2])
                
                print(f"{pair} done")
                
            print(f"{max_depth}, {min_samples_leaf}, {max_features} done: {100.*np.mean(all_alphas['Val'])}")
            
            break
        break
    break
    
# End timer
end = time.perf_counter()

print(f"RF loop run in {time.strftime('%H:%M:%S', time.gmtime(end - start))}.")

print_model_results_table(all_pairs, all_alphas, all_sharpes, f"Random forest results using ${num_lags}$ lags", f"tab:rf_{num_lags}")
