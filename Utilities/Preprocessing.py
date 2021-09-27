
import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
from Datasets_and_sampling import *
from Models import *
from Training import *
from Exponential_smoothing import *
from Frac_diff import *



def denan_and_reduce(data_raw, minimum_data_rows, fx_only = False):
    
    # Check how much data we have for each column
    data_ends = {}
    for col_name in data_raw.columns.tolist():
        series_bools = ~pd.to_numeric(data_raw[col_name], errors = 'coerce').isnull()
        data_ends[col_name] = np.sum(series_bools)

    # Select columns we want to include
    columns_include = []
    
    for col_name in data_raw.columns.tolist():
        if data_ends[col_name] >= minimum_data_rows:
            columns_include.append(col_name)

    # Remove non-FX columns if required
    if fx_only:
        columns_include = [col_name for col_name in columns_include if 'Curncy' in col_name[0]]
        
    # Hack to include only first column
    first_col_only = False
    if first_col_only is True:
        columns_include = [columns_include[0]]
    
    # Apply column selection to data
    data_subset = data_raw[columns_include][-minimum_data_rows:]

    # Remove rows that are all NaNs (e.g. weekends)
    data_denan = data_subset[~data_subset.isnull().all(1)]
    
    # Replace other NaNs with previous values
    data_denan = data_denan.fillna(method = 'ffill')
    
    # Any remaining NaNs (as may be found on the first row) can just be forward-filled
    data_denan = data_denan.fillna(method = 'bfill')
    
    # Check for NaNs
    if data_denan.isnull().values.any():
        raise RuntimeError("denan_and_reduce: NaNs found where there should be none")
    
    return data_denan


def get_datasets(asset_mode):
    
    # Chosen settings
    minimum_data_rows = 8000
    fx_only = False
    val_frac = 0.8
    
    # Load data
    data_raw_df = load_raw_data(asset_mode)
    
    # Remove NaNs and select columns
    data_all_df = denan_and_reduce(data_raw_df, minimum_data_rows, fx_only)
    
    # Train/val/test split
    data_nontest_df, data_test_df = data_splitter(data_all_df)
    data_train_df, data_val_df = data_splitter(data_nontest_df, frac = val_frac)
    
    return data_all_df, data_train_df, data_val_df, data_test_df


def take_log_returns_np(np_array):
    
    np_array_logret = (np.log(np_array) - np.log(np.roll(np_array, 1)))[1:]
    
    return np_array_logret


def take_log_returns(data_all_df, data_train_df, data_val_df, data_test_df):
    
    # Note down columns
    columns = data_all_df.columns.tolist()
    
    # Compute daily differences for rates/vols and log returns for the others
    rates_cols = [col_name for col_name in columns if ('Govt' in col_name or '_vol' in col_name)]
    non_rates_cols = [col_name for col_name in columns if not ('Govt' in col_name or '_vol' in col_name)]
    
    # Take diffs for the rates data
    data_rates = data_all_df[rates_cols]
    data_rates_diffs = (data_rates - data_rates.shift(1))[1:]
    
    # Take log returns for the rest
    data_non_rates = data_all_df[non_rates_cols]
    data_non_rates_logret = (np.log(data_non_rates) - np.log(data_non_rates.shift(1)))[1:]
    
    # Join the data back together and put columns back in the original order
    data_all_df = pd.concat([data_non_rates_logret, data_rates_diffs], axis=1)
    data_all_df = data_all_df[columns]
    
    # Retrieve the new train/val/test sets from the new data using their indices (thereby avoiding any issues
    # at the joins such as rows that would need to be removed)
    data_train_df = data_all_df.loc[data_train_df.index[1:]]
    data_val_df = data_all_df.loc[data_val_df.index]
    data_test_df = data_all_df.loc[data_test_df.index]
    
    return data_all_df, data_train_df, data_val_df, data_test_df


def frac_diff_data(data_all_df, data_train_df, data_val_df, data_test_df):
    
    # Here we tune the order of differencing to the training data and apply to the others
    frac_diff_ob = FracdiffStat_df(data_train_df)
    
    # Note down columns
    columns = data_all_df.columns.tolist()
    
    # Print the tuned orders of differencing
    for col_idx in range(len(columns)):
        print(f"{columns[col_idx]}: fractional differencing order {frac_diff_ob.fracdiffstat.d_[col_idx]:.2f}")
    
    # The differenced training data can be extracted from the tuned object (it has already
    # been truncated at the start due to the differencing)
    data_train_df = frac_diff_ob.data_fitted_df
    
    # Apply the tuned differencing orders to all the data
    data_all_df = frac_diff_ob.apply_fracdiff(data_all_df)
    
    # Retrieve the new val/test sets from the new data using their indices (thereby avoiding any issues
    # at the joins such as rows that would need to be removed)
    data_val_df = data_all_df.loc[data_val_df.index]
    data_test_df = data_all_df.loc[data_test_df.index]
    
    return data_all_df, data_train_df, data_val_df, data_test_df
    
    
def smooth_data(data_all_df, data_train_df, data_val_df, data_test_df, verbose = True):
    
    # Here we tune the smoothing to the training data and apply to the others
    smoother = ExponentialSmoothing_df(data_train_df)
    
    # Note down columns
    columns = data_all_df.columns.tolist()
    
    # Print the tuned smoothing parameters
    if verbose:
        for col_idx in range(len(columns)):
            smoothing_level = smoother.model_data[columns[col_idx]]['params'].params['smoothing_level']
            smoothing_trend = smoother.model_data[columns[col_idx]]['params'].params['smoothing_trend']
            print(f"{columns[col_idx]}: smoothing level {smoothing_level:.2f}, smoothing trend {smoothing_trend:.2f}")
    
    # The smoothed training data can be extracted from the tuned object
    data_train_df = smoother.data_fitted_df
    
    # Apply the tuned smoothing to all the data
    data_all_df = smoother.apply_exp_smooth(data_all_df)
    
    # Retrieve the new val/test sets from the new data using their indices (thereby avoiding any issues
    # at the joins such as rows that would need to be removed)
    data_val_df = data_all_df.loc[data_val_df.index]
    data_test_df = data_all_df.loc[data_test_df.index]
    
    return data_all_df, data_train_df, data_val_df, data_test_df


def scale_data(X_train, X_val, X_test):
    
    # Set scaler range
    scaler_range = (-0.1, 0.1)
    
    # Create scaler and fit to training data
    scaler = pp.MinMaxScaler(feature_range = scaler_range)
    X_train = scaler.fit_transform(X_train)
    
    # Apply tuned scaler to val/test sets
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Print the scaling values
    print(f"Data scaled from {(scaler.data_min_, scaler.data_max_)} to {scaler_range}.")
    
    # Return sets
    return X_train, X_val, X_test


def laggify_data(X, num_lags):
    
    X_lagged = np.zeros((X.shape[0] - num_lags, X.shape[1] * (num_lags + 1)))
    
    for col_idx in range(X.shape[1]):
        for lag in range(num_lags + 1):
            for t in range(X.shape[0] - num_lags):
                X_lagged[t, col_idx * (num_lags + 1) + lag] = X[num_lags + t - lag - 1, col_idx]
    
    return X_lagged


def get_preproc_Xs(data_all_df, data_train_df, data_val_df, data_test_df, diff_mode = None, smoothing = False, scaling = False):
    
    # Transform the data as requested
    if diff_mode is None:
        data_train_df = data_all_df.loc[data_train_df.index[1:]]  # Chop the first row so the size of Y agrees
    elif diff_mode == 'logret':
        data_all_df, data_train_df, data_val_df, data_test_df = take_log_returns(data_all_df, data_train_df, data_val_df, data_test_df)
    elif diff_mode == 'fracdiff':
        data_all_df, data_train_df, data_val_df, data_test_df = frac_diff_data(data_all_df, data_train_df, data_val_df, data_test_df)
    else:
        raise ValueError("prepare_datasets: diff_mode value not recognised.")
    
    # Apply smoothing if requested
    if smoothing is True:
        data_all_df, data_train_df, data_val_df, data_test_df = smooth_data(data_all_df, data_train_df, data_val_df, data_test_df)
    elif smoothing is False:
        pass
    else:
        raise ValueError("prepare_datasets: smoothing value not recognised.")
    
    # Convert to numpy arrays
    X_train = data_train_df.to_numpy()
    X_val = data_val_df.to_numpy()
    X_test = data_test_df.to_numpy()
    
    # Apply scaling if requested
    if scaling is True:
        X_train, X_val, X_test = scale_data(X_train, X_val, X_test)
    elif scaling is False:
        pass
    else:
        raise ValueError("prepare_datasets: scaling value not recognised.")
    
    # Return the pre-processed data sets
    return X_train, X_val, X_test


def get_preproc_Ys(data_all_df, data_train_df, data_val_df, data_test_df, label_name):
    
    # Get the column needed
    labels_all_df = data_all_df[[label_name]]
    labels_train_df = data_train_df[[label_name]]
    labels_val_df = data_val_df[[label_name]]
    labels_test_df = data_test_df[[label_name]]
    
    # Calculate log returns
    labels_all_df, labels_train_df, labels_val_df, labels_test_df = take_log_returns(
        labels_all_df,
        labels_train_df,
        labels_val_df,
        labels_test_df
    )
    
    # Convert to numpy
    Y_train = labels_train_df.to_numpy()
    Y_val = labels_val_df.to_numpy()
    Y_test = labels_test_df.to_numpy()
    
    # Return the pre-processed data sets
    return Y_train, Y_val, Y_test


def get_preprocessed_datasets(label_name, asset_mode, diff_mode = None, smoothing = False, scaling = False):
    
    # Get datasets
    data_all_df, data_train_df, data_val_df, data_test_df = get_datasets(asset_mode)
    
    # Get pre-processed Ys
    Y_train, Y_val, Y_test = get_preproc_Ys(
        data_all_df,
        data_train_df,
        data_val_df,
        data_test_df,
        label_name
    )
    
    # Get pre-processed Xs
    X_train, X_val, X_test = get_preproc_Xs(
        data_all_df,
        data_train_df,
        data_val_df,
        data_test_df,
        diff_mode,
        smoothing,
        scaling
    )
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def stitch_tail(first_np, second_np, num_stitches):
        
    first_tail_plus_second_np = np.concatenate((first_np[-num_stitches:], second_np), axis = 0)
    
    return first_tail_plus_second_np


def reduce_dimension(data_np, compression_mode, asset_mode, diff_mode, smoothing, scaling):
    
    if compression_mode is None:
        
        data_compressed = data_np
        
    elif compression_mode == 'PCA':
        raise NotImplementedError("reduce_dimension: PCA not implemented")
    elif compression_mode == 'AE':
        
        data_dim = data_np.shape[1]
        compression_dim = 6
        
        # Load trained autoencoder
        model_name = f"autoencoder_{data_dim}_{compression_dim}_1"
        num_epochs = 200
        model_string = f"{model_name}_{num_epochs}_{asset_mode}_{diff_mode}_{smoothing}_{scaling}"
        
        try:
            autoencoder = load_model(model_string)
        except:
            raise ValueError("reduce_dimension: autoencoder not found.")
            
        data_torch = AutoencoderDataset(data_np)
        
        data_compressed = run_model_on_data(autoencoder.encoder, data_torch).detach().cpu().numpy()
        
    else:
        raise ValueError("reduce_dimension: compression_mode not recognised")
        
    return data_compressed
    
    
def get_category_cutoffs(data_np):
    
    cutoffs = np.percentile(data_np, [49.9, 50.1], axis = 0)
    
    return cutoffs
    
    
def categorise(data_np, cutoffs):
    
    cat_data_np = np.empty_like(data_np)
    
    for j in range(data_np.shape[1]):
        for i in range(data_np.shape[0]):
            if data_np[i, j] < cutoffs[0, j]:
                cat_data_np[i, j] = 0
            elif data_np[i, j] > cutoffs[1, j]:
                cat_data_np[i, j] = 2
            else:
                cat_data_np[i, j] = 1
    
    return cat_data_np
