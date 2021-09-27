
import numpy as np
import datetime as dt
from Preprocessing import *



# Get the fraction of predictions with correct sign
def sign_stats(y_true, y_preds):
    
    num_pos = np.mean(y_true > 0.)
    pred_pos = np.mean(y_preds > 0.)
    
    tot = y_true.shape[0]
    tp = np.sum((y_preds > 0.) * (y_true > 0.))
    fp = np.sum((y_preds > 0.) * (~(y_true > 0.)))
    fn = np.sum((~(y_preds > 0.)) * (y_true > 0.))
    
    acc = np.mean(y_preds * y_true > 0.)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return num_pos, pred_pos, acc, precision, recall
    

# If the model predicts a +ve return for EURUSD, go all-in on EUR, otherwise hold only USD.
def pnl_backtest(y_true, y_preds):

    # Number of periods
    num_periods = y_true.shape[0]
    
    # Start with one unit of term ccy
    initial_term_capital = 1.
    sign_strat = np.empty(num_periods + 1)
    buy_hold_strat = np.empty(num_periods + 1)
    sign_strat[0] = initial_term_capital
    buy_hold_strat[0] = initial_term_capital
    current_asset = 'term'
    
    # Run the simulation
    for t in range(num_periods):
        
        sign_strat[t+1] = sign_strat[t]
        buy_hold_strat[t+1] = buy_hold_strat[t] * np.exp(y_true[t])[0]
        
        if y_preds[t] > 0.:
            
            # If we already hold base ccy, there is no cost, else we have to buy and pay the spread
            if current_asset == 'term':
                
                # Take costs into account very simplistically by using FX bid-offers
                current_asset = 'base'
                cost = 0.
                sign_strat[t] -= cost
            
            # Once we have base ccy, we get the return of the ccy pair on our capital
            port_perf = np.exp(y_true[t])[0]
            
        else:
            
            # If we are holding base ccy, we have to sell and pay the spread, else there is no cost
            if current_asset == 'base':
                current_asset = 'term'
                cost = 0.
                sign_strat[t] -= cost
            
            # We do not receive the return; our capital stays the same
            port_perf = 1.
        
        sign_strat[t+1] *= port_perf

    return {'Sign_strat' : sign_strat, 'Buy_hold_strat' : buy_hold_strat}


# If the model predicts a large +ve return for EURUSD, go all-in on EUR, switch to hold USD if a large -ve return is predicted
def pnl_backtest_cat(y_true, y_preds):

    # Number of periods
    num_periods = y_true.shape[0]
    
    # Start with one unit of term ccy
    initial_term_capital = 1.
    sign_strat = np.empty(num_periods + 1)
    buy_hold_strat = np.empty(num_periods + 1)
    sign_strat[0] = initial_term_capital
    buy_hold_strat[0] = initial_term_capital
    current_asset = 'term'
    
    # Run the simulation
    for t in range(num_periods):
        
        sign_strat[t+1] = sign_strat[t]
        buy_hold_strat[t+1] = buy_hold_strat[t] * np.exp(y_true[t])[0]
        
        if y_preds[t] == 2:
            
            # If we already hold base ccy, there is no cost, else we have to buy and pay the spread
            if current_asset == 'term':
                
                # Take costs into account very simplistically by using FX bid-offers
                current_asset = 'base'
                cost = 0.
                sign_strat[t] -= cost
            
            # Once we have base ccy, we get the return of the ccy pair on our capital
            port_perf = np.exp(y_true[t])[0]
            
        elif y_preds[t] == 1:
            
            # Keep current position
            if current_asset == 'base':
                port_perf = np.exp(y_true[t])[0]
            else:
                port_perf = 1.
            
        elif y_preds[t] == 0:
            
            # If we already hold term ccy, there is no cost, else we have to buy and pay the spread
            if current_asset == 'base':
                
                # Take costs into account very simplistically by using FX bid-offers
                current_asset = 'term'
                cost = 0.
                sign_strat[t] -= cost
            
            # Once we have base ccy, we get the return of the ccy pair on our capital
            port_perf = 1.
            
        else:
            raise ValueError(f"pnl_backtest_cat: prediction expected to be 0, 1 or 2 but found {y_preds[t]}")
        
        sign_strat[t+1] *= port_perf

    return {'Sign_strat' : sign_strat, 'Buy_hold_strat' : buy_hold_strat}


# Annualisation factor (don't worry about day count conventions too much)
def annualisation_factor(data_df, backtest_results):
    
    # Hack!
    start_index = data_df.index[0]
    end_index = data_df.index[-1]
    if type(start_index) is not str:
        start_index = start_index.strftime('%d/%m/%Y')
        end_index = end_index.strftime('%d/%m/%Y')
    
    # Get the total number of calendar days in the period
    start_date = dt.datetime.strptime(start_index, '%d/%m/%Y')
    end_date = dt.datetime.strptime(end_index, '%d/%m/%Y')
    total_days = (end_date - start_date).days
    
    ann_frac = 365. / total_days
    
    return ann_frac


# Alpha will mean the improvement over the underlying asset
def calc_alpha(data_df, backtest_results):
    
    # Annualisation factor
    ann_frac = annualisation_factor(data_df, backtest_results)
    
    # Find the relative performance of the sign strat over the performance of the asset itself
    buy_hold_perf = backtest_results['Buy_hold_strat'][-1] / backtest_results['Buy_hold_strat'][0]
    sign_perf = backtest_results['Sign_strat'][-1] / backtest_results['Sign_strat'][0]
    rel_perf = sign_perf / buy_hold_perf
    
    # Annualise the performance
    ann_perf = np.power(rel_perf, ann_frac)
    
    # Calculate the annualised return
    alpha = ann_perf - 1.
    
    return alpha


# Sharpe will mean alpha over volatility (we do not include 'risk-free' returns)
def calc_rel_sharpe(data_df, backtest_results):
    
    # Calculate performance
    buy_hold_perf = backtest_results['Buy_hold_strat'][-1] / backtest_results['Buy_hold_strat'][0]
    sign_perf = backtest_results['Sign_strat'][-1] / backtest_results['Sign_strat'][0]
    
    # Annualise the performance
    ann_frac = annualisation_factor(data_df, backtest_results)
    buy_hold_perf_ann = np.power(buy_hold_perf, ann_frac)
    sign_perf_ann = np.power(sign_perf, ann_frac)
    
    # Convert to return
    buy_hold_ret_ann = buy_hold_perf_ann - 1.
    sign_ret_ann = sign_perf_ann - 1.
    
    # Calculate volatility
    buy_hold_log_returns = take_log_returns_np(backtest_results['Buy_hold_strat'])
    sign_log_returns = take_log_returns_np(backtest_results['Sign_strat'])
    buy_hold_drift = np.mean(buy_hold_log_returns)
    sign_drift = np.mean(sign_log_returns)
    buy_hold_variance_ann = (365. / buy_hold_log_returns.shape[0]) * np.sum((buy_hold_log_returns - buy_hold_drift)**2)
    sign_variance_ann = (365. / sign_log_returns.shape[0]) * np.sum((sign_log_returns - sign_drift)**2)
    buy_hold_vol = np.sqrt(buy_hold_variance_ann)
    sign_vol = np.sqrt(sign_variance_ann)
    
    if sign_vol <= 0.:
        raise RuntimeError(f"calc_rel_sharpe: cannot divide by sign_vol = {sign_vol}")
    
    # Create ratio of Sharpes
    buy_hold_sharpe = buy_hold_ret_ann / buy_hold_vol
    sign_sharpe = sign_ret_ann / sign_vol
    ratio_sharpes = sign_sharpe / buy_hold_sharpe
    
    return ratio_sharpes
