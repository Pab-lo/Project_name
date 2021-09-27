
import numpy as np
import pandas as pd
import fracdiff as fd
from statsmodels.tsa.stattools import adfuller


# Carry out ADF unit-root test
def adf_test(data_array):
    
    adf, pvalue, _, _, _, _ = adfuller(data_array)
    
    return adf, pvalue



# Class to apply fractional differencing to a dataframe
class FracdiffStat_df():
    
    def __init__(self, data_tofit_df):
        
        # Mode "valid" chops off the first few rows for which there is not enough history for
        # all the terms of the fractional difference; mode "full" keeps these rows but there is
        # a boundary effect at the start due to the insufficient history
        self.fracdiffstat = fd.FracdiffStat(
            window = 12,
            mode = "valid",
            precision = 1.e-8,
            upper = 10.,
            lower = 0.
        )
        
        # Tune the order of differencing to the data and record the resulting data
        data_fitted_np = self.fracdiffstat.fit_transform(data_tofit_df.to_numpy())
        self.data_fitted_df = pd.DataFrame(
            data_fitted_np,
            columns = data_tofit_df.columns,
            index = data_tofit_df.index[-data_fitted_np.shape[0]:]
        )
        
        # If any of the orders of differencing were not found, throw (may need to change upper/lower
        # binary search limits above)
        if np.isnan(np.sum(self.fracdiffstat.d_)):
            raise RuntimeError("FracdiffStat_df::__init__: Failed to tune order of differencing")
        
        # If any of the tuned differencing orders are positive integers, the array returned by
        # self.fracdiffstat.transform may not be the same shape as the method's input; to avoid
        # this, we add a small amount
        small_amount = 1.e-2
        for i in range(len(self.fracdiffstat.d_)):
            diff_order = self.fracdiffstat.d_[i]
            if diff_order.is_integer() is True and diff_order > 0.:
                self.fracdiffstat.d_[i] += small_amount
        
        
    def apply_fracdiff(self, data_df):
        
        # Apply the tuned orders of differencing to the data and rebuild dataframe structure
        data_diff_np = self.fracdiffstat.transform(data_df.to_numpy())
        data_diff_df = pd.DataFrame(
            data_diff_np,
            columns = data_df.columns,
            index = data_df.index[-data_diff_np.shape[0]:]
        )
        
        return data_diff_df
        