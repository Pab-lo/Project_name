
import statsmodels.api as sm
import sklearn.model_selection as skms
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from Preprocessing import *



class ExponentialSmoothing_df():
    
    def __init__(self, data_tofit_df):
    
        # Get columns and make container for model results
        columns = data_tofit_df.columns.tolist()
        self.model_data = {col_name : {} for col_name in columns}
        
        # Create an array to store the smoothed data
        data_fitted_np = np.zeros(data_tofit_df.shape)
        
        # We must use one model per column
        for col_idx in range(len(columns)):
            
            # Create model object and store it
            model = sm.tsa.ExponentialSmoothing(
                data_tofit_df[columns[col_idx]].to_numpy(),
                trend = 'add'
            )
            self.model_data[columns[col_idx]]['model'] = model
            
            # Fit to data and store results
            results = model.fit()
            self.model_data[columns[col_idx]]['params'] = results
            
            # Smooth this column
            data_fitted_np[:, col_idx] = model.predict(
                results.params,
                start = 0,
                end = data_tofit_df.shape[0] - 1
            )
            
        # Combine the smoothed columns into a dataframe
        self.data_fitted_df = pd.DataFrame(
            data_fitted_np,
            columns = data_tofit_df.columns,
            index = data_tofit_df.index
        )
    
    
    def apply_exp_smooth(self, data_df):
        
        # Apply the tuned smoothing parameters to the data and rebuild dataframe structure
        columns = data_df.columns.tolist()
        data_smoothed_np = np.zeros(data_df.shape)
        
        for col_idx in range(len(columns)):
            
            model = sm.tsa.ExponentialSmoothing(
                data_df[columns[col_idx]].to_numpy(),
                trend = 'add',
            )
            
            results = model.fit(
                optimized = False,
                smoothing_level = self.model_data[columns[col_idx]]['params'].params['smoothing_level'],
                smoothing_trend = self.model_data[columns[col_idx]]['params'].params['smoothing_trend'],
                initial_level = self.model_data[columns[col_idx]]['params'].params['initial_level'],
                initial_trend = self.model_data[columns[col_idx]]['params'].params['initial_trend']
            )
            
            data_smoothed_np[:, col_idx] = model.predict(
                self.model_data[columns[col_idx]]['params'].params,
                start = 0,
                end = data_df.shape[0] - 1
            )
            
        data_smoothed_df = pd.DataFrame(
            data_smoothed_np,
            columns = data_df.columns,
            index = data_df.index
        )
        
        return data_smoothed_df
