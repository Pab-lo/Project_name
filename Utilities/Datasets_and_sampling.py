
import numpy as np
import pandas as pd
import os
import torch
torch.manual_seed(0)



## General data utilities

def load_raw_data(asset_mode = 'all'):
    
    if asset_mode == 'all':
        sne_df = load_raw_stock_data()
        macro_df = load_raw_macro_data()
        raw_df = sne_df.merge(macro_df, how = 'left', left_index = True, right_index = True)
    elif asset_mode == 'macro':
        raw_df = load_raw_macro_data()
    elif asset_mode == 'stock':
        raw_df = load_raw_stock_data()
    else:
        raise ValueError("load_raw_data: asset_mode not recognised")
        
    return raw_df


def load_raw_macro_data():
    
    # Load from csv
    data_path = r"..\..\Data\Project_data.csv"
    data_raw = pd.read_csv(data_path, header = [0, 1], index_col = 0)
    data_raw = data_raw[::-1]  # reverse
    
    # Extract only PX_LAST entries
    data_raw = data_raw.xs('PX_LAST', axis = 1, level = 1)
    
    return data_raw


def load_raw_stock_data():
    
    # Set an absolute date range
    start = "01/01/1970"
    end = "13/08/2021"
    date_range = pd.date_range(start = start, end = end)
    
    # Create an empty dataframe with the date range as index
    sne_df = pd.DataFrame([], index = date_range)
    
    # Write the stock data path
    sne_path = r"..\..\Data\Single_name_equities"
    directory = os.fsencode(sne_path)
    
    # Loop through the files and strap on the relevant column for each
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            
            # Load all the data for the stock
            stock_data = pd.read_csv(os.path.join(sne_path, filename), header = 0, index_col = 0)
            
            # Split out the different columns
            stock_open = stock_data['Open']
            stock_high = stock_data['High']
            stock_low = stock_data['Low']
            stock_close = stock_data['Close']
            stock_adj_close = stock_data['Adj Close']
            
            # Compute an estimate of volatility using Garman-Klass-Yang-Zhang
            vol_window = 12
            
            stock_previous_close = stock_close.shift(1)
            term1 = ((np.log(stock_open) - np.log(stock_previous_close))**2).rolling(vol_window).mean()
            term2 = 0.5 * ((np.log(stock_high) - np.log(stock_low))**2).rolling(vol_window).mean()
            term3 = -(2. * np.log(2.) - 1.) * ((np.log(stock_close) - np.log(stock_open))**2).rolling(vol_window).mean()
            
            gkyz_vol = np.sqrt(252. / vol_window) * np.sqrt(term1 + term2 + term3)
            
            # Add the spot and vol columns to the overall dataframe
            stock_adj_close.name = filename[:-4]
            gkyz_vol.name = f"{filename[:-4]}_vol"
            sne_df = sne_df.merge(stock_adj_close, how = 'left', left_index = True, right_index = True)
            sne_df = sne_df.merge(gkyz_vol, how = 'left', left_index = True, right_index = True)
            
        else:
            continue
            
    return sne_df


def data_splitter(data, frac = 0.6):
    
    ## Find cut index
    set_size = int(frac * data.shape[0])
    
    # Split into two parts
    first_set = data.iloc[:set_size]
    second_set = data.iloc[set_size:]
    
    return first_set, second_set
    
    
def get_dataloader(pytorch_dataset, batch_size, shuffle, sampling_weights = None):

    # Optional use of weighted random sampler    
    if sampling_weights is not None:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
            sampling_weights,
            num_samples = len(sampling_weights),
            replacement = True
        )
    else:
        sampler = None
    
    dataloader = torch.utils.data.DataLoader(
        pytorch_dataset,
        batch_size = batch_size,
        sampler = sampler,
        shuffle = shuffle
    )
    
    return dataloader



## Pytorch dataset implementations

class AutoencoderDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_array):
        
        self.data_array = data_array
        
        
    def __len__(self):
        
        return self.data_array.__len__()

    
    def __getitem__(self, index):
        
        diagonal_pair = (torch.FloatTensor(self.data_array[index]),
                         torch.FloatTensor(self.data_array[index]))
        
        return diagonal_pair


# Sample a multi-dim sequence from the array
class PricesDatasetMultiDim(torch.utils.data.Dataset):
    
    def __init__(self, X, Y, x_len, y_len = 1):
        
        self.X = X
        self.Y = Y
        self.x_len = x_len
        self.y_len = y_len
    
    
    def __len__(self):
        
        return self.X.__len__() - (self.x_len + self.y_len - 1)
    
    
    def __getitem__(self, index):
        
        window_pair = (torch.FloatTensor(self.X[index:index + self.x_len]),
                       torch.FloatTensor(self.Y[index + self.x_len:index + self.x_len + self.y_len]))
        
        return window_pair


# Sample a 1-dim sequence from any column of the array
class PricesDatasetOneDim(torch.utils.data.Dataset):
    
    def __init__(self, X, Y, x_len, y_len = 1):
        
        # We can't just stack the columns to make a big 1-dim array as then we might sample sequences
        # that cross the join points, which we don't want to do
        self.X = X
        self.Y = Y
        self.x_len = x_len
        self.y_len = y_len
        self.sequences_per_column = self.X.shape[0] - (self.x_len + self.y_len - 1)
        
        
    def __len__(self):
        
        return self.sequences_per_column * self.X.shape[1]

    
    def __getitem__(self, index):
        
        # We ignore Y here and take the target values from X
        col_idx = index // self.sequences_per_column
        pos_idx = index % self.sequences_per_column
        window_pair = (torch.FloatTensor(self.X[pos_idx:pos_idx + self.x_len, col_idx])[:, None],
                       torch.FloatTensor(self.X[pos_idx + self.x_len:pos_idx + self.x_len + self.y_len, col_idx])[:, None])
        
        return window_pair
