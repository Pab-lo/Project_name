
from Preprocessing import *
from Plotting import *


data_all_df, data_train_df, data_val_df, data_test_df = get_datasets(asset_mode = 'all')

#label_name = 'EURUSD Curncy'
label_name = 'AAPL_vol'

plot_spot(data_train_df[label_name].to_numpy(), data_train_df.index)
plot_spot(data_val_df[label_name].to_numpy(), data_val_df.index)
