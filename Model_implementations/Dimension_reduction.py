
import time
import sys
sys.path.append("../Utilities")
from Preprocessing import *
from Models import *
from Training import *
from Plotting import *



## Get pre-processed and split data

# Choose data configs
label_name = 'GBPUSD Curncy'  # not used by model so any pair works
asset_mode = 'macro'
#label_name = 'AAPL'
#asset_mode = 'stock'
diff_mode = 'logret'
#diff_mode = 'fracdiff'
smoothing = False
scaling = False

# Get data
X_train, _, X_val, _, _, _ = get_preprocessed_datasets(label_name, asset_mode, diff_mode, smoothing, scaling)



## Create list of candidate models

data_dim = X_train.shape[1]
compressed_dim = 6

models_list = [
    AutoEncoder1Layer(data_dim, compressed_dim),
    AutoEncoder2Layer(data_dim, compressed_dim),
]

models_data = {model.name : {} for model in models_list}



## Train the models

num_epochs = 200

# Start timer
start = time.perf_counter()

for model in models_list:
    
    print(f"Training model {model.name}.\n")
    model_start = time.perf_counter()
    
    models_data[model.name]['loss_by_epoch'] = train_autoencoder(
        model,
        X_train,
        X_val,
        num_epochs,
        verbose = True
    )
    
    model_end = time.perf_counter()
    print(f"Training time for model {model.name}: {time.strftime('%H:%M:%S', time.gmtime(model_end - model_start))}.\n")
    
# End timer
end = time.perf_counter()
print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(end - start))}")



## Run the models on the validation set

for model in models_list:
    
    encoded_spots = model.encoder(torch.FloatTensor(X_val)).detach().numpy()
    
    models_data[model.name]['Encoded_spots'] = encoded_spots
    models_data[model.name]['Decoded_spots'] = model.decoder(torch.FloatTensor(encoded_spots)).detach().numpy()



## Plot compressed data and training and validation loss for each model
for model in models_list:
    plot_compressed_data(models_data[model.name]['Encoded_spots'])
    plot_learning_curves(models_data[model.name]['loss_by_epoch'])



## Save trained models to disk
for model in models_list:
    model_string = f"{model.name}_{num_epochs}_{asset_mode}_{diff_mode}_{smoothing}_{scaling}"
    save_model(model, model_string)
