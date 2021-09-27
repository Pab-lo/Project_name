
import numpy as np
import torch
torch.manual_seed(0)
from Datasets_and_sampling import *



## Training routines

def optimise_model(
    model,
    data_train_torch,
    shuffle,
    loss_function,
    optimiser,
    num_epochs,
    data_val_torch,
    Y_val,
    verbose = False,
    categorical = False
    ):
    
    # Create dataloader object
    batch_size = 200
    dataloader = get_dataloader(data_train_torch, batch_size, shuffle)
    
    # Container for loss
    loss_by_epoch = {'Train_loss' : {}, 'Val_loss' : {}}
    
    # Deploy model
    model.to(torch.device("cpu"))
    
    # Loop through epochs
    for epoch in range(1, num_epochs + 1):
        
        epoch_train_losses = []
        
        # Loop through mini-batches
        for idx, batch in enumerate(dataloader):
            
            # Get xs and ys
            x_batch, y_batch = batch
            
            # Zero the parameter gradients
            optimiser.zero_grad()
            
            # Forward pass
            batch_outputs = model(x_batch)
            
            # Get the right data type for the targets
            if categorical is False:
                #y_true = torch.flatten(y_batch).reshape(-1, 1)
                y_true = y_batch.reshape(-1, 1)
            else:
                y_true = torch.flatten(y_batch).type(torch.LongTensor)
            
            # Propagate the loss backwards
            batch_train_loss = loss_function(batch_outputs, y_true)
            batch_train_loss.backward()
            epoch_train_losses.append(batch_train_loss.item())
            
            # Take a step
            optimiser.step()
        
        # Record losses
        train_loss = np.mean(epoch_train_losses)
        if categorical is False:
                #Y_val_torch = torch.flatten(torch.FloatTensor(Y_val)).reshape(-1, 1)
                Y_val_torch = torch.FloatTensor(Y_val)
        else:
                Y_val_torch = torch.flatten(torch.LongTensor(Y_val))
        val_loss = calc_val_loss(model, data_val_torch, Y_val_torch, loss_function)
        
        loss_by_epoch['Train_loss'][epoch] = train_loss
        loss_by_epoch['Val_loss'][epoch] = val_loss
        
        if verbose is True and epoch % 5 == 0:
            print(f"Epoch {epoch} train loss: {loss_by_epoch['Train_loss'][epoch]}, val loss: {loss_by_epoch['Val_loss'][epoch]}")
    
    return loss_by_epoch


def train_autoencoder(
    model,
    X_train,
    X_val,
    num_epochs,
    verbose = False
    ):
    
    # Create Pytorch datasets
    X_train_torch = AutoencoderDataset(X_train)
    X_val_torch = AutoencoderDataset(X_val)
    
    # Choose loss and optimiser
    loss_function = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = 1.e-2)
    
    # Data will be shuffled
    shuffle = True
    
    # Train model
    loss_by_epoch = optimise_model(model, X_train_torch, shuffle, loss_function, optimiser, num_epochs, X_val_torch, X_val, verbose)
        
    return loss_by_epoch


def train_forecasting_model(
    model,
    series_type,
    X_train,
    Y_train,
    X_val,
    Y_val,
    num_lags,
    num_epochs,
    verbose = True,
    categorical = False
    ):
    
    # Val data should immediately follow train data in time; we join them here
    X_train_tail_plus_val = np.concatenate((X_train[-num_lags:], X_val), axis = 0)
    Y_train_tail_plus_val = np.concatenate((Y_train[-num_lags:], Y_val), axis = 0)
    
    # Create Pytorch datasets
    if series_type == 'multi_dim':
        data_train_torch = PricesDatasetMultiDim(X_train, Y_train, x_len = num_lags)
        data_val_torch = PricesDatasetMultiDim(X_train_tail_plus_val, Y_train_tail_plus_val, x_len = num_lags)
    elif series_type == 'one_dim':
        data_train_torch = PricesDatasetOneDim(X_train, Y_train, x_len = num_lags)
        data_val_torch = PricesDatasetOneDim(X_train_tail_plus_val, Y_train_tail_plus_val, x_len = num_lags)
    
    # Choose loss and optimiser
    if categorical is False:
        loss_function = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr = 1.e-3)
    else:
        loss_function = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr = 1.e-2)
    
    # Data will not be shuffled
    shuffle = False
    
    # Train model
    loss_by_epoch = optimise_model(model, data_train_torch, shuffle, loss_function, optimiser, num_epochs, data_val_torch, Y_val, verbose, categorical)
        
    return loss_by_epoch


def calc_val_loss(model, data_val_torch, true_values, loss_function):
    
    # Make predictions on val data
    val_preds = run_model_on_data(model, data_val_torch)
    
    # Calculate loss    
    val_loss = loss_function(val_preds, true_values).item()
    
    # Return loss
    return val_loss


def run_model_on_data(model, data_test_torch):
    
    dataloader = get_dataloader(data_test_torch, batch_size = data_test_torch.__len__(), shuffle = False)
    for idx, batch in enumerate(dataloader):
        preds = model(batch[0].to(torch.device("cpu")))
    
    return preds
