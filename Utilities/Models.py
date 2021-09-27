
import numpy as np
import torch
from torch import nn
torch.manual_seed(0)



## Model related functions and structures


# Model loading function
def load_model(model_string):

    print(f"Loading trained model from Serialised_objects\model_{model_string}")
    model = torch.load(r"..\..\Serialised_objects\model_" + model_string)
    print(f"Model loaded.")
    
    return model



# Model saving
def save_model(model, model_string):
    
    print(f"Saving trained model to disk...")
    torch.save(model, r"..\..\Serialised_objects\model_" + model_string)
    print(f"Model saved.")



# A generic RNN/GRU/LSTM implementation for regression
class ForecastingModel(nn.Module):
    
    def __init__(self, data_dim, hidden_dim, num_layers, model_type = 'RNN', drop_prob = 0.):
        
        super(ForecastingModel, self).__init__()
        
        self.name = f"{data_dim}_{hidden_dim}_{num_layers}_{model_type}"
        
        # Recurrent component
        if model_type == 'RNN':
            self.core_layers = nn.RNN(data_dim, hidden_dim, num_layers, batch_first = True, dropout = drop_prob)
        elif model_type == 'GRU':
            self.core_layers = nn.GRU(data_dim, hidden_dim, num_layers, batch_first = True, dropout = drop_prob)
        elif model_type == 'LSTM':
            self.core_layers = nn.LSTM(data_dim, hidden_dim, num_layers, batch_first = True, dropout = drop_prob)
        else:
            raise ValueError("ForecastingModel.__init__: unrecognised model type.")
        
        # Linear output
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Dropout layers
        self.last_rnn_layer_dropout = torch.nn.Dropout(p = drop_prob)
        self.linear_dropout = torch.nn.Dropout(p = drop_prob)
        
        # Xavier initialisation (not used)
        use_Xavier = False
        if use_Xavier is True:
            for layer_p in self.core_layers._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        #print("bef", p, self.core_layers.__getattr__(p))
                        torch.nn.init.xavier_uniform_(self.core_layers.__getattr__(p))
                        #print("aft", p, self.core_layers.__getattr__(p))

            torch.nn.init.xavier_uniform_(self.output_layer.weight)
        
        
    def forward(self, x_sequence):
        
        core_out, hidden_neurons = self.core_layers(x_sequence)
        last_rnn_layer_dropout_out = self.last_rnn_layer_dropout(core_out)
        linear_out = self.output_layer(last_rnn_layer_dropout_out[:, last_rnn_layer_dropout_out.size(1) - 1, :])
        linear_dropout_out = self.linear_dropout(linear_out)
        
        return linear_dropout_out



# A generic RNN/GRU/LSTM implementation for classification
class ForecastingModelCat(ForecastingModel):
    
    def __init__(self, data_dim, hidden_dim, num_layers, model_type = 'RNN', drop_prob = 0., num_classes = 3):
        
        super(ForecastingModelCat, self).__init__(data_dim, hidden_dim, num_layers, model_type, drop_prob)
        
        # Linear output
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
        # Softmax
        self.softy = torch.nn.Softmax(dim = 1)
        
    def forward(self, x_sequence):
        
        core_out, hidden_neurons = self.core_layers(x_sequence)
        last_rnn_layer_dropout_out = self.last_rnn_layer_dropout(core_out)
        linear_out = self.output_layer(last_rnn_layer_dropout_out[:, last_rnn_layer_dropout_out.size(1) - 1, :])
        linear_dropout_out = self.linear_dropout(linear_out)
        softy_out = self.softy(linear_dropout_out)
        
        return softy_out



# A fully dense network with one encoding and one decoding layer
class AutoEncoder1Layer(nn.Module):
    
    def __init__(self, data_dim, compression_dim):
        
        super(AutoEncoder1Layer, self).__init__()
        
        self.name = f"autoencoder_{data_dim}_{compression_dim}_1"
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features = data_dim, out_features = compression_dim),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features = compression_dim, out_features = data_dim)
        )
        
    def forward(self, x):
        
        z = self.encoder(x)
        x_out = self.decoder(z)
        
        return x_out


    
# A fully dense network with two encoding and two decoding layers
class AutoEncoder2Layer(nn.Module):
    
    def __init__(self, data_dim, compression_dim):
        
        super(AutoEncoder2Layer, self).__init__()
        
        self.name = f"autoencoder_{data_dim}_{compression_dim}_2"
        
        intermediate_dim = int(0.5 * (data_dim + compression_dim))
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features = data_dim, out_features = intermediate_dim),
            nn.Sigmoid(),
            nn.Linear(in_features = intermediate_dim, out_features = compression_dim),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features = compression_dim, out_features = intermediate_dim),
            nn.Sigmoid(),
            nn.Linear(in_features = intermediate_dim, out_features = data_dim)
        )
        
    def forward(self, x):
        
        z = self.encoder(x)
        x_out = self.decoder(z)
        
        return x_out
