import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_V3(nn.Module):
    """
    Multilayer CNN with 1D convolutions
    """
    def __init__(
        self,
        in_channels,
        L_in,
        output_size,
        depth=2,
        filter_size=3, 
        n_filters=64, 
        n_neurons=64, 
        dropout=0.2,
        activation='relu',
    ):
        super().__init__()
        self.depth = depth
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        padding = int(np.floor(filter_size / 2))
        
        if depth == 1:
            self.conv1 = nn.Conv1d(in_channels, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(int(L_in * (n_filters) / 2), n_neurons)
            self.fc1_drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(n_neurons, 1)
    
        elif depth == 2:
            self.conv1 = nn.Conv1d(in_channels, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(int(L_in * (n_filters) / 4), n_neurons)
            self.fc1_drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(n_neurons, 1)
            
        elif depth == 3:
            self.conv1 = nn.Conv1d(in_channels, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool3 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(int(L_in * (n_filters) / 8), n_neurons)
            self.fc1_drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(n_neurons, 1)
    
    def forward(self, x):
        # x: tensor (batch_size, L_in, in_channels)
        x = x.transpose(1,2) # swap time and feature axes
        
        x = self.pool1(self.activation(self.conv1(x)))
        if self.depth == 2 or self.depth == 3:
            x = self.pool2(self.activation(self.conv2(x)))
        if self.depth == 3:
            x = self.pool3(self.activation(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # flatten
        x = self.activation(self.fc1_drop(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

class RNN_V2(nn.Module):
    """
    Multi-layer LSTM network
    """
    def __init__(
        self, 
        input_size,
        input_length,
        output_size,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        n_neurons=64,
        activation='relu',
    ):
        super().__init__()
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        
        self.lstm = nn.LSTM(int(input_size), int(hidden_size), int(num_layers), batch_first=True)
        self.fc1 = nn.Linear(hidden_size, n_neurons)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_neurons, output_size)
    
    def forward(self, x):
        # x: tensor (batch_size, T, input_size)
        # h_all: (batch_size, T, hidden_size)
        h_0, c_0 = self.init_hidden(x)
        h_all, (h_T, c_T) = self.lstm(x, (h_0, c_0))
        output = h_T[-1]
        output = self.activation(self.fc1_drop(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output
    
    def init_hidden(self, x):
        batch_size = x.size(0)
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device))