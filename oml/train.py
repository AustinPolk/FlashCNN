import pandas as pd
import torch
from model import FlashPoint
from torch import nn, optim
import numpy as np
import torch.utils.data as Data

def create_tensors(from_data: pd.DataFrame, variables: list):
    stations = from_data['STATION'].unique()
    station_tensors = {}
    station_dates = {}
    for station in stations:
        station_specific = from_data[from_data['STATION'] == station]
        date_lookup = {}
        index = pd.to_datetime(station_specific['DATE']).dt.strftime('%Y-%m-%d')
        for i in range(len(index)):
            date_lookup[index.iloc[i]] = i
        station_dates[station] = date_lookup
        station_tensors[station] = torch.from_numpy(station_specific[variables].values).float().unsqueeze(0)

    return station_tensors, station_dates

def create_datasets(tensor: torch.Tensor, lookback: int, lookahead: int):
    first_idx = lookback + 1
    last_idx = tensor.size()[1] - lookahead - 1
    
    X = []
    Y = []

    for idx in range(first_idx, last_idx, 1):
        b1, b2 = idx - lookback - 1, idx - 1
        input_features = tensor[:, b1:b2]
        a1, a2 = idx, idx + lookahead
        output_features = tensor[-1, a1:a2]

        X.append(input_features)
        Y.append(output_features)
    
    return torch.stack(X, dim=0), torch.stack(Y, dim=0)

def split(X: torch.Tensor, Y: torch.Tensor, date_lookup: dict, training_start: str, training_end: str, validation_start: str, validation_end: str):
    return ((X[date_lookup[training_start]:date_lookup[training_end]], Y[date_lookup[training_start]:date_lookup[training_end]]),
            (X[date_lookup[validation_start]:date_lookup[validation_end]], Y[date_lookup[validation_start]:date_lookup[validation_end]]))

def train(model: FlashPoint, training: tuple, validation: tuple, learning_rate=0.001, batch_size=8, epochs=100):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_X, train_Y = training
    val_X, val_Y = validation

    dataset = Data.TensorDataset(train_X, train_Y)
    loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.eval()
    with torch.no_grad():
        Y_pred = model(val_X)
        val_rmse = np.sqrt(loss_fn(Y_pred, val_Y))
        print(f'Current model test RMSE: {val_rmse}')

    training_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)

    checkpoint_model = model.copy()

    best_val_rmse = val_rmse
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            Y_pred = model(train_X)
            train_rmse = np.sqrt(loss_fn(Y_pred, train_Y))
            training_losses[epoch] = np.square(train_rmse)
            Y_pred = model(val_X)
            val_rmse = np.sqrt(loss_fn(Y_pred, val_Y))
            validation_losses[epoch] = np.square(val_rmse)
        
        if True:#(epoch + 1) % 5 == 0:
            print("Epoch %d: train RMSE %.4f, val RMSE %.4f" % (epoch+1, train_rmse, val_rmse))
    
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            print("Checkpointing at epoch %d with val RMSE %.4f" % (epoch + 1, val_rmse))
            checkpoint_model.copy_from(model)

    return checkpoint_model, training_losses, validation_losses

    

    