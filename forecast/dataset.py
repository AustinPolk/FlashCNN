import torch

def create_cnn_dataset(dataset, num_lag_features):
    X, y = [], []
    for i in range(len(dataset)-num_lag_features):
        feature = dataset[i:i+num_lag_features]
        target = dataset[i+1]
        X.append(feature)
        y.append(target)
    return torch.stack(X, dim=0), torch.stack(y, dim=0)

def create_lstm_dataset(dataset, num_lag_features):
    X, y = [], []
    for i in range(len(dataset)-num_lag_features):
        feature = dataset[i:i+num_lag_features]
        target = dataset[i+1:i+num_lag_features+1]
        X.append(feature)
        y.append(target)
    return torch.stack(X, dim=0), torch.stack(y, dim=0)