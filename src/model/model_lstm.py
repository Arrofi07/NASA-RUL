import torch
import torch.nn as nn
from src.model.train import train_torch_model


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size=64, num_layers=2):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        return self.fc(out)
    
def train_lstm(train_loader, val_loader, input_size, device):

    model = LSTMModel(input_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = torch.nn.MSELoss()

    train_torch_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device
    )

    return model