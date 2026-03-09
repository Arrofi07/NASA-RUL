import torch
import torch.nn as nn
from src.model.train import train_torch_model


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = out[:, -1, :]

        return self.fc(out)


def train_lstm(config, data):

    device = config["training"]["device"]

    params = config["models"]["lstm"]

    model = LSTMModel(
        input_size=config["data"]["input_size"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"]
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"]
    )

    criterion = torch.nn.MSELoss()

    train_torch_model(
        model,
        data["dl"]["train_loader"],
        data["dl"]["val_loader"],
        optimizer,
        criterion,
        device,
        epochs=config["training"]["epochs"]
    )

    return model