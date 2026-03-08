import torch.nn as nn
import torch
from src.model.train import train_torch_model

class TCN(nn.Module):

    def __init__(self, input_size, num_channels=64):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(input_size, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        out = self.network(x)

        out = out.squeeze(-1)

        return self.fc(out)
    


def train_tcn(config, data):

    device = config["training"]["device"]

    params = config["models"]["tcn"]

    model = TCN(
        input_size=config["data"]["input_size"],
        num_channels=params["num_channels"]
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = torch.nn.MSELoss()

    train_torch_model(
        model,
        data["dl"]["train_loader"],
        data["dl"]["val_loader"],
        optimizer,
        criterion,
        device
    )

    return model

