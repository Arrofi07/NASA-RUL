import torch.nn as nn
from src.model.train import train_torch_model
import torch

class TransformerModel(nn.Module):

    def __init__(self, input_size, d_model=64, nhead=4):

        super().__init__()

        self.embedding = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):

        x = self.embedding(x)

        x = x.permute(1, 0, 2)

        out = self.transformer(x)

        out = out[-1]

        return self.fc(out)
    


def train_transformer(config, data):

    device = config["training"]["device"]

    params = config["models"]["transformer"]

    model = TransformerModel(
        input_size=config["data"]["input_size"],
        d_model=params["d_model"],
        nhead=params["nhead"]
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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