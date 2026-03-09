import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self, input_size):

        super(LSTMModel, self).__init__()

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(0.3)

        self.lstm3 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True
        )

        self.dropout3 = nn.Dropout(0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(32, 32)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):

        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out, _ = self.lstm3(out)

        # take last timestep
        out = out[:, -1, :]

        out = self.dropout3(out)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)

        return out
    
"""
Input: (batch, 30, 20)

LSTM1 → (batch, 30, 128)
Dropout

LSTM2 → (batch, 30, 64)
Dropout

LSTM3 → (batch, 30, 32)

Take last timestep
→ (batch, 32)

Dense(32)
→ (batch, 32)

Dense(1)
→ (batch, 1)
"""