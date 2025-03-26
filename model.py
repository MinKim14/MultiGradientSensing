import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(),
        )

    def forward(self, x):
        out = self.linear(x)
        return out


# LSTM model
class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(SimpleLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            dropout=0.3,
            num_layers=num_layers,
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size),
            nn.Dropout(0.3),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size),
            # nn.Softmax(),
        )

    def forward(self, x):
        emb, _ = self.lstm(x)

        emb = torch.cat([emb, x], dim=2)
        out = self.linear(emb[:, -1, :])
        # return out
        return out, emb[:, -1, :]


class SimpleGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(SimpleGRUModel, self).__init__()
        self.lstm = nn.GRU(
            input_size,
            hidden_size,
            batch_first=True,
            dropout=0.1,
            num_layers=num_layers,
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(),
        )

    def forward(self, x):
        emb, _ = self.lstm(x)
        out = self.linear(emb[:, -1, :])
        return out, emb[:, -1, :]


class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(SimpleLinearModel, self).__init__()

        self.first_linear = nn.ModuleList()
        self.first_linear.add_module(
            "linear_{}".format(0),
            nn.Sequential(
                nn.Linear(input_size * 32, hidden_size),
                nn.Dropout(0.1),
                nn.ReLU(),
            ),
        )
        for i in range(num_layers - 2):
            self.first_linear.add_module(
                "linear_{}".format(i + 1),
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                ),
            )

        self.first_linear.add_module(
            "linear_{}".format(num_layers - 1),
            nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Softmax(),
            ),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for i, layer in enumerate(self.first_linear):
            x = layer(x)
        return x, x
