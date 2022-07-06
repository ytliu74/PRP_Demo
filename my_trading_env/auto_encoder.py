import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.4),
            nn.Tanh(),
            nn.Linear(1024, output_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, input_dim),
        )

    def forward(self, x):
        encoded = self.decoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
