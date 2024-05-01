import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, latent_size):
        super(Autoencoder, self).__init__()
        self.L = latent_size
        self.encoder = nn.Sequential(
            nn.Linear(151, 64),
            nn.ReLU(),
            nn.Linear(64, self.L),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.L, 64),
            nn.ReLU(),
            nn.Linear(64, 151),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoencoderLSTM(nn.Module):
    def __init__(self, latent_size):
        super(AutoencoderLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size=151, hidden_size=latent_size, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),  # Fully connected layer
            nn.ReLU(),  # Activation function
            nn.Linear(64, 151),  # Output layer
            nn.Sigmoid()  # Sigmoid activation to restrict output to [0, 1]
        )

    def forward(self, x):
        # Encoder
        _, (hidden, cell) = self.encoder(x)
        # Decoder
        output = self.decoder(hidden[-1])  # Use only the hidden state of the last LSTM layer as input
        return output
