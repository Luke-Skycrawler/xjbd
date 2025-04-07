import torch
from torch import nn
from torch import optim

class Encoder(nn.Module):
    def __init__(self, n_modes, n_nodes, n_latent = 32):

        super().__init__()
        self.n_modes = n_modes
        self.n_nodes = n_nodes
        self.n_latent = n_latent
        

        self.fc0 = nn.Linear(self.n_modes, self.n_latent)
        self.fc1 = nn.Linear(self.n_latent, self.n_latent)
        self.fc2 = nn.Linear(self.n_latent, self.n_nodes * 4)
        self.mlp = nn.Sequential(
            # nn.Linear(self.n_modes, self.n_latent), 
            self.fc0,
            nn.ReLU(),
            # nn.Linear(self.n_latent, self.n_latent), 
            self.fc1,
            nn.ReLU(),
            # nn.Linear(self.n_latent, self.n_nodes * 4)
            self.fc2
        )

        # nn.init.kaiming_normal(self.mlp)
        for layer in [self.fc0, self.fc1, self.fc2]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        s = self.mlp(x).flatten()

        return s
        # q = s[:3 * self.n_nodes].reshape(-1, 3)
        # r = s[3 * self.n_nodes:].reshape(-1) 
        # return q, r
        

