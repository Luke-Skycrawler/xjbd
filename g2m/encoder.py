import torch
from torch import nn
from torch import optim
from g2m.medial import SlabMesh
import numpy as np 
class Encoder(nn.Module):
    def __init__(self, n_modes, n_nodes, n_latent = 128):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()
        self.n_modes = n_modes
        self.n_nodes = n_nodes
        self.n_latent = n_latent
        

        # self.fc0 = nn.Linear(self.n_modes, self.n_latent)
        # self.fc1 = nn.Linear(self.n_latent, self.n_latent)
        # self.fc2 = nn.Linear(self.n_latent, self.n_nodes * 4)
        # self.mlp = nn.Sequential(
        #     # nn.Linear(self.n_modes, self.n_latent), 
        #     self.fc0,
        #     nn.ReLU(),
        #     # nn.Linear(self.n_latent, self.n_latent), 
        #     self.fc1,
        #     nn.ReLU(),
        #     # nn.Linear(self.n_latent, self.n_nodes * 4)
        #     self.fc2
        # )

        model = "effel"
        self.slabmesh = SlabMesh(f"assets/{model}/ma/{model}.ma")
        VR = np.hstack([self.slabmesh.V, self.slabmesh.R.reshape(-1, 1)])
        self.VR = torch.tensor(VR.reshape(-1), dtype=torch.float32).to(device)

        layer_widths_encoder = [self.n_modes, 120, 60, 30]
        layer_widths_decoder = [self.n_nodes * 4 + 30, self.n_nodes * 4 + 60, self.n_nodes * 4 + 60, self.n_nodes * 4]

        self.layers_encoder = []
        self.layers_decoder = []

        self.fcs = []

        for i in range(len(layer_widths_encoder) - 1):
            input_dim = layer_widths_encoder[i]
            output_dim = layer_widths_encoder[i + 1]
            non_linear_cond = True
            layer = nn.Linear(input_dim, output_dim)
            self.layers_encoder.append(layer)
            self.fcs.append(layer)
            if non_linear_cond:
                self.layers_encoder.append(nn.ReLU())

        
        for i in range(len(layer_widths_decoder) - 1):
            input_dim = layer_widths_decoder[i]
            output_dim = layer_widths_decoder[i + 1]
            non_linear_cond = i < len(layer_widths_decoder) - 2
            layer = nn.Linear(input_dim, output_dim)
            self.layers_decoder.append(layer)
            self.fcs.append(layer)
            if non_linear_cond:
                self.layers_decoder.append(nn.ReLU())
        

        # self.mlp = nn.Sequential(*self.layers)
        self.encoder = nn.Sequential(*self.layers_encoder)
        self.decoder = nn.Sequential(*self.layers_decoder)

        # nn.init.kaiming_normal(self.mlp)
        # for layer in [self.fc0, self.fc1, self.fc2]:
        for layer in self.fcs:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        s = self.encoder(x)
        if len(s.shape) == 1:
            VR = self.VR
        else:
            VR = self.VR.unsqueeze(0).expand(x.size(0), -1)
        decoder_input = torch.cat([s, VR], dim=-1)
        y = self.decoder(decoder_input)

        return y.flatten()
        # q = s[:3 * self.n_nodes].reshape(-1, 3)
        # r = s[3 * self.n_nodes:].reshape(-1) 
        # return q, r
        

