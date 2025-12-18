import torch
from torch import nn
from torch import optim
from g2m.medial import SlabMesh
import numpy as np 
from g2m.utils import dqs_Q
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
class Encoder(nn.Module):
    def __init__(self, n_modes, n_nodes, n_latent = 128):
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
        self.weights = np.load(f"data/W_medial_{model}.npy")[:, self.n_modes // 12]
        VR = np.hstack([self.slabmesh.V, self.slabmesh.R.reshape(-1, 1)])
        self.VR = torch.tensor(VR.reshape(-1), dtype=torch.float32).to(device)

        layer_widths_encoder = [self.n_modes, 120, 60, 30]
        layer_widths_decoder = [self.n_nodes * 4 + 40, self.n_nodes * 4 + 60, self.n_nodes * 4 + 60, self.n_nodes * 4]

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

    def Gq(self, q):
        x, y, z, w = q
        return torch.tensor([
            [w, -z, y, -x],
            [z, w, -x, -y],
            [-y, x, w, -z],
        ])
    
    def Hq(self, q):
        x, y, z, w = q
        return torch.tensor([
            [w, z, -y, -x],
            [-z, w, x, -y],
            [y, -x, w, -z],
        ])

    def Rq(self, q):
        return self.Gq(q) @ self.Hq(q).T

    def dqs(self, x): 
        qm4 = x.reshape((-1, 4))
        qs = self.weights @ qm4
        # nomalize qs by rows
        qs = qs / torch.norm(qs, p = 2, dim = 1)
        y = torch.zeros((self.n_nodes, 4), dtype = torch.float32).to(device)
        for i in range(self.VR.shape[0] // 4):
            q = qs[i]
            ri = self.Rq(q)
            vi = self.VR[i * 4 : i * 4 + 3]
            yi = ri @ vi
            y[i, :3] = yi
            y[i, 3] = self.VR[i * 4 + 3]
        return y.reshape(-1)
        
    def forward(self, x):
        # s = self.encoder(x)
        s = x
        if len(s.shape) == 1:
            VR = self.VR
        else:
            VR = self.VR.unsqueeze(0).expand(x.size(0), -1)

        y_dqs = self.dqs(s)
        # decoder_input = torch.cat([s, VR], dim=-1)
        decoder_input = torch.cat([s, VR], dim=-1)
        y = self.decoder(decoder_input)

        # return (y + y_dqs).flatten()
        return y_dqs
        # q = s[:3 * self.n_nodes].reshape(-1, 3)
        # r = s[3 * self.n_nodes:].reshape(-1) 
        # return q, r
        

class DQSEncoder(nn.Module): 
    def __init__(self, n_modes, n_nodes, mid = None):
        super().__init__()
        self.n_modes = n_modes
        self.n_nodes = n_nodes
        model = "effel"
        self.slabmesh = SlabMesh(f"assets/{model}/ma/{model}.ma")
        self.mid = mid
        
        weights = np.load(f"data/W_medial_{model}.npy")[:, 1:11]

        weights_tets = np.load(f"data/W_{model}.npy")[:, 1: 11] 
        _, self.Q_range, self.Q_min = dqs_Q(weights_tets)
        
        weights -= self.Q_min
        weights_comp = np.hstack([weights, 1.0 - weights])
        self.weights = torch.tensor(weights_comp, dtype=torch.float32).to(device)
        V = self.slabmesh.V.copy()
        if mid is not None: 
            V -= mid
        VR = np.hstack([V, self.slabmesh.R.reshape(-1, 1)])
        self.VR = torch.tensor(VR.reshape(-1), dtype=torch.float32).to(device)

    def Gq(self, q):
        x, y, z, w = q
        return torch.tensor([
            [w, -z, y, -x],
            [z, w, -x, -y],
            [-y, x, w, -z],
        ])
    
    def Hq(self, q):
        x, y, z, w = q
        return torch.tensor([
            [w, z, -y, -x],
            [-z, w, x, -y],
            [y, -x, w, -z],
        ])

    def Rq(self, q):
        return self.Gq(q) @ self.Hq(q).T

    def dqs(self, x): 
        qm4p1 = x.reshape((-1, 4))
        qm4p2 = torch.zeros_like(qm4p1)
        qm4p2[:, 3] = 1.0
        qm4 = torch.cat([qm4p1, qm4p2], dim = 0)
        qs = self.weights @ qm4
        # nomalize qs by rows
        qs = qs / torch.linalg.vector_norm(qs, dim = 1, keepdim = True)
        y = torch.zeros((self.n_nodes, 4), dtype = torch.float32).to(device)
        for i in range(self.VR.shape[0] // 4):
            q = qs[i]
            ri = self.Rq(q)
            vi = self.VR[i * 4 : i * 4 + 3]
            yi = ri @ vi
            y[i, :3] = yi
            y[i, 3] = self.VR[i * 4 + 3]
        return y.reshape(-1)
        
    def forward(self, x):
        y_dqs = self.dqs(x)
        return y_dqs
