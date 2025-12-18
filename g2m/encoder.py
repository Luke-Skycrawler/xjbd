import torch
from torch import nn
from torch import optim
from g2m.medial import SlabMesh
import numpy as np 
from g2m.utils import dqs_Q
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mid_effel = np.array([0.19910182, 0.35107037, 0.1984188])
# device = "cpu"

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
        self.VR = torch.tensor(VR, dtype=torch.float32).to(device)
        self.qm4 = torch.zeros((self.n_modes * 2, 4), dtype = torch.float32).to(device)
        self.qm4[:, -1] = 1.0
        self.y = torch.zeros((self.n_nodes, 4), dtype = torch.float32).to(device)
        


    def Gq(self, q):
        x, y, z, w = q
        return torch.tensor([
            [w, -z, y, -x],
            [z, w, -x, -y],
            [-y, x, w, -z],
        ], device=device)
    
    def Hq(self, q):
        x, y, z, w = q
        return torch.tensor([
            [w, z, -y, -x],
            [-z, w, x, -y],
            [y, -x, w, -z], 
        ], device = device)

    def Rq(self, q):
        return self.Gq(q) @ self.Hq(q).T


    def quat_to_rotmat(self, q):
        """
        q: (B, V, 4) tensor of unit quaternions (x, y, z, w)
        returns: (B, V, 3, 3) rotation matrices
        """
        x, y, z, w = q.unbind(dim=-1)

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        R = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
        ], dim=-1)

        return R.view(q.shape[:-1] + (3, 3))

    def dqs(self, x): 
        """
        x: (B, M, 4)  batch of mode quaternions
        returns: (B, V, 4)
        """
        M = 10
        qm4p1 = x.reshape((-1, M, 4))

        qm4 = self.qm4.unsqueeze(0).expand(qm4p1.shape[0], -1, -1).clone()
        qm4[:, :self.n_modes] = qm4p1
        # self.qm4[:self.n_modes] = qm4p1
        # qs = self.weights @ self.qm4

        qs = torch.einsum('vm,bmf->bvf', self.weights, qm4)
        qs = qs / torch.linalg.vector_norm(qs, dim = -1, keepdim = True)

        # gpt wrote this part to batch the dqs computation
        R = self.quat_to_rotmat(qs)                 
        # Extract vertex positions
        v = self.VR[:, :3]                           
        # Rotate all vertices at once
        # y = torch.bmm(R, v.unsqueeze(-1)).squeeze(-1) 
        y = torch.matmul(R, v.unsqueeze(-1)).squeeze(-1)
        # Preserve homogeneous coord
        w = self.VR[:, 3:4].unsqueeze(0).expand(y.shape[0], -1, -1)
        y = torch.cat([y, w], dim=-1)            

        return y.view(y.shape[0], -1)

        # old dqs implementation
        for i in range(self.VR.shape[0] // 4):
            q = qs[i]
            ri = self.Rq(q)
            vi = self.VR[i * 4 : i * 4 + 3]
            yi = ri @ vi
            self.y[i, :3] = yi
            self.y[i, 3] = self.VR[i * 4 + 3]
        return self.y.reshape(-1)
        
    def forward(self, x):
        y_dqs = self.dqs(x)
        return y_dqs


class Encoder(DQSEncoder):
    def __init__(self, n_modes, n_nodes, n_latent = 128, mid = mid_effel):
        super().__init__(n_modes, n_nodes, mid)
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

        layer_widths_encoder = [self.n_modes, 120, 60, 30]
        layer_widths_decoder = [self.n_nodes * 4 + 40, self.n_nodes * 4 + 60, self.n_nodes * 4 + 60, self.n_nodes * 4]

        self.layers_encoder = []
        self.layers_decoder = []

        self.fcs = []

        # for i in range(len(layer_widths_encoder) - 1):
        #     input_dim = layer_widths_encoder[i]
        #     output_dim = layer_widths_encoder[i + 1]
        #     non_linear_cond = True
        #     layer = nn.Linear(input_dim, output_dim)
        #     self.layers_encoder.append(layer)
        #     self.fcs.append(layer)
        #     if non_linear_cond:
        #         self.layers_encoder.append(nn.ReLU())

        
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

        # self.encoder = nn.Sequential(*self.layers_encoder)
        self.decoder = nn.Sequential(*self.layers_decoder)

        # nn.init.kaiming_normal(self.mlp)
        # for layer in [self.fc0, self.fc1, self.fc2]:
        for layer in self.fcs:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        
    def forward(self, x):
        # s = self.encoder(x)
        s = x
        vr = self.VR.flatten()
        if len(s.shape) == 1:
            VR = vr
        else:
            VR = vr.unsqueeze(0).expand(x.size(0), -1)

        y_dqs = self.dqs(s)
        # decoder_input = torch.cat([s, VR], dim=-1)
        decoder_input = torch.cat([s, VR], dim=-1)
        y = self.decoder(decoder_input)

        return (y + y_dqs).flatten()
        # return y_dqs
        # q = s[:3 * self.n_nodes].reshape(-1, 3)
        # r = s[3 * self.n_nodes:].reshape(-1) 
        # return q, r
        
