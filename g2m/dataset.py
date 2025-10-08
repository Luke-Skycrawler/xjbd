import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from g2m.utils import euler_to_affine, dqs_Q
class RestShapeDataset(Dataset):
    def __init__(self, Q, V0):
        self.Q = Q
        self.V0 = V0

        self.n_modes = Q.shape[1]
        self.nv = V0.shape[0]

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.V0, torch.zeros(self.n_modes) 

    
class PQDataset(Dataset):
    def __init__(self, name = "10000_1e-3", start = 0, end = None):
        
        self.q = np.load(f"data/pqsample/{name}.npy")[start: end].astype(np.float32)
        Q = np.load(f"data/W_effel.npy")[:, 1:11]
        self.Q, self.Q_range = dqs_Q(Q)

        if self.q.shape[1] == 120: 
            self.q_120d = self.q
        elif self.q.shape[1] == 36:

            q = self.q.reshape((self.q.shape[0], 12, 3))
            self.q_120d = np.zeros((self.q.shape[0], 120), np.float32)
            for id in range(self.q.shape[0]):
                self.q_120d[id] = euler_to_affine(q[id], self.Q_range)

        self.p_prime = np.load(f"data/pqsample/p_{name}.npy")[start: end].astype(np.float32)

        print(f"in dataset: q shape, p shape = {self.q.shape}, {self.p_prime.shape}")
        self.n_modes = self.q_120d.shape[1]
        self.n_nodes = self.p_prime.shape[1]
        print(f"in dataset: n_modes, n_nodes = {self.n_modes}, {self.n_nodes}")


    def __len__(self):
        return self.q_120d.shape[0]


    def __getitem__(self, idx):
        return torch.from_numpy(self.p_prime[idx]), torch.from_numpy(self.q_120d[idx])

    
