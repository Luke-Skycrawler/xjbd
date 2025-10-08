import polyscope as ps
import polyscope.imgui as gui
import numpy as np 
from igl import lbs_matrix, dqs
from scipy.spatial.transform import Rotation as R
from g2m.viewer import MedialViewerInterface
from g2m.encoder import Encoder
from g2m.medial import SlabMesh
from g2m.utils import dqs_Q, euler_to_quat, euler_to_affine
import torch
import os
from fast_cd import RodLBSWeight
model = "effel"
dataset = ["10000_1e-3", "36d_2000_pi"]
class PSViewer:
    def __init__(self, Q, V0, F):
        self.Q = Q[:, 1:11]
        self.V0 = V0
        self.F = F

        self.ui_deformed_mode = 0
        self.ui_dqs = 0
        self.ui_sample_id = 0
        self.n_modes = self.Q.shape[1]
        self.B = lbs_matrix(self.V0, self.Q)
        self.T = np.zeros((4 * self.n_modes, 3))
        
        self.ui_magnitude = self.T[self.idx_to_T(self.ui_deformed_mode)]
        self.ui_exponent = 0


        self.ui_Rx = 0.0
        self.ui_Ry = 0.0 
        self.ui_Rz = 0.0


        self.Q, self.Q_range = dqs_Q(self.Q)

        self.t = np.zeros((self.n_modes * 2, 3))
        self.q = np.zeros((self.n_modes * 2, 4))
        self.q[:, 3] = 1.0


        self.ps_mesh = ps.register_surface_mesh("rod", V0, F)
        self.ps_mesh.add_scalar_quantity("weight", Q[:, 0], enabled = True)

        # load pretrained encoder
        self.n_modes = 120
        self.n_nodes = 150
        # name = "10000_1e-3"
        name = dataset[1]
        checkpoint = 6000

        self.encoder = Encoder(self.n_modes, self.n_nodes)
        self.encoder.load_state_dict(torch.load(f"data/{name}_{checkpoint}.pth"))
        self.encoder.eval()

        self.q_samples = np.load(f"data/pqsample/{name}.npy")

        self.q_120d = np.zeros((self.q_samples.shape[0], 120), np.float32)
        if self.q_samples.shape[1] == 120:
            self.q_120d = self.q_samples
        elif self.q_samples.shape[1] == 36:
            for id in range(self.q_samples.shape[0]):
                self.q_120d[id] = euler_to_affine(self.q_samples[id], self.Q_range)
        
    def current_magnitude(self):
        return self.T[self.idx_to_T(self.ui_deformed_mode)]
    def idx_to_T(self, idx):
        i = idx // 3
        j = idx % 3
        return i, j

    def to_dqs_weight(self, Q):
        return np.hstack([Q, 1.0 - Q])

    def compute_V(self):
        # return self.B @ self.T + self.V0

        if self.ui_dqs == 1:
            return dqs(self.V0.astype(float), self.Q, self.q, self.t)
        else:
            return self.B @ self.T+ self.V0
        
    def callback(self):
        changed, self.ui_sample_id = gui.InputInt("sample id", self.ui_sample_id)

        if changed: 
            q = self.q_samples[self.ui_sample_id]
            if self.q_samples.shape[1] == 36:
                quats = euler_to_quat(q, self.Q_range)
                self.q[:] = quats
            elif self.q_samples.shape[1] == 120:
                self.T[:] = self.q_120d[self.ui_sample_id].reshape(self.T.shape)

        self.V_deform = self.compute_V()

        self.ps_mesh.update_vertex_positions(self.V_deform)

        changed, self.ui_deformed_mode = gui.InputInt("#mode", self.ui_deformed_mode, step = 1)
        if changed:
            self.ui_magnitude = self.T[self.idx_to_T(self.ui_deformed_mode)]
        changed, self.ui_dqs = gui.SliderInt("dqs", self.ui_dqs, v_max=2)
        changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = 0.0, v_max = 10)
        changed, self.ui_exponent = gui.SliderInt("Exponent", self.ui_exponent, v_min = -5, v_max = 5)

        changed, self.ui_Rx = gui.SliderFloat("Rx", self.ui_Rx, v_min = -np.pi, v_max = np.pi)
        changed, self.ui_Ry = gui.SliderFloat("Ry", self.ui_Ry, v_min = -np.pi, v_max = np.pi)
        changed, self.ui_Rz = gui.SliderFloat("Rz", self.ui_Rz, v_min = -np.pi, v_max = np.pi)
        
        self.ps_mesh.add_scalar_quantity("weight", self.Q[:, self.ui_deformed_mode], enabled = True)

class PSViewerMedialSocket(MedialViewerInterface, PSViewer):
    def __init__(self, Q, V0, F):
        self.slabmesh = SlabMesh(f"assets/{model}/ma/{model}.ma")

        self.E = self.slabmesh.E
        self.V_rest = self.slabmesh.V
        self.R_rest = self.slabmesh.R

        self.V_medial = np.zeros_like(self.V_rest)
        self.R = np.zeros_like(self.R_rest)
        
        super().__init__(Q, V0, F)


    def callback(self):
        super().callback()

        q_input = self.q_120d[self.ui_sample_id]

        with torch.no_grad():
            # with wp.ScopedTimer("inference"):
            p = self.encoder(torch.from_numpy(q_input.astype(np.float32)))
            pnp = p.numpy().reshape(-1, 4)

            self.V_medial[:] = pnp[:, :3]
            self.R[:] = pnp[:, 3]

        self.update_medial()



if __name__ == "__main__":
    ps.init()
    ps.set_ground_plane_mode("none")
    rod = RodLBSWeight()
    # rod = RodLBSWeightBC()
    lam, Q = None, None
    if not os.path.exists(f"data/W_{model}.npy"):
    # if True:
        lam, Q = rod.eigs()
        np.save(f"data/W_{model}.npy", Q)
    else:
        Q = np.load(f"data/W_{model}.npy")
    
    viewer = PSViewerMedialSocket(Q, rod.V0, rod.F)
    ps.set_user_callback(viewer.callback)
    ps.show()
