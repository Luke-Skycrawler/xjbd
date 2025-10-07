import polyscope as ps
import polyscope.imgui as gui
import numpy as np 
from igl import lbs_matrix, dqs
from scipy.spatial.transform import Rotation as R
from g2m.viewer import MedialViewerInterface
from g2m.encoder import Encoder
from g2m.medial import SlabMesh
import torch
import os
from fast_cd import RodLBSWeight

model = "effel"

class PSViewer:
    def __init__(self, Q, V0, F):
        self.Q = Q[:, 1:11]
        self.V0 = V0
        self.F = F

        self.ui_deformed_mode = 0
        self.ui_dqs = False
        self.ui_sample_id = 0
        self.n_modes = self.Q.shape[1]
        self.B = lbs_matrix(self.V0, self.Q)
        self.T = np.zeros((4 * self.n_modes, 3))
        
        self.ui_magnitude = self.T[self.idx_to_T(self.ui_deformed_mode)]
        self.ui_exponent = 0


        self.ui_Rx = 0.0
        self.ui_Ry = 0.0 
        self.ui_Rz = 0.0

        self.t = np.zeros((2, 3))
        self.q = np.zeros((2, 4))
        self.q[:, 3] = 1.0


        Q = np.zeros((self.Q.shape))
        Q[:] = self.Q[:] 
        # Q_max = np.max(np.abs(Q), axis = 0, keepdims = True)
        # Q /= Q_max
        
        Q_max_signed = np.max(Q, axis = 0, keepdims = True)
        Q_min = np.min(Q, axis = 0, keepdims = True)
        Q_range = Q_max_signed - Q_min
        Q[:, 1:] -= Q_min[:, 1:] 
        Q[:, 1:] /= Q_range[:, 1:]
        
        self.Q = Q

        self.ps_mesh = ps.register_surface_mesh("rod", V0, F)
        self.ps_mesh.add_scalar_quantity("weight", Q[:, 0], enabled = True)

        # load pretrained encoder
        self.n_modes = 120
        self.n_nodes = 150
        name = "10000_1e-3"
        checkpoint = 5000

        self.encoder = Encoder(self.n_modes, self.n_nodes)
        self.encoder.load_state_dict(torch.load(f"data/{name}_{checkpoint}.pth"))
        self.encoder.eval()
        self.q_samples = np.load(f"data/pqsample/{name}.npy")
        
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

        rotation = R.from_euler('xyz', [self.ui_Rx, self.ui_Ry, self.ui_Rz], degrees = False)
        q = rotation.as_quat()
        self.q[self.ui_deformed_mode // 12] = q
        
        Q = self.to_dqs_weight(self.Q[:, self.ui_deformed_mode: self.ui_deformed_mode + 1])
        if self.ui_dqs == 1:
            return dqs(self.V0.astype(float), Q, self.q, self.t)
        else:
            return self.B @ self.T+ self.V0
        
    def callback(self):
        self.V_deform = self.compute_V()

        self.ps_mesh.update_vertex_positions(self.V_deform)

        changed, self.ui_deformed_mode = gui.InputInt("#mode", self.ui_deformed_mode, step = 1)
        if changed:
            self.ui_magnitude = self.T[self.idx_to_T(self.ui_deformed_mode)]
        changed, self.ui_dqs = gui.SliderInt("dqs", self.ui_dqs, v_max=2)
        changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = 0.0, v_max = 10)
        changed, self.ui_exponent = gui.SliderInt("Exponent", self.ui_exponent, v_min = -5, v_max = 5)

        changed, self.ui_sample_id = gui.InputInt("sample id", self.ui_sample_id)

        if changed: 
            self.T = self.q_samples[self.ui_sample_id].reshape(self.T.shape)

        # changed = gui.Button("Random")
        # if changed:
        #     mag = np.abs(self.ui_magnitude) * pow(10, self.ui_exponent)
        #     rand_T = np.random.uniform(-mag, mag, (4 * self.n_modes, 3))
        #     self.T = rand_T

        # self.T[self.idx_to_T(self.ui_deformed_mode)] = self.ui_magnitude


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

        q_input = self.q_samples[self.ui_sample_id]
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
