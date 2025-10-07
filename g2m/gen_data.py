import polyscope as ps 
import polyscope.imgui as gui 
import numpy as np
import warp as wp 
from fem.interface import Rod
from scipy.sparse import bsr_matrix, csr_matrix, bmat, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import null_space
from scipy.io import savemat, loadmat
from scipy.spatial.transform import Rotation as R
from warp.sparse import bsr_axpy, bsr_set_from_triplets, bsr_zeros
from fem.fem import Triplets
from fem.params import rho
from igl import lbs_matrix, massmatrix, dqs
import igl
import os
from g2m.viewer import MedialViewerInterface
from g2m.bary_centric import TetBaryCentricCompute
from g2m.naive_fitter import Fitter
from fast_cd import RodLBSWeight
# model = "bunny"
# model = "windmill"
model = "effel"
from stretch import eps
class PSViewer:
    def __init__(self, Q, V0, F):
        self.Q = Q[:, 1:11]
        self.V0 = V0
        self.F = F

        self.ui_deformed_mode = 0
        self.ui_dqs = False
        self.n_modes = self.Q.shape[1]
        self.B = lbs_matrix(self.V0, self.Q)
        self.T = np.zeros((4 * self.n_modes, 3))
        # self.T[:3, :] = np.eye(3)
        
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
        elif self.ui_dqs == 2: 
            rr = rotation.as_matrix().T
            rrt = np.vstack([rr, np.zeros((1, 3))])
            start = 0
            end = start + 4
            T = np.zeros((8, 3))
            T[start: end, :] = rrt
            T[start + 4: end + 3, :] = np.eye(3)
            B = lbs_matrix(self.V0, Q)
            return B @ T
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

        changed = gui.Button("Random")
        if changed:
            mag = np.abs(self.ui_magnitude) * pow(10, self.ui_exponent)
            rand_T = np.random.uniform(-mag, mag, (4 * self.n_modes, 3))
            self.T = rand_T

        # self.T[self.idx_to_T(self.ui_deformed_mode)] = self.ui_magnitude


        changed, self.ui_Rx = gui.SliderFloat("Rx", self.ui_Rx, v_min = -np.pi, v_max = np.pi)
        changed, self.ui_Ry = gui.SliderFloat("Ry", self.ui_Ry, v_min = -np.pi, v_max = np.pi)
        changed, self.ui_Rz = gui.SliderFloat("Rz", self.ui_Rz, v_min = -np.pi, v_max = np.pi)
        
        self.ps_mesh.add_scalar_quantity("weight", self.Q[:, self.ui_deformed_mode], enabled = True)

class PSViewerMedialSocket(MedialViewerInterface, PSViewer):
    def __init__(self, model, Q, V0, F):
        self.tbtt = TetBaryCentricCompute(model)

        self.V_rest = self.tbtt.slabmesh.V
        self.R_rest = self.tbtt.slabmesh.R
        self.E = self.tbtt.slabmesh.E
        V, T = self.tbtt.V, self.tbtt.T
        self.fitter = Fitter(V, T, self.tbtt.slabmesh)

        self.V_medial = self.fitter.fitted_V
        self.R = self.fitter.fitted_R
        self.ui_fitter = False
        super().__init__(Q, V0, F)

    def callback(self):
        super().callback()
        self.tbtt.deform(self.V_deform)
        changed, self.ui_fitter = gui.Checkbox("fitter", self.ui_fitter)
        if self.ui_fitter:
            V, R = self.fitter.V2p(self.V_deform)
        self.update_medial()

    def q2p(self, q):
        self.T = q.reshape(self.T.shape)
        self.V_deform = self.compute_V()
        self.tbtt.deform(self.V_deform)
        V, R = self.fitter.V2p(self.V_deform)
        p = np.hstack([V, R.reshape(-1, 1)])

        return p

    def gen_samples(self, filename):
        qs = np.load(f"data/pqsample/{filename}.npy")
        ps = np.zeros((qs.shape[0], self.fitter.nv_medial, 4))
        for i in range(qs.shape[0]):
            p = self.q2p(qs[i])
            ps[i] = p
            if i %  100 == 0: 
                print(f"{i} done")
        np.save(f"data/pqsample/p_{filename}.npy", ps)
            
        
        
def vis_weights(): 
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.init()
    rod = RodLBSWeight()    
    if True:
        Q = np.load(f"data/W_{model}.npy")
    
    viewer = PSViewerMedialSocket(model, Q, rod.V0, rod.F)
    # ps.set_user_callback(viewer.callback)
    viewer.gen_samples("10000_1e-3")

if __name__ == "__main__":
    vis_weights()
    