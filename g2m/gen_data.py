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
from g2m.utils import dqs_Q, euler_to_quat
from fast_cd import RodLBSWeight
# model = "bunny"
# model = "windmill"
model = "effel"

dataset = ["10000_1e-3", "36d_2000_pi"]

from stretch import eps
class PSViewer:
    def __init__(self, Q, V0, F, filename):
        self.Q = Q[:, 1:11]
        self.V0 = V0
        self.F = F

        self.ui_deformed_mode = 0
        self.ui_dqs = 1
        self.n_modes = self.Q.shape[1]
        self.B = lbs_matrix(self.V0, self.Q)
        self.T = np.zeros((4 * self.n_modes, 3))
        # self.T[:3, :] = np.eye(3)
        
        self.ui_magnitude = self.T[self.idx_to_T(self.ui_deformed_mode)]
        self.ui_exponent = 0


        self.ui_Rx = 0.0
        self.ui_Ry = 0.0 
        self.ui_Rz = 0.0

        self.t = np.zeros((self.n_modes * 2, 3))
        self.q = np.zeros((self.n_modes * 2, 4))
        self.q[:, 3] = 1.0

        self.Q, self.Q_range = dqs_Q(self.Q)

        self.ps_mesh = ps.register_surface_mesh("rod", V0, F)
        self.ps_mesh.add_scalar_quantity("weight", Q[:, 0], enabled = True)

        self.cnt_samples = 0

        self.qs = np.load(f"data/pqsample/{filename}.npy")
        self.ps = np.zeros((self.qs.shape[0], self.tbtt.slabmesh.nv, 4))
        self.filename = filename
        
    def current_magnitude(self):
        return self.T[self.idx_to_T(self.ui_deformed_mode)]
    def idx_to_T(self, idx):
        i = idx // 3
        j = idx % 3
        return i, j

    def to_dqs_weight(self, Q):
        return np.hstack([Q, 1.0 - Q])

    def compute_V(self, q):        
        if self.ui_dqs == 1:
            quats = euler_to_quat(q, self.Q_range)
            self.q = quats
            return dqs(self.V0.astype(float), self.Q, quats, self.t)
        else:
            # q: 120 float
            self.T = q.reshape(self.T.shape)
            # return self.B @ self.T + self.V0
            return self.B @ self.T+ self.V0
        
    def display(self):
        q = self.qs[self.cnt_samples]
        self.V_deform = self.compute_V(q)

        self.ps_mesh.update_vertex_positions(self.V_deform)

    def callback(self):

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

        self.display()

class PSDataExamine(MedialViewerInterface, PSViewer):
    def __init__(self, model, Q, V0, F, filename):
        self.tbtt = TetBaryCentricCompute(model)

        self.V_rest = self.tbtt.slabmesh.V
        self.R_rest = self.tbtt.slabmesh.R
        self.E = self.tbtt.slabmesh.E
        V, T = self.tbtt.V, self.tbtt.T

        self.V_medial = np.copy(self.V_rest)
        self.R = np.copy(self.R_rest)

        self.fitted_vr = np.load(f"data/pqsample/p_{filename}.npy")
        super().__init__(Q, V0, F, filename)


    def callback(self):  
        changed, self.cnt_samples = gui.InputInt("input #", self.cnt_samples, step = 1)
        if changed:
            self.display()
            p = self.fitted_vr[self.cnt_samples]
            self.V_medial = p[:, :3]
            self.R = p[:, 3]
            self.update_medial()
    
class PSViewerMedialSocket(MedialViewerInterface, PSViewer):
    def __init__(self, model, Q, V0, F, filename):
        self.tbtt = TetBaryCentricCompute(model)

        self.V_rest = self.tbtt.slabmesh.V
        self.R_rest = self.tbtt.slabmesh.R
        self.E = self.tbtt.slabmesh.E
        V, T = self.tbtt.V, self.tbtt.T
        self.fitter = Fitter(V, T, self.tbtt.slabmesh)

        self.V_medial = self.fitter.fitted_V
        self.R = self.fitter.fitted_R
        self.ui_fitter = True
        super().__init__(Q, V0, F, filename)

    def callback(self):
        super().callback()
        self.tbtt.deform(self.V_deform)
        changed, self.ui_fitter = gui.Checkbox("fitter", self.ui_fitter)
        if self.ui_fitter:
            V, R = self.fitter.V2p(self.V_deform)
        self.update_medial()
        p = np.hstack([self.V_medial, self.R.reshape(-1, 1)])
        self.ps[self.cnt_samples] = p
        self.cnt_samples += 1
        if self.cnt_samples % 100 == 0: 
            print(f"{self.cnt_samples} done")
        if self.cnt_samples == self.qs.shape[0]:
            np.save(f"data/pqsample/p_{self.filename}.npy", self.ps)
            quit()

    def q2p(self, q):
        self.V_deform = self.compute_V(q)
        self.ps_mesh.update_vertex_positions(self.V_deform)
        self.tbtt.deform(self.V_deform)
        V, R = self.fitter.V2p(self.V_deform)
        p = np.hstack([V, R.reshape(-1, 1)])
        self.V_medial = V
        self.R = R
        self.update_medial()
        return p

    def gen_samples(self):
        # no gui version
        qs = self.qs
        ps = self.ps
        filename = self.filename
        for i in range(qs.shape[0]):
            p = self.q2p(qs[i])
            ps[i] = p
            if i %  100 == 0: 
                print(f"{i} done")
        np.save(f"data/pqsample/p_{filename}.npy", ps)
            
        
        
def gen_data(): 
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.init()
    rod = RodLBSWeight()    
    if True:
        Q = np.load(f"data/W_{model}.npy")
    
    viewer = PSViewerMedialSocket(model, Q, rod.V0, rod.F, dataset[1])
    ps.set_user_callback(viewer.callback)
    # viewer.gen_samples(dataset[1])
    ps.show()

def examine_data():
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.init()
    rod = RodLBSWeight()    
    Q = np.load(f"data/W_{model}.npy")
    
    viewer = PSDataExamine(model, Q, rod.V0, rod.F, dataset[1])
    ps.set_user_callback(viewer.callback)
    # viewer.gen_samples(dataset[1])
    ps.show()

if __name__ == "__main__":
    # gen_data()
    examine_data()
    