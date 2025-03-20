import polyscope as ps 
import polyscope.imgui as gui 
import numpy as np
import warp as wp 
from warp.sparse import BsrMatrix
from fem.interface import Rod
from scipy.sparse import bsr_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from warp.sparse import bsr_axpy, bsr_set_from_triplets, bsr_zeros
from fem.fem import Triplets
from igl import lbs_matrix
import os

class PSViewer:
    def __init__(self, Q, V0, F):
        self.Q = Q
        self.V0 = V0
        self.F = F
        self.ps_mesh = ps.register_surface_mesh("rod", V0, F)

        self.ui_deformed_mode = 0

        self.n_modes = Q.shape[1]
        self.B = lbs_matrix(self.V0, Q)
        self.T = np.zeros((4 * self.n_modes, 3))
        self.T[:3, :] = np.eye(3)
        
        self.ui_magnitude = self.T[self.idx_to_T(self.ui_deformed_mode)]

    def current_magnitude(self):
        return self.T[self.idx_to_T(self.ui_deformed_mode)]
    def idx_to_T(self, idx):
        i = idx // 3
        j = idx % 3
        return i, j

    def callback(self):
        self.V_deform = self.B @ self.T

        self.ps_mesh.update_vertex_positions(self.V_deform)

        changed, self.ui_deformed_mode = gui.InputInt("#mode", self.ui_deformed_mode, step = 1)
        if changed:
            self.ui_magnitude = self.T[self.idx_to_T(self.ui_deformed_mode)]

        changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = 0.0, v_max = 4)
        self.T[self.idx_to_T(self.ui_deformed_mode)] = self.ui_magnitude


@wp.struct
class CSRTriplets:
    rows: wp.array(dtype = int)
    cols: wp.array(dtype = int) 
    vals: wp.array(dtype = float)

@wp.kernel
def compute_Hw(triplets: Triplets, triplets_Hw: CSRTriplets):
    i = wp.tid()
    ii = triplets.rows[i]
    jj = triplets.cols[i]
    mat = triplets.vals[i]

    triplets_Hw.rows[i] = ii
    triplets_Hw.cols[i] = jj
    triplets_Hw.vals[i] = wp.trace(mat)


    
class RodLBSWeight(Rod):
    def __init__(self):
        self.filename = "assets/bar2.tobj"
        super().__init__()
        self.define_Hw()

    def eigs(self):
        K = self.to_scipy_csr()
        # print("start weight space eigs")
        with wp.ScopedTimer("weight space eigs"):
            lam, Q = eigsh(K, k = 10, which = "SM", tol = 1e-4)
            # Q_norm = np.linalg.norm(Q, axis = 0, ord = np.inf, keepdims = True)
            # Q /= Q_norm
        return lam, Q
        
    def to_scipy_csr(self):
        ii = self.Hw.offsets.numpy()
        jj = self.Hw.columns.numpy()
        values = self.Hw.values.numpy()

        csr = csr_matrix((values, jj, ii), shape = (self.n_nodes, self.n_nodes))
        return csr
        
    def define_Hw(self):
        self.triplets_Hw = CSRTriplets()
        self.triplets_Hw.rows = wp.zeros((self.n_tets * 4 * 4,), dtype = int)
        self.triplets_Hw.cols = wp.zeros_like(self.triplets_Hw.rows)
        self.triplets_Hw.vals = wp.zeros((self.n_tets * 4 *4), dtype = float)
        
        self.Hw = bsr_zeros(self.n_nodes, self.n_nodes, float)
        wp.launch(compute_Hw, (self.n_tets * 4 * 4,), inputs = [self.triplets, self.triplets_Hw])
        bsr_set_from_triplets(self.Hw, self.triplets_Hw.rows, self.triplets_Hw.cols, self.triplets_Hw.vals)

def vis_weights(): 
    model = "bar2"
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.init()
    rod = RodLBSWeight()
    lam, Q = None, None
    if not os.path.exists(f"data/W_{model}.npy"):
    # if True:
        lam, Q = rod.eigs()
        np.save(f"data/W_{model}.npy", Q)
    else:
        Q = np.load(f"data/W_{model}.npy")
    
    viewer = PSViewer(Q, rod.V0, rod.F)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    vis_weights()
    