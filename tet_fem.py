import warp as wp
import numpy as np
from warp.sparse import bsr_set_from_triplets, bsr_zeros, BsrMatrix
from fem.params import *
from scipy.sparse.linalg import eigsh
from scipy.sparse import bsr_matrix
from fem.geometry import TOBJLoader

import polyscope as ps
import polyscope.imgui as gui

@wp.struct 
class Triplets:
    rows: wp.array(dtype = int)
    cols: wp.array(dtype = int)
    vals: wp.array(dtype = wp.mat33)

@wp.func
def xq_tet(geo: FEMMesh, e: int):
    x0 = geo.xcs[geo.T[e, 0]]
    x1 = geo.xcs[geo.T[e, 1]]
    x2 = geo.xcs[geo.T[e, 2]]
    x3 = geo.xcs[geo.T[e, 3]]

    return 0.25 * (x0 + x1 + x2 + x3)

@wp.func
def bf_tet(geo: FEMMesh, e: int, _i: int, x: wp.vec3):
    bi = 0.25
    n = normal(geo, e, _i)
    x0 = geo.xcs[geo.T[e, _i]]
    k = 0.75 / (wp.dot(x0 - x, n))
    dbidx = n * k
    
    return bi, dbidx

@wp.func
def normal(geo: FEMMesh, e: int, _i: int) -> wp.vec3:
    v0 = geo.T[e, 0]
    v1 = geo.T[e, 1]
    v2 = geo.T[e, 2]

    v = geo.T[e, _i]
    x = geo.xcs[v]
    if _i == 0:
        v0 = geo.T[e, 3]
    if _i == 1:
        v1 = geo.T[e, 3]
    if _i == 2:
        v2 = geo.T[e, 3]

    x0 = geo.xcs[v0]
    x1 = geo.xcs[v1]
    x2 = geo.xcs[v2]

    n = wp.cross(x1 - x0, x2 - x0)
    n /= wp.length(n)
    # n = (x1 - x0).cross(x2 - x0).normalized()
    # if n.dot(x - x0) < 0:
    if wp.dot(n, x - x0) < 0.0:
        n = -n
    return n

@wp.kernel
def compute_W(geo: FEMMesh, W: wp.array(dtype = float)): 
    e = wp.tid()

    x0 = geo.xcs[geo.T[e, 0]]
    x1 = geo.xcs[geo.T[e, 1]]
    x2 = geo.xcs[geo.T[e, 2]]
    x3 = geo.xcs[geo.T[e, 3]]

    Dm = wp.mat33(x0 - x3, x1 - x3, x2 - x3)    
    W[e] = wp.abs(wp.determinant(Dm)) / 6.0

@wp.kernel
def tet_kernel_sparse(geo: FEMMesh, triplets: Triplets, W: wp.array(dtype = float)):
    e, ii, jj = wp.tid()
    x = xq_tet(geo, e)
    i = geo.T[e, ii]
    j = geo.T[e, jj]
    bi, dbidx = bf_tet(geo, e, ii, x)
    bj, dbjdx = bf_tet(geo, e, jj, x)
    a = wp.mat33(0.0)
    for k in range(3):
        ek = wp.vec3(0.0)
        ek[k] = 1.0
        grad_v = wp.outer(ek, dbidx)
        # grad_v = ti.Vector.unit(3, k, ti.f32).outer_product(dbidx)
        eps = (grad_v + wp.transpose(grad_v)) / 2.0
        c = wp.trace(eps) * lam * dbjdx + 2.0 * mu * wp.transpose(eps) @ dbjdx

        a[k] = c * W[e]

    idx = e * 16 + ii * 4 + jj
    triplets.rows[idx] = i
    triplets.cols[idx] = j
    triplets.vals[idx] = a

class TetFEM:
    def __init__(self):
        if not hasattr(self, "filename"):
            self.filename = "assets/bar2.tobj"
        super().__init__()
        self.W = wp.zeros((self.n_tets), dtype = float)
        self.geo = FEMMesh()
        self.geo.n_nodes = self.n_nodes
        self.geo.n_tets = self.n_tets
        self.geo.xcs = self.xcs
        self.geo.T = self.T
        self.define_K_sparse()


    def define_K_sparse(self):
        wp.launch(compute_W, self.n_tets, inputs = [self.geo, self.W])
        self.tet_kernel_sparse()
        self.K_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        
        bsr_set_from_triplets(self.K_sparse, self.triplets.rows, self.triplets.cols, self.triplets.vals)

    def tet_kernel_sparse(self):
        self.triplets = Triplets()
        self.triplets.rows = wp.zeros((self.n_tets * 4 * 4), dtype = int)
        self.triplets.cols = wp.zeros_like(self.triplets.rows)
        self.triplets.vals = wp.zeros((self.n_tets * 4 * 4), dtype = wp.mat33)

        wp.launch(tet_kernel_sparse, (self.n_tets,4, 4), inputs = [self.geo, self.triplets, self.W])
    
    def eigs_sparse(self):
        K = self.to_scipy_bsr()
        print("start eigs")
        
        lam, Q = eigsh(K, k = 10, which = "SM", tol = 1e-4)
        return lam, Q


    def to_scipy_bsr(self, mat: BsrMatrix = None):
        if mat is None:
            mat = self.K_sparse
        ii = mat.offsets.numpy()
        jj = mat.columns.numpy()
        values = mat.values.numpy()
        shape = (mat.nrow * 3, mat.ncol * 3) 
        print(f"shape = {shape}, values = {values.shape}, ii = {ii.shape}, jj = {jj.shape}")
        bsr = bsr_matrix((values, jj, ii), shape = mat.shape, blocksize = (3 , 3))
        # return bsr.toarray()
        return bsr

class PSViewer:
    def __init__(self, Q, V0, F):
        self.Q = Q
        self.V0 = V0
        self.F = F
        self.ps_mesh = ps.register_surface_mesh("rod", V0, F)

        self.ui_deformed_mode = 0

        self.ui_magnitude = 2
    def callback(self):
        Qi = self.Q[:, self.ui_deformed_mode]

        disp = self.ui_magnitude * Qi
        disp = disp.reshape((-1, 3))

        self.V_deform = self.V0 + disp 

        self.ps_mesh.update_vertex_positions(self.V_deform)

        changed, self.ui_deformed_mode = gui.InputInt("#mode", self.ui_deformed_mode, step = 1)

        changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = 0.0, v_max = 4)

class Rod(TetFEM, TOBJLoader):
    '''
    NOTE: need to have self.filename predefined before calling super().__init__()
    '''
    def __init__(self):
        super().__init__()
        self.V = self.xcs.numpy()
        self.F = self.indices.numpy()
        self.mid = np.mean(self.V, axis = 0)
        self.V0 = self.V - self.mid


def vis_eigs():
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.init()
    # rod = Rod("assets/elephant.mesh")
    rod = Rod()
    lam, Q = None, None
    if True:
        lam, Q = rod.eigs_sparse()
        # lam, Q = rod.eigs_sparse()
    # if True:
    #     K = rod.K
    #     savemat(f"K_{model}.mat", {"K": K})
    #     quit()
    else:
        Q = np.load(f"Q_{model}.npy")

    mid, V0, F = rod.mid, rod.V0, rod.F

    viewer = PSViewer(Q, V0, F)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    vis_eigs()