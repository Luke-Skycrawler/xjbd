import warp as wp
from fem.params import *
from fem.geometry import TOBJLoader
from fem.fem import Triplets, compute_Dm
from warp.sparse import bsr_set_from_triplets, bsr_zeros, bsr_mm, bsr_transposed, BsrMatrix
from scipy.sparse import bsr_matrix, diags
from scipy.sparse.linalg import eigsh
import igl

import polyscope as ps
import polyscope.imgui as gui

@wp.kernel
def assemble_D(geo: FEMMesh, triplets: Triplets, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float)):
    e = wp.tid()
    # i0 = geo.T[e, 0]
    # i1 = geo.T[e, 1]
    # i2 = geo.T[e, 2]
    # i3 = geo.T[e, 3]

    # x0 = geo.xcs[i0]
    # x1 = geo.xcs[i1]
    # x2 = geo.xcs[i2]
    # x3 = geo.xcs[i3]

    mat34 = wp.matrix(dtype=float, shape=(3, 4))
    mat34[0,  0] = 1.0
    mat34[1,  1] = 1.0
    mat34[2,  2] = 1.0

    mat34[0, 3] = -1.0
    mat34[1, 3] = -1.0
    mat34[2, 3] = -1.0

    Di = wp.transpose(Bm[e]) @ mat34
    w = wp.sqrt(W[e] * (lam + mu * 2.0 / 3.0))
    for ii in range(3):
        for jj in range(4):
            c = geo.T[e, jj]
            r = e * 3 + ii

            triplets.rows[r * 4 + jj] = r
            triplets.cols[r * 4 + jj] = c
            triplets.vals[r * 4 + jj] = wp.identity(3, float) * Di[ii, jj] * w
    

class ADMM_PD(TOBJLoader):
    '''
    base class for ADMM-PD solvers, 
    notations in accordance to ADMM \superset ProjectiveDynamics:FastSimulationofGeneralConstitutiveModels
    '''

    def __init__(self):
        self.filename = "assets/bar2.tobj"
        super().__init__()
        self.geo = FEMMesh()
        self.geo.n_nodes = self.n_nodes
        self.geo.n_tets = self.n_tets
        self.geo.xcs = self.xcs
        self.geo.T = self.T

        self.Bm = wp.zeros((self.n_tets, ), dtype=wp.mat33)
        self.W = wp.zeros((self.n_tets,), dtype=float)


        self.define_M()
        self.compute_Dm()
        self.define_D()

    def compute_Dm(self):
        wp.launch(compute_Dm, (self.n_tets,), inputs = [self.geo, self.Bm, self.W])

    def define_D(self):

        # triplets
        self.triplets = Triplets()
        triplets_size = self.n_tets * 3 * 4
        self.triplets.rows = wp.zeros((triplets_size,), dtype=int)
        self.triplets.cols = wp.zeros_like(self.triplets.rows)
        self.triplets.vals = wp.zeros((triplets_size,), dtype=wp.mat33)

        self.D = bsr_zeros(self.n_tets * 3, self.n_nodes, wp.mat33)
        wp.launch(assemble_D, (n_tets, ), inputs=[self.geo, self.triplets, self.Bm, self.W])
        bsr_set_from_triplets(self.D, self.triplets.rows, self.triplets.cols, self.triplets.vals)
        self.A = bsr_mm(bsr_transposed(self.D, ), self.D)
        

    def add_constraints(self):
        pass
    
    def eigs_sparse(self):
        
        K = self.to_scipy_bsr(self.A)
        print("start eigs")
        lam, Q = eigsh(K, k = 10, which = "SM", tol = 1e-4)
        return lam, Q

    def to_scipy_bsr(self, mat: BsrMatrix):
        ii = mat.offsets.numpy()
        jj = mat.columns.numpy()
        values = mat.values.numpy()
        shape = (mat.nrow * 3, mat.ncol * 3) 
        nnz = mat.nnz
        print(f"shape = {shape}, values = {values.shape}, ii = {ii.shape}, jj = {jj.shape}")
        bsr = bsr_matrix((values[:nnz], jj[:nnz], ii), shape = mat.shape, blocksize = (3 , 3))
        return bsr

    def define_M(self):
        M1 = igl.massmatrix(self.xcs.numpy(), self.T.numpy(), igl.MASSMATRIX_TYPE_BARYCENTRIC)
        M1 = M1.toarray()
        print(M1.shape)
        mdiag = np.diag(M1)
        mdiag3 = np.repeat(mdiag, 3)
        self.M_sparse = diags(mdiag3)


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




def vis_eigs():
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.init()
    rod = ADMM_PD()
    lam, Q = None, None
    lam, Q = rod.eigs_sparse()
    V0 = rod.xcs.numpy()
    F = rod.indices.numpy()
    viewer = PSViewer(Q, V0, F)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    vis_eigs()