import warp as wp
import numpy as np
from fem.params import *
from fem.geometry import TOBJLoader
from fem.fem import Triplets, compute_Dm
from warp.sparse import bsr_set_from_triplets, bsr_zeros, bsr_mm, bsr_transposed, BsrMatrix
from scipy.sparse import bsr_matrix, diags
from scipy.sparse.linalg import eigsh, splu
import igl
from stretch import PSViewer
import polyscope as ps
import polyscope.imgui as gui

'''
reference:
[1] ADMM \superset Projective Dynamics: Fast Simulation of General Constitutive Models
[2] Quasi-Newton Methods for Real-time Simulation of Hyperelastic Materials

'''
stiffness = 1e7

@wp.struct 
class ADMMState:
    x: wp.array(dtype = wp.vec3)
    x0: wp.array(dtype = wp.vec3)
    xdot: wp.array(dtype = wp.vec3)
    dx: wp.array(dtype = wp.vec3)

@wp.kernel
def assemble_D(geo: FEMMesh, triplets: Triplets, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float)):
    '''
    selector matrix for ADMM/PD global stage

    Di_{3 \times 4} = Bm ^ T @ (I_3, (-1, -1, -1) ^ T)
    where Bm = Dm ^ -1 

    Di @ (x0, x1, x2, x3)_{3 \times 4}^T = F_e^T 
    '''
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
    # w = 1.0
    for ii in range(3):
        for jj in range(4):
            c = geo.T[e, jj]
            r = e * 3 + ii

            triplets.rows[r * 4 + jj] = r
            triplets.cols[r * 4 + jj] = c
            triplets.vals[r * 4 + jj] = wp.identity(3, float) * Di[ii, jj] * w
    
@wp.kernel
def pin_constraints(triplets: Triplets, indices: wp.array(dtype = int), n_tets: int):
    i = wp.tid()
    c = indices[i]
    offset = n_tets * 4 * 3
    idx = i + offset
    r = n_tets * 3 + i
    triplets.rows[idx] = r
    triplets.cols[idx] = c
    triplets.vals[idx] = wp.identity(3, float) * wp.sqrt(stiffness)
    
@wp.func
def inertia_y(states: ADMMState, i: int, dt: float) -> wp.vec3:
    x = states.x0[i]
    xdot = states.xdot[i]
    y = x + dt * xdot

    return y

@wp.kernel
def compute_pi(geo: FEMMesh, states: ADMMState, Bm: wp.array(dtype = wp.mat33), p: wp.array(dtype = wp.mat33), Fi: wp.array(dtype = wp.mat33)):
    e = wp.tid()

    t0 = states.x[geo.T[e, 0]]
    t1 = states.x[geo.T[e, 1]]
    t2 = states.x[geo.T[e, 2]]
    t3 = states.x[geo.T[e, 3]]
    
    Ds = wp.mat33(t0 - t3, t1 - t3, t2 - t3)
    
    F = Ds @ Bm[e]
    U = wp.mat33(0.0)
    sigma = wp.vec3(0.0)
    V = wp.mat33(0.0)
    wp.svd3(F, U, sigma, V)
    # handle inversion 
    mid = wp.identity(3, float)
    if wp.determinant(F) < 0.0:
        mid[2, 2] = -1.0

    pp = U @ mid @  wp.transpose(V)
    # must transpose
    p[e] = wp.transpose(pp)
    Fi[e] = F

@wp.kernel
def update_x0_xdot(state: ADMMState, h: float):
    i = wp.tid()
    state.xdot[i] = (state.x[i] - state.x0[i]) / h
    state.x0[i] = state.x[i]


@wp.kernel
def add_dx(state: ADMMState, alpha :float):
    i = wp.tid()
    state.x[i] -= state.dx[i] * alpha


class ADMM_PD(TOBJLoader):
    '''
    base class for ADMM-PD solvers, 
    notations in accordance to [1]
    '''

    def __init__(self, filename = "assets/bar2.tobj", h = 1e-2):
        self.filename = filename
        super().__init__()
        self.geo = FEMMesh()
        self.geo.n_nodes = self.n_nodes
        self.geo.n_tets = self.n_tets
        self.geo.xcs = self.xcs
        self.geo.T = self.T
        self.F = self.indices.numpy()

        self.Bm = wp.zeros((self.n_tets, ), dtype=wp.mat33)
        self.W = wp.zeros((self.n_tets,), dtype=float)


        self.define_M()
        self.compute_Dm()
        self.define_D()

        # initialze functions regarding the dynamics last
        self.h = h
        self.define_states()
        self.prefactor()
    
        self.p = wp.zeros(dtype = wp.mat33, shape=(self.n_tets,))
        x0 = self.xcs.numpy()
        self.p_pinned = x0[self.constraints.numpy()].reshape(-1)
        self.def_grad = wp.zeros_like(self.p)

    def prefactor(self):
        self.A = self.to_scipy_bsr(self.L) + self.M_sparse / (self.h * self.h)
        self.solver = splu(self.A)
        self.b = np.zeros((self.n_nodes * 3,), dtype=float)
        
    def define_states(self):
        self.states = ADMMState()
        self.states.x = wp.zeros_like(self.xcs)
        self.states.x0 = wp.zeros_like(self.xcs)
        self.states.xdot = wp.zeros_like(self.xcs)
        self.states.dx = wp.zeros_like(self.xcs)
        self.reset()
    
    def reset(self):
        self.frame = 0
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)
        self.states.xdot.zero_()

    def compute_Dm(self):
        wp.launch(compute_Dm, (self.n_tets,), inputs = [self.geo, self.Bm, self.W])

    def get_constraint_size(self):
        '''
        mark vertices with x coord < -0.5 + eps for constraint set
        '''
        # return 0
        eps = 1e-3
        
        v_rst = self.xcs.numpy()

        x_rst = v_rst[:, 0]
        select = np.abs(x_rst + 0.5) < eps
        # select = np.abs(x_rst) < -eps
        n_constraints = np.sum(select)
        self.constraints = wp.zeros((n_constraints,), dtype = int)
        self.constraints.assign(np.arange(n_nodes)[select])
        self.n_pinned_constraints = n_constraints
        return n_constraints
        
    def define_D(self):
        # triplets
        n_constraints = self.get_constraint_size()
        self.triplets = Triplets()
        triplets_size = self.n_tets * 3 * 4 + n_constraints
        self.triplets.rows = wp.zeros((triplets_size,), dtype=int)
        self.triplets.cols = wp.zeros_like(self.triplets.rows)
        self.triplets.vals = wp.zeros((triplets_size,), dtype=wp.mat33)

        self.add_constraints()


        self.D = bsr_zeros(self.n_tets * 3 + n_constraints, self.n_nodes, wp.mat33)

        W = self.W.numpy()
        
        wp.launch(assemble_D, (n_tets, ), inputs=[self.geo, self.triplets, self.Bm, self.W])
        bsr_set_from_triplets(self.D, self.triplets.rows, self.triplets.cols, self.triplets.vals)
        # J = D^T S, which happens to be identity
        S_diag = np.repeat(np.sqrt(W * (lam + mu * 2.0 / 3.0)), 9)
        S_diag = np.concatenate([S_diag, np.sqrt(stiffness) * np.ones(self.n_pinned_constraints * 3)])
        self.S_sparse = diags(S_diag)

        self.J = self.to_scipy_bsr(self.D).T @ self.S_sparse
        self.L = bsr_mm(bsr_transposed(self.D, ), self.D)
        self.L_scipy = self.to_scipy_bsr(self.L)

    def add_constraints(self):
        # return
        offset = self.n_tets * 4 * 3
        wp.launch(pin_constraints, (self.n_pinned_constraints, ),inputs = [self.triplets, self.constraints, n_tets])
    

    def eigs_sparse(self):
        
        K = self.to_scipy_bsr(self.L)
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
        V = self.xcs.numpy()
        T = self.T.numpy()
        # self.M is a vector composed of diagonal elements 
        self.Mnp = igl.massmatrix(V, T, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal() * rho
        M_diag = np.repeat(self.Mnp, 3)
        self.M_sparse = diags(M_diag)


    '''
    dynamic functions 
    '''
    def local_project(self):
        '''
        compute pi for all constraints 
        '''

        n_voxel_constraints = self.n_tets
        
        # p = wp.zeros((n_voxel_constraints,), dtype=wp.mat33)
        # F = wp.zeros_like(p)
        wp.launch(compute_pi, (n_voxel_constraints,), inputs = [self.geo, self.states, self.Bm, self.p, self.def_grad])
        # pnp = self.p.numpy()
        # print("pnp sample : ", pnp[0])
        
    def compute_y(self):
        y = self.states.x0.numpy() + self.h * self.states.xdot.numpy()
        # gravity = np.zeros_like(y)
        # gravity[:, 1] = -g 
        y += gravity_np * (self.h * self.h)
        return y.reshape(-1)

    def compute_gradient(self, y):
        '''
        eq.8 in [2]
        \nabla g(x) = 1 / h^2 M(x - y) + D^T D x - D^T S p(x)
        '''
        x = self.states.x.numpy().reshape(-1)
        p = np.concatenate([self.p.numpy().reshape(-1), self.p_pinned])
        term = 1.0 / (self.h * self.h)
        self.b[:] = term * self.M_sparse @ (x - y) + self.L_scipy @ x - self.J @ p
        # self.b[:] = self.J @ p + self.M_sparse @ y * term
        # self.b[:] = self.M_sparse @ y * term + self.L_scipy @ y
        
    def update_x(self):
        dx = self.solver.solve(self.b)
        self.states.dx.assign(dx)
        # self.states.x.assign(dx)
        wp.launch(add_dx, (self.n_nodes,), inputs = [self.states, 1.0])

    def step(self):
        n_iters = 10
        y = self.compute_y()
        
        for iter in range(n_iters):
            self.local_project()
            self.compute_gradient(y)
            self.update_x()
            iter += 1
            
        self.update_x0_xdot()
        self.frame += 1

    def update_x0_xdot(self):
        wp.launch(update_x0_xdot, dim = (self.n_nodes,), inputs = [self.states, self.h])
        
# class PSViewer:
#     def __init__(self, Q, V0, F):
#         self.Q = Q
#         self.V0 = V0
#         self.F = F
#         self.ps_mesh = ps.register_surface_mesh("rod", V0, F)

#         self.ui_deformed_mode = 0

#         self.ui_magnitude = 2
#     def callback(self):
#         Qi = self.Q[:, self.ui_deformed_mode]

#         disp = self.ui_magnitude * Qi 
#         disp = disp.reshape((-1, 3))

#         self.V_deform = self.V0 + disp 

#         self.ps_mesh.update_vertex_positions(self.V_deform)

#         changed, self.ui_deformed_mode = gui.InputInt("#mode", self.ui_deformed_mode, step = 1)

#         changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = 0.0, v_max = 4)




def vis_eigs():
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.config.max_unroll = 0
    wp.init()
    rod = ADMM_PD()
    lam, Q = rod.eigs_sparse()
    V0 = rod.xcs.numpy()
    F = rod.indices.numpy()
    viewer = PSViewer(Q, V0, F)
    ps.set_user_callback(viewer.callback)
    ps.show()

def test_pd():
    ps.init() 
    ps.set_ground_plane_mode("none")
    wp.config.max_unroll = 0
    wp.init()
    rod = ADMM_PD()
    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    # vis_eigs()
    test_pd()