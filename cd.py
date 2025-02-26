import igl 
import numpy as np

import polyscope as ps
import polyscope.imgui as gui
import warp as wp
from stretch import RodBC, NewtonState, Triplets, set_M_diag
from fem.fem import tet_kernel_sparse
from warp.sparse import *
from warp.optim.linear import bicgstab
import os
gravity = wp.vec3(0, -10.0, 0)
h = 1e-2
rho = 1e3
def load():
    
    # V, _, _, F, _, _ = igl.read_obj("assets/elephant.obj")
    # default obj
    C, BE, _, _, _, _ = igl.read_tgf("assets/elephant_handles.tgf")

    # default weight and handles

    # C, BE, _, _, _, _ = igl.read_tgf("assets/elephant.tgf")
    # W = igl.read_dmat("assets/elephant-weights.dmat")
    
    T = igl.read_dmat("assets/elephant-anim.dmat")
    verts, tets, _= igl.read_mesh("assets/elephant.mesh")

    # deprecated code to find closest vertices to control points
    # updated handles are stored in `elephant_handles.tgf`

    # cid = []
    # for c in C:
    #     distances = np.linalg.norm(verts - c.reshape(1, 3), axis = 1)
    #     idx = np.argmin(distances)
    #     cid.append(idx)
    # cid = np.array(cid)
    # print(cid)
    # print(verts[cid])

    boundary = igl.boundary_facets(tets)
    V = verts
    F = boundary

    # bone-based weights
    if os.path.exists("data/comp_W.npy"):
        W = np.load("data/comp_W.npy")
    else:
        ok, b, bc = igl.boundary_conditions(verts, tets, C, np.zeros(0, dtype = int), BE, np.zeros((0, 0), dtype = int), np.zeros((0, 0), dtype = int))
        print(ok)
        bbw = igl.BBW(2, 8)
        W = bbw.solve(V, tets, b, bc)
        W_sum = np.sum(W, axis = 1)
        W = W / W_sum.reshape(-1, 1)
        np.save("data/comp_W.npy", W)

    return V, F, W, C, BE, T

@wp.kernel
def compute_rhs(states: NewtonState, h: float, M: wp.array(dtype = float), b: wp.array(dtype = wp.vec3), u_rig: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    # b[i] += -(M[i] / h) * ((u_rig[i] - states.x0[i]) / h - states.xdot[i]) + gravity * M[i]
    b[i] += -(M[i] / h) * ((u_rig[i] - states.x0[i]) / h - states.xdot[i]) + gravity * M[i]
    b[i] *= (h * h)

@wp.kernel
def fill_J_triplets(xcs: wp.array(dtype = wp.vec3), W: wp.array2d(dtype = float), M: wp.array(dtype = float), triplets: Triplets, n_nodes: int):
    i, j, k = wp.tid()

    xx = W.shape[0]
    yy = W.shape[1]

    idx = (i * yy + j) * 4 + k
    triplets.rows[idx] = i
    triplets.cols[idx] = n_nodes + j * 4 + k
    xk = float(0.0)
    if k == 3:
        xk = 1.0
    else: 
        xk = xcs[i][k]
    triplets.vals[idx] = wp.diag(wp.vec3(W[i, j])) * xk * M[i]


@wp.kernel
def x_gets_u_comp_plus_u_rig(x: wp.array(dtype = wp.vec3), u_rig: wp.array(dtype = wp.vec3), u_comp: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    x[i] = u_rig[i] + u_comp[i]


class CompRodBC (RodBC):
    '''
    notation implemented as complementaray dynamics algorithm 2
    '''
    def __init__(self, M, W, h):
        super().__init__(h, "assets/elephant.mesh")
        self.MM = M
        self.n_handles = W.shape[1]
        self.WW = wp.zeros(W.shape, dtype = float)
        self.WW.assign(W)
        self.u_rig = wp.zeros_like(self.states.x)
        self.u_comp = wp.zeros_like(self.states.x)
        self.u_rig.assign(self.states.x)
        # self.sys_dim = self.n_nodes + self.n_handles * 4
        self.sys_dim = self.n_nodes
        self.dxdlam = wp.zeros((self.sys_dim,), dtype = wp.vec3)
        # can compute J at first as it only depends on weights & rest positions
        self.define_M_ext()
        self.compute_CT()


    def set_bc_fixed(self):
        pass

    def step(self):
        newton_iter = True
        n_iter = 0
        max_iter = 5
        # while n_iter < max_iter:
        while newton_iter:
            self.compute_A()
            self.compute_rhs()

            self.solve()
            # wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, 1.0])
            
            
            # line search stuff, not converged yet
            alpha = self.line_search()
            if alpha == 0.0:
                break

            dxnp = self.states.dx.numpy()
            norm_dx = np.linalg.norm(dxnp)
            newton_iter = norm_dx > 1e-3 and n_iter < max_iter
            print(f"norm = {np.linalg.norm(dxnp)}, {n_iter}")
            n_iter += 1
        self.update_x0_xdot()

    # def step(self, Vf):
    #     self.u_rig.assign(Vf)
    #     wp.launch(x_gets_u_comp_plus_u_rig, (self.n_nodes, ), inputs = [self.states.x, self.u_rig, self.u_comp])

    #     while True:
    #         self.compute_K()
    #         self.compute_rhs()
    #         self.compute_Q()
    #         self.assemble_sys()
    #         self.solve()
    #         # FIXME: no line search for now 
    #         break

    #     wp.launch(x_gets_u_comp_plus_u_rig, (self.n_nodes, ), inputs = [self.states.x, self.u_rig, self.u_comp])
    #     self.update_x0_xdot()

    # def solve(self):
    #     self.dxdlam.zero_()
    #     # wp.copy(self.dxdlam, self.u_comp)
    #     # self.sys_matrix = self.K_sparse
    #     bicgstab(self.sys_matrix, self.b, self.dxdlam, 1e-6, maxiter = 100)
    #     print("norm u_comp = ", np.linalg.norm(self.dxdlam.numpy()))
    #     wp.copy(self.u_comp, self.dxdlam, count = self.u_comp.shape[0])
        


    # def compute_K(self):
    #     '''
    #     partial Psi / partial u_c = -self.b
    #     partial^2 Psi / partial u_c^2 = self.K_sparse
    #     '''

    #     self.triplets.vals.zero_()
    #     self.b.zero_()
    #     wp.launch(x_gets_u_comp_plus_u_rig, (self.n_nodes, ), inputs = [self.states.x, self.u_rig, self.u_comp])
    #     wp.launch(tet_kernel_sparse, (self.n_tets * 4 * 4,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.triplets, self.b])
        
    #     # set K from triplets
    #     self.K_sparse = bsr_zeros(self.sys_dim, self.sys_dim, wp.mat33)
    #     bsr_set_zero(self.K_sparse)
    #     bsr_set_from_triplets(self.K_sparse, self.triplets.rows, self.triplets.cols, self.triplets.vals)

    def define_M_ext(self):
        '''
        compute once in the constructor 

        almost the same as in stretch.py EXCEPT the M_sparse dimension is `sys_dim = n_nodes + n_handles * 4`
        '''

        self.M_sparse = bsr_zeros(self.sys_dim, self.sys_dim, wp.mat33)
        M_diag = wp.zeros((self.n_nodes,), dtype = wp.mat33)
        wp.launch(set_M_diag, (self.n_nodes,), inputs = [self.M, M_diag])
        bsr_set_diag(self.M_sparse, M_diag, self.sys_dim, self.sys_dim)

    def compute_Q(self):
        '''
        Q = K + M / h^2
        paper has a typo here in algorithm 2 (Q <- K + h ^ 2 M) 
        '''
        h = self.h
        bsr_axpy(self.M_sparse, self.K_sparse, 1.0 / (h * h))
        self.Q = self.K_sparse

    def compute_A(self):
        self.compute_K()

        h = self.h
        # bsr_axpy(self.M_sparse, self.K_sparse, 1.0 / (h * h))
        bsr_axpy(self.M_sparse, self.K_sparse, 1.0, (h * h))
        self.A = self.K_sparse

    def compute_rhs(self):
        '''
        simplified linear elesticity version, -K u_rig - M/h ((u_rig - u0) / h - udot) + f


        l = -g + M / h (( u^r - u_0) / h - u_dot) + f
        '''
        # self.b = wp.zeros(self.sys_dim, dtype = wp.vec3)
        # u_r = wp.zeros_like(self.b)
        # wp.copy(u_r, self.u_rig)
        # bsr_mv(self.K_sparse, u_r, self.b, -1.0)

        wp.launch(compute_rhs, (self.n_nodes,), inputs = [self.states, self.h, self.M, self.b, self.u_rig])
    
    def compute_CT(self):
        '''
        compute once in the constructor

        C^T = M J
        put the matrix in upper right corner
        '''
        self.J_triplets = Triplets()
        self.J_triplets.rows = wp.zeros((self.n_handles * self.n_nodes * 4,), dtype = int)
        self.J_triplets.cols = wp.zeros_like(self.J_triplets.rows)  
        self.J_triplets.vals = wp.zeros((self.n_handles * self.n_nodes * 4,), dtype = wp.mat33)
        # W = wp.zeros(self.W.shape, dtype = float)
        # W.assign(self.W)
        # M = wp.zeros((self.n_nodes, ), dtype = float)
        # wp.copy(M, self.M)
        # print(f"M shape = {self.M.shape}")
        wp.launch(fill_J_triplets, (self.WW.shape[0], self.WW.shape[1], 4), inputs = [self.xcs, self.WW, self.M, self.J_triplets, self.n_nodes])

        # asssemble [0, C^T; C, 0]
        self.C = bsr_zeros(self.sys_dim, self.sys_dim, wp.mat33)        
        bsr_set_from_triplets(self.C, self.J_triplets.rows, self.J_triplets.cols, self.J_triplets.vals)
        bsr_axpy(bsr_transposed(self.C), self.C, 1.0, 1.0)

    def assemble_sys(self):
        bsr_axpy(self.C, self.Q, 1.0, 1.0)
        self.sys_matrix = self.Q
    
class PSViewer:
    def __init__(self, V, F, W, T, C, BE):
        self.V = V
        self.F = F
        self.W = W
        self.T = T
        self.C = C
        self.BE = BE

        self.M = igl.lbs_matrix(V, W)

        self.ps_mesh = ps.register_surface_mesh("mesh", self.V, self.F)
        ps.set_user_callback(self.callback)

        self.frame = 0
        self.n_frames = T.shape[1]
        self.skeleton = ps.register_curve_network("skeleton", C, BE)

        self.ui_rest = False

        self.sim = CompRodBC(self.M, W, h)

    def callback(self):
        changed, self.ui_rest = gui.Checkbox("Rest", self.ui_rest)
        Tf = self.T[:, self.frame].reshape(3, -1).T
        Vf = self.M @ Tf

        # self.sim.step(Vf)
        self.sim.step()
        if self.ui_rest:
            self.ps_mesh.update_vertex_positions(self.V)
        else :
            Vf = self.sim.states.x.numpy()
            self.ps_mesh.update_vertex_positions(Vf)
        
        print(f"frame = {self.frame}")
        self.frame += 1
        self.frame = self.frame % self.n_frames
        

if __name__ == "__main__":
    ps.init()
    np.printoptions(precision = 3)
    V, F, W, C, BE, T = load()
    viewer = PSViewer(V, F, W, T, C, BE)
    ps.show()
        