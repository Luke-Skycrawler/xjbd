import warp as wp
import numpy as np
from fem.params import *
from tet_fem import xq_tet, Rod, PSViewer
from scipy.sparse.linalg import splu, spsolve, eigsh
from warp.sparse import bsr_set_from_triplets, bsr_zeros, BsrMatrix
from scipy.sparse import diags
import polyscope as ps
import polyscope.imgui as gui
import os
import igl


constrained = False
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

    n = wp.vec3(wp.cross(x1 - x0, x2 - x0))
    n /= wp.length(n)
    # n = (x1 - x0).cross(x2 - x0).normalized()
    # if n.dot(x - x0) < 0:
    if wp.dot(n, x - x0) < 0.0:
        n = -n
    return n

@wp.func
def bf_tet(geo: FEMMesh, e: int, _i: int, x: wp.vec3) -> wp.vec3:
    bi = 0.25
    n = normal(geo, e, _i)
    x0 = geo.xcs[geo.T[e, _i]]
    k = 0.75 / (wp.dot(x0 - x, n))
    dbidx = n * k
    
    return dbidx

@wp.func
def C(geo: FEMMesh, e: int, aa: int, bb: int, cc: int, w: float) -> wp.vec3:
    '''
    C(a,b,c) = \int nabla phi_a (nabla phi_b dot nabla phi_c) dV \in R^3
    '''
    xq = xq_tet(geo, e)
    dbdx = bf_tet(geo, e, bb, xq)
    dcdx = bf_tet(geo, e, cc, xq)
    dadx = bf_tet(geo, e, aa, xq)
    # return wp.vec3(dbdx)
    return dadx * w * wp.dot(dbdx, dcdx)

@wp.struct 
class Quadplets:
    rows: wp.array(dtype = int)
    cols: wp.array(dtype = int)
    ls: wp.array(dtype = int)
    vals: wp.array2d(dtype = wp.mat33)
    # vals: wp.array(dtype = wp.mat33)

@wp.func
def add_tensor_1(a:int, b: int, c: int, vec: wp.vec3, idx: int, quadplets: Quadplets):
    for i in range(3):
        ei = wp.vec3(0.0)
        ei[i] = 1.0
        quadplets.vals[idx, i] = wp.outer(vec, ei)
    quadplets.rows[idx] = a
    quadplets.cols[idx] = b
    quadplets.ls[idx] = c

@wp.func
def add_tensor_2(a: int, b: int, c: int, vec: wp.vec3, idx: int, quadplets: Quadplets):
    for i in range(3):
        ei = wp.vec3(0.0)
        ei[i] = 1.0
        quadplets.vals[idx, i] = wp.outer(ei, vec)
    quadplets.rows[idx] = a
    quadplets.cols[idx] = b
    quadplets.ls[idx] = c

@wp.func
def add_tensor_3(a: int, b: int, c: int, vec: wp.vec3, idx: int, quadplets: Quadplets):
    for i in range(3):
        quadplets.vals[idx, i] = wp.diag(wp.vec3(1.0)) * vec[i]
    quadplets.rows[idx] = a
    quadplets.cols[idx] = b
    quadplets.ls[idx] = c

@wp.func
def ix(e:int, a:int, b:int, c:int):
    return c + 4 * (b + 4 * (a + 4 * e))

@wp.kernel
def precompute_C(geo: FEMMesh, vec: wp.array(dtype= wp.vec3), W: wp.array(dtype = float)):
    e, aa, bb, cc = wp.tid()
    dw = W[e]
    D = C(geo, e, aa, bb, cc, dw)
    idx = ix(e, aa, bb, cc)
    vec[idx] = D


@wp.kernel
def compute_H(geo: FEMMesh, quadplets: Quadplets, W: wp.array(dtype =float), precomputed_C: wp.array(dtype = wp.vec3)):
    '''
    H_e^{a, b, c} = C1(a, b, c) + C2(b, c, a) + C2(c, b, a) \otimes I3 + I3 \otimes (C1(c, b, a) + C2(a, c, b) + C2(b, c, a)) + \Sigma_{1 ... 3} (e_i \otimes (C1(b, c, a) + C2(a, b, c) + C2(c, b, a) \otimes ei))

    H_e \in R^{3x3x3}
    C1(a, b, c) = lambda * C(a, b, c)
    C2(a, b, c) = mu * C(a, b, c)
    '''
    e, aa, bb, cc = wp.tid()
    # dw = W[e]
    # D = 0.5 * lam * C(geo, e, cc, aa, bb, dw) + mu * C(geo, e, aa, bb, cc, dw)
    D = 0.5 * lam * precomputed_C[ix(e, cc, aa, bb)] + mu * precomputed_C[ix(e, aa, bb, cc)]
    a = geo.T[e, aa]
    b = geo.T[e, bb]
    c = geo.T[e, cc]

    idx = ix(e, aa, bb, cc)
    add_tensor_1(c, a, b, D, idx * 4 + 0, quadplets)
    add_tensor_1(c, b, a, D, idx * 4 + 1, quadplets)
    
    CC = lam * precomputed_C[ix(e, aa, bb, cc)] + mu * (precomputed_C[ix(e, cc, aa, bb)] + precomputed_C[ix(e, bb, aa, cc)])

    add_tensor_2(c, a, b, CC, idx * 4 + 2, quadplets)
    add_tensor_3(c, b, a, CC, idx * 4 + 3, quadplets)

class MDRod(Rod):
    def __init__(self):
        super().__init__()
        self.define_M()
        self.md_precompute_once()

    def compute_H(self):
        self.quadplets = Quadplets()
        n_q = self.n_tets * 4 * 4 * 4
        self.quadplets.rows = wp.zeros((n_q * 4), dtype = int)
        self.quadplets.cols = wp.zeros_like(self.quadplets.rows)
        self.quadplets.ls = wp.zeros_like(self.quadplets.rows)
        self.quadplets.vals = wp.zeros((n_q * 4, 3), dtype = wp.mat33)
        c = wp.zeros((n_q), dtype = wp.vec3)
        wp.launch(precompute_C, (self.n_tets,4, 4, 4), inputs = [self.geo, c, self.W])
        wp.launch(compute_H, (self.n_tets, 4, 4, 4), inputs = [self.geo, self.quadplets, self.W, c])
    
    def md_precompute_once(self):
        self.compute_H()
        self.sort_quadplets()
        self.K_bar = self.to_scipy_bsr().tocsr()

    def modal_derivatives(self, Psi_i, Psi_j, lam_i = 0.0):
        '''
        K Phi_ij = - (H : Psi_j) Psi_i
        '''
        if constrained: 
            K = self.K_bar[75:, 75:]
            Psi_i = Psi_i[75:]
            # NOTE: don't do Psi_j[75:]. it will mess up the 3d tensor loop
        else: 
            '''
            K_bar = K - lam_i M + Psi_i Psi_i ^ T
            '''
            K = self.K_bar - lam_i * self.M_sparse + np.outer(Psi_i, Psi_i)
        
        # K_solve = splu(K, )
        H_ddot_Psi = self.compute_H_ddot_Psi(Psi_j)
        # H_ddot_Psi = self.to_scipy_bsr(H_ddot_Psi)
        b = H_ddot_Psi @ Psi_i
        if not constrained: 
            term = self.M_sparse @ np.outer(Psi_i, Psi_i) - np.identity(self.n_nodes * 3)
            b = term @ b
        Phi_ij = spsolve(K, -b,)

        phiij = np.zeros((self.n_nodes * 3,))
        if constrained:
            phiij[75:] = Phi_ij
        else: 
            phiij = Phi_ij

        print(f"b norm = {np.linalg.norm(b)}, Phi ij @ K - b = {np.linalg.norm(K @ Phi_ij + b)} ")
        return phiij

    def sort_quadplets(self):
        lnp = self.quadplets.ls.numpy()
        indices = np.argsort(lnp)

        rnp = self.quadplets.rows.numpy()
        cnp = self.quadplets.cols.numpy()
        
        rnp = rnp[indices]
        cnp = cnp[indices]
        lnp = lnp[indices]
        self.quadplets.rows.assign(rnp)
        self.quadplets.cols.assign(cnp)
        self.quadplets.ls.assign(lnp)

        vnp = self.quadplets.vals.numpy()
        
        self.quadplets.vals.assign(vnp[indices])

        unique_vals, start_indices = np.unique(lnp, return_index=True)
        # buckets = np.split(lnp, start_indices[1:])  # skip first index (0)
        self.start_indices = start_indices

    def compute_H_ddot_Psi(self, Psi_j):
        '''
        H:a = \Sigma H_{i, j, l} a_l
        '''
        H = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33, )
        r_bucket = np.split(self.quadplets.rows.numpy(), self.start_indices[1:])
        c_bucket = np.split(self.quadplets.cols.numpy(), self.start_indices[1:])
        # l_bucket = np.split(self.quadplets.ls.numpy(), self.start_indices[1:])
        v_bucket = np.split(self.quadplets.vals.numpy(), self.start_indices[1:])
        
        # for l in range(25, self.n_nodes):
        for l in range(self.n_nodes):
            rows = wp.array(r_bucket[l], int)
            cols = wp.array(c_bucket[l], int)
            

            for k in range(3):
                Hl = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
                values = wp.array(v_bucket[l][:, k], dtype = wp.mat33)
                bsr_set_from_triplets(Hl, rows, cols, values)
                al = Psi_j[l*3 + k]
                H += Hl * al
        H = self.to_scipy_bsr(H).tocsr()
        if constrained:
            H = H[75:, 75:]
        # H = self.to_scipy_bsr(H)
        return H

    def define_M(self):
        V = self.xcs.numpy()
        T = self.T.numpy()
        # self.M is a vector composed of diagonal elements 
        self.Mnp = igl.massmatrix(V, T, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal() * rho
        M_diag = np.repeat(self.Mnp, 3)
        self.M_sparse = diags(M_diag)

    def get_constraint(self):
        v_rst = self.xcs.numpy()
        x_rst = v_rst[:, 0]
        # w = np.zeros((self.n_nodes * 3))
        # w = np.ones_like(v_rst, dtype = int)
        w = np.ones_like(v_rst, dtype = int)
        w[x_rst < -0.5 + 1e-3] = 0
        
        w = w.reshape((-1))
        # indices = np.where(w == 1)[0]
        # J = np.identity(self.n_nodes * 3)[w]
        return w

    def eigs_sparse(self):
        
        K = self.to_scipy_bsr().tocsr()
        tilde_M = self.M_sparse

        # w = self.get_constraint()
        # na1 = null_space(w)
        # na1 = w

        # indices = w 
        if constrained:
            tilde_K = K[75:, 75:]
            mm = np.repeat(self.Mnp, 3)
            mm = mm[75:]
            tilde_M = diags(mm)
        else:
            tilde_K = K
        
        # tilde_M = M[w, w]
        # tilde_K = na1.T @ K @ na1
        # tilde_M = na1.T @ M @ na1

        lam, Ql = eigsh(tilde_K, k = 10, M = tilde_M, which = "SM")

        Q = np.zeros((self.n_nodes * 3, 10))
        if constrained:
            Q[75:] = Ql
        else:
            Q = Ql
        # Ql = na1 @ Ql
        # Q = Ql[: self.n_nodes * 3]

        return lam, Q
        
def vis_eigs():
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.init()
    # rod = Rod("assets/elephant.mesh")
    rod = MDRod()
    lam, Q = rod.eigs_sparse()
    i, j = 6, 8
    assert i <= j, "i <= j"
    Psi_i = Q[:, i]
    Psi_j = Q[:, j]
    # if not os.path.exists(f"Phi_{i}{j}.npy"):
    if True:
        phiij = rod.modal_derivatives(Psi_i, Psi_j, lam[i])
        
        np.save(f"Phi_{i}{j}.npy", phiij)
    else: 
        phiij = np.load(f"Phi_{i}{j}.npy")

    Q[:, 0: 1] = phiij.reshape((-1, 1))
    mid, V0, F = rod.mid, rod.V0, rod.F

    viewer = PSViewer(Q, V0, F)
    # np.save(f"Q_{model}.npy", Q)
    ps.set_user_callback(viewer.callback)
    ps.show()

def compute_md():
    rod = MDRod()
    lam, Q = rod.eigs_sparse()
    # np.save(f"lambda_{model}.npy", lam)
    # np.save(f"Q_{model}.npy", Q)
    # quit()
    for i in range(6, 10):
        for j in range(i, 10):
            
            Psi_i = Q[:, i]
            Psi_j = Q[:, j]
            if True:
                phiij = rod.modal_derivatives(Psi_i, Psi_j, lam[i])
                np.save(f"Phi_{i}{j}.npy", phiij)
    


class MDViewer:
    def __init__(self, Q, V0, F):
        self.Q = Q
        # self.Phi = Phi
        self.V0 = V0
        self.F = F
        self.ps_mesh = ps.register_surface_mesh("rod", V0, F)

        self.ui_deformed_mode = 6
        self.ui_mode_j = 5

        self.ui_magnitude = 2
    def callback(self):
        '''
        if mode j < 6, display linear modes Psi i
        if mode j >= 6, display modal derivatives Phi ij
        '''


        changed, self.ui_deformed_mode = gui.InputInt("#mode", self.ui_deformed_mode, step = 1)
        self.ui_deformed_mode = min(self.ui_deformed_mode, 9)
        changed, self.ui_mode_j = gui.InputInt("#mode j", self.ui_mode_j, step = 1)
        self.ui_mode_j = min(self.ui_mode_j, 9)

        changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = 0.0, v_max = 4)
        
        Qi = self.Q[:, self.ui_deformed_mode]
        if self.ui_mode_j < 6:
            disp = Qi
        else:
            i = min(self.ui_deformed_mode, self.ui_mode_j)
            j = max(self.ui_deformed_mode, self.ui_mode_j)
            Phi_ij = np.load(f"Phi_{i}{j}.npy")
            disp = Phi_ij

        disp = self.ui_magnitude * disp
        disp = disp.reshape((-1, 3))

        self.V_deform = self.V0 + disp 

        self.ps_mesh.update_vertex_positions(self.V_deform)

def view_md():
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.init()
    rod = MDRod()
    mid, V0, F = rod.mid, rod.V0, rod.F
    lam, Q = rod.eigs_sparse()

    viewer = MDViewer(Q, V0, F)
    # np.save(f"Q_{model}.npy", Q)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__" :
    # vis_eigs()
    # compute_md()
    view_md()