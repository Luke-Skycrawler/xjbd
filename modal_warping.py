import warp as wp 
from fem.params import FEMMesh
from fem.interface import Rod
from fem.params import *
import polyscope as ps
import polyscope.imgui as gui
import numpy as np
import warp.sparse
from scipy.io import savemat, loadmat
from scipy.sparse.linalg import eigsh
import os
import igl
from scipy.sparse import diags

model = "bar2"
save_Psi_only = False
@wp.struct
class ModalWarpingData: 
    W: wp.array(dtype = wp.vec3)
    cnt: wp.array(dtype = int)
    psi: wp.array(dtype = wp.vec3)
    u: wp.array(dtype = wp.vec3)

@wp.struct
class Triplet:
    row: wp.array(dtype = int)
    col: wp.array(dtype = int)
    values: wp.array(dtype = wp.mat33)

@wp.func
def normal(geo: FEMMesh, e: int, _i: int):
    '''normal of the face opposite to vetex _i (pointing _i)'''
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
    n = wp.normalize(wp.cross(x1 - x0, x2 - x0))
    if wp.dot(n, x - x0) < 0:
        n = -n
    return n
    
@wp.func
def bary_center(geo: FEMMesh, e: int) -> wp.vec3:
    x0 = geo.xcs[geo.T[e, 0]]
    x1 = geo.xcs[geo.T[e, 1]]
    x2 = geo.xcs[geo.T[e, 2]]
    x3 = geo.xcs[geo.T[e, 3]]
    return 0.25 * (x0 + x1 + x2 + x3)

@wp.func
def nabla_bf_tet(geo: FEMMesh, e: int, _i: int):
    '''gradient of bilinear tent function'''
    n = normal(geo, e, _i)
    x = bary_center(geo, e)
    x0 = geo.xcs[geo.T[e, _i]]
    k = 0.75 / (wp.dot(x0 - x, n))
    dbidx = n * k
    return dbidx

@wp.func 
def shape_func(geo: FEMMesh, e: int):
    x0 = geo.xcs[geo.T[e, 0]]   
    x1 = geo.xcs[geo.T[e, 1]]
    x2 = geo.xcs[geo.T[e, 2]]
    x3 = geo.xcs[geo.T[e, 3]]

    Dm = wp.mat33(x0 - x3, x1 - x3, x2 - x3)
    inv_Dm = wp.inverse(Dm)
    return inv_Dm
    
@wp.func 
def W_e(geo: FEMMesh, e:int, modal_w: ModalWarpingData): 

    # same as nabla_bf_tet, use either 
    # inv_Dm = shape_func(geo, e)
    # grad_t0 = inv_Dm[0]
    # grad_t1 = inv_Dm[1]
    # grad_t2 = inv_Dm[2]
    # grad_t3 = -inv_Dm[0] - inv_Dm[1] - inv_Dm[2]
    
    grad_t0 = nabla_bf_tet(geo, e, 0)
    grad_t1 = nabla_bf_tet(geo, e, 1)
    grad_t2 = nabla_bf_tet(geo, e, 2)
    grad_t3 = nabla_bf_tet(geo, e, 3)


    i0 = geo.T[e, 0]
    i1 = geo.T[e, 1]
    i2 = geo.T[e, 2]
    i3 = geo.T[e, 3]


    wp.atomic_add(modal_w.W, i0, grad_t0 * 0.5)
    wp.atomic_add(modal_w.W, i1, grad_t1 * 0.5)
    wp.atomic_add(modal_w.W, i2, grad_t2 * 0.5)
    wp.atomic_add(modal_w.W, i3, grad_t3 * 0.5)

    wp.atomic_add(modal_w.cnt, i0, 1)
    wp.atomic_add(modal_w.cnt, i1, 1)   
    wp.atomic_add(modal_w.cnt, i2, 1)
    wp.atomic_add(modal_w.cnt, i3, 1)

@wp.func
def select(i: int, a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3) -> wp.vec3:
    ret = a
    if i == 1:
        ret = b
    elif i == 2:
        ret = c
    else:
        ret = d
    return ret

@wp.func
def skew(w: wp.vec3) -> wp.mat33:
    return wp.mat33(
        0.0, -w[2], w[1], 
        w[2], 0.0, -w[0], 
        -w[1], w[0], 0.0
    )
@wp.kernel
def compute_sparse_W(geo: FEMMesh, modal_w: ModalWarpingData, triplets: Triplet):
    tid = wp.tid()
    e = tid // 4
    j = tid % 4

    # grad_t0 = nabla_bf_tet(geo, e, 0) * 0.5
    # grad_t1 = nabla_bf_tet(geo, e, 1) * 0.5
    # grad_t2 = nabla_bf_tet(geo, e, 2) * 0.5
    # grad_t3 = nabla_bf_tet(geo, e, 3) * 0.5

    jj = geo.T[e, j]
    # w = select(j, grad_t0, grad_t1, grad_t2, grad_t3) / float(icnt)
    w = nabla_bf_tet(geo, e, j) * 0.5
    # w = grad_t3
    # w = wp.select(j == 0, wp.select(j == 1, wp.select(j == 2, grad_t3, grad_t2), grad_t1), grad_t0) / float(icnt)

    for i in range(4):
        ii = geo.T[e, i]
        icnt = modal_w.cnt[ii]
        value = wp.skew(w) / float(icnt)
        row = ii
        col = jj
        # value = wp.skew(select(j, grad_t0, grad_t1, grad_t2, grad_t3)) / float(icnt)

        triplet_id = 16 * e + j * 4 + i
        triplets.row[triplet_id] = row
        triplets.col[triplet_id] = col 
        triplets.values[triplet_id] = value

@wp.kernel
def compute_W(geo: FEMMesh, modal_w: ModalWarpingData):
    e = wp.tid()
    W_e(geo, e, modal_w)
        
@wp.kernel
def average_W(modal_w : ModalWarpingData):
    i = wp.tid()
    if modal_w.cnt[i] > 0:
        modal_w.W[i] /= float(modal_w.cnt[i])

@wp.kernel
def compute_Psi(geo: FEMMesh, modal_w: ModalWarpingData, Phi: wp.array(dtype = wp.vec3), q_k: float):
    e = wp.tid()
    i0 = geo.T[e, 0]
    i1 = geo.T[e, 1]
    i2 = geo.T[e, 2]
    i3 = geo.T[e, 3]

    dw = wp.cross(modal_w.W[i0], Phi[i0]) + wp.cross(modal_w.W[i1], Phi[i1]) + wp.cross(modal_w.W[i2], Phi[i2]) + wp.cross(modal_w.W[i3], Phi[i3])

    wp.atomic_add(modal_w.psi, i0, dw * q_k)
    wp.atomic_add(modal_w.psi, i1, dw * q_k)
    wp.atomic_add(modal_w.psi, i2, dw * q_k)
    wp.atomic_add(modal_w.psi, i3, dw * q_k)

@wp.kernel
def displace_u(modal_w: ModalWarpingData, Phi: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    w_k = modal_w.psi[i]
    norm_wk = wp.length(w_k)

    term1 = (1.0 - wp.cos(norm_wk)) / norm_wk
    term2 = (1.0 - wp.sin(norm_wk) / norm_wk)

    wkx = wp.skew(w_k / norm_wk)
    w_hat = wp.normalize(w_k)
    modal_w.u[i] = Phi[i] + wp.cross(w_hat, wp.cross(w_hat, Phi[i])) * term2 + wp.cross(w_hat, Phi[i]) * term1

@wp.kernel
def displace_u_new(modal_w: ModalWarpingData, Phi: wp.array(dtype = wp.vec3), q: float):
    i = wp.tid()
    w_k = modal_w.psi[i] * q
    norm_wk = wp.length(w_k)

    term1 = (1.0 - wp.cos(norm_wk)) / norm_wk
    term2 = (1.0 - wp.sin(norm_wk) / norm_wk)

    wkx = wp.skew(w_k / norm_wk)
    w_hat = wp.normalize(w_k)
    phii = Phi[i] * q
    modal_w.u[i] = phii + wp.cross(w_hat, wp.cross(w_hat, phii)) * term2 + wp.cross(w_hat, phii) * term1


class ModalWarpingRod(Rod):
    def __init__(self, filename = default_tobj):
        self.filename = filename
        super().__init__()

        W = wp.zeros((self.n_nodes), dtype = wp.vec3)
        cnt = wp.zeros((self.n_nodes), dtype = int)
        # R = wp.zeros((self.n_nodes), dtype = wp.mat33)
        psi = wp.zeros_like(W)
        u = wp.zeros_like(W)

        self.modal_w = ModalWarpingData()
        self.modal_w.W = W
        self.modal_w.cnt = cnt
        self.modal_w.psi = psi
        self.modal_w.u = u 
        # self.modal_w.R = R

        self.Phi = wp.zeros_like(W)



        self.compute_W()
        self.compute_Q()

        self.W = warp.sparse.bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        self.triplets = Triplet()
        rows = wp.zeros((self.n_tets * 4 * 4, ), dtype = int)
        cols = wp.zeros((self.n_tets * 4 * 4, ), dtype = int)
        vals = wp.zeros((self.n_tets * 4 * 4, ), dtype = wp.mat33)
        self.triplets.row = rows
        self.triplets.col = cols
        self.triplets.values = vals


        self.precompute_Psi()
        
    def compute_Q(self):
        self.define_M()
        self.lam, self.Q = self.eigs_sparse()

    def define_M(self):
        V = self.xcs.numpy()
        T = self.T.numpy()
        # self.M is a vector composed of diagonal elements 
        self.Mnp = igl.massmatrix(V, T, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal() * rho
        M_diag = np.repeat(self.Mnp, 3)
        self.M_sparse = diags(M_diag)

    def eigs_sparse(self): 
        update = False
        K = self.to_scipy_bsr()
        M = self.M_sparse
        print("start eigs")
        dim = K.shape[0]
        # if dim >= 3000:
        if True:
            self.eigs_export(K, M)
            print("dimension exceeds scipy capability, switching to matlab")
            with wp.ScopedTimer("matlab eigs"):
                if update or not os.path.exists(f"data/eigs_lma/Q_{model}.mat"):
                    import matlab.engine
                    eng = matlab.engine.start_matlab()
                    eng.lma(model)
                data = loadmat(f"data/eigs_lma/Q_{model}.mat")
                Q = data["Vv"].astype(np.float64)
                lam = data["D"].astype(np.float64)
        else: 
            with wp.ScopedTimer("weight space eigs"):
                lam, Q = eigsh(K, k = 10, which = "SM")
        return lam, Q

    def eigs_export(self, K, M):
        f = f"data/eigs_lma/{model}.mat"
        savemat(f, {"K": K, "M": M}, long_field_names= True)
        print(f"exported matrices to {f}")

    
    def compute_sparse_W(self):
        wp.launch(compute_sparse_W, dim = (self.n_tets * 4,), inputs = [self.geo, self.modal_w, self.triplets])
        warp.sparse.bsr_set_zero(self.W)
        warp.sparse.bsr_set_from_triplets(self.W, self.triplets.row, self.triplets.col, self.triplets.values)

    def compute_W(self):
        wp.launch(compute_W, dim = (self.n_tets,), inputs = [self.geo, self.modal_w])
        wp.launch(average_W, dim = (self.n_nodes,), inputs = [self.modal_w])
        # cnt = np.min(self.modal_w.cnt.numpy())
        # print("cnt min = ", cnt)
        # assert cnt > 0

    def precompute_Psi(self): 
        self.Psi = np.zeros_like(self.Q)
        for mode in range(self.Q.shape[1]):
            self.Psi[:, mode] = self.compute_Psi(mode)
        
        if save_Psi_only: 
            f = f"data/lma_weight/Psi_{model}.npy"
            np.save(f, self.Psi)
            print(f"saved Psi to {f}")
            quit()
            
    def compute_Psi(self, mode):
        Qi = self.Q[:, mode]

        # disp = Qi.reshape((-1, 3)) * q_k
        disp = Qi.reshape((-1, 3)) 

        self.Phi.assign(disp)

        # return self.Phi.numpy() * q_k
        self.modal_w.psi.zero_()
        
        self.compute_sparse_W()
        w = warp.sparse.bsr_mv(self.W, self.Phi)
        return w.numpy().reshape(-1)
    
    def compute_Psi_mix(self, u): 
        disp = u.reshape((-1, 3))   
        self.Phi.assign(disp)
        self.modal_w.psi.zero_()
        self.compute_sparse_W()
        w = warp.sparse.bsr_mv(self.W, self.Phi)
        return w.numpy().reshape(-1)

    def compute_displacement(self, mode, q_k):
        # wp.copy(self.modal_w.psi, w)   
        Qi = self.Q[:, mode]
        disp = Qi.reshape((-1, 3)) 
        self.Phi.assign(disp)    
        self.modal_w.psi.assign(self.Psi[:, mode].reshape((-1, 3)))
        wp.launch(displace_u_new, dim = (self.n_nodes, ), inputs = [self.modal_w, self.Phi, q_k])
        uprime = self.modal_w.u.numpy()
        return uprime

    def compute_displacement_mix(self, u):
        disp = u.reshape((-1, 3))   
        psi = self.compute_Psi_mix(u)
        self.Phi.assign(disp)
        self.modal_w.psi.assign(psi.reshape((-1, 3)))
        wp.launch(displace_u, dim = (self.n_nodes, ), inputs = [self.modal_w, self.Phi])
        uprime = self.modal_w.u.numpy()
        return uprime


class MWViewer:
    def __init__(self, rod: ModalWarpingRod):
        self.Q = rod.Q
        self.V0 = rod.V0
        self.F = rod.F
        self.magnitude = 1.0
        self.ps_mesh = ps.register_surface_mesh("rod", self.V0, self.F)

        self.ui_deformed_mode = 9

        self.ui_magnitude = 0.0
        self.qs = np.zeros((self.Q.shape[1], ))
        self.ui_use_modal_warping = True

        self.rod = rod
        self.V_deform = np.zeros_like(self.V0)

    def callback(self):
        self.control_panel()
        self.display()

    def control_panel(self):
        changed, self.ui_use_modal_warping = gui.Checkbox("Use Modal Warping", self.ui_use_modal_warping)


        self.ps_mesh.update_vertex_positions(self.V_deform)

        changed, self.ui_deformed_mode = gui.InputInt("#mode", self.ui_deformed_mode, step = 1)

        changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = -8.0, v_max = 8.0)
        self.qs[self.ui_deformed_mode] = self.ui_magnitude
        
        if gui.Button("save"): 
            mode = self.ui_deformed_mode

            # disp = self.rod.compute_displacement(self.ui_deformed_mode, self.ui_magnitude) if self.ui_use_modal_warping else (self.Q[:, self.ui_deformed_mode] * self.ui_magnitude).reshape((-1, 3))

            

            # V_deform = self.V0 + disp 
            V_deform = self.V_deform
            igl.write_obj(f"mode{mode}_pos.obj", V_deform, self.F)

            disp = self.rod.compute_displacement(self.ui_deformed_mode, -self.ui_magnitude) if self.ui_use_modal_warping else (self.Q[:, self.ui_deformed_mode] * -self.ui_magnitude).reshape((-1, 3))
            V_deform = self.V0 + disp 
            igl.write_obj(f"mode{mode}_neg.obj", V_deform, self.F)
    def display(self):
        # disp = self.rod.compute_Psi(self.ui_deformed_mode, self.ui_magnitude) if self.ui_use_modal_warping else (self.Q[:, self.ui_deformed_mode] * self.ui_magnitude).reshape((-1, 3))
        # disp = self.rod.compute_displacement(self.ui_deformed_mode, self.ui_magnitude) if self.ui_use_modal_warping else (self.Q[:, self.ui_deformed_mode] * self.ui_magnitude).reshape((-1, 3))
        blend = np.dot(self.Q, self.qs)
        disp = self.rod.compute_displacement_mix(blend) 

        
        self.V_deform = self.V0 + disp 

if __name__ == "__main__":
    wp.init()
    ps.init()
    ps.set_ground_plane_mode("none")

    rod = ModalWarpingRod(f"assets/{model}/{model}.tobj")

    viewer = MWViewer(rod)
    ps.set_user_callback(viewer.callback)

    ps.show()
    



