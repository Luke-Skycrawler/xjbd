import warp as wp 
from fem.params import FEMMesh
from fem.interface import Rod
from fem.params import *
import polyscope as ps
import polyscope.imgui as gui
import numpy as np
import warp.sparse
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
    

@wp.kernel
def compute_sparse_W(geo: FEMMesh, modal_w: ModalWarpingData, triplets: Triplet):
    tid = wp.tid()
    e = tid // 4
    j = tid % 4

    jj = geo.T[e, j]

    # same as nabla_bf_tet, use either 
    # inv_Dm = shape_func(geo, e)
    # grad_t0 = inv_Dm[0]
    # grad_t1 = inv_Dm[1]
    # grad_t2 = inv_Dm[2]
    # grad_t3 = -inv_Dm[0] - inv_Dm[1] - inv_Dm[2]    

    w = nabla_bf_tet(geo, e, j) * 0.5

    for i in range(4):
        ii = geo.T[e, i]
        icnt = modal_w.cnt[ii]
        value = wp.skew(w) / float(icnt)
        row = ii
        col = jj

        triplet_id = 16 * e + j * 4 + i
        triplets.row[triplet_id] = row
        triplets.col[triplet_id] = col 
        triplets.values[triplet_id] = value

@wp.kernel
def compute_cnt(geo: FEMMesh, modal_w: ModalWarpingData):
    e = wp.tid()
    i0 = geo.T[e, 0]
    i1 = geo.T[e, 1]
    i2 = geo.T[e, 2]
    i3 = geo.T[e, 3]

    wp.atomic_add(modal_w.cnt, i0, 1)
    wp.atomic_add(modal_w.cnt, i1, 1)
    wp.atomic_add(modal_w.cnt, i2, 1)
    wp.atomic_add(modal_w.cnt, i3, 1)

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



        self.compute_cnt()
        self.compute_Q()

        self.W = warp.sparse.bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        self.triplets = Triplet()
        rows = wp.zeros((self.n_tets * 4 * 4, ), dtype = int)
        cols = wp.zeros((self.n_tets * 4 * 4, ), dtype = int)
        vals = wp.zeros((self.n_tets * 4 * 4, ), dtype = wp.mat33)
        self.triplets.row = rows
        self.triplets.col = cols
        self.triplets.values = vals

        self.compute_sparse_W()

    def compute_Q(self):
        self.lam, self.Q = self.eigs_sparse()

    def compute_sparse_W(self):
        wp.launch(compute_sparse_W, dim = (self.n_tets * 4,), inputs = [self.geo, self.modal_w, self.triplets])
        warp.sparse.bsr_set_zero(self.W)
        warp.sparse.bsr_set_from_triplets(self.W, self.triplets.row, self.triplets.col, self.triplets.values)

    def compute_cnt(self):
        wp.launch(compute_cnt, dim = (self.n_tets, ), inputs = [self.geo, self.modal_w])

    def compute_Psi(self, mode, q_k):
        Qi = self.Q[:, mode]

        disp = Qi.reshape((-1, 3)) * q_k

        self.Phi.assign(disp)

        # return self.Phi.numpy() * q_k
        
        warp.sparse.bsr_mv(self.W, self.Phi, self.modal_w.psi)
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

        self.ui_deformed_mode = 6

        self.ui_magnitude = 2

        self.ui_use_modal_warping = True

        self.rod = rod

    def callback(self):
        changed, self.ui_use_modal_warping = gui.Checkbox("Use Modal Warping", self.ui_use_modal_warping)

        disp = self.rod.compute_Psi(self.ui_deformed_mode, self.ui_magnitude) if self.ui_use_modal_warping else (self.Q[:, self.ui_deformed_mode] * self.ui_magnitude).reshape((-1, 3))

        
        self.V_deform = self.V0 + disp 

        self.ps_mesh.update_vertex_positions(self.V_deform)

        changed, self.ui_deformed_mode = gui.InputInt("#mode", self.ui_deformed_mode, step = 1)

        changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = 0.0, v_max = 4)

if __name__ == "__main__":
    wp.init()
    ps.init()
    ps.set_ground_plane_mode("none")

    rod = ModalWarpingRod()

    viewer = MWViewer(rod)
    ps.set_user_callback(viewer.callback)

    ps.show()
    



