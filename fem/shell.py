import warp as wp 
from .fem import SifakisFEM, Triplets, FEMMesh
from .geometry import TOBJComplex
from warp.sparse import bsr_zeros, bsr_set_from_triplets, bsr_set_zero, bsr_axpy, bsr_mv
import numpy as np 
from stretch import PSViewer, RodBCBase
from .params import *
import polyscope as ps 
import igl 
h = 1e-2
default_shell = "assets/shell/shell.obj"
ksu = 1e5
ksv = 1e5
kb = 1e2
h_fd = 1e-4
eps = 1e-4

'''
implements [1] with adaptations from [2] 
reference:
[1]: Large Steps in Cloth Simulation
[2]: Dynamic Deformables: Implementation and Production Practicalities, SIGGRAPH course 2022
[3]: A Quadratic Bending Model for Inextensible Surfaces
'''

@wp.struct 
class EdgeTopoloby: 
    ev: wp.array2d(dtype = int)
    ef: wp.array2d(dtype = int)


@wp.func 
def cot(c: float): 
    '''
    c: cosine of angle 
    '''
    s = wp.sqrt(1.0 - c * c)
    return c / s

@wp.func 
def cij(ei: wp.vec3, ej: wp.vec3): 
    '''
    cotangent of ei, ej sharing an vertex
    ei, ej points out from the shared vertex 
    '''
    ein = wp.normalize(ei)
    ejn = wp.normalize(ej)
    c = wp.dot(ein, ejn)
    return cot(c) 
    
@wp.kernel
def quadratic_bending(geo: FEMMesh, topo: EdgeTopoloby, triplets: Triplets, W: wp.array(dtype = float)):
    '''
    bending energy for edge eij = (vi, vj) shared by two triangles (vi, vj, vk) and (vj, vi, vl)
    E_bend = 1/2 x^T Q x

    refer to the picture in [3] for notation 
    '''
    ei = wp.tid()
    

    # find the two triangles sharing this edge 
    t0 = topo.ef[ei, 0]
    t1 = topo.ef[ei, 1]

    ii = topo.ev[ei, 0]
    jj = topo.ev[ei, 1]

    

    if t0 != -1 and t1 != -1:
        # find the opposite vertices 
        se = ii + jj 
        kk = geo.T[t0, 0] + geo.T[t0, 1] + geo.T[t0, 2] - se
        ll = geo.T[t1, 0] + geo.T[t1, 1] + geo.T[t1, 2] - se

        x0 = geo.xcs[ii]
        x1 = geo.xcs[jj]
        x2 = geo.xcs[kk]
        x3 = geo.xcs[ll]

        c03 = cij(x0 - x1, x2 - x1)
        c04 = cij(x0 - x1, x3 - x1)
        c01 = cij(x1 - x0, x2 - x0)
        c02 = cij(x1 - x0, x3 - x0)
        
        K0 = wp.vec4(c03 + c04, c01 + c02, -c01 - c03, -c02 - c04)
        i33 = wp.identity(3, dtype = float)
        idx = wp.vec4i(ii, jj, kk, ll)
        a0 = wp.abs(W[t0])
        a1 = wp.abs(W[t1])
        
        term = 1.5 / (a0 + a1)
        for _i in range(4):
            for _j in range(4):
                triplets.rows[ei * 16 + _i * 4 + _j] = idx[_i]
                triplets.cols[ei * 16 + _i * 4 + _j] = idx[_j]

                a = K0[_i] * K0[_j] * term * i33
                if should_fix(geo.xcs[idx[_i]]) or should_fix(geo.xcs[idx[_j]]):
                    a = wp.mat33(0.0)
                triplets.vals[ei * 16 + _i * 4 + _j] = a
    

@wp.kernel
def shell_kernel_sparse(x: wp.array(dtype = wp.vec3), geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float), triplets: Triplets, b: wp.array(dtype = wp.vec3)):
    '''
    E_stretch = 1/2 (ksu Cu^2 + ksv Cv^2) * area
    Cu, Cv = |wu| - 1, |wv| - 1
    fn = -auv [ksu (partial Cu/p pn) Cu + ksv (partial Cv /p pn) Cv]
    '''
    e = wp.tid()
    
    ii = geo.T[e, 0]
    ij = geo.T[e, 1]
    ik = geo.T[e, 2]
    

    dp1 = x[ij] - x[ii]
    dp2 = x[ik] - x[ii]
    Bme = Bm[e]

    dudv = wp.mat22(
        Bme[0, 0], Bme[0, 1],
        Bme[1, 0], Bme[1, 1]
    )

    wuwv = wp.transpose(wp.matrix_from_cols(dp1, dp2) @ dudv)
    wu = wuwv[0]
    wv = wuwv[1] 
    C_stretch = wp.vec2(wp.length(wu) - 1.0, wp.length(wv) - 1.0)
    J = W[e] * 2.0
    ae = wp.abs(W[e])

    Cu = C_stretch[0]
    Cv = C_stretch[1]
    
    wu_unit = wp.normalize(wu)
    wv_unit = wp.normalize(wv)
    C_shear = wp.dot(wu_unit, wv_unit)

    dwudpi = (geo.xcs[ij].z - geo.xcs[ik].z) / J
    dwvdpi = (geo.xcs[ik].x - geo.xcs[ij].x) / J
    dwudpj = (geo.xcs[ik].z - geo.xcs[ii].z) / J
    dwvdpj = (geo.xcs[ii].x - geo.xcs[ik].x) / J
    dwudpk = (geo.xcs[ii].z - geo.xcs[ij].z) / J
    dwvdpk = (geo.xcs[ij].x - geo.xcs[ii].x) / J
    
    uuT = wp.outer(wu_unit, wu_unit)
    vvT = wp.outer(wv_unit, wv_unit)
    shear_common_u = (wp.identity(3, float) - uuT) @ wv_unit / (Cu + 1.0)
    shear_common_v = (wp.identity(3, float) - vvT) @ wu_unit / (Cv + 1.0)

    fi = -ae * (ksu * dwudpi * wu_unit * Cu  + ksv * dwvdpi * wv_unit * Cv + C_shear * shear_common_u * dwudpi + C_shear * shear_common_v * dwvdpi)

    fj = -ae * (ksu * dwudpj * wu_unit * Cu  + ksv * dwvdpj * wv_unit * Cv + C_shear * shear_common_u * dwudpj + C_shear * shear_common_v * dwvdpj)

    fk = -(fi + fj)


    wp.atomic_add(b, ii, fi)
    wp.atomic_add(b, ij, fj)
    wp.atomic_add(b, ik, fk)


    aii = ae * (ksu * dwudpi * dwudpi * uuT + ksv * dwvdpi * dwvdpi * vvT)
    aij = ae * (ksu * dwudpi * dwudpj * uuT + ksv * dwvdpi * dwvdpj * vvT)
    ajj = ae * (ksu * dwudpj * dwudpj * uuT + ksv * dwvdpj * dwvdpj * vvT)

    if Cu >= 0.0: 
        commonu = Cu / (Cu + 1.0) * ksu * ae * (wp.identity(3, float) - uuT) 

        aii += commonu * dwudpi * dwudpi
        aij += commonu * dwudpi * dwudpj
        ajj += commonu * dwudpj * dwudpj

    if Cv >= 0.0:
        commonv = Cv / (Cv + 1.0) * ksv * ae * (wp.identity(3, float) - vvT) 

        aii += commonv * dwvdpi * dwvdpi
        aij += commonv * dwvdpi * dwvdpj
        ajj += commonv * dwvdpj * dwvdpj


    aik = -aii - aij
    aji = wp.transpose(aij)
    ajk = -aji - ajj
    aki = wp.transpose(aik)
    akj = wp.transpose(ajk)
    akk = -aki - akj

    cnt = e * 9
    for _i in range(3):
        for _j in range(3):
            triplets.cols[cnt + _i * 3 + _j] = geo.T[e, _i]
            triplets.rows[cnt + _i * 3 + _j] = geo.T[e, _j]

    triplets.vals[cnt + 0] = aii 
    triplets.vals[cnt + 1] = aij
    triplets.vals[cnt + 2] = aik
    triplets.vals[cnt + 3] = aji
    triplets.vals[cnt + 4] = ajj
    triplets.vals[cnt + 5] = ajk
    triplets.vals[cnt + 6] = aki
    triplets.vals[cnt + 7] = akj
    triplets.vals[cnt + 8] = akk

@wp.kernel
def compute_Dm(geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float)): 
    e = wp.tid()

    x0 = geo.xcs[geo.T[e, 0]]
    x1 = geo.xcs[geo.T[e, 1]]
    x2 = geo.xcs[geo.T[e, 2]]


    ui = x0[0]
    uj = x1[0]
    uk = x2[0]
    vi = x0[2]
    vj = x1[2]
    vk = x2[2]

    Dm = wp.matrix_from_rows(
        wp.vec2(uj - ui, uk - ui), 
        wp.vec2(vj - vi, vk - vi))

    inv_Dm = wp.inverse(Dm)
    Bm[e] = wp.matrix_from_rows(
        wp.vec3(inv_Dm[0, 0], inv_Dm[0, 1], 0.0),
        wp.vec3(inv_Dm[1, 0], inv_Dm[1, 1], 0.0),
        wp.vec3(0.0, 0.0, 1.)
    )
    W[e] = (wp.determinant(Dm) / 2.0)


class BW98ThinShell(SifakisFEM): 
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        # n_tets is actually n_triangles for shell 

    def compute_Dm(self): 
        '''
        compute the area
        '''
        wp.launch(compute_Dm, (self.n_tets, ), inputs = [self.geo, self.Bm, self.W])
        # self.W.fill_(1.0)        
        areas = self.W.numpy()

        print(f"areas: min = {areas.min()}, max = {areas.max()}")

    def tet_kernel_sparse(self):
        self.triplets = Triplets()
        self.triplets.rows = wp.zeros((self.n_tets * 3 * 3), dtype = int)
        self.triplets.cols = wp.zeros_like(self.triplets.rows)
        self.triplets.vals = wp.zeros((self.n_tets * 3 * 3), dtype = wp.mat33)

        wp.launch(shell_kernel_sparse, (self.n_tets,), inputs = [self.xcs, self.geo, self.Bm, self.W, self.triplets, self.b]) 
    
@wp.func 
def should_fix(x: wp.vec3): 
    p1 = wp.abs(x[0] + 0.5) < eps and wp.abs(x[2] + 0.5) < eps
    p2 = wp.abs(x[0] - 0.5) < eps and wp.abs(x[2] + 0.5) < eps
    return p1 or p2 

@wp.kernel
def set_b_fixed(geo: FEMMesh,b: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    # set fixed points rhs to 0
    if should_fix(geo.xcs[i]): 
        b[i] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def set_K_fixed(geo: FEMMesh, triplets: Triplets):
    eij = wp.tid()
    i = triplets.rows[eij]
    j = triplets.cols[eij]
    
    if should_fix(geo.xcs[i]) or should_fix(geo.xcs[j]):        
        if i == j:
            triplets.vals[eij] = wp.identity(3, dtype = float)
        else:
            triplets.vals[eij] = wp.mat33(0.0)

# @wp.kernel
# def set_Q_fixed(geo: FEMMesh, triplets: Triplets):
#     eij = wp.tid()
#     i = triplets.rows[eij]
#     j = triplets.cols[eij]
    
#     if should_fix(geo.xcs[i]) or should_fix(geo.xcs[j]):        
#         triplets.vals[eij] = wp.mat33(0.0)

class Shell(RodBCBase, BW98ThinShell, TOBJComplex):
    def __init__(self, h, meshes_filename = [default_shell], transforms = [np.identity(4, dtype = float)]): 
        self.meshes_filename = meshes_filename
        self.transforms = transforms

        super().__init__(h) 
        
        self.V = self.xcs.numpy()
        self.F = self.indices.numpy().reshape(-1, 3)
        self.define_bending_stiffness()

    def compute_K(self):
        self.triplets.vals.zero_()
        self.b.zero_()
        wp.launch(shell_kernel_sparse, (self.n_tets,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.triplets, self.b]) 
        # now self.b has the elastic forces

        self.set_bc_fixed_hessian()
        bsr_set_zero(self.K_sparse)
        bsr_set_from_triplets(self.K_sparse, self.triplets.rows, self.triplets.cols, self.triplets.vals) 
        bsr_axpy(self.Q, self.K_sparse, kb, beta = 1.0)       
    
    def set_bc_fixed_hessian(self):
        wp.launch(set_K_fixed, (self.n_tets * 3 * 3,), inputs = [self.geo, self.triplets])

    def compute_K_fd(self):
        # wp.launch(shell_kernel_sparse, (self.n_tets,), inputs = [self.xcs, self.geo, self.Bm, self.W, self.triplets, self.b]) 
        # b0 = self.b.numpy().copy()
        x0 = self.xcs.numpy().reshape(-1)
        K_fd = np.zeros((self.n_nodes * 3, self.n_nodes * 3), dtype = float)
        for i in range(9):
            ei = np.zeros((12,), dtype = float)
            ei[i + 3] = 1.0 
            x = x0 + h_fd * ei 
            self.states.x.assign(x.reshape(-1, 3)) 
            wp.launch(shell_kernel_sparse, (self.n_tets,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.triplets, self.b]) 
            bip = self.b.numpy()
            
            self.states.x.assign((x0 - h_fd * ei).reshape(-1, 3))
            wp.launch(shell_kernel_sparse, (self.n_tets,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.triplets, self.b]) 
            bim = self.b.numpy()
            Ki = bip - bim
            K_fd[:, i + 3] = Ki.reshape(-1)
        return -K_fd / (h_fd * 2.)
    
    def set_bc_fixed_grad(self): 
        wp.launch(set_b_fixed, (self.n_nodes,), inputs = [self.geo, self.b])

    def define_bending_stiffness(self):
        F = self.T.numpy()
        ev, fe, ef = igl.edge_topology(self.V, F)
        print(f"ev shape = {ev.shape}, fe shape = {fe.shape}, ef shape = {ef.shape}")
        topo = EdgeTopoloby() 
        topo.ev = wp.from_numpy(ev, dtype = int)
        topo.ef = wp.from_numpy(ef, dtype = int)
        
        n_edges = ef.shape[0]
        triplets = Triplets()
        triplets.rows = wp.zeros((n_edges * 4 * 4), dtype = int)
        triplets.cols = wp.zeros_like(triplets.rows)
        triplets.vals = wp.zeros((n_edges * 4 * 4), dtype = wp.mat33)
        
        wp.launch(quadratic_bending, (n_edges,), inputs = [self.geo, topo, triplets, self.W])
        self.Q = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        # wp.launch(set_Q_fixed, (n_edges * 4 * 4,), inputs = [self.geo, triplets])
        bsr_set_from_triplets(self.Q, triplets.rows, triplets.cols, triplets.vals)

        bb = wp.zeros_like(self.b)
        bsr_mv(self.Q, self.xcs, bb)
        print(f"bending force norm from rest shape = {np.linalg.norm(bb.numpy())}")
            
    def compute_rhs(self):
        bsr_mv(self.Q, self.states.x, self.b, alpha = -kb, beta = 1.0)
        super().compute_rhs()

def test_shell_mesh(): 
    shell = Shell(h)
    xcs = shell.xcs.numpy()
    # uu = shell.u.numpy()
    # uv = shell.v.numpy()

    # erru = xcs[:, 0] - (uu - 0.5) 
    # errv = xcs[:, 2] - (uv - 0.5)
    
    # print("erru", np.linalg.norm(erru))
    # print("errv", np.linalg.norm(errv))

    viewer = PSViewer(shell) 
    # K_fd = shell.compute_K_fd()
    # np.save("K_fd.npy", K_fd)


    # wp.copy(shell.states.x, shell.xcs)
    # shell.compute_K()
    # bsr_K = shell.to_scipy_bsr()
    # K = bsr_K.toarray() 
    # # set numpy print precision
    # np.set_printoptions(precision=1, suppress=True)
    # print(f"K_fd = {K_fd[3:, 3:]}")
    # print(f"K = {K[3:, 3:]}")

    
    ps.set_user_callback(viewer.callback)
    ps.set_ground_plane_mode("none")
    ps.show()


if __name__ == "__main__": 
    wp.config.max_unroll = 1
    wp.init()
    ps.init()
    test_shell_mesh()

