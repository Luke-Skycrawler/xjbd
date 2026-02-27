import warp as wp 
from .fem import SifakisFEM, Triplets, FEMMesh
from .geometry import TOBJComplex
from warp.sparse import bsr_zeros, bsr_set_from_triplets, bsr_set_zero
import numpy as np 
from stretch import PSViewer, RodBCBase, should_fix
from .params import *
import polyscope as ps 

h = 1e-2
default_shell = "assets/shell/shell.obj"
ksu = 1e5
ksv = 1e5
h_fd = 1e-4

@wp.kernel
def shell_kernel_sparse(x: wp.array(dtype = wp.vec3), geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float), triplets: Triplets, b: wp.array(dtype = wp.vec3)):
    '''
    fn = -auv [ksu p Cu/p pn Cu + ksv p Cv /p pn Cv]
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

    dwudpi = (geo.xcs[ij].z - geo.xcs[ik].z) / J
    dwvdpi = (geo.xcs[ik].x - geo.xcs[ij].x) / J
    dwudpj = (geo.xcs[ik].z - geo.xcs[ii].z) / J
    dwvdpj = (geo.xcs[ii].x - geo.xcs[ik].x) / J
    dwudpk = (geo.xcs[ii].z - geo.xcs[ij].z) / J
    dwvdpk = (geo.xcs[ij].x - geo.xcs[ii].x) / J
    

    fi = -ae * (ksu * dwudpi * wu_unit * Cu  + ksv * dwvdpi * wv_unit * Cv)

    fj = -ae * (ksu * dwudpj * wu_unit * Cu  + ksv * dwvdpj * wv_unit * Cv)

    fk = -(fi + fj)


    wp.atomic_add(b, ii, fi)
    wp.atomic_add(b, ij, fj)
    wp.atomic_add(b, ik, fk)


    uuT = wp.outer(wu_unit, wu_unit)
    vvT = wp.outer(wv_unit, wv_unit)
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
    

@wp.kernel
def set_K_fixed(geo: FEMMesh, triplets: Triplets):
    eij = wp.tid()
    e = eij // 9
    ii = (eij // 3) % 3
    jj = eij % 3

    i = geo.T[e, ii]
    j = geo.T[e, jj]
    
    if should_fix(geo.xcs[i]) or should_fix(geo.xcs[j]):        
        if ii == jj:
            triplets.vals[eij] = wp.identity(3, dtype = float)
        else:
            triplets.vals[eij] = wp.mat33(0.0)

class Shell(RodBCBase, BW98ThinShell, TOBJComplex):
    def __init__(self, h, meshes_filename = [default_shell], transforms = [np.identity(4, dtype = float)]): 
        self.meshes_filename = meshes_filename
        self.transforms = transforms

        super().__init__(h) 
        
        self.V = self.xcs.numpy()
        self.F = self.indices.numpy().reshape(-1, 3)

    def compute_K(self):
        self.triplets.vals.zero_()
        self.b.zero_()
        wp.launch(shell_kernel_sparse, (self.n_tets,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.triplets, self.b]) 
        # now self.b has the elastic forces

        self.set_bc_fixed_hessian()
        bsr_set_zero(self.K_sparse)
        bsr_set_from_triplets(self.K_sparse, self.triplets.rows, self.triplets.cols, self.triplets.vals)        
    
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

