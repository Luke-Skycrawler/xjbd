import numpy as np 
import warp as wp 
from .fem import Triplets
from warp.sparse import bsr_zeros, bsr_set_from_triplets, BsrMatrix, bsr_set_diag
from warp.optim.linear import cg
from .geometry import TOBJComplex
from stretch import set_M_diag, NewtonState, PSViewer, compute_rhs, add_dx, update_x0_xdot
import igl
from .params import *
import polyscope as ps

h = 1e-2
eps = 1e-4
ksu = 1e6
ksv = 1e6
default_shell = "assets/shell/shell.obj"

'''
implements [1]
reference:
[1]: Large Steps in Cloth Simulation
[2]: Dynamic Deformables: Implementation and Production Practicalities, SIGGRAPH course 2022
[3]: A Quadratic Bending Model for Inextensible Surfaces
'''

@wp.struct 
class ThinShell: 
    xcs: wp.array(dtype = wp.vec3)
    indices: wp.array(dtype = int)
    u: wp.array(dtype = float)
    v: wp.array(dtype = float)

@wp.kernel
def compute_dudv(geo: ThinShell, dudv: wp.array(dtype = wp.mat22), auv: wp.array(dtype = float)):
    # triangle element
    e = wp.tid()
    i = geo.indices[e * 3 + 0]
    j = geo.indices[e * 3 + 1] 
    k = geo.indices[e * 3 + 2]
    
    du1 = geo.u[j] - geo.u[i]
    du2 = geo.u[k] - geo.u[i]
    
    dv1 = geo.v[j] - geo.v[i]
    dv2 = geo.v[k] - geo.v[i]

    Dm = wp.mat22(du1, du2, dv1, dv2)
    dudv[e] = wp.inverse(Dm)    
    auv[e] = wp.abs(wp.determinant(Dm))

@wp.kernel
def stretch_shear_kernel(x: wp.array(dtype = wp.vec3), geo: ThinShell, dudv: wp.array(dtype = wp.mat22), auv: wp.array(dtype = float), triplets: Triplets, b: wp.array(dtype = wp.vec3)):
    '''
    fn = -auv [ksu p Cu/p pn Cu + ksv p Cv /p pn Cv]
    '''

    e = wp.tid()
    
    ii = geo.indices[e * 3 + 0]
    ij = geo.indices[e * 3 + 1]
    ik = geo.indices[e * 3 + 2]
    

    dp1 = x[ij] - x[ii]
    dp2 = x[ik] - x[ii]
    
    wuwv = wp.transpose(wp.matrix_from_cols(dp1, dp2) @ dudv[e])
    wu = wuwv[0]
    wv = wuwv[1] 
    C_stretch = wp.vec2(wp.length(wu) - 1.0, wp.length(wv) - 1.0)
    Cu = C_stretch[0]
    Cv = C_stretch[1]
    
    wu_unit = wp.normalize(wu)
    wv_unit = wp.normalize(wv)

    # ae = auv[e]
    ae = 1.0

    dwudpi = (geo.v[ij] - geo.v[ik]) / ae
    dwvdpi = (geo.u[ik] - geo.u[ij]) / ae
    dwudpj = (geo.v[ik] - geo.v[ii]) / ae
    dwvdpj = (geo.u[ii] - geo.u[ik]) / ae
    dwudpk = (geo.v[ii] - geo.v[ij]) / ae
    dwvdpk = (geo.u[ij] - geo.u[ii]) / ae
    

    fi = -ae * (ksu * dwudpi * wu_unit * Cu  + ksv * dwvdpi * wv_unit * Cv)

    fj = -ae * (ksu * dwudpj * wu_unit * Cu  + ksv * dwvdpj * wv_unit * Cv)

    fk = -(fi + fj)


    wp.atomic_add(b, ii, fi)
    wp.atomic_add(b, ij, fj)
    wp.atomic_add(b, ik, fk)

    aii = ae * (ksu * wp.outer(dwudpi * wu_unit, dwudpi * wu_unit) + ksv * wp.outer(dwvdpi * wv_unit, dwvdpi * wv_unit))
    aij = ae * (ksu * wp.outer(dwudpi * wu_unit, dwudpj * wu_unit) + ksv * wp.outer(dwvdpi * wv_unit, dwvdpj * wv_unit))
    ajj = ae * (ksu * wp.outer(dwudpj * wu_unit, dwudpj * wu_unit) + ksv * wp.outer(dwvdpj * wv_unit, dwvdpj * wv_unit))

    if Cu >= 0.0: 
        commonu = 1.0 / wp.length(wu) * (wp.identity(3, float) - wp.outer(wu_unit, wu_unit)) * ksu * ae @ Cu

        aii += commonu * dwudpi * dwudpi
        aij += commonu * dwudpi * dwudpj
        ajj += commonu * dwudpj * dwudpj

    if Cv >= 0.0:
        commonv = 1.0 / wp.length(wv) * (wp.identity(3, float) - wp.outer(wv_unit, wv_unit)) * ksv * ae @ Cv

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
    for _j in range(3):
        for _i in range(3):
            triplets.rows[cnt + _i * 3 + _j] = geo.indices[e * 3 + _i]
            triplets.cols[cnt + _i * 3 + _j] = geo.indices[e * 3 + _j]

    triplets.vals[cnt + 0] = aii 
    triplets.vals[cnt + 1] = aij
    triplets.vals[cnt + 2] = aik
    triplets.vals[cnt + 3] = aji
    triplets.vals[cnt + 4] = ajj
    triplets.vals[cnt + 5] = ajk
    triplets.vals[cnt + 6] = aki
    triplets.vals[cnt + 7] = akj
    triplets.vals[cnt + 8] = akk

    
@wp.func 
def should_fix(x: wp.vec3): 
    p1 = wp.abs(x[0] + 0.5) < eps and wp.abs(x[2] + 0.5) < eps
    p2 = wp.abs(x[0] - 0.5) < eps and wp.abs(x[2] + 0.5) < eps
    return p1 or p2 


@wp.kernel
def _set_b_fixed(geo: ThinShell,b: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    # set fixed points rhs to 0
    if should_fix(geo.xcs[i]): 
        b[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _set_K_fixed(geo: ThinShell, triplets: Triplets):
    eij = wp.tid()
    e = eij // 9
    ii = (eij // 3) % 3
    jj = eij % 3

    i = geo.indices[e * 3 + ii]
    j = geo.indices[e * 3 + jj]
    
    if should_fix(geo.xcs[i]) or should_fix(geo.xcs[j]):        
        if ii == jj:
            triplets.vals[eij] = wp.identity(3, dtype = float)
        else:
            triplets.vals[eij] = wp.mat33(0.0)
            
class BW98ThinShellBase: 
    def __init__(self):
        '''
        The base class should have n_nodes, n_face, indices: wp.array, xcs: wp.array, and uv coordinates defined
        '''
        super().__init__()
        
        self.b = wp.zeros((self.n_nodes, ), dtype = wp.vec3)
        self.dudv = wp.zeros((self.n_faces, ), dtype = wp.mat22)
        self.geo = ThinShell()
        self.geo.xcs = self.xcs 
        self.geo.indices = self.indices
        self.geo.u = wp.zeros((self.n_nodes, ), dtype = float)
        self.geo.v = wp.zeros((self.n_nodes, ), dtype = float)
        
        self.auv = wp.zeros((self.n_faces, ), dtype = float)
        self.define_K_sparse()

    def define_K_sparse(self): 
        self.compute_dudv()
        # print("auv min = ", self.auv.numpy().min())
        auvnp = self.auv.numpy()        
        print(f"auv stats, min = {auvnp.min()}, max = {auvnp.max()}, mean = {auvnp.mean()}")
        self.auv.fill_(1.0)
        
        self.triplets = Triplets()
        self.triplets.rows = wp.zeros((self.n_faces * 3 * 3, ), dtype = int)
        self.triplets.cols = wp.zeros_like(self.triplets.rows)
        self.triplets.vals = wp.zeros((self.n_faces * 3 * 3, ), dtype = wp.mat33)

        self.face_kernel_sparse()
        self.K_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        bsr_set_from_triplets(self.K_sparse, self.triplets.rows, self.triplets.cols, self.triplets.vals)

    def compute_dudv(self):
        unp = self.u.numpy()
        vnp = self.v.numpy()
        print(f'examing indices: {self.indices.numpy()[:12]}')
        print(f"u stats: min = {unp.min()}, max = {unp.max()}, mean = {unp.mean()}")
        print(f"v stats: min = {vnp.min()}, max = {vnp.max()}, mean = {vnp.mean()}")
        wp.launch(compute_dudv, dim = (self.n_faces, ), inputs = [self.geo, self.dudv, self.auv])
    

    def face_kernel_sparse(self, x: wp.array = None): 
        if x is None:
            x = self.geo.xcs
        wp.launch(stretch_shear_kernel, dim = (self.n_faces,), inputs = [x, self.geo, self.dudv, self.auv, self.triplets, self.b])

class BW98ThinShellDynamic(BW98ThinShellBase):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.states = NewtonState()
        self.states.x = wp.zeros_like(self.geo.xcs)
        self.states.x0 = wp.zeros_like(self.geo.xcs)
        self.states.dx = wp.zeros_like(self.geo.xcs)
        self.states.xdot = wp.zeros_like(self.geo.xcs)
        self.states.Psi = wp.zeros((self.n_faces, ), dtype = float)

        self.reset()
        self.h = h
        self.define_M()

    def define_M(self):
        V = self.xcs.numpy()
        T = self.indices.numpy().reshape((-1, 3))
        # self.M is a vector composed of diagonal elements 
        self.Mnp = igl.massmatrix(V, T, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
        self.M = wp.zeros((self.n_nodes,), dtype = float)
        self.M.assign(self.Mnp * rho)

        self.M_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        M_diag = wp.zeros((self.n_nodes,), dtype = wp.mat33)
        wp.launch(set_M_diag, (self.n_nodes,), inputs = [self.M, M_diag])
        bsr_set_diag(self.M_sparse, M_diag)

    def reset(self):
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)

        self.frame = 0

    def line_search(self): 
        wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, alpha])
        return 1.0 

    def solve(self):
        self.states.dx.zero_()
        cg(self.K_sparse, self.b, self.states.dx, 1e-4, use_cuda_graph = True)



    def compute_A(self):
        self.face_kernel_sparse(self.states.x)
        self.set_bc_fixed_hessian()
        bsr_set_from_triplets(self.K_sparse, self.triplets.rows, self.triplets.cols, self.triplets.vals) 

    def set_bc_fixed_hessian(self):
        wp.launch(_set_K_fixed, dim = (self.n_faces * 9,), inputs = [self.geo, self.triplets])

    def compute_rhs(self): 
        wp.launch(compute_rhs, (self.n_nodes, ), inputs = [self.states, self.h, self.M, self.b])
        self.set_bc_fixed_grad()
        # self.b.zero_()

    def set_bc_fixed_grad(self): 
        wp.launch(_set_b_fixed, (self.n_nodes,), inputs = [self.geo, self.b])
    
    def step(self):
        newton_iter = True
        n_iter = 0
        max_iter = 8
        # while n_iter < max_iter:
        while newton_iter:
            self.compute_A()
            self.compute_rhs()

            self.solve()
            
            
            # line search stuff, not converged yet
            alpha = self.line_search()
            if alpha == 0.0:
                break

            dxnp = self.states.dx.numpy()
            norm_dx = np.linalg.norm(dxnp)
            newton_iter = norm_dx > 1e-6 and n_iter < max_iter
            print(f"norm = {np.linalg.norm(dxnp)}, {n_iter}")
            n_iter += 1
        self.update_x0_xdot()

    def update_x0_xdot(self):
        wp.launch(update_x0_xdot, dim = (self.n_nodes,), inputs = [self.states, self.h])

class Shell(BW98ThinShellDynamic, TOBJComplex):
    def __init__(self, h, meshes_filename = [default_shell], transforms = [np.identity(4, dtype = float)]): 
        self.meshes_filename = meshes_filename
        self.transforms = transforms

        super().__init__(h) 
        self.V = self.xcs.numpy()
        self.F = self.indices.numpy().reshape(-1, 3)

def test_quasi_static_drape(): 
    ps.init()
    shell = Shell(h)
    viewer = PSViewer(shell) 
    ps.set_user_callback(viewer.callback)
    ps.show()

def test_shell_mesh(): 
    shell = Shell(h)
    xcs = shell.xcs.numpy()
    uu = shell.u.numpy()
    uv = shell.v.numpy()

    erru = xcs[:, 0] - (uu - 0.5) 
    errv = xcs[:, 2] - (uv - 0.5)
    
    print("erru", np.linalg.norm(erru))
    print("errv", np.linalg.norm(errv))
    
if __name__ == "__main__": 
    wp.config.max_unroll = 1
    wp.init()
    # test_shell_mesh()
    test_quasi_static_drape()