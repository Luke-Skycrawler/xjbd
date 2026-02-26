import numpy as np 
import warp as wp 
from .fem import Triplets
from warp.sparse import bsr_zeros, bsr_set_from_triplets, BsrMatrix, bsr_set_diag
from .geometry import TOBJComplex
from stretch import set_M_diag, NewtonState, PSViewer
import igl
from .params import *
import polyscope as ps

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
    auv[e] = wp.determinant(Dm)

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

    ae = auv[e]
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

    aii = -ae * (ksu * wp.outer(dwudpi * wu_unit, dwudpi * wu_unit) + ksv * wp.outer(dwvdpi * wv_unit, dwvdpi * wv_unit))
    aij = -ae * (ksu * wp.outer(dwudpi * wu_unit, dwudpj * wu_unit) + ksv * wp.outer(dwvdpi * wv_unit, dwvdpj * wv_unit))
    ajj = -ae * (ksu * wp.outer(dwudpj * wu_unit, dwudpj * wu_unit) + ksv * wp.outer(dwvdpj * wv_unit, dwvdpj * wv_unit))

    if Cu > 0.0: 

        commonu = 1.0 / wp.length(wu) * (wp.identity(3, float) - wp.outer(wu_unit, wu_unit)) * ksu * -ae @ Cu

        aii += commonu * dwudpi * dwudpi
        aij += commonu * dwudpi * dwudpj
        ajj += commonu * dwudpj * dwudpj

    if Cv > 0.0:
        commonv = 1.0 / wp.length(wv) * (wp.identity(3, float) - wp.outer(wv_unit, wv_unit)) * ksv * -ae @ Cv

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
    
    triplets.vals[e * 9 + 0] = aii 
    triplets.vals[e * 9 + 1] = aij
    triplets.vals[e * 9 + 2] = aik
    triplets.vals[e * 9 + 3] = aji
    triplets.vals[e * 9 + 4] = ajj
    triplets.vals[e * 9 + 5] = ajk
    triplets.vals[e * 9 + 6] = aki
    triplets.vals[e * 9 + 7] = akj
    triplets.vals[e * 9 + 8] = akk

    
    

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
        
        self.auv = wp.zeros((self.n_faces))
        self.define_K_sparse()

    def define_K_sparse(self): 
        self.compute_dudv()
        
        self.triplets = Triplets()
        self.triplets.rows = wp.zeros((self.n_faces * 3 * 3), dtype = int)
        self.triplets.cols = wp.zeros_like(self.triplets.rows)
        self.triplets.vals = wp.zeros((self.n_faces * 3 * 3), dtype = wp.mat33)

        self.face_kernel_sparse()
        self.K_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        bsr_set_from_triplets(self.K_sparse, self.triplets.rows, self.triplets.cols, self.triplets.vals)

    def compute_dudv(self):
        wp.launch(compute_dudv, dim = (self.n_nodes, ), inputs = [self.geo, self.dudv, self.auv])
    

    def face_kernel_sparse(self, x: wp.array = None): 
        if x is None:
            x = self.geo.xcs
        wp.launch(stretch_shear_kernel, dim = (self.n_faces,), inputs = [x, self.geo, self.dudv, self.auv, self.triplets, self.b])

    def drape_quasi_static(self):
        pass

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
        T = self.T.numpy()
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
    
class Shell(BW98ThinShellBase, TOBJComplex):
    def __init__(self, meshes_filename = [default_shell], transforms = [np.identity(4, dtype = float)]): 
        self.meshes_filename = meshes_filename
        self.transforms = transforms

        super().__init__() 

def test_quasi_static_drape(): 
    ps.init()
    shell = Shell()
    viewer = PSViewer() 
    shell.drape_quasi_static()
    ps.set_user_callback(viewer.callback)
    ps.show()

def test_shell_mesh(): 
    shell = Shell()
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
    test_shell_mesh()
    # test_quasi_static_drape()