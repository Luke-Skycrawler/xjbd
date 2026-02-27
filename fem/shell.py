import warp as wp 
from .fem import SifakisFEM, Triplets, FEMMesh
from .geometry import TOBJComplex
from warp.sparse import bsr_zeros, bsr_set_from_triplets
import numpy as np 
from stretch import PSViewer, RodBCBase
from .params import *
import polyscope as ps 

h = 1e-2
default_shell = "assets/shell/shell.obj"
ksu = 1e3
ksv = 1e3

@wp.kernel
def shell_kernel_sparse(x: wp.array(dtype = wp.vec3), geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float), triplets: Triplets, b: wp.array(dtype = wp.vec3)):
    '''
    fn = -auv [ksu p Cu/p pn Cu + ksv p Cv /p pn Cv]
    '''

    e = wp.tid()
    
    ii = geo.indices[e * 3 + 0]
    ij = geo.indices[e * 3 + 1]
    ik = geo.indices[e * 3 + 2]
    

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
    Cu = C_stretch[0]
    Cv = C_stretch[1]
    
    wu_unit = wp.normalize(wu)
    wv_unit = wp.normalize(wv)

    # ae = auv[e]
    ae = 1.0

    dwudpi = (geo.xcs[ij].z - geo.xcs[ik].z) / ae
    dwvdpi = (geo.xcs[ik].x - geo.xcs[ij].x) / ae
    dwudpj = (geo.xcs[ik].z - geo.xcs[ii].z) / ae
    dwvdpj = (geo.xcs[ii].x - geo.xcs[ik].x) / ae
    dwudpk = (geo.xcs[ii].z - geo.xcs[ij].z) / ae
    dwvdpk = (geo.xcs[ij].x - geo.xcs[ii].x) / ae
    

    fi = -ae * (ksu * dwudpi * wu_unit * Cu  + ksv * dwvdpi * wv_unit * Cv)

    fj = -ae * (ksu * dwudpj * wu_unit * Cu  + ksv * dwvdpj * wv_unit * Cv)

    fk = -(fi + fj)


    wp.atomic_add(b, ii, fi)
    wp.atomic_add(b, ij, fj)
    wp.atomic_add(b, ik, fk)

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
    W[e] = wp.abs(wp.determinant(Dm)) / 2.0


class BW98ThinShell(SifakisFEM): 
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        # n_tets is actually n_triangles for shell 

    def compute_Dm(self): 
        '''
        compute the area
        '''
        wp.launch(compute_Dm, (self.n_tets, ), inputs = [self.geo, self.Bm, self.W])
        print(f"areas = {self.W.numpy()}")

    def tet_kernel_sparse(self):
        self.triplets = Triplets()
        self.triplets.rows = wp.zeros((self.n_tets * 3 * 3), dtype = int)
        self.triplets.cols = wp.zeros_like(self.triplets.rows)
        self.triplets.vals = wp.zeros((self.n_tets * 3 * 3), dtype = wp.mat33)

        wp.launch(shell_kernel_sparse, (self.n_tets * 3 * 3,), inputs = [self.xcs, self.geo, self.Bm, self.W, self.triplets, self.b]) 


class Shell(RodBCBase, BW98ThinShell, TOBJComplex):
    def __init__(self, h, meshes_filename = [default_shell], transforms = [np.identity(4, dtype = float)]): 
        self.meshes_filename = meshes_filename
        self.transforms = transforms

        super().__init__(h) 
        
        self.V = self.xcs.numpy()
        self.F = self.indices.numpy().reshape(-1, 3)


def test_shell_mesh(): 
    shell = Shell(h)
    xcs = shell.xcs.numpy()
    uu = shell.u.numpy()
    uv = shell.v.numpy()

    erru = xcs[:, 0] - (uu - 0.5) 
    errv = xcs[:, 2] - (uv - 0.5)
    
    # print("erru", np.linalg.norm(erru))
    # print("errv", np.linalg.norm(errv))

    viewer = PSViewer(shell) 
    ps.set_user_callback(viewer.callback)
    ps.show()


if __name__ == "__main__": 
    wp.config.max_unroll = 1
    wp.init()
    ps.init()
    test_shell_mesh()

