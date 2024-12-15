import warp as wp 
import numpy as np 

from scipy.linalg import eigh, null_space
import igl
from .params import *


@wp.kernel
def compute_Dm(geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float)): 
    e = wp.tid()

    x0 = geo.xcs[geo.T[e, 0]]
    x1 = geo.xcs[geo.T[e, 1]]
    x2 = geo.xcs[geo.T[e, 2]]
    x3 = geo.xcs[geo.T[e, 3]]

    Dm = wp.mat33(x0 - x3, x1 - x3, x2 - x3)    
    inv_Dm = wp.inverse(Dm)
    Bm[e] = inv_Dm
    W[e] = wp.abs(wp.determinant(Dm)) / 6.0


    
@wp.func
def PK1(F: wp.mat33, dF: wp.mat33) -> wp.mat33:
    F_inv_T = wp.transpose(wp.inverse(F))
    B = wp.inverse(F) @ dF
    det_F = wp.determinant(F)
    
    return mu * dF + (mu - lam * wp.log(det_F)) * F_inv_T @ wp.transpose(dF) @ F_inv_T + (lam * wp.trace(B)) * F_inv_T 


@wp.kernel
def tet_kernel(geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float), a: wp.array2d(dtype = float)):

    e = wp.tid()
    for _j in range(4):
        t0 = geo.xcs[geo.T[e, 0]]
        t1 = geo.xcs[geo.T[e, 1]]
        t2 = geo.xcs[geo.T[e, 2]]
        t3 = geo.xcs[geo.T[e, 3]]
        
        Ds = wp.mat33(t0 - t3, t1 - t3, t2 - t3)
        
        F = Ds @ Bm[e]
        for k in range(3):
            
            dDs = wp.mat33(0.0)
            if _j < 3: 
                dDs[k, _j] = -1.0
            else: 
                dDs[k, 0] = 1.0
                dDs[k, 1] = 1.0
                dDs[k, 2] = 1.0
            dF = dDs @ Bm[e]
            dP = PK1(F, dF)
            dH = -W[e] * dP @ wp.transpose(Bm[e])
            
            for _i in range(4):
                i = geo.T[e, _i]
                j = geo.T[e, _j]
                df = wp.vec3(0.0)
                if _i == 3: 
                    df= -wp.vec3(dH[0, 0] + dH[0, 1] + dH[0, 2], dH[1, 0] + dH[1, 1] + dH[1, 2], dH[2, 0] + dH[2, 1] + dH[2, 2])

                else:
                    df = wp.vec3(dH[0, _i], dH[1, _i], dH[2, _i])
                
                for l in range(3):
                    a[i * 3 + l, j * 3 + k] += df[l]
class SifakisFEM:
    '''

    '''
    def __init__(self):
        super().__init__()
        n_unknowns = 3 * self.n_nodes
        self.a = wp.zeros((n_unknowns, n_unknowns), dtype = wp.float32)
        self.Bm = wp.zeros((self.n_tets), dtype = wp.mat33)
        self.W = wp.zeros((self.n_tets), dtype = wp.float32)
        self.geo = FEMMesh()
        self.geo.n_nodes = self.n_nodes
        self.geo.n_tets = self.n_tets
        self.geo.xcs = self.xcs
        self.geo.T = self.T
        
        self.define_K()

    def define_K(self):
        self.compute_Dm()
        self.a.zero_()
        self.tet_kernel()
        self.K = self.a.numpy()
        
        M1 = igl.massmatrix(self.xcs.numpy(), self.T.numpy(), igl.MASSMATRIX_TYPE_BARYCENTRIC)
        M1 = M1.toarray()
        print(M1.shape)
        mdiag = np.diag(M1)
        mdiag3 = np.repeat(mdiag, 3)
        self.M = np.diag(mdiag3)

    def compute_Dm(self):
        wp.launch(compute_Dm, (self.n_tets, ), inputs = [self.geo, self.Bm, self.W])


    def tet_kernel(self): 
        wp.launch(tet_kernel, (self.n_tets,), inputs = [self.geo, self.Bm, self.W, self.a]) 

    def eigs(self):
        lam, Q = eigh(self.K, self.M)
        return lam, Q
