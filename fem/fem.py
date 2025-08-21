import warp as wp 
import numpy as np 
# import cupy
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
# from cupy.linalg import eigh
import igl
from .params import *
from warp.sparse import bsr_axpy, bsr_set_from_triplets, bsr_zeros, BsrMatrix
from scipy.sparse import bsr_matrix

from .neo_hookean import PK1, tangent_stiffness, psi
from .linear_elasticity import PK1, tangent_stiffness, psi
# from .stvk import PK1, tangent_stiffness, psi

from .params import lam, mu
lam0, mu0 = lam, mu
lam1, mu1 = lam * 100.0, mu * 10.0
precompute = True
@wp.struct 
class Triplets:
    rows: wp.array(dtype = int)
    cols: wp.array(dtype = int)
    vals: wp.array(dtype = wp.mat33)

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

    

# still works, deprecated due to high compile time
# @wp.kernel
# def tet_kernel1(geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float), a: wp.array2d(dtype = float)):

#     e = wp.tid()
#     for _j in range(4):
#         t0 = geo.xcs[geo.T[e, 0]]
#         t1 = geo.xcs[geo.T[e, 1]]
#         t2 = geo.xcs[geo.T[e, 2]]
#         t3 = geo.xcs[geo.T[e, 3]]
        
#         Ds = wp.mat33(t0 - t3, t1 - t3, t2 - t3)
        
#         F = Ds @ Bm[e]
#         for k in range(3):
            
#             dDs = wp.mat33(0.0)
#             if _j < 3: 
#                 dDs[k, _j] = -1.0
#             else: 
#                 dDs[k, 0] = 1.0
#                 dDs[k, 1] = 1.0
#                 dDs[k, 2] = 1.0
#             dF = dDs @ Bm[e]
#             dP = tangent_stiffness(F, dF)
#             dH = -W[e] * dP @ wp.transpose(Bm[e])
            
#             for _i in range(4):
#                 i = geo.T[e, _i]
#                 j = geo.T[e, _j]
#                 df = wp.vec3(0.0)
#                 if _i == 3: 
#                     df= -wp.vec3(dH[0, 0] + dH[0, 1] + dH[0, 2], dH[1, 0] + dH[1, 1] + dH[1, 2], dH[2, 0] + dH[2, 1] + dH[2, 2])

#                 else:
#                     df = wp.vec3(dH[0, _i], dH[1, _i], dH[2, _i])
                
#                 for l in range(3):
#                     a[i * 3 + l, j * 3 + k] += df[l]

    
@wp.kernel
def tet_kernel(x: wp.array(dtype = wp.vec3), geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float), a: wp.array2d(dtype = float), b: wp.array(dtype = wp.vec3)):

    ej = wp.tid()
    e = ej // 16
    _j = ej % 4
    _i = (ej // 4) % 4
    t0 = x[geo.T[e, 0]]
    t1 = x[geo.T[e, 1]]
    t2 = x[geo.T[e, 2]]
    t3 = x[geo.T[e, 3]]
    
    Ds = wp.mat33(t0 - t3, t1 - t3, t2 - t3)
    
    F = Ds @ Bm[e]
        
    ll = lam0
    mm = mu0

    center = (t0 + t1 + t2 + t3) / 4.0

    # rest shape
    #####################################
    if precompute:
        xx = center[0]
        xy = center[2]
        if wp.sqrt(xx * xx + xy * xy) < 0.2:
            # inside skirt of the watermill
            ll = lam1 
            mm = mu1

    # transformed
    #####################################
    else: 
        xx = center[0] - 3.
        xy = center[2] + 2.
        if wp.sqrt(xx * xx + xy * xy) < 0.2 * 0.1:    
            ll = lam1
            mm = mu1

    P = PK1(F, ll, mm)
    H = -W[e] * P @ wp.transpose(Bm[e])

    # forces are columns of H
    # transpose H so that forces can be fetched with H[_i]
    H = wp.transpose(H)
    
    i = geo.T[e, _i]
    j = geo.T[e, _j]

    for k in range(3):
        
        dDs = wp.mat33(0.0)
        if _j < 3: 
            dDs[k, _j] = -1.0
        else: 
            dDs[k, 0] = 1.0
            dDs[k, 1] = 1.0
            dDs[k, 2] = 1.0

        dF = dDs @ Bm[e]
        dP = tangent_stiffness(F, dF, ll, mm)
        dH = -W[e] * dP @ wp.transpose(Bm[e])
        df = wp.vec3(0.0)
        if _i == 3: 
            df= -wp.vec3(dH[0, 0] + dH[0, 1] + dH[0, 2], dH[1, 0] + dH[1, 1] + dH[1, 2], dH[2, 0] + dH[2, 1] + dH[2, 2])

        else:
            df = wp.vec3(dH[0, _i], dH[1, _i], dH[2, _i])
        
        for l in range(3):
            a[i * 3 + l, j * 3 + k] += df[l]

    df = wp.vec3(0.0)
    if _i == 3: 
        df = -H[0] - H[1] - H[2]
    else: 
        df = H[_i]
    
    if _j == 0:
        wp.atomic_add(b, i, df)

@wp.kernel
def tet_kernel_sparse(x: wp.array(dtype = wp.vec3), geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float), triplets: Triplets, b: wp.array(dtype = wp.vec3)):

    ej = wp.tid()
    e = ej // 16
    _j = ej % 4
    _i = (ej // 4) % 4
    t0 = x[geo.T[e, 0]]
    t1 = x[geo.T[e, 1]]
    t2 = x[geo.T[e, 2]]
    t3 = x[geo.T[e, 3]]
    
    Ds = wp.mat33(t0 - t3, t1 - t3, t2 - t3)
    
    F = Ds @ Bm[e]
    
    a = wp.mat33(0.0)

    i = geo.T[e, _i]

    ll = lam0 
    mm = mu0

    center = (t0 + t1 + t2 + t3) / 4.0


    # rest shape
    #####################################
    if precompute:
        xx = center[0]
        xy = center[2]
        if wp.sqrt(xx * xx + xy * xy) < 0.2:
            # inside skirt of the watermill
            ll = lam1 
            mm = mu1

    # transformed
    #####################################
    else: 
        xx = center[0] - 3.
        xy = center[2] + 2.
        if wp.sqrt(xx * xx + xy * xy) < 0.2 * 0.1:    
            ll = lam1
            mm = mu1
        
    P = PK1(F, ll, mm)
    H = -W[e] * P @ wp.transpose(Bm[e])

    # forces are columns of H
    # transpose H so that forces can be fetched with H[_i]
    H = wp.transpose(H)
    
    j = geo.T[e, _j]
    for k in range(3):
        
        dDs = wp.mat33(0.0)
        if _j < 3: 
            dDs[k, _j] = -1.0
        else: 
            dDs[k, 0] = 1.0
            dDs[k, 1] = 1.0
            dDs[k, 2] = 1.0
        dF = dDs @ Bm[e]

        dP = tangent_stiffness(F, dF, ll, mm)
        # dP
        dH = -W[e] * dP @ wp.transpose(Bm[e])

        df = wp.vec3(0.0)
        if _i == 3: 
            df= -wp.vec3(dH[0, 0] + dH[0, 1] + dH[0, 2], dH[1, 0] + dH[1, 1] + dH[1, 2], dH[2, 0] + dH[2, 1] + dH[2, 2])

        else:
            df = wp.vec3(dH[0, _i], dH[1, _i], dH[2, _i])
        
        for l in range(3):
            a[l, k] += df[l]

    cnt = ej

    triplets.rows[cnt] = i
    triplets.cols[cnt] = j
    triplets.vals[cnt] = a

    fi = wp.vec3(0.0)
    if _i == 3: 
        fi = -H[0] - H[1] - H[2]
    else:
        fi = H[_i]

    if _j == 0:
        wp.atomic_add(b, i, fi)
                
class SifakisFEM:
    '''

    '''
    def __init__(self):
        super().__init__()
        n_unknowns = 3 * self.n_nodes
        self.b = wp.zeros((self.n_nodes, ), dtype = wp.vec3)
        self.Bm = wp.zeros((self.n_tets), dtype = wp.mat33)
        self.W = wp.zeros((self.n_tets), dtype = wp.float32)
        self.geo = FEMMesh()
        self.geo.n_nodes = self.n_nodes
        self.geo.n_tets = self.n_tets
        self.geo.xcs = self.xcs
        self.geo.T = self.T
        
        # use either
        # self.define_K()
        self.define_K_sparse()

    def define_K(self):
        self.compute_Dm()
        self.a.zero_()
        self.tet_kernel()
        self.K = self.a.numpy()
        
        M1 = igl.massmatrix(self.xcs.numpy(), self.T.numpy(), igl.MASSMATRIX_TYPE_BARYCENTRIC)
        M1 = M1.toarray()
        print("defining M: shape = ", M1.shape)
        mdiag = np.diag(M1)
        mdiag3 = np.repeat(mdiag, 3)
        self.M = np.diag(mdiag3)

    def tet_kernel_sparse(self):
        self.triplets = Triplets()
        self.triplets.rows = wp.zeros((self.n_tets * 4 * 4), dtype = int)
        self.triplets.cols = wp.zeros_like(self.triplets.rows)
        self.triplets.vals = wp.zeros((self.n_tets * 4 * 4), dtype = wp.mat33)

        wp.launch(tet_kernel_sparse, (self.n_tets * 4 * 4,), inputs = [self.xcs, self.geo, self.Bm, self.W, self.triplets, self.b]) 

    def define_K_sparse(self):
        self.compute_Dm()
        self.tet_kernel_sparse()
        self.K_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        
        bsr_set_from_triplets(self.K_sparse, self.triplets.rows, self.triplets.cols, self.triplets.vals)
        
        # self.K = self.to_scipy_bsr().toarray()
        

    def to_scipy_bsr(self, mat: BsrMatrix = None):
        if mat is None:
            mat = self.K_sparse
        ii = mat.offsets.numpy()
        jj = mat.columns.numpy()
        values = mat.values.numpy()
        shape = (mat.nrow * 3, mat.ncol * 3) 
        print(f"shape = {shape}, values = {values.shape}, ii = {ii.shape}, jj = {jj.shape}")
        bsr = bsr_matrix((values, jj, ii), shape = mat.shape, blocksize = (3 , 3))
        # return bsr.toarray()
        return bsr

    def compute_Dm(self):
        wp.launch(compute_Dm, (self.n_tets, ), inputs = [self.geo, self.Bm, self.W])


    def tet_kernel(self): 
        # deprecated tet_kernel1 
        # wp.launch(tet_kernel1, (self.n_tets,), inputs = [self.geo, self.Bm, self.W, self.a]) 


        wp.launch(tet_kernel_sparse, (self.n_tets * 4 * 4,), inputs = [self.xcs, self.geo, self.Bm, self.W, self.a, self.b]) 


    def eigs(self):
        # lam, Q = eigh(self.K, self.M)
        print("start eigs")
        lam, Q = eigh(self.K)
        return lam, Q
        # lam, Q = eigh(cupy.array(self.K))
        # return lam.get(), Q.get()
    
    def eigs_sparse(self):
        K = self.to_scipy_bsr()
        print("start eigs")
        
        lam, Q = eigsh(K, k = 10, which = "SM", tol = 1e-4)
        return lam, Q
