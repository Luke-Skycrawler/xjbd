import numpy as np 
import warp as wp 
from .fem import Triplets

ksu = 1e6
ksv = 1e6

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

    
    

class BW98ThinShell: 
    def __init__(self):
        super().__init__()
        
        self.b = wp.zeros((self.n_nodes, ), dtype = wp.vec3)
        self.dudv = wp.zeros((self.n_nodes, ), dtype = wp.mat22)
        self.geo = ThinShell()
        self.geo.xcs = self.xcs 
        self.geo.indices = self.indices

        self.define_K_sparse()

    def define_K_sparse(self): 
        pass 



