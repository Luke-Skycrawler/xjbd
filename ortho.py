import warp as wp 
import numpy as np
from fem.params import mu
kappa = mu * 0.25
@wp.func
def hessian(F: wp.mat33):
    H = wp.matrix(
        shape = (12, 12), dtype = float
    )
    FT = wp.transpose(F)
    for i in range(3):
        for j in range(3):
            h = wp.mat33(0.0)
            qi = FT[i]
            qj = FT[j]

            if i == j:
                h = 2.0 * wp.outer(qi, qi) + wp.diag(wp.vec3(1.0, 1.0, 1.0) * (wp.dot(qi, qi) - 1.0))
                for k in range(3):
                    if k!= i:
                        qk = FT[k]
                        h += wp.outer(qk, qk)
            else: 
                h = wp.diag(wp.vec3(1.0, 1.0, 1.0)) * wp.dot(qj, qi) + wp.outer(qj, qi)
            
            h *= 4.0 * kappa
            for ii in range(3):
                for jj in range(3):
                    H[3 * i + ii, 3 * j + jj] = h[ii, jj]
    return H

@wp.func
def gradient(F: wp.mat33):
    ret = wp.vector(length = 12, dtype = float)
    FT = wp.transpose(F)
    for i in range(3):
        g = wp.vec3(0.0)
        for j in range(3):
            qi = FT[i]
            qj = FT[j]
            g += (wp.dot(qi, qj) - wp.select(i == j, 0.0, 1.0)) * qj
        g *= 4.0 * kappa
        for ii in range(3):
            ret[i * 3 + ii] = g[ii]

    return ret
            
@wp.kernel(enable_backward=False)
def test_hessian():
    F = wp.diag(wp.vec3(1.0, 1.0, 0.99))
    H = hessian(F)
    print(H)

@wp.kernel(enable_backward= False)
def fill_gh(g: wp.array(dtype = float), H: wp.array2d(dtype = float), F: wp.array(dtype = wp.mat33)):
    gg = gradient(F[0])
    hh = hessian(F[0])
    
    for ii in range(9):
        g[ii] = gg[ii]
        for jj in range(9):
            H[ii, jj] = hh[ii, jj]
        

class OrthogonalEnergy:
    def __init__(self):
        pass

    def analyze(self, ff):
        g = wp.zeros(12, float)
        h = wp.zeros((12, 12), float)
        F = wp.zeros((1, ), dtype = wp.mat33)

        F.assign(ff.reshape(1, 3, 3))
        wp.launch(fill_gh, (1), inputs = [g, h, F])
        gg = g.numpy().reshape(-1)
        hh = h.numpy()
        return gg, hh
    # def gradient(self, ff):
    #     g = wp.zeros(12, float)
    #     h = wp.zeros((12, 12), float)
    #     F = wp.zeros((1, ), dtype = wp.mat33)

    #     F.assign(ff.reshape(1, 3, 3))
    #     wp.launch(fill_gh, (1), inputs = [g, h, F])
    #     gg = g.numpy().reshape(-1)
    #     return gg
    #     # return wp.zeros((12), dtype = float)
    #     return np.zeros((12), dtype = float)
    
    # def hessian(self, ff):
    #     g = wp.zeros(12, float)
    #     h = wp.zeros((12, 12), float)
    #     # F = wp.zeros((1, ), dtype = wp.mat33)

    #     # F.assign(ff.reshape(1, 3, 3))

    #     F = wp.array(ff, dtype = wp.mat33)
    #     wp.launch(fill_gh, (1), inputs = [g, h, F])
    #     hh = h.numpy()
    #     return hh
    #     # return wp.zeros((12, 12), dtype = float)
    #     return np.zeros((12, 12), dtype = float)

    
if __name__ == "__main__":
    wp.config.max_unroll = 0
    wp.init()
    wp.launch(test_hessian, (1), )
            