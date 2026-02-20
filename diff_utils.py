import warp as wp 
from warp.types import vector
from fem.params import * 
import numpy as np
from scipy.spatial.transform import Rotation as R

h = 1e-3
@wp.func 
def vec33(m: wp.mat33):
    return wp.vector(
        m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1], m[2, 2], length = 9, dtype = float
    )

@wp.func 
def inv_vec(v: vector(9, dtype = float)) -> wp.mat33:
    return wp.matrix(
        v[0], v[3], v[6],
        v[1], v[4], v[7],
        v[2], v[5], v[8],
        shape = (3, 3),
        dtype = float
    )
@wp.func 
def dPdF(F: wp.mat33):
    '''
    9x9 matrix of linear elasticity vec(partial P / partial F)
    P = 2 mu eps + lam tr(eps) I
    '''
    z33 = wp.mat33(0.0)
    i33 = wp.identity(3, dtype = float)
    f00 = z33
    f00[0, 0] = 2.0 
    f11 = z33
    f11[1, 1] = 2.0
    f22 = z33
    f22[2, 2] = 2.0

    f10 = z33
    f10[1, 0] = 1.0 
    f10[0, 1] = 1.0
    
    f20 = z33
    f20[2, 0] = 1.0 
    f20[0, 2] = 1.0

    f21 = z33 
    f21[2, 1] = 1.0
    f21[1, 2] = 1.0

    return wp.matrix_from_cols(
        vec33(f00 * mu + lam * i33), 
        vec33(f10 * mu), 
        vec33(f20 * mu), 
        vec33(f10 * mu),
        vec33(f11 * mu + lam * i33), 
        vec33(f21 * mu),
        vec33(f20 * mu),
        vec33(f21 * mu),
        vec33(f22 * mu + lam * i33)
    )


@wp.func   
def dFdx(i: int, Bm: wp.mat33): 
    '''
    vec(partial F / partial x_i), 9x3 matrix 
    '''
    z33 = wp.mat33(0.0)
    ret = wp.matrix(0.0, shape = (9, 3), dtype = float)
    if i == 3: 
        o3 = wp.vec3(1.0)
        z3 = wp.vec3(0.0)
        f03 = -wp.matrix_from_rows(o3, z3, z3)
        f13 = -wp.matrix_from_rows(z3, o3, z3)
        f23 = -wp.matrix_from_rows(z3, z3, o3)
        ret = wp.matrix_from_cols(
            vec33(f03 @ Bm),
            vec33(f13 @ Bm),
            vec33(f23 @ Bm)
        )
    else: 
        f0i = z33
        f0i[0, i] = 1.0 
        f1i = z33 
        f1i[1, i] = 1.0
        f2i = z33
        f2i[2, i] = 1.0

        ret = wp.matrix_from_cols(
            vec33(f0i @ Bm),
            vec33(f1i @ Bm),
            vec33(f2i @ Bm)
        )
    return ret

@wp.func 
def def_grad(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, Bm: wp.mat33):
    Ds = wp.matrix_from_cols(x0 - x3, x1 - x3, x2 - x3)
    return Ds @ Bm

@wp.kernel
def template_fd(x: wp.array(dtype = wp.vec3), dx: wp.array(dtype = wp.vec3), dF_ret: wp.array(dtype = wp.mat33), dF_fd_ret: wp.array(dtype = wp.mat33)): 
    Bm = wp.identity(3, dtype = float)
    j = wp.tid()
    dFdxii = dFdx(0, Bm) 
    dF = dFdxii @ dx[j] * h * 2.0
    x0 = x[j * 4 + 0]
    x1 = x[j * 4 + 1]
    x2 = x[j * 4 + 2]
    x3 = x[j * 4 + 3]
    
    dF_fd = def_grad(x0 + dx[j] * h, x1, x2, x3, Bm) - def_grad(x0 - dx[j] * h, x1, x2, x3, Bm)
    dF_fd_ret[j] = wp.transpose(dF_fd)
    dF_ret[j] = inv_vec(dF)

    
    

def test_against_fd(): 
    n_tests = 10
    rng = np.random.default_rng(42)

    x34 = np.eye(4, 3, dtype = float)
    # generate random orthogonal matrix
    xnp = np.vstack([np.vstack([R.random(random_state=rng).as_matrix(), np.zeros((1, 3))]) for _ in range(n_tests)])

    xwp = wp.array(xnp, dtype = wp.vec3)
    dxnp = np.random.rand(n_tests, 3).astype(float)
    dxwp = wp.array(dxnp, dtype = wp.vec3)
    dF_ret = wp.zeros((n_tests, ), dtype = wp.mat33)
    dF_fd_ret = wp.zeros((n_tests, ), dtype = wp.mat33)
    wp.launch(template_fd, dim = (n_tests, ), inputs = [xwp, dxwp, dF_ret, dF_fd_ret])
    dF_analytic = dF_ret.numpy()
    dF_fd = dF_fd_ret.numpy()
    print("dF: ", dF_analytic)
    print("dF_fd: ", dF_fd)
    print("diff norm", np.linalg.norm(dF_analytic - dF_fd, axis =  (1, 2)))
    print("dF norm: ", np.linalg.norm(dF_analytic, axis = (1, 2)))
    

if __name__ == "__main__":
    wp.config.max_unroll = 1
    wp.init()
    test_against_fd()

