import warp as wp 
from warp.types import vector
from fem.params import * 
import numpy as np
from scipy.spatial.transform import Rotation as R
from fem.linear_elasticity import PK1 
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
def dHdx0(F: wp.mat33, Bm: wp.mat33, i: int): 
    dFdx0i = dFdx0(i, Bm, F)
    dPdx0i = wp.transpose(dPdF(F) @ dFdx0i)
    m0 = inv_vec(dPdx0i[0])
    m1 = inv_vec(dPdx0i[1])
    m2 = inv_vec(dPdx0i[2])
    Bm_inv = wp.inverse(Bm)
    Bmt_inv = wp.transpose(Bm_inv)

    H = compute_H(F, Bm_inv)
    f0, f1, f2 = dDsdx(i)
    return wp.matrix_from_cols(
        vec33((m0 - H @ wp.transpose(f0)) @  Bmt_inv),
        vec33((m1 - H @ wp.transpose(f1)) @  Bmt_inv),
        vec33((m2 - H @ wp.transpose(f2)) @  Bmt_inv)
    )    

@wp.func
def compute_H(F: wp.mat33, Bm_inv: wp.mat33):
    # treating We = -1 for now
    return PK1(F) @ wp.transpose(Bm_inv)

@wp.func 
def dHdx(F: wp.mat33, Bm_inv: wp.mat33, i: int): 
    '''
    H = -We P Bm^-T
    elastic forces are the columns of H
    treating We = -1 for now
    '''
    dFdxii = dFdx(i, Bm_inv)
    dPdxii = wp.transpose(dPdF(F) @ dFdxii)
    m0 = inv_vec(dPdxii[0])
    m1 = inv_vec(dPdxii[1])
    m2 = inv_vec(dPdxii[2])
    Bmt_inv = wp.transpose(Bm_inv)
    return wp.matrix_from_cols(
        vec33(m0 @ Bmt_inv),
        vec33(m1 @ Bmt_inv),
        vec33(m2 @ Bmt_inv)
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
def dDsdx(i:int): 
    z33 = wp.mat33(0.0)
    f0 = z33
    f1 = z33 
    f2 = z33 
    if i == 3: 
        o3 = wp.vec3(1.0)
        z3 = wp.vec3(0.0)
        f0 = -wp.matrix_from_rows(o3, z3, z3)
        f1 = -wp.matrix_from_rows(z3, o3, z3)
        f2 = -wp.matrix_from_rows(z3, z3, o3)
    else: 
        f0[0, i] = 1.0
        f1[1, i] = 1.0
        f2[2, i] = 1.0
    return f0, f1, f2

@wp.func 
def dFdx0(i:int, Bm: wp.mat33, F: wp.mat33): 
    '''
    vec(partial F / partial rest x), 9x3 matrix
    '''
    Bm_inv = wp.inverse(Bm)

    f0, f1, f2 = dDsdx(i)
    ret = wp.matrix_from_cols(
        vec33(-F @ f0 @ Bm_inv), 
        vec33(-F @ f1 @ Bm_inv),
        vec33(-F @ f2 @ Bm_inv)
    )
    return ret

@wp.func   
def dFdx(i: int, Bm_inv: wp.mat33): 
    '''
    vec(partial F / partial x_i), 9x3 matrix 
    '''
    f0, f1, f2 = dDsdx(i)
    ret = wp.matrix_from_cols(
        vec33(f0 @ Bm_inv),
        vec33(f1 @ Bm_inv),
        vec33(f2 @ Bm_inv)
    )
    return ret

@wp.func 
def def_grad(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, Bm_inv: wp.mat33):
    Ds = wp.matrix_from_cols(x0 - x3, x1 - x3, x2 - x3)
    return Ds @ Bm_inv

@wp.func 
def def_grad_rest(Dm: wp.mat33, x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3): 
    '''
    Dm: deformed shape
    x0, x1, x2, x3: rest shape
    '''
    Bm = wp.matrix_from_cols(x0 - x3, x1 - x3, x2 - x3)
    Bm_inv = wp.inverse(Bm)
    return Dm @ Bm_inv
    
@wp.kernel
def template_fd_F(x: wp.array(dtype = wp.vec3), dx: wp.array(dtype = wp.vec3), dF_ret: wp.array(dtype = wp.mat33), dF_fd_ret: wp.array(dtype = wp.mat33)): 
    Bm_inv = wp.identity(3, dtype = float)
    j = wp.tid()
    dFdxii = dFdx(0, Bm_inv) 
    dF = dFdxii @ dx[j] * h * 2.0
    x0 = x[j * 4 + 0]
    x1 = x[j * 4 + 1]
    x2 = x[j * 4 + 2]
    x3 = x[j * 4 + 3]
    
    dF_fd = def_grad(x0 + dx[j] * h, x1, x2, x3, Bm_inv) - def_grad(x0 - dx[j] * h, x1, x2, x3, Bm_inv)
    dF_fd_ret[j] = wp.transpose(dF_fd)
    dF_ret[j] = inv_vec(dF)

@wp.kernel
def template_fd_dFdx0(x: wp.array(dtype = wp.vec3), dx: wp.array(dtype = wp.vec3), dF_ret: wp.array(dtype = wp.mat33), dF_fd_ret: wp.array(dtype = wp.mat33)): 
    Dm = wp.identity(3, dtype = float)
    j = wp.tid()
    x0 = x[j * 4 + 0]
    x1 = x[j * 4 + 1]
    x2 = x[j * 4 + 2]
    x3 = x[j * 4 + 3]
    F = def_grad_rest(Dm, x0, x1, x2, x3)
    Bm = wp.matrix_from_cols(x0 - x3, x1 - x3, x2 - x3)

    dFdxii = dFdx0(0, Bm, F) 
    dF = dFdxii @ dx[j] * h * 2.0
    
    dF_fd = def_grad_rest(Dm, x0 + dx[j] * h, x1, x2, x3) - def_grad_rest(Dm, x0 - dx[j] * h, x1, x2, x3)
    dF_fd_ret[j] = wp.transpose(dF_fd)
    dF_ret[j] = inv_vec(dF)

@wp.kernel
def template_fd_P(x: wp.array(dtype = wp.vec3), dx: wp.array(dtype = wp.vec3), dP_ret: wp.array(dtype = wp.mat33), dP_fd_ret: wp.array(dtype = wp.mat33)): 
    Bm_inv = wp.identity(3, dtype = float)
    j = wp.tid()
    dFdxii = dFdx(0, Bm_inv) 
    
    x0 = x[j * 4 + 0]
    x1 = x[j * 4 + 1]
    x2 = x[j * 4 + 2]
    x3 = x[j * 4 + 3]
    
    F = def_grad(x0, x1, x2, x3, Bm_inv)
    dPdxii = dPdF(F) @ dFdxii 
    dP = dPdxii @ dx[j] * h * 2.0

    dP_fd = PK1(def_grad(x0 + dx[j] * h, x1, x2, x3, Bm_inv)) - PK1(def_grad(x0 - dx[j] * h, x1, x2, x3, Bm_inv))
    dP_fd_ret[j] = wp.transpose(dP_fd)
    dP_ret[j] = inv_vec(dP)

@wp.kernel
def template_fd_H(x: wp.array(dtype = wp.vec3), dx: wp.array(dtype = wp.vec3), dH_ret: wp.array(dtype = wp.mat33), dH_fd_ret: wp.array(dtype = wp.mat33)): 
    Bm_inv = wp.identity(3, dtype = float)
    j = wp.tid()
    
    x0 = x[j * 4 + 0]
    x1 = x[j * 4 + 1]
    x2 = x[j * 4 + 2]
    x3 = x[j * 4 + 3]
    
    F = def_grad(x0, x1, x2, x3, Bm_inv)

    dHdxii = dHdx(F, Bm_inv, 0) 
    dH = dHdxii @ dx[j] * h * 2.0
    
    Bmt_inv = wp.transpose(Bm_inv)
    dH_fd = PK1(def_grad(x0 + dx[j] * h, x1, x2, x3, Bm_inv)) @ Bmt_inv - PK1(def_grad(x0 - dx[j] * h, x1, x2, x3, Bm_inv)) @ Bmt_inv
    dH_fd_ret[j] = wp.transpose(dH_fd)
    dH_ret[j] = inv_vec(dH)

@wp.kernel
def template_fd_dHdx0(x: wp.array(dtype = wp.vec3), dx: wp.array(dtype = wp.vec3), dH_ret: wp.array(dtype = wp.mat33), dH_fd_ret: wp.array(dtype = wp.mat33)): 
    Dm = wp.identity(3, dtype = float)
    j = wp.tid()
    x0 = x[j * 4 + 0]
    x1 = x[j * 4 + 1]
    x2 = x[j * 4 + 2]
    x3 = x[j * 4 + 3]
    F = def_grad_rest(Dm, x0, x1, x2, x3)
    Bm = wp.matrix_from_cols(x0 - x3, x1 - x3, x2 - x3)
    Bm_minus = wp.matrix_from_cols(x0 - x3 - dx[j] * h, x1 - x3, x2 - x3)
    Bm_plus = wp.matrix_from_cols(x0 - x3 + dx[j] * h, x1 - x3, x2 - x3)
    dHdxii = dHdx0(F, Bm, 0) 
    dH = dHdxii @ dx[j] * h * 2.0
    
    dH_fd = compute_H(def_grad_rest(Dm, x0 + dx[j] * h, x1, x2, x3), wp.inverse(Bm_plus)) - compute_H(def_grad_rest(Dm, x0 - dx[j] * h, x1, x2, x3), wp.inverse(Bm_minus))
    dH_fd_ret[j] = wp.transpose(dH_fd)
    dH_ret[j] = inv_vec(dH)

def test_against_fd(test, n_tests = 10): 
    rng = np.random.default_rng(42)

    x34 = np.eye(4, 3, dtype = float)
    # generate random orthogonal matrix
    xnp = np.vstack([np.vstack([R.random(random_state=rng).as_matrix(), np.zeros((1, 3))]) for _ in range(n_tests)])

    xwp = wp.array(xnp, dtype = wp.vec3)
    dxnp = np.random.rand(n_tests, 3).astype(float)
    dxwp = wp.array(dxnp, dtype = wp.vec3)
    dF_ret = wp.zeros((n_tests, ), dtype = wp.mat33)
    dF_fd_ret = wp.zeros((n_tests, ), dtype = wp.mat33)
    if test == "dFdx": 
        wp.launch(template_fd_F, dim = (n_tests, ), inputs = [xwp, dxwp, dF_ret, dF_fd_ret])
    elif test == "dPdx": 
        wp.launch(template_fd_P, dim = (n_tests, ), inputs = [xwp, dxwp, dF_ret, dF_fd_ret])
    elif test == "dFdx0":
        wp.launch(template_fd_dFdx0, dim = (n_tests, ), inputs = [xwp, dxwp, dF_ret, dF_fd_ret])
    elif test == "dHdx": 
        wp.launch(template_fd_H, dim = (n_tests, ), inputs = [xwp, dxwp, dF_ret, dF_fd_ret])
    elif test == "dHdx0": 
        wp.launch(template_fd_dHdx0, dim = (n_tests, ), inputs = [xwp, dxwp, dF_ret, dF_fd_ret])
    dF_analytic = dF_ret.numpy()
    dF_fd = dF_fd_ret.numpy()
    if n_tests == 1:
        print("dF: ", dF_analytic)
        print("dF_fd: ", dF_fd)

    diff = np.linalg.norm(dF_analytic - dF_fd, axis =  (1, 2))
    dF_norm = np.linalg.norm(dF_analytic, axis = (1, 2))
    print("diff norm", diff)
    print("dF norm: ", dF_norm)
    
    return diff, dF_norm

if __name__ == "__main__":
    wp.config.max_unroll = 1
    wp.init()
    tests = ["dFdx", "dPdx", "dFdx0", "dHdx", "dHdx0"]
    for test in tests:
        diff, tot = test_against_fd(test)
        print(f"\n{test} relative error: {np.mean(diff / tot):.2e}\n")

